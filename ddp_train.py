import json
import os
import time
from pathlib import Path

import click
from tqdm import tqdm

from huggingface_hub import snapshot_download

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from safetensors.torch import load_file

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from checkpointing import get_checkpoints_dir, load_checkpoint, save_checkpoint
from gpt import GPTModel


class BigTrainDataset(Dataset):

    def __init__(self, all_tokens, seq_length, minibatch_size):
        self.xs = all_tokens[:-1].reshape(-1, minibatch_size, seq_length)
        self.ys = all_tokens[1:].reshape(-1, minibatch_size, seq_length)

    def __getitem__(self, ix):
        return (self.xs[ix], self.ys[ix])

    def __len__(self):
        return self.xs.shape[0]


def download_dataset(datasets_dir, dataset_name):
    download_path = snapshot_download(
        f"{dataset_name}",
        repo_type="dataset",
        local_dir=datasets_dir / dataset_name,
        allow_patterns="*"
    )
    return Path(download_path)


def load_dataset(
    dataset_dir, split,
    min_tokens, start_token,
    world_size, minibatch_size,
    seq_length
):
    full_dataset = load_file(dataset_dir / f"{split}.safetensors")["tokens"]
    if start_token > len(full_dataset):
        raise Exception(f"start_token {start_token} is past the end of the dataset")

    one_full_batch_tokens = world_size * minibatch_size * seq_length

    if min_tokens == -1:
        available_tokens = len(full_dataset) - start_token
        available_batches = (available_tokens // one_full_batch_tokens)
        tokens_needed = available_batches * one_full_batch_tokens
    else:
        batches_for_just_over_min = (min_tokens // one_full_batch_tokens) + 1
        tokens_needed = batches_for_just_over_min * one_full_batch_tokens

    # Note that we need one extra token for our Ys.
    tokens_needed += 1

    if len(full_dataset) < start_token + tokens_needed:
        raise Exception(f"Not enough tokens (wanted {start_token + tokens_needed}, got {len(full_dataset)})")

    return BigTrainDataset(
        full_dataset[start_token:start_token + tokens_needed],
        seq_length, minibatch_size,
    )


def get_training_data(run_dir):
    checkpoints_dir = get_checkpoints_dir(run_dir)

    train_losses = []
    val_losses = []
    best_global_step = None
    for item in checkpoints_dir.iterdir():
        if item.name == "latest":
            continue

        meta = json.loads((item / "meta.json").read_text())
        if item.name == "best":
            best_global_step = meta["global_step"]
            continue

        train_losses.append((meta["global_step"], meta["train_loss"]))
        val_losses.append((meta["global_step"], meta["val_loss"]))

    train_losses.sort(key=lambda x: x[0])
    val_losses.sort(key=lambda x: x[0])

    return train_losses, val_losses, best_global_step


def generate_training_chart(run_dir):
    train_points, val_points, best_global_step = get_training_data(run_dir)

    plt.title("TRAINING RUN LOSS")
    plt.xkcd()
    plt.rcParams['font.family'] = "xkcd"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    train_epochs, train_losses = zip(*train_points)
    val_epochs, val_losses = zip(*val_points)
    ax.plot(train_epochs, train_losses, label="TRAINING LOSS", marker="o")
    ax.plot(val_epochs, val_losses, label="VALIDATION LOSS", marker="s")

    ax.axvline(
        best_global_step, color="red", linestyle="--", linewidth=1.5,
        label="BEST GLOBAL STEP"
    )

    ax.set_title("TRAINING RUN LOSS")
    ax.set_xlabel("GLOBAL STEP")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("LOSS")
    ax.legend()

    fig.tight_layout()
    image_file = run_dir / "big-training-run-chart.png"
    fig.savefig(image_file, bbox_inches="tight")
    plt.close(fig)


def calculate_loss(logits, targets):
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten()
    )


def train(
    run_dir,
    model, optimizer, scaler,
    train_ds, val_ds,
    start_global_step, best_loss,
    validation_interval, validation_batches
):
    device = next(model.parameters()).device

    torch.set_float32_matmul_precision("high")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    total_global_steps = len(train_ds) // world_size

    print(f"Starting rank {rank} training at global step {start_global_step}")
    train_losses = []
    start_time = time.time()
    tokens_seen_this_rank = 0

    progress_bar = tqdm(
        range(start_global_step, total_global_steps),
        disable=(rank != 0)
    )
    for global_step in progress_bar:
        model.train()
        inputs, targets = train_ds[global_step * world_size + rank]
        inputs = inputs.to(device).to(torch.long)
        targets = targets.to(device).to(torch.long)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(inputs)

            train_loss = calculate_loss(logits, targets)

        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(train_loss.item())

        minibatch_size, sequence_length = inputs.shape
        tokens_seen_this_rank += minibatch_size * sequence_length

        if rank == 0:
            elapsed_time = time.time() - start_time
            tokens_per_sec = (tokens_seen_this_rank * world_size) / elapsed_time
            progress_bar.set_postfix(
                loss=f"{train_loss.item():.3f}",
                tps=f"{tokens_per_sec:,.0f}"
            )


        is_eval_iter = (
            (global_step % validation_interval == 0)
            or (global_step == total_global_steps - 1)
        )
        if is_eval_iter:
            dist.barrier()

            if rank == 0:
                print("\n\n\nValidation/checkpoint")
                model.eval()

                base_model = model.module
                with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    val_losses = []
                    for val_ix in tqdm(range(validation_batches)):
                        val_inputs, val_targets = val_ds[val_ix]
                        val_inputs = val_inputs.to(device).to(torch.long)
                        val_targets = val_targets.to(device).to(torch.long)
                        val_logits = base_model(val_inputs)
                        val_losses.append(
                            calculate_loss(val_logits, val_targets).item()
                        )
                    val_loss = sum(val_losses) / len(val_losses)

                if best_loss is None or val_loss < best_loss:
                    is_best = True
                    best_loss = val_loss
                else:
                    is_best = False

                avg_train_loss = sum(train_losses) / len(train_losses)
                train_losses = []

                save_checkpoint(
                    run_dir,
                    f"iteration-{global_step}",
                    base_model, optimizer, scaler,
                    avg_train_loss, val_loss,
                    global_step,
                    is_best
                )
                generate_training_chart(run_dir)

                model.train()
                print("\nContinuing training")

            dist.barrier()

    end_time = time.time()
    elapsed_time = end_time - start_time

    if rank == 0:
        print(f"\n\n\nTraining complete in {elapsed_time:,.3f} seconds")
        total_tokens_seen = tokens_seen_this_rank * world_size
        print(f"Tokens seen: {total_tokens_seen:,.0f}")
        print(f"Throughput: {total_tokens_seen / elapsed_time:,.0f} tokens/second")
        print(f"Final train loss: {avg_train_loss:.3f}")
        print(f"Final val loss: {val_loss:.3f}")


@click.command()
@click.argument("run")
@click.argument("datasets_dir_path")
@click.argument("checkpoint", default=None)
def main(run, datasets_dir_path, checkpoint):
    run_dir = Path(__file__).resolve().parent / "runs" / run
    if not run_dir.is_dir():
        raise Exception(f"Could not find run dir {run_dir}")

    model_conf_file = run_dir / "model.json"
    if not model_conf_file.is_file():
        raise Exception(f"Could not find model config in {model_conf_file}")
    with open(model_conf_file, "r") as f:
        model_conf = json.load(f)

    train_conf_file = run_dir / "train.json"
    if not train_conf_file.is_file():
        raise Exception(f"Could not find train config in {train_conf_file}")
    with open(train_conf_file, "r") as f:
        train_conf = json.load(f)

    # Which of the one-per-GPU processes are we on this machine?
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set ourselves up to use the GPU with the ID that matches our local rank
    torch.accelerator.set_device_index(local_rank)

    # Get the accelerator object associated with that GPU,
    # and the associated backend object (eg. `nccl` for CUDA):
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)

    # Initialize torch.distributed; set the device ID explicitly
    # to avoid warnings in `dist.barrier`
    dist.init_process_group(backend, device_id=local_rank)

    model = GPTModel(model_conf).to(local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )

    scaler = torch.amp.GradScaler()

    datasets_dir = Path(datasets_dir_path)
    if not datasets_dir.is_dir():
        raise Exception(f"{datasets_dir_path} is not a directory")
    dataset_dir = download_dataset(datasets_dir, train_conf["dataset"])

    train_ds = load_dataset(
        dataset_dir, "train",
        train_conf["min_train_tokens"], train_conf["start_train_token"],
        dist.get_world_size(), train_conf["minibatch_size"],
        model_conf["context_length"]
    )
    val_ds = load_dataset(
        dataset_dir, "validation",
        -1, train_conf["start_val_token"],
        dist.get_world_size(), train_conf["minibatch_size"],
        model_conf["context_length"]
    )

    if checkpoint:
        global_step, best_loss = load_checkpoint(
            run_dir, checkpoint, model, optimizer, scaler
        )
    else:
        global_step = 0
        best_loss = None

    ddp_model = DDP(model, device_ids=[local_rank])

    train(
        run_dir,
        ddp_model, optimizer, scaler,
        train_ds, val_ds,
        global_step, best_loss,
        train_conf["validation_interval"], train_conf["validation_batches"],
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
