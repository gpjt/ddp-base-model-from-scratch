import json
import os
import random
import time
from pathlib import Path

import click
from tqdm import tqdm

from huggingface_hub import snapshot_download

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

from safetensors.torch import load_file

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from checkpointing import get_checkpoints_dir, load_checkpoint, save_checkpoint
from gpt import GPTModel


class BigTrainDataset(Dataset):

    def __init__(self, all_tokens, seq_length, microbatch_size):
        self.xs = all_tokens[:-1].reshape(-1, microbatch_size, seq_length)
        self.ys = all_tokens[1:].reshape(-1, microbatch_size, seq_length)

    def __getitem__(self, ix):
        return (self.xs[ix], self.ys[ix])

    def __len__(self):
        return self.xs.shape[0]


def download_dataset(dataset_dir, dataset_name):
    snapshot_download(
        f"{dataset_name}",
        repo_type="dataset",
        local_dir=dataset_dir,
        allow_patterns="*"
    )


def load_dataset(
    dataset_dir, split,
    min_tokens, start_token,
    world_size, microbatch_size,
    seq_length
):
    full_dataset = load_file(dataset_dir / f"{split}.safetensors")["tokens"]
    if start_token > len(full_dataset):
        raise Exception(f"start_token {start_token} is past the end of the dataset")

    one_full_batch_tokens = world_size * microbatch_size * seq_length

    if min_tokens == -1:
        available_tokens = len(full_dataset) - start_token
        available_batches = (available_tokens // one_full_batch_tokens)
        tokens_needed = available_batches * one_full_batch_tokens
    else:
        if min_tokens % one_full_batch_tokens == 0:
            tokens_needed = min_tokens
        else:
            batches_for_just_over_min = (min_tokens // one_full_batch_tokens) + 1
            tokens_needed = batches_for_just_over_min * one_full_batch_tokens

    # Note that we need one extra token for our Ys.
    tokens_needed += 1

    if len(full_dataset) < start_token + tokens_needed:
        raise Exception(f"Not enough tokens (wanted {start_token + tokens_needed}, got {len(full_dataset)})")

    return BigTrainDataset(
        full_dataset[start_token:start_token + tokens_needed],
        seq_length, microbatch_size,
    )


def get_training_data(run_dir):
    checkpoints_dir = get_checkpoints_dir(run_dir)

    min_train_losses = []
    max_train_losses = []
    avg_train_losses = []
    max_grad_norms = []
    avg_grad_norms = []
    frac_clipped = []
    best_global_step = None
    for item in checkpoints_dir.iterdir():
        if item.name == "latest":
            continue

        meta = json.loads((item / "meta.json").read_text())
        if item.name == "best":
            best_global_step = meta["global_step"]
            continue

        min_train_losses.append((meta["global_step"], meta["min_train_loss"]))
        max_train_losses.append((meta["global_step"], meta["max_train_loss"]))
        avg_train_losses.append((meta["global_step"], meta["avg_train_loss"]))

        if meta.get("max_grad_norms") is not None:
            max_grad_norms.append((meta["global_step"], meta["max_grad_norms"]))
        if meta.get("avg_grad_norms") is not None:
            avg_grad_norms.append((meta["global_step"], meta["avg_grad_norms"]))
        if meta.get("frac_clipped") is not None:
            frac_clipped.append((meta["global_step"], meta["frac_clipped"]))
            

    min_train_losses.sort(key=lambda x: x[0])
    max_train_losses.sort(key=lambda x: x[0])
    avg_train_losses.sort(key=lambda x: x[0])
    max_grad_norms.sort(key=lambda x: x[0])
    avg_grad_norms.sort(key=lambda x: x[0])
    frac_clipped.sort(key=lambda x: x[0])

    return (
        min_train_losses, max_train_losses, avg_train_losses, 
        max_grad_norms, avg_grad_norms, frac_clipped,
        best_global_step
    )
        

def generate_training_chart(run_dir):
    (
        min_train_points, max_train_points, avg_train_points, 
        max_grad_norms, avg_grad_norms, frac_clipped,
        best_global_step
    ) = get_training_data(run_dir)

    plt.xkcd()

    font_family = None
    for f in font_manager.fontManager.ttflist:
        if "xkcd" in f.name.lower():
            font_family = f.name
            break

    if font_family is not None:
        plt.rcParams['font.family'] = font_family

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    train_epochs, min_train_losses = zip(*min_train_points)
    _, max_train_losses = zip(*max_train_points)
    _, avg_train_losses = zip(*avg_train_points)

    ax.fill_between(
        train_epochs,
        min_train_losses,
        max_train_losses,
        color="lightblue",
        alpha=0.25,
        label="MIN–MAX RANGE",
    )

    ax.plot(train_epochs, avg_train_losses, label="AVG TRAINING LOSS", marker="o")

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
    clipping_max_norm,
    train_ds,
    start_global_step, best_loss,
    checkpoint_interval,
    do_checkpoints=True,
    max_steps=None,
):
    device = next(model.parameters()).device

    torch.set_float32_matmul_precision("high")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if max_steps is not None:
        total_global_steps = max_steps
    else:
        total_global_steps = len(train_ds) // world_size

    print(f"Starting rank {rank} training at global step {start_global_step}")
    train_losses = []
    grad_norms = []
    clipped_steps = []
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

        if clipping_max_norm is not None:
            scaler.unscale_(optimizer)
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_max_norm).item()
            grad_norms.append(pre_clip_norm)
            clipped_steps.append(pre_clip_norm > clipping_max_norm)

        scaler.step(optimizer)
        scaler.update()
        train_losses.append(train_loss.item())

        microbatch_size, sequence_length = inputs.shape
        tokens_seen_this_rank += microbatch_size * sequence_length

        if rank == 0:
            elapsed_time = time.time() - start_time
            tokens_per_sec = (tokens_seen_this_rank * world_size) / elapsed_time
            progress_bar.set_postfix(
                loss=f"{train_loss.item():.3f}",
                tps=f"{tokens_per_sec:,.0f}"
            )


        is_checkpoint_iter = do_checkpoints and (
            (global_step % checkpoint_interval == 0)
            or (global_step == total_global_steps - 1)
        )
        if is_checkpoint_iter:
            dist.barrier()

            if rank == 0:
                print("\n\n\nCheckpoint")
                base_model = model.module
            
                min_train_loss = min(train_losses)
                max_train_loss = max(train_losses)
                avg_train_loss = sum(train_losses) / len(train_losses)
                train_losses = []

                if best_loss is None or avg_train_loss < best_loss:
                    is_best = True
                    best_loss = avg_train_loss
                else:
                    is_best = False

                if clipping_max_norm is not None:
                    max_grad_norms = max(grad_norms)
                    avg_grad_norms = sum(grad_norms) / len(grad_norms)
                    frac_clipped = sum(b for b in clipped_steps if b) / len(clipped_steps)
                else:
                    max_grad_norms = None
                    avg_grad_norms = None
                    frac_clipped = None
                grad_norms = []
                clipped_steps = []

                save_checkpoint(
                    run_dir,
                    f"iteration-{global_step}",
                    base_model, optimizer, scaler,
                    min_train_loss, max_train_loss, avg_train_loss,
                    max_grad_norms, avg_grad_norms, frac_clipped,
                    global_step,
                    is_best
                )
                generate_training_chart(run_dir)

                model.train()
                print("\nContinuing training")

            dist.barrier()

    end_time = time.time()
    elapsed_time = end_time - start_time

    if do_checkpoints and rank == 0:
        print(f"\n\n\nTraining complete in {elapsed_time:,.3f} seconds")
        total_tokens_seen = tokens_seen_this_rank * world_size
        print(f"Tokens seen: {total_tokens_seen:,.0f}")
        print(f"Throughput: {total_tokens_seen / elapsed_time:,.0f} tokens/second")
        print(f"Final train loss: {avg_train_loss:.3f}")


def check_batch_size_works(
    batch_size,
    run_dir, dataset_dir,
    model, optimizer, scaler,
    train_conf, model_conf
):
    if dist.get_rank() == 0:
        print(f"Trying to train with batch size {batch_size}")
    train_ds = load_dataset(
        dataset_dir, "train",
        train_conf["min_train_tokens"], train_conf["start_train_token"],
        dist.get_world_size(), batch_size,
        model_conf["context_length"]
    )

    try:
        train(
            run_dir,
            model, optimizer, scaler,
            None,
            train_ds,
            start_global_step=0, best_loss=None,
            checkpoint_interval=None,
            do_checkpoints=False, max_steps=3
        )
        if dist.get_rank() == 0:
            print(f"Batch size {batch_size} worked OK")
        return True
    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise
        torch.cuda.empty_cache()
        if dist.get_rank() == 0:
            print(f"Batch size {batch_size} OOMed")
        return False


def binary_chop_batch_sizes(
    run_dir, dataset_dir,
    model, optimizer, scaler, local_rank,
    train_conf, model_conf
):
    ddp_model = DDP(model, device_ids=[local_rank])

    def _check_batch_size_works(batch_size):
        return check_batch_size_works(
            batch_size, run_dir, dataset_dir,
            ddp_model, optimizer, scaler,
            train_conf, model_conf
        )

    smallest_fail = 70
    if _check_batch_size_works(smallest_fail):
        raise Exception(
            f"Batch size of {smallest_fail} worked!  Nice machine, "
            "but you're going to have to alter the code."
        )

    largest_win = 1
    if not _check_batch_size_works(largest_win):
        raise Exception(
            f"Batch size of {largest_win} didn't work :-(.  "
            "Gonna need a bigger box"
        )

    while smallest_fail - largest_win > 1:
        midpoint = largest_win + ((smallest_fail - largest_win) // 2)
        if _check_batch_size_works(midpoint):
            largest_win = midpoint
        else:
            smallest_fail = midpoint

    return largest_win


def load_datasets_and_train(
    run_dir,
    model, optimizer, scaler, local_rank,
    dataset_dir,
    train_conf, model_conf,
    checkpoint,
):
    train_ds = load_dataset(
        dataset_dir, "train",
        train_conf["min_train_tokens"], train_conf["start_train_token"],
        dist.get_world_size(), train_conf["microbatch_size"],
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
        train_conf.get("clipping_max_norm"),
        train_ds,
        global_step, best_loss,
        checkpoint_interval=train_conf["checkpoint_interval"],
        do_checkpoints=True,
    )


@click.command()
@click.argument("run")
@click.argument("datasets_dir_path")
@click.argument("checkpoint", default=None)
@click.option("--find-max-microbatch-size", "-f", is_flag=True)
def main(run, datasets_dir_path, checkpoint, find_max_microbatch_size):
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

    # Set all of the random seeds
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = GPTModel(model_conf).to(local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )

    scaler = torch.amp.GradScaler()

    datasets_dir = Path(datasets_dir_path)
    dataset_name = train_conf["dataset"]
    dataset_dir = datasets_dir / dataset_name
    if local_rank == 0:
        if not datasets_dir.exists():
            datasets_dir.mkdir()
        if not datasets_dir.is_dir():
            raise Exception(f"{datasets_dir_path} is not a directory")
        download_dataset(dataset_dir, dataset_name)
    dist.barrier()

    if find_max_microbatch_size:
        max_microbatch_size = binary_chop_batch_sizes(
            run_dir, dataset_dir,
            model, optimizer, scaler, local_rank,
            train_conf, model_conf
        )
        if dist.get_rank() == 0:
            print(f"Max microbatch size was {max_microbatch_size}")
    else:
        load_datasets_and_train(
            run_dir,
            model, optimizer, scaler, local_rank,
            dataset_dir,
            train_conf, model_conf,
            checkpoint,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
