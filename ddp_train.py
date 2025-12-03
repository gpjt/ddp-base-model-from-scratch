import json
import os
from pathlib import Path

import click
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from safetensors.torch import load_file

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from checkpointing import get_checkpoints_dir, load_checkpoint, save_checkpoint
from gpt import GPTModel


VAL_AND_CHECKPOINT_INTERVAL = 2000


class BigTrainDataset(Dataset):

    def __init__(self, all_tokens, seq_length, minibatch_size):
        self.xs = all_tokens[:-1].reshape(-1, minibatch_size, seq_length)
        self.ys = all_tokens[1:].reshape(-1, minibatch_size, seq_length)

    def __getitem__(self, ix):
        return (self.xs[ix], self.ys[ix])

    def __len__(self):
        return self.xs.shape[0]


def load_dataset(run_dir, split, seq_length, minibatch_size):
    return BigTrainDataset(
        load_file(run_dir / "datasets" / f"{split}.safetensors")["tokens"],
        seq_length, minibatch_size,
    )


def get_training_data(run_dir):
    checkpoints_dir = get_checkpoints_dir(run_dir)

    train_losses = []
    val_losses = []
    best_train_ds_offset = None
    for item in checkpoints_dir.iterdir():
        if item.name == "latest":
            continue

        meta = json.loads((item / "meta.json").read_text())
        if item.name == "best":
            best_train_ds_offset = meta["train_ds_offset"]
            continue

        train_losses.append((meta["train_ds_offset"], meta["train_loss"]))
        val_losses.append((meta["train_ds_offset"], meta["val_loss"]))

    train_losses.sort(key=lambda x: x[0])
    val_losses.sort(key=lambda x: x[0])

    return train_losses, val_losses, best_train_ds_offset


def generate_training_chart(run_dir):
    train_points, val_points, best_train_ds_offset = get_training_data(run_dir)

    plt.title("TRAINING RUN LOSS")
    plt.xkcd()
    plt.rcParams['font.family'] = "xkcd"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    train_epochs, train_losses = zip(*train_points)
    val_epochs, val_losses = zip(*val_points)
    ax.plot(train_epochs, train_losses, label="TRAINING LOSS", marker="o")
    ax.plot(val_epochs, val_losses, label="VALIDATION LOSS", marker="s")

    ax.axvline(
        best_train_ds_offset, color="red", linestyle="--", linewidth=1.5,
        label="BEST ITERATION"
    )

    ax.set_title("TRAINING RUN LOSS")
    ax.set_xlabel("ITERATION")
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
    train_ds_offset, best_loss
):
    device = next(model.parameters()).device

    torch.set_float32_matmul_precision("high")

    print(f"Starting training at dataset offset {train_ds_offset}")
    train_losses = []
    for ix in tqdm(range(train_ds_offset, len(train_ds))):
        model.train()
        inputs, targets = train_ds[ix]
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

        if (ix % VAL_AND_CHECKPOINT_INTERVAL == 0) or (ix == len(train_ds) - 1):
            print("Validation/checkpoint")
            model.eval()
            base_model = model.module
            with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                val_losses = []
                for val_inputs, val_targets in tqdm(val_ds):
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
                f"iteration-{ix}",
                base_model, optimizer, scaler,
                avg_train_loss, val_loss,
                ix,
                is_best
            )
            generate_training_chart(run_dir)

            model.train()
            print("Continuing training")

    dist.destroy_process_group()


@click.command()
@click.argument("run")
@click.argument("checkpoint", default=None)
def main(run, checkpoint):
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

    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)
    rank = dist.get_rank()
    print(f"On rank {rank}.")
    device_id = rank % torch.accelerator.device_count()

    model = GPTModel(model_conf).to(device_id)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )

    scaler = torch.amp.GradScaler()

    train_ds = load_dataset(
        run_dir, "train",
        model_conf["context_length"], train_conf["minibatch_size"]
    )
    val_ds = load_dataset(
        run_dir, "validation",
        model_conf["context_length"], train_conf["minibatch_size"]
    )

    if checkpoint:
        train_ds_offset, best_loss = load_checkpoint(
            run_dir, checkpoint, model, optimizer, scaler
        )
    else:
        train_ds_offset = 0
        best_loss = None

    ddp_model = DDP(model, device_ids=[device_id])

    train(
        run_dir,
        ddp_model, optimizer, scaler,
        train_ds, val_ds,
        train_ds_offset, best_loss
    )


if __name__ == "__main__":
    main()
