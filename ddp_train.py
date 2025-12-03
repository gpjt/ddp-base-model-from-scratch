import json
import os
from pathlib import Path

import click

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt import GPTModel


def train(model_conf):
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)
    rank = dist.get_rank()
    print(f"On rank {rank}.")
    device_id = rank % torch.accelerator.device_count()

    model = GPTModel(model_conf).to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    dist.destroy_process_group()


@click.command()
@click.argument("run")
def main(run):
    run_dir = Path(__file__).resolve().parent / "runs" / run
    if not run_dir.is_dir():
        raise Exception(f"Could not find run dir {run_dir}")

    model_conf_file = run_dir / "model.json"
    if not model_conf_file.is_file():
        raise Exception(f"Could not find model config in {model_conf_file}")
    with open(model_conf_file, "r") as f:
        model_conf = json.load(f)

    train(model_conf)


if __name__ == "__main__":
    main()
