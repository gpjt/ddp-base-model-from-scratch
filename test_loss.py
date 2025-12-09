import json
from pathlib import Path

import click
from tqdm import tqdm

import torch

from safetensors.torch import load_file
from ddp_train import calculate_loss, download_dataset, load_dataset
from gpt import GPTModel


@click.command()
@click.argument("datasets_dir_path")
@click.argument("model_config_path")
@click.argument("model_safetensors_path")
def main(datasets_dir_path, model_config_path, model_safetensors_path):
    datasets_dir = Path(datasets_dir_path)
    if not datasets_dir.is_dir():
        raise Exception(f"{datasets_dir_path} is not a directory")
    dataset_dir = download_dataset(datasets_dir, "gpjt/fineweb-gpt2-tokens")

    if not Path(model_config_path).is_file():
        raise Exception(f"Could not fine model config at {model_config_path}")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    if not Path(model_safetensors_path).is_file():
        raise Exception(f"Could not fine model safetensors at {model_safetensors_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(model_config)
    model.load_state_dict(load_file(model_safetensors_path))
    model.to(device)

    world_size = 1
    batches = 3200
    batch_size = 6
    seq_len = 1024
    test_ds = load_dataset(
        dataset_dir, "validation",
        batches * batch_size * seq_len, 50_000_000,
        world_size, batch_size,
        seq_len
    )

    model.eval()
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.float16):
        test_losses = []
        for test_inputs, test_targets in tqdm(test_ds):
            test_inputs = test_inputs.to(device).to(torch.long)
            test_targets = test_targets.to(device).to(torch.long)
            test_logits = model(test_inputs)
            test_losses.append(
                calculate_loss(test_logits, test_targets).item()
            )
        test_loss = sum(test_losses) / len(test_losses)

    print(f"Loss against our test dataset: {test_loss:.3f}")


if __name__ == "__main__":
    main()
