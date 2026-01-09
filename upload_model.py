import json
from pathlib import Path

import click

from safetensors.torch import load_file

from hf_model_wrapper import HFGPTModel


@click.command()
@click.argument("model_config_path")
@click.argument("model_safetensors_path")
@click.argument("hf_model_name")
def main(model_config_path, model_safetensors_path, hf_model_name):
    if not Path(model_config_path).is_file():
        raise Exception(f"Could not find model config at {model_config_path}")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    if not Path(model_safetensors_path).is_file():
        raise Exception(f"Could not find model safetensors at {model_safetensors_path}")

    model = HFGPTModel(model_config)
    model.load_state_dict(load_file(model_safetensors_path))

    model.push_to_hub(hf_model_name)


if __name__ == "__main__":
    main()
