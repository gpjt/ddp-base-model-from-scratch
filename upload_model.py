import json
from pathlib import Path

import click

from transformers import AutoTokenizer
from safetensors.torch import load_file

from hf_wrapper.configuration_gpjtgpt2 import GPJTGPT2Config
from hf_wrapper.modeling_gpjtgpt2 import GPJTGPT2Model, GPJTGPT2ModelForCausalLM


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

    GPJTGPT2Config.register_for_auto_class()
    GPJTGPT2Model.register_for_auto_class("AutoModel")
    GPJTGPT2ModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = GPJTGPT2ModelForCausalLM(GPJTGPT2Config(model_config))
    model.model.load_state_dict(load_file(model_safetensors_path))

    model.push_to_hub(hf_model_name)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.push_to_hub(hf_model_name)


if __name__ == "__main__":
    main()
