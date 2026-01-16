import io
import json
from pathlib import Path

import click
import jinja2

from huggingface_hub import HfApi
from transformers import AutoTokenizer
from safetensors.torch import load_file

from hf_wrapper.configuration_gpjtgpt2 import GPJTGPT2Config
from hf_wrapper.modeling_gpjtgpt2 import GPJTGPT2Model, GPJTGPT2ModelForCausalLM

MY_DIR = Path(__file__).resolve().parent


@click.command()
@click.argument("model_config_path")
@click.argument("model_safetensors_path")
@click.argument("train_config_path")
@click.argument("hf_model_name")
def main(model_config_path, model_safetensors_path, train_config_path, hf_model_name):
    if not Path(model_config_path).is_file():
        raise Exception(f"Could not find model config at {model_config_path}")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    if not Path(model_safetensors_path).is_file():
        raise Exception(f"Could not find model safetensors at {model_safetensors_path}")

    if not Path(train_config_path).is_file():
        raise Exception(f"Could not find train config at {train_config_path}")
    with open(train_config_path, "r") as f:
        train_config = json.load(f)

    GPJTGPT2Config.register_for_auto_class()
    GPJTGPT2Model.register_for_auto_class("AutoModel")
    GPJTGPT2ModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    config = GPJTGPT2Config(model_config)
    config.auto_map = {
        "AutoConfig": "configuration_gpjtgpt2.GPJTGPT2Config",
        "AutoModel": "modeling_gpjtgpt2.GPJTGPT2Model",
        "AutoModelForCausalLM": "modeling_gpjtgpt2.GPJTGPT2ModelForCausalLM",
    }
    config.architectures = ["GPJTGPT2ModelForCausalLM"]

    model = GPJTGPT2ModelForCausalLM(config)
    model.model.load_state_dict(load_file(model_safetensors_path))

    model.push_to_hub(hf_model_name)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.push_to_hub(hf_model_name)

    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(MY_DIR / "hf_wrapper" / "templates")
    )
    readme_template = jinja_env.get_template("README.md")
    readme = readme_template.render(
        hf_model_name=hf_model_name,
        parameters=sum(p.numel() for p in model.model.parameters()),
        model_config=model_config,
        train_config=train_config,
    )
    api = HfApi()
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=hf_model_name,
        repo_type="model",
        commit_message="Add README/model card",
    )



if __name__ == "__main__":
    main()
