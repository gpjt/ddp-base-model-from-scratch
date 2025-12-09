import json
import math
from pathlib import Path

import click

import tiktoken
import torch

from safetensors.torch import load_file
from gpt import GPTModel


@click.command()
@click.argument("model_config_path")
@click.argument("model_safetensors_path")
def main(model_config_path, model_safetensors_path):
    if not Path(model_config_path).is_file():
        raise Exception(f"Could not fine model config at {model_config_path}")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    if not Path(model_safetensors_path).is_file():
        raise Exception(f"Could not find model safetensors at {model_safetensors_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(model_config)
    model.load_state_dict(load_file(model_safetensors_path))
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    input_text = "Every effort moves you"
    tokens = tokenizer.encode(input_text)

    num_tokens = 20
    temperature = 1.4
    top_k = 25
    with torch.no_grad():
        for ix in range(num_tokens):
            input_tensor = torch.tensor(
                tokens, dtype=torch.long, device=device
            ).unsqueeze(0)
            output_tensor = model(input_tensor)
            logits = output_tensor[:, -1, :]
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(-math.inf).to(logits.device),
                logits
            )
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)

    print(tokenizer.decode(tokens))




if __name__ == "__main__":
    main()
