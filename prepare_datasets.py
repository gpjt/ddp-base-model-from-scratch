import shutil
from pathlib import Path

import click
import tiktoken
import torch
from tqdm import tqdm
from datasets import load_dataset
from safetensors.torch import save_file


def build_and_save_dataset_tensor(ds, tokenizer, path):
    print(f"Building dataset for {path}")
    results = []
    num_tokens = 0
    batch_size = 1000
    for ix in tqdm(range(0, len(ds), batch_size)):
        texts = ds[ix:ix + batch_size]["text"]
        text_tokens = tokenizer.encode_batch(
            texts,
            allowed_special={'<|endoftext|>'}
        )
        all_tokens = []
        for toks in text_tokens:
            all_tokens.extend(toks)
            all_tokens.append(tokenizer.eot_token)
        results.append(torch.tensor(all_tokens, dtype=torch.uint16))
        num_tokens += len(all_tokens)
    result = torch.cat(results)
    print(f"Saving {result.shape[0]} tokens to {path}")
    save_file({"tokens": result}, path)


@click.command()
@click.argument("input_dataset_dir")
@click.argument("output_dataset_dir")
def main(input_dataset_dir, output_dataset_dir):
    splits = load_dataset(
        "parquet",
        data_files=f"./{input_dataset_dir}/sample/10BT/*.parquet",
        split={"train": "train[:99%]", "validation": "train[99%:]"}
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset_dir = Path(__file__).resolve().parent / output_dataset_dir
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir()

    build_and_save_dataset_tensor(
        splits["validation"], tokenizer,
        dataset_dir / "validation.safetensors"
    )
    build_and_save_dataset_tensor(
        splits["train"], tokenizer,
        dataset_dir / "train.safetensors"
    )


if __name__ == "__main__":
    main()
