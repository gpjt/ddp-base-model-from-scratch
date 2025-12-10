# Based on code from:
#   "Build a Large Language Model (from Scratch)"
#   Copyright 2023-2025 Sebastian Raschka
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications copyright 2025 Giles Thomas


import json
import os
from functools import partial
from pathlib import Path

import click
import requests
from tqdm import tqdm

import tiktoken
import torch
from openai import OpenAI
from torch.utils.data import Dataset, DataLoader

from safetensors.torch import load_file
from ddp_train import calculate_loss
from gpt import GPTModel


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        text_data = requests.get(url).text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    data = json.loads(text_data)
    return data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, ix):
        return self.encoded_texts[ix]

    def __len__(self):
        return len(self.encoded_texts)


def custom_collate_fn(
    batch, pad_token_id=50256, ignore_index=-100,
    allowed_max_length=None, device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst = []
    targets_lst = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def get_data_splits():
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    return train_data, val_data, test_data


def get_data_loaders(train_data, val_data, tokenizer, device):
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    num_workers = 0
    batch_size = 8

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = calculate_loss(logits, target_batch)

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return val_loss


def train_model(
    model, train_loader, val_loader,
    optimizer, device,
    eval_iter
):
    last_val_loss = None
    last_params = None
    for epoch in range(100):
        model.train()
        for input_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)

        if last_val_loss is None or last_val_loss > val_loss:
            last_val_loss = val_loss
            last_params = model.state_dict()
            print("Val loss still decreasing, continuing")
        else:
            print("Val loss rising, bailing out")
            break

    model.load_state_dict(last_params)


def generate(
    model, idx, max_new_tokens, context_size,
    temperature=0.0, top_k=None, eos_id=None
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx


def query_model(prompt):
    client = OpenAI()

    response = client.responses.create(
        model="gpt-5.1",
        input=prompt
    )

    return response.output_text


def generate_model_scores(json_data):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}` "
            f"on a scale of 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


@click.command()
@click.argument("model_config_path")
@click.argument("model_safetensors_path")
def main(model_config_path, model_safetensors_path):
    if not Path(model_config_path).is_file():
        raise Exception(f"Could not find model config at {model_config_path}")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    if not Path(model_safetensors_path).is_file():
        raise Exception(f"Could not find model safetensors at {model_safetensors_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(model_config)
    model.load_state_dict(load_file(model_safetensors_path))
    model.to(device)

    train_data, val_data, test_data = get_data_splits()

    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader = get_data_loaders(
        train_data, val_data, tokenizer, device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )

    train_model(
        model, train_loader, val_loader, optimizer, device,
        eval_iter=5
    )

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=torch.tensor(
                tokenizer.encode(input_text),
                dtype=torch.long, device=device
            ).unsqueeze(0),
            max_new_tokens=256,
            context_size=model_config["context_length"],
            eos_id=50256,
        )
        generated_text = tokenizer.decode(token_ids.squeeze(0).tolist())

        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        test_data[i]["model_response"] = response_text

    for entry in test_data[:3]:
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}` "
            f"on a scale of 0 to 100, where 100 is the best score."
        )
        print("\nDataset response:")
        print(">>", entry["output"])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt))
        print("\n-----------------------------------------")

    scores = generate_model_scores(test_data)
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}\n")


if __name__ == "__main__":
    main()
