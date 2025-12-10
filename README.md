# ddp-base-model-from-scratch

Code to pre-train a GPT-style language model from scratch using PyTorch Distributed Data Parallel (DDP), on a tokenised slice of the FineWeb / FineWeb-Edu datasets.

It’s designed as a small, educational-ish “base model pre-train” pipeline that you can:

- run on a single multi-GPU machine (via `torchrun`)
- point at a pre-tokenised dataset on Hugging Face
- monitor via loss curves and simple evaluation scripts
- extend or tweak for your own experiments

---

## Contents

- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Dataset options](#dataset-options)  
  - [Using the published FineWeb tokens](#using-the-published-fineweb-tokens)  
  - [Rebuilding the token dataset yourself](#rebuilding-the-token-dataset-yourself)  
- [Configuring a training run](#configuring-a-training-run)  
  - [`model.json`](#modeljson)  
  - [`train.json`](#trainjson)  
- [Running DDP training](#running-ddp-training)  
  - [Basic example](#basic-example)  
  - [Resuming from a checkpoint](#resuming-from-a-checkpoint)  
- [Evaluating and sampling](#evaluating-and-sampling)  
  - [Quick smoke test generation](#quick-smoke-test-generation)  
  - [Loss on a validation slice](#loss-on-a-validation-slice)  
  - [Instruction-following / IF test](#instruction-following--if-test)  
- [Lambda Labs / cloud notes](#lambda-labs--cloud-notes)  
- [License](#license)

---

## Features

- GPT-2-style transformer implemented directly in `gpt.py`
- Tokenised FineWeb / FineWeb-Edu dataset stored as `safetensors` with a single `tokens` vector
- Multi-GPU training using PyTorch Distributed Data Parallel (`torch.distributed`, `DDP`)
- Automatic dataset download using `huggingface_hub.snapshot_download`
- Checkpointing with symlinks to `best` and `latest`, plus a loss-over-time PNG chart
- Small helper scripts to:
  - rebuild the tokenised dataset from raw FineWeb parquet
  - compute loss against a held-out validation slice
  - generate sample text
  - run an instruction-following “sanity check” using a small SFT dataset

---

## Requirements

- **Python**: `>=3.13` (as per `pyproject.toml`; earlier 3.x may work but is not the default target)
- **PyTorch** with CUDA (for DDP you’ll want at least 1 GPU; usually several)
- **Hugging Face account** and token (to download datasets via `huggingface_hub`)
- A POSIX-like environment (Linux, WSL, or similar)

Python dependencies are declared in `pyproject.toml` and include:

- `torch`
- `huggingface-hub`
- `datasets`
- `safetensors`
- `tiktoken`
- `click`
- `matplotlib`
- `tqdm`
- `openai` (only for the IF test script)

---

## Installation

The project is set up to work nicely with [uv](https://docs.astral.sh/uv/), but you can also use plain `pip`.

### 1. Clone the repo

```bash
git clone https://github.com/gpjt/ddp-base-model-from-scratch.git
cd ddp-base-model-from-scratch
```

### 2. Install dependencies (with `uv`)

If you don’t already have `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repo root:

```bash
uv sync
```

This creates a virtual environment and installs everything from `pyproject.toml`.

To run scripts under `uv`, use:

```bash
uv run python some_script.py ...
```

(If you prefer `pip`, you can do something like `python -m venv .venv && source .venv/bin/activate && pip install -e .`, but the project is tuned for `uv`.)

---

## Dataset options

The training code expects **tokenised datasets stored on Hugging Face** in `safetensors` format, containing a single `tokens` tensor.

There are two main ways to get such a dataset:

1. Use the already-tokenised **FineWeb** datasets published under the `gpjt` account.
2. Rebuild the tokens yourself from raw FineWeb / FineWeb-Edu parquet data and push to your own HF dataset repo.

### Using the published FineWeb tokens

There are two relevant datasets on Hugging Face:

- `gpjt/fineweb-gpt2-tokens`
- `gpjt/fineweb-edu-gpt2-tokens`

These contain GPT-2-tokenised versions of FineWeb / FineWeb-Edu.  
The training script will download whichever dataset name you specify in `train.json` (see below) using `huggingface_hub.snapshot_download`.

You’ll pass a **local datasets directory** as an argument at runtime; the script will create subfolders there as needed.

For example:

- You run: `torchrun ... ddp_train.py my-run ./datasets`
- `train.json` says `"dataset": "gpjt/fineweb-edu-gpt2-tokens"`
- The code downloads into: `./datasets/gpjt/fineweb-edu-gpt2-tokens`
- Inside that folder it expects `train.safetensors` and `validation.safetensors` containing a `tokens` array.

### Rebuilding the token dataset yourself

If you want to regenerate the token dataset from the original FineWeb parquet:

1. **Download FineWeb / FineWeb-Edu parquet**

   The helper script `download-fineweb-10b.py` downloads the 10B-token samples for both FineWeb and FineWeb-Edu from the `HuggingFaceFW` organisation.

   ```bash
   uv run python download-fineweb-10b.py
   ```

   This creates e.g.:

   - `./fineweb/`
   - `./fineweb-edu/`

   with a `sample/10BT/*.parquet` layout.

2. **Convert parquet to token tensors**

   Use `prepare_datasets.py` to load parquet files, tokenize with `tiktoken`’s GPT-2 tokenizer, and save to `safetensors` files.

   Example:

   ```bash
   # Build tokenised datasets from the FineWeb-Edu 10BT sample
   uv run python prepare_datasets.py fineweb-edu fineweb-edu-gpt2-tokens
   ```

   This will create an `./fineweb-edu-gpt2-tokens` directory containing:

   - `train.safetensors`  (with a `tokens` tensor)
   - `validation.safetensors` (same)

3. **(Optional) Push to Hugging Face**

   The training and evaluation scripts expect to download from HF via a **dataset repo name** (e.g. `gpjt/fineweb-edu-gpt2-tokens`). If you want to use your freshly built dataset with the unmodified scripts, you’ll want to:

   - Create a dataset on Hugging Face: e.g. `yourname/your-fineweb-tokens`
   - Upload `train.safetensors` and `validation.safetensors`
   - Set `"dataset": "yourname/your-fineweb-tokens"` in `train.json` (see below)

   Alternatively, you can customise `ddp_train.py` to read directly from a local directory instead of calling `snapshot_download`.

---

## Configuring a training run

Each run lives in its own subdirectory under `runs/` and is controlled by two JSON files:

- `runs/<run-name>/model.json`
- `runs/<run-name>/train.json`

You can create as many runs as you like; each will keep its own checkpoints and plots.

### `model.json`

This defines the actual GPT-style model. The keys map directly onto `GPTModel` in `gpt.py`.

Typical fields:

- `vocab_size` — usually `50257` for GPT-2 tokenizer
- `emb_dim` — embedding dimension (e.g. `768` for GPT-2 small)
- `context_length` — maximum sequence length (e.g. `1024`)
- `n_heads` — number of attention heads
- `n_layers` — number of transformer blocks
- `drop_rate` — dropout probability
- `qkv_bias` — boolean for linear layer bias in Q/K/V projections

Example (roughly GPT-2 small-ish):

```json
{
  "vocab_size": 50257,
  "emb_dim": 768,
  "context_length": 1024,
  "n_heads": 12,
  "n_layers": 12,
  "drop_rate": 0.1,
  "qkv_bias": false
}
```

### `train.json`

This controls how much data you train on and how validation/checkpointing works.

Fields:

- `dataset`: **Hugging Face dataset repo name** (e.g. `"gpjt/fineweb-edu-gpt2-tokens"`)
- `min_train_tokens`: minimum number of tokens to pull for training (`-1` means “use as many as possible from `start_train_token` onwards, in full global batches”)
- `start_train_token`: starting token index for training (e.g. `0`)
- `start_val_token`: starting token index for validation (so train/val don’t overlap)
- `minibatch_size`: per-GPU mini-batch size
- `validation_interval`: how often (in global steps) to run validation & checkpoint
- `validation_batches`: how many batches to use when computing validation loss

Example:

```json
{
  "dataset": "gpjt/fineweb-edu-gpt2-tokens",
  "min_train_tokens": -1,
  "start_train_token": 0,
  "start_val_token": 50000000,
  "minibatch_size": 6,
  "validation_interval": 100,
  "validation_batches": 100
}
```

The code computes how many tokens to pull such that each DDP **global batch** is:

- `world_size * minibatch_size * context_length` tokens

and it only ever uses a multiple of that, plus **one extra token** for the label shift.

---

## Running DDP training

Training entry point: **`ddp_train.py`**

CLI:

- `run` — the name of the run, i.e. the subdirectory under `runs/`
- `datasets_dir_path` — local root directory where HF datasets will be downloaded
- `checkpoint` (optional) — name of a checkpoint directory or symlink under `runs/<run>/checkpoints` to resume from

### Basic example

1. Create a run directory and configs, for example:

   - `runs/163m-fineweb-edu/model.json`
   - `runs/163m-fineweb-edu/train.json`

2. Decide how many GPUs you want to use (say 4):

3. Run training with `torchrun`:

```bash
# From the repo root
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  ddp_train.py \
    163m-fineweb-edu \
    ./datasets
```

What happens:

- `torchrun` spawns 4 processes with `LOCAL_RANK` set appropriately.
- `ddp_train.py`:
  - Reads `runs/163m-fineweb-edu/model.json` and `train.json`
  - Calls `torch.accelerator.set_device_index(local_rank)` and initialises DDP
  - Downloads the dataset named in `train.json["dataset"]` into `./datasets/<dataset-name>`
  - Creates a `BigTrainDataset` from the token vector
  - Trains, periodically:
    - runs validation on the same rank-0 model
    - saves checkpoints in `runs/163m-fineweb-edu/checkpoints`
    - updates `best` and `latest` symlinks
    - regenerates a plot `runs/163m-fineweb-edu/big-training-run-chart.png`

At the end, rank 0 prints:

- elapsed training time
- total tokens seen
- throughput (tokens/second)
- final train and validation loss

### Resuming from a checkpoint

To resume from the latest checkpoint:

1. Find the run directory, e.g. `runs/163m-fineweb-edu/`
2. The checkpointing code creates symlinks:
   - `runs/<run>/checkpoints/latest`
   - `runs/<run>/checkpoints/best`

You can resume from `latest` like this:

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  ddp_train.py \
    163m-fineweb-edu \
    ./datasets \
    latest
```

Or from a specific checkpoint directory name, e.g.:

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  ddp_train.py \
    163m-fineweb-edu \
    ./datasets \
    20251209Z143355-iteration-1000
```

Internally, `load_checkpoint` will:

- load the model, optimizer and scaler state
- set `global_step` to `meta["global_step"] + 1`
- reload the best validation loss from `checkpoints/best/meta.json`

---

## Evaluating and sampling

### Quick smoke test generation

Script: **`test_smoke.py`**

This is a small text-generation smoke test that:

- loads a config and weights
- runs a short sampling loop with temperature and top-k
- prints the decoded tokens to stdout

Usage:

```bash
uv run python test_smoke.py \
  runs/163m-fineweb-edu/model.json \
  runs/163m-fineweb-edu/checkpoints/best/model.safetensors
```

It uses the GPT-2 tokenizer (`tiktoken`) and starts from:

> Every effort moves you

By default it:

- generates `num_tokens = 20` new tokens
- uses `temperature = 1.4`
- `top_k = 25`

You can tweak those in the script if you like.

### Loss on a validation slice

Script: **`test_loss.py`**

This script computes the average cross-entropy loss over a slice of the validation set from a tokenised dataset.

It:

- downloads `gpjt/fineweb-gpt2-tokens` into the provided datasets directory
- loads a fixed number of batches from validation starting at a fixed token offset
- reports mean loss

Usage:

```bash
uv run python test_loss.py \
  ./datasets \
  runs/163m-fineweb-edu/model.json \
  runs/163m-fineweb-edu/checkpoints/best/model.safetensors
```

Notes:

- It currently hard-codes:
  - dataset name: `"gpjt/fineweb-gpt2-tokens"`
  - total tokens and offsets (e.g. `batches = 3200`, `batch_size = 6`, `seq_len = 1024`, `start_token = 50_000_000`)
- If you want to evaluate against a different dataset or slice, edit those values in `test_loss.py`.

### Instruction-following / IF test

Script: **`test_ift.py`**

This is a more elaborate script based on the instruction-tuning chapter from *Build a Large Language Model (from Scratch)*. It:

1. Downloads a small instruction-following dataset (`instruction-data.json`) from Sebastian Raschka’s repo.
2. Builds a PyTorch `Dataset` over instruction/response triples using the GPT-2 tokenizer.
3. Fine-tunes your base model a bit on that instruction data.
4. Generates responses on a held-out test split.
5. Calls the OpenAI Responses API to **score** those responses against the reference outputs.

Usage:

```bash
export OPENAI_API_KEY=sk-...   # required for scoring

uv run python test_ift.py \
  runs/163m-fineweb-edu/model.json \
  runs/163m-fineweb-edu/checkpoints/best/model.safetensors
```

The script:

- runs a simple “stop-when-validation-loss-rises” training loop on instruction data
- generates responses with a `generate` helper
- prints a few example comparisons (reference vs model response vs score)
- computes an average score over the test set

You can also uncomment or adapt the `generate_model_scores` / printing sections to save scores somewhere else if needed.

---

## Lambda Labs / cloud notes

There’s a small helper script for setting up a fresh GPU VM (e.g. at Lambda Labs):

Script: **`setup_lambda.sh`**

It:

- installs `uv`
- installs the XKCD font so that Matplotlib’s `plt.xkcd()` and font selection in `ddp_train.py` work cleanly

Usage (on a new VM):

```bash
bash setup_lambda.sh
uv sync
```

After that, the DDP commands above should Just Work™, assuming your driver / CUDA setup is correct and `torch` sees your GPUs.

---

## License

- The core GPT model is **based on** the implementation from *Build a Large Language Model (From Scratch)* by Sebastian Raschka and is under the Apache 2.0 license (see `LICENSE` and headers in `gpt.py` / `test_ift.py`).
- Modifications and additional scripts are copyright © 2025 Giles Thomas, also under Apache 2.0.

See `LICENSE` for full details.
