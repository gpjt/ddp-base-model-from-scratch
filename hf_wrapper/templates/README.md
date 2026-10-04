---
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
tags:
  - lore
  - low-rank
  - causal-lm
datasets:
  - {{ train_config['dataset'] }}
---

# Model Card for {{ hf_model_name }}

This model is {{ hf_model_name }}, a trained-from-scratch base model based on
the GPT-2-style architecture from [Sebastian Raschka](https://sebastianraschka.com/)'s book
"[Build a Large Language Model (from Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)",
but extended to optionally use low-rank matrices for the embeddings and the output head --
a technique I've taken to calling LoRE (for Low Rank Embeddings).  This idea was
[suggested](https://huggingface.co/gpjt/jax-with-mha-bias-fw-fwedu-5050-DEPRECATED/discussions/1) by
[`AndrewThompson1233`](https://huggingface.co/AndrewThompson1233).

You can read more about it in the blog post: [Low-rank vocab matrices](https://www.gilesthomas.com/2026/10/low-rank-vocab-matrices) (coming soon!)


## LoRE settings for this model

See the blog post for information about what these mean:

* LoRE enabled for token embeddings: {{ model_config["lore"]["input_embeddings"] }}
* LoRE enabled for output head: {{ model_config["lore"]["output_head"] }}
* LoRE rank: {{ model_config["lore"]["rank"] }}
* Smart initialization: {{ model_config["lore"]["smart_initialize"] }}


## Model Details

### Model Description

- **Developed by:** [Giles Thomas](https://huggingface.co/gpjt), based on code by [Sebastian Raschka](https://huggingface.co/rasbt)
- **Model type:** GPT-2 style Transformer-based causal LLM.
- **License:** [Apache 2](https://huggingface.co/models?license=license:apache-2.0&sort=downloads)
- **Parameters:** {{ "{0:,}".format(parameters) }}
- **Context length:** {{ "{0:,}".format(model_config['context_length']) }}
- **Transformers embedding dimension:** {{ "{0:,}".format(model_config['emb_dim']) }}
- **MHA heads:** {{ model_config['n_heads'] }}
- **Layers:** {{ model_config['n_layers'] }}
- **QKV bias:** {{ model_config['qkv_bias'] }}
- **Weight tying:** {{ model_config.get('tie_weights', False) }}

Don't have high expectations for the model!  It has 163M parameters -- or even fewer if it's using LoRE on either the embeddings
or the output head -- and was trained on roughly the Chinchilla-optimal number of tokens (~20x the number of parameters)
for a non-LoRE version, which means that it doesn't know
many facts and is not terribly smart.  If you want to do serious work, use a serious model (I like
[Qwen's](https://huggingface.co/Qwen)).  But if you want to build on this and see what you can do with a 2020-vintage
LLM, please do feel free to play with it!


### Model Sources

- **Repository:** [gpjt/ddp-base-model-from-scratch](https://github.com/gpjt/ddp-base-model-from-scratch)
- **Blog post:** [Low-rank vocab matrices](https://www.gilesthomas.com/2026/10/low-rank-vocab-matrices) (coming soon!)

## How to Get Started with the Model

You can download and run the model for inference directly:

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="{{ hf_model_name }}", trust_remote_code=True)
out = pipe(
    "Every effort moves you",
    max_new_tokens=20,
    do_sample=True,
    temperature=1.4,
    top_k=25,
)
print(out[0]["generated_text"])
```

Note that because it uses custom code, you'll need to set `trust_remote_code` to `True`.

It supports `AutoTokenizer`, `AutoModel` and `AutoModelForCausalLM`:

```python
>>> from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("{{ hf_model_name }}")
>>> model = AutoModel.from_pretrained("{{ hf_model_name }}", trust_remote_code=True)
>>> llm_model = AutoModelForCausalLM.from_pretrained("{{ hf_model_name }}", trust_remote_code=True)
```

You can also fine-tune it; [this notebook](https://github.com/gpjt/ddp-base-model-from-scratch/blob/main/hf_train.ipynb) has an example.

Again, don't expect too much from this model!  It's a 163M or fewer-parameter GPT-2 one, trained on a limited
number of tokens.  It's [both dumb and ignorant](https://www.gilesthomas.com/2026/01/llm-from-scratch-30-digging-into-llm-as-a-judge) ;-)


## Training Details

- **Machine type:** 8x A100 on Lambda, with 40 GiB VRAM per GPU.
- **Tokens:** 3,260,252,160 (the target of 3,260,190,720 -- exactly 20x the
    non-LoRE model's parameter count -- rounded up to the nearest complete batch).
- **Dataset:** [{{ train_config['dataset'] }}](https://huggingface.co/datasets/{{ train_config['dataset'] }})
- **Micro-batch size:** {{ train_config['microbatch_size'] }}
- **Global batch size:** 96
- **Dropout:** {{ model_config['drop_rate'] }}
- **Gradient clipping:** {{ train_config.get('clipping_max_norm') }}
- **Learning rate:** {{ train_config.get('learning_rate', 0.0004) }}
- **Schedule learning rate:** {{ train_config.get('schedule_learning_rate', False) }}
- **Weight decay:** {{ train_config.get('weight_decay', 0.1) }}

