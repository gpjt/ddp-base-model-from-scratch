---
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
tags:
  - gpjt-llm-from-scratch
datasets:
  - {{ train_config['dataset'] }}
---

# Model Card for {{ hf_model_name }}

This model is {{ hf_model_name }}, a trained-from-scratch base model using
the GPT-2-style architecture from [Sebastian Raschka](https://sebastianraschka.com/)'s book
"[Build a Large Language Model (from Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)".


## Model Details

### Model Description

- **Developed by:** [Giles Thomas](https://huggingface.co/gpjt), based on code by [Sebastian Raschka](https://huggingface.co/rasbt)
- **Model type:** GPT-2 style transformers-based causal LLM.
- **License:** [Apache 2](https://huggingface.co/models?license=license:apache-2.0&sort=downloads)
- **Parameters:** {{ "{0:,}".format(parameters) }}
- **Context length:** {{ "{0:,}".format(model_config['context_length']) }}
- **Embedding dimensions:** {{ "{0:,}".format(model_config['emb_dim']) }}
- **MHA heads:** {{ model_config['n_heads'] }}
- **Layers:** {{ model_config['n_layers'] }}
- **QKV bias:** {{ model_config['qkv_bias'] }}
- **Weight tying:** No.

Don't have high expectations for the model!  It has only 163M parameters (the GPT-2 "small" size)
and was trained on roughly the Chinchilla-optimal number of tokens (~20x the number of parameters), which means that it doesn't know
many facts and is not terribly smart.  If you want to do serious work, use a serious model (I like
[Qwen's](https://huggingface.co/Qwen)).  But if you want to build on this and see what you can do with a 2020-vintage
LLM, please do feel free to play with it!


### Model Sources

- **Repository:** [gpjt/ddp-base-model-from-scratch](https://github.com/gpjt/ddp-base-model-from-scratch)
- **Blog post:** [Writing an LLM from scratch, part 29 -- using DistributedDataParallel to train a base model from scratch in the cloud](https://www.gilesthomas.com/2026/01/llm-from-scratch-29-ddp-training-a-base-model-in-the-cloud)

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

Again, don't expect too much from this model!  It's a 163M-parameter GPT-2 one, trained on a limited
number of tokens.  It's [both dumb and ignorant](https://www.gilesthomas.com/2026/01/llm-from-scratch-30-digging-into-llm-as-a-judge) ;-)


## Training Details

- **Machine type:** TODO
- **Tokens:**  3,260,190,720 (Chinchilla-optimal of 20x parameters) rounded up to the nearest batch.
- **Dataset:** [{{ train_config['dataset'] }}](https://huggingface.co/datasets/{{ train_config['dataset'] }})
- **Micro-batch size:** {{ train_config['microbatch_size'] }}
- **Global batch size:** TODO
- **Dropout:** {{ model_config['drop_rate'] }}

