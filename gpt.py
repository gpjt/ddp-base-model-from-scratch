# Based on code from:
#   "Build a Large Language Model (from Scratch)"
#   Copyright 2023-2025 Sebastian Raschka
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications copyright 2025, 2026 Giles Thomas

import json

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_in, d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False
    ):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)

        return context_vec



class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))


    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift



class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"])
        )


    def forward(self, x):
        return self.layers(x), None



class MixtureOfExperts(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg["moe"]["num_experts"]
        self.num_active_experts = cfg["moe"]["num_active_experts"]
        if self.num_active_experts < 2:
            raise Exception(
                f"Can't train with `num_active_experts` < 2 (got {self.num_active_experts})"
            )
        if self.num_active_experts > self.num_experts:
            raise Exception(
                f"{self.num_active_experts=} is larger than {self.num_experts=}"
            )
        self.router = nn.Linear(cfg["emb_dim"], self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(cfg) for _ in range(self.num_experts)
        ])
        self.last_routing_logits = None


    def log_routing_logits(self, f, step, name):
        json.dump(
            {
                "step": step,
                "name": name,
                "routing_logits": self.last_routing_logits.tolist(),
            },
            f
        )
        f.write("\n")
        self.last_routing_logits = None


    def forward(self, xs):
        # See moe-calculations-explainer.ipynb for an explanation of how this
        # all works.
        routing_logits = self.router(xs)

        # We're stashing away a copy of the logits (with the compute graph
        # detached) so that we can log it later on.
        self.last_routing_logits = routing_logits.detach()

        top_k_values, top_k_indices = torch.topk(
            routing_logits,
            k=self.num_active_experts,
            dim=-1
        )
        top_k_routing_logits = torch.full_like(routing_logits, -torch.inf)
        top_k_routing_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
        expert_weights = torch.softmax(top_k_routing_logits, dim=-1)

        all_outputs = torch.zeros_like(xs)
        for expert_ix, expert in enumerate(self.experts):
            this_expert_mask = expert_weights[:, :, expert_ix] > 0
            this_expert_results, _ = expert(xs[this_expert_mask])
            this_expert_weights = expert_weights[this_expert_mask, expert_ix].unsqueeze(1)
            all_outputs[this_expert_mask] += this_expert_results * this_expert_weights

        return all_outputs, (routing_logits, expert_weights)



class TransformersBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.is_moe = "moe" in cfg
        if self.is_moe:
            self.ff = MixtureOfExperts(cfg)
        else:
            self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])


    def forward(self, inputs):
        x, moe_routing_info = inputs
        if self.is_moe and moe_routing_info is None:
            moe_routing_info = []

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x, this_block_moe_routing_info = self.ff(x)
        if self.is_moe:
            moe_routing_info.append(this_block_moe_routing_info)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x, moe_routing_info



class GPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformersBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        if cfg.get("tie_weights", False):
            self.out_head.weight = self.tok_emb.weight

        self.log_moe_balance_file = None
        if cfg.get("moe") and cfg["moe"].get("log_moe_balance_file"):
            self.log_moe_balance_file = cfg["moe"]["log_moe_balance_file"]


    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds

        x = self.drop_emb(x)
        x, moe_routing_info = self.trf_blocks((x, None))
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits, moe_routing_info


    def log_routing_logits(self, step):
        if not self.log_moe_balance_file:
            return

        with open(self.log_moe_balance_file, "a") as f:
            for name, module in self.named_modules():
                if isinstance(module, MixtureOfExperts):
                    module.log_routing_logits(f, step, name)


