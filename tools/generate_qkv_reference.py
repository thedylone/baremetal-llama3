#!/usr/bin/env python3
"""Generate reference RMSNorm + QKV projection outputs for C++ verification.

Loads weights from the original safetensors files, computes
embeddings -> RMSNorm -> Q/K/V in float32, and writes the results
to a binary file that the C++ test can load and compare against.

Usage:
    python tools/generate_qkv_reference.py
"""

import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_DIR = "assets/llama3"
OUTPUT_PATH = "tests/data/qkv_reference.bin"
SENTENCE = "Hello world"
EPS = 1e-5


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    token_ids = tokenizer.encode(SENTENCE, add_special_tokens=True)
    print(f"Token IDs ({len(token_ids)}): {token_ids}")

    with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    def load_tensor(name: str) -> torch.Tensor:
        shard = weight_map[name]
        with safe_open(f"{MODEL_DIR}/{shard}", framework="pt") as f:
            return f.get_tensor(name).float()

    embed = load_tensor("model.embed_tokens.weight")
    ln_weight = load_tensor("model.layers.0.input_layernorm.weight")
    q_weight = load_tensor("model.layers.0.self_attn.q_proj.weight")
    k_weight = load_tensor("model.layers.0.self_attn.k_proj.weight")
    v_weight = load_tensor("model.layers.0.self_attn.v_proj.weight")

    hidden_size = embed.shape[1]
    kv_dim = k_weight.shape[0]

    ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    embeddings = embed[ids_tensor]
    normed = rms_norm(embeddings, ln_weight, EPS)
    Q = normed @ q_weight.T
    K = normed @ k_weight.T
    V = normed @ v_weight.T

    seq_len = len(token_ids)
    print(f"seq_len={seq_len}, hidden_size={hidden_size}, kv_dim={kv_dim}")
    print(f"normed[0][:10]: {normed[0][:10].tolist()}")
    print(f"Q[0][:10]: {Q[0][:10].tolist()}")
    print(f"K[0][:10]: {K[0][:10].tolist()}")
    print(f"V[0][:10]: {V[0][:10].tolist()}")

    # Binary format:
    #   seq_len(i32), hidden_size(i32), kv_dim(i32), token_ids(i32 * seq_len)
    #   normed, Q, K, V  (all float32)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(struct.pack("<iii", seq_len, hidden_size, kv_dim))
        for tid in token_ids:
            f.write(struct.pack("<i", tid))
        for tensor in (normed, Q, K, V):
            f.write(tensor.detach().numpy().tobytes())

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
