#!/usr/bin/env python3
"""Generate reference RoPE outputs for C++ verification.

Computes embeddings -> RMSNorm -> Q/K/V -> RoPE in float32 using
the same rotate_half convention as HuggingFace Llama, and writes
Q_rope and K_rope to a binary file.

Usage:
    python tools/generate_rope_reference.py
"""

import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_DIR = "assets/llama3"
OUTPUT_PATH = "tests/data/rope_reference.bin"
SENTENCE = "Hello world"
EPS = 1e-5
ROPE_BASE = 500000.0
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the first half with the second half (HuggingFace convention)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE: q * cos + rotate_half(q) * sin."""
    return q * cos + rotate_half(q) * sin


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    token_ids = tokenizer.encode(SENTENCE, add_special_tokens=True)
    seq_len = len(token_ids)
    print(f"Token IDs ({seq_len}): {token_ids}")

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

    hidden_size = embed.shape[1]
    kv_dim = k_weight.shape[0]

    # Forward: embed -> RMSNorm -> Q/K
    ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    embeddings = embed[ids_tensor]
    normed = rms_norm(embeddings, ln_weight, EPS)
    Q = normed @ q_weight.T  # [seq, hidden_size]
    K = normed @ k_weight.T  # [seq, kv_dim]

    # Build cos/sin tables [seq_len, head_dim/2]
    half_dim = HEAD_DIM // 2
    i = torch.arange(half_dim, dtype=torch.float32)
    theta = 1.0 / (ROPE_BASE ** (2.0 * i / HEAD_DIM))  # [half_dim]
    positions = torch.arange(seq_len, dtype=torch.float32)  # [seq_len]
    angles = positions.unsqueeze(1) * theta.unsqueeze(0)  # [seq_len, half_dim]
    cos_table = torch.cos(angles)  # [seq_len, half_dim]
    sin_table = torch.sin(angles)  # [seq_len, half_dim]

    # Expand cos/sin to full head_dim: [seq_len, head_dim]
    cos_full = torch.cat([cos_table, cos_table], dim=-1)  # [seq_len, head_dim]
    sin_full = torch.cat([sin_table, sin_table], dim=-1)  # [seq_len, head_dim]

    # Reshape Q to [seq, num_heads, head_dim], apply RoPE per head, flatten back
    Q_heads = Q.view(seq_len, NUM_HEADS, HEAD_DIM)
    Q_rope = apply_rope(Q_heads, cos_full.unsqueeze(1), sin_full.unsqueeze(1))
    Q_rope = Q_rope.view(seq_len, hidden_size)

    K_heads = K.view(seq_len, NUM_KV_HEADS, HEAD_DIM)
    K_rope = apply_rope(K_heads, cos_full.unsqueeze(1), sin_full.unsqueeze(1))
    K_rope = K_rope.view(seq_len, kv_dim)

    print(f"Q_rope[0][:10]: {Q_rope[0][:10].tolist()}")
    print(f"K_rope[0][:10]: {K_rope[0][:10].tolist()}")

    # Binary format:
    #   seq_len(i32), hidden_size(i32), kv_dim(i32), token_ids(i32 * seq_len)
    #   Q_rope, K_rope  (float32)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(struct.pack("<iii", seq_len, hidden_size, kv_dim))
        for tid in token_ids:
            f.write(struct.pack("<i", tid))
        f.write(Q_rope.detach().numpy().tobytes())
        f.write(K_rope.detach().numpy().tobytes())

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
