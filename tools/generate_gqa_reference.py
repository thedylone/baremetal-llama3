#!/usr/bin/env python3
import argparse
import json
import math
import os
import struct
from pathlib import Path

import numpy as np


def read_qkv_reference(path: Path):
    with path.open("rb") as f:
        seq_len = struct.unpack("<i", f.read(4))[0]
        hidden_size = struct.unpack("<i", f.read(4))[0]
        kv_dim = struct.unpack("<i", f.read(4))[0]

        token_ids = np.frombuffer(f.read(seq_len * 4), dtype=np.int32).copy()

        n_normed = seq_len * hidden_size
        n_q = seq_len * hidden_size
        n_k = seq_len * kv_dim
        n_v = seq_len * kv_dim

        _normed = np.frombuffer(f.read(n_normed * 4), dtype=np.float32).copy()
        q = np.frombuffer(f.read(n_q * 4), dtype=np.float32).copy()
        k = np.frombuffer(f.read(n_k * 4), dtype=np.float32).copy()
        v = np.frombuffer(f.read(n_v * 4), dtype=np.float32).copy()

    return {
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "kv_dim": kv_dim,
        "token_ids": token_ids,
        "q": q.reshape(seq_len, hidden_size),
        "k": k.reshape(seq_len, kv_dim),
        "v": v.reshape(seq_len, kv_dim),
    }


def read_rope_reference(path: Path):
    with path.open("rb") as f:
        seq_len = struct.unpack("<i", f.read(4))[0]
        hidden_size = struct.unpack("<i", f.read(4))[0]
        kv_dim = struct.unpack("<i", f.read(4))[0]

        token_ids = np.frombuffer(f.read(seq_len * 4), dtype=np.int32).copy()

        q_rope = np.frombuffer(
            f.read(seq_len * hidden_size * 4), dtype=np.float32
        ).copy()
        k_rope = np.frombuffer(
            f.read(seq_len * kv_dim * 4), dtype=np.float32
        ).copy()

    return {
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "kv_dim": kv_dim,
        "token_ids": token_ids,
        "q_rope": q_rope.reshape(seq_len, hidden_size),
        "k_rope": k_rope.reshape(seq_len, kv_dim),
    }


def try_read_heads_from_config(config_path: Path):
    if not config_path.exists():
        return None, None
    with config_path.open("r") as f:
        cfg = json.load(f)

    num_heads = cfg.get("num_attention_heads", None)
    num_kv_heads = cfg.get("num_key_value_heads", None)
    if num_heads is None and "n_head" in cfg:
        num_heads = cfg["n_head"]
    if num_kv_heads is None and "n_kv_head" in cfg:
        num_kv_heads = cfg["n_kv_head"]
    return num_heads, num_kv_heads


def causal_softmax(scores: np.ndarray) -> np.ndarray:
    # scores: [S, S]
    s = scores.shape[0]
    mask = np.triu(np.ones((s, s), dtype=bool), k=1)
    scores = scores.copy()
    scores[mask] = -1e6  # match CUDA kernel behavior
    row_max = scores.max(axis=1, keepdims=True)
    ex = np.exp(scores - row_max)
    ex_sum = ex.sum(axis=1, keepdims=True) + 1e-6
    return ex / ex_sum


def compute_gqa(q_rope, k_rope, v, seq_len, num_heads, num_kv_heads, head_dim):
    # reshape to [S, H, D], [S, H_kv, D], [S, H_kv, D]
    qh = q_rope.reshape(seq_len, num_heads, head_dim)
    kh = k_rope.reshape(seq_len, num_kv_heads, head_dim)
    vh = v.reshape(seq_len, num_kv_heads, head_dim)

    out = np.zeros((seq_len, num_heads, head_dim), dtype=np.float32)
    heads_per_kv = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    for h in range(num_heads):
        kv_h = h // heads_per_kv
        q_mat = qh[:, h, :]  # [S, D]
        k_mat = kh[:, kv_h, :]  # [S, D]
        v_mat = vh[:, kv_h, :]  # [S, D]

        scores = (q_mat @ k_mat.T) * scale  # [S, S]
        alpha = causal_softmax(scores)  # [S, S]
        out[:, h, :] = alpha @ v_mat  # [S, D]

    return out.reshape(seq_len, num_heads * head_dim)


def write_gqa_reference(
    path: Path,
    seq_len,
    hidden_size,
    kv_dim,
    num_heads,
    num_kv_heads,
    head_dim,
    token_ids,
    out,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<i", seq_len))
        f.write(struct.pack("<i", hidden_size))
        f.write(struct.pack("<i", kv_dim))
        f.write(struct.pack("<i", num_heads))
        f.write(struct.pack("<i", num_kv_heads))
        f.write(struct.pack("<i", head_dim))
        f.write(token_ids.astype(np.int32).tobytes())
        f.write(out.astype(np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Generate GQA reference binary for C++ tests."
    )
    parser.add_argument("--qkv-ref", default="tests/data/qkv_reference.bin")
    parser.add_argument("--rope-ref", default="tests/data/rope_reference.bin")
    parser.add_argument("--out", default="tests/data/gqa_reference.bin")
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--config", default="assets/llama3/dumped/config.json")
    args = parser.parse_args()

    qkv = read_qkv_reference(Path(args.qkv_ref))
    rope = read_rope_reference(Path(args.rope_ref))

    assert qkv["seq_len"] == rope["seq_len"], "seq_len mismatch"
    assert qkv["hidden_size"] == rope["hidden_size"], "hidden_size mismatch"
    assert qkv["kv_dim"] == rope["kv_dim"], "kv_dim mismatch"
    assert np.array_equal(
        qkv["token_ids"], rope["token_ids"]
    ), "token_ids mismatch"

    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads

    if num_heads is None or num_kv_heads is None:
        cfg_heads, cfg_kv_heads = try_read_heads_from_config(Path(args.config))
        if num_heads is None:
            num_heads = cfg_heads
        if num_kv_heads is None:
            num_kv_heads = cfg_kv_heads

    if num_heads is None or num_kv_heads is None:
        raise ValueError(
            "Could not infer num_heads/num_kv_heads. "
            "Pass --num-heads and --num-kv-heads explicitly."
        )

    seq_len = qkv["seq_len"]
    hidden_size = qkv["hidden_size"]
    kv_dim = qkv["kv_dim"]

    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    head_dim = hidden_size // num_heads

    if kv_dim != num_kv_heads * head_dim:
        raise ValueError("kv_dim must equal num_kv_heads * head_dim")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    out = compute_gqa(
        rope["q_rope"].astype(np.float32),
        rope["k_rope"].astype(np.float32),
        qkv["v"].astype(np.float32),
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
    )

    write_gqa_reference(
        Path(args.out),
        seq_len,
        hidden_size,
        kv_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        qkv["token_ids"],
        out,
    )

    print(f"Wrote {args.out}")
    print(
        f"seq_len={seq_len} hidden_size={hidden_size} kv_dim={kv_dim} "
        f"num_heads={num_heads} num_kv_heads={num_kv_heads} head_dim={head_dim}"
    )


if __name__ == "__main__":
    main()
