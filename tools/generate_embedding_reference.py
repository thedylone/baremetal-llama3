#!/usr/bin/env python3
"""Generate reference embedding outputs for C++ verification.

Loads the embedding table from the original safetensors files,
looks up token embeddings in float32, and writes the results
to a binary file that the C++ test can load and compare against.

Usage:
    python tools/generate_embedding_reference.py
"""

import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_DIR = "assets/llama3"
OUTPUT_PATH = "tests/data/embedding_reference.bin"
SENTENCE = "The quick brown fox jumps over the lazy dog."


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    token_ids = tokenizer.encode(SENTENCE, add_special_tokens=True)
    print(f"Token IDs ({len(token_ids)}): {token_ids}")

    with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    shard = weight_map["model.embed_tokens.weight"]
    with safe_open(f"{MODEL_DIR}/{shard}", framework="pt") as f:
        embed = f.get_tensor("model.embed_tokens.weight").float()

    seq_len = len(token_ids)
    hidden_size = embed.shape[1]
    ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    embeddings = embed[ids_tensor]  # [seq, hidden]

    print(f"seq_len={seq_len}, hidden_size={hidden_size}")
    print(f"embeddings[0][:10]: {embeddings[0][:10].tolist()}")

    # Binary format:
    #   seq_len(i32), hidden_size(i32), token_ids(i32 * seq_len),
    #   embeddings(float32 * seq_len * hidden_size)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(struct.pack("<ii", seq_len, hidden_size))
        for tid in token_ids:
            f.write(struct.pack("<i", tid))
        f.write(embeddings.detach().numpy().tobytes())

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
