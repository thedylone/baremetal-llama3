#!/usr/bin/env python3
"""Generate reference forward-pass output for C++ verification.

Runs a single forward pass through the Llama 3 model using PyTorch,
saves the logits and argmax token to a binary file.

Usage:
    python tools/generate_forward_reference.py
"""

import struct
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "assets/llama3"
OUTPUT_PATH = "tests/data/forward_reference.bin"
SENTENCE = "Yellow Fruit Like a"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    token_ids = tokenizer.encode(SENTENCE, add_special_tokens=True)
    seq_len = len(token_ids)
    print(f"Input: \"{SENTENCE}\"")
    print(f"Token IDs ({seq_len}): {token_ids}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32
    )
    model.eval()

    with torch.no_grad():
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        outputs = model(input_ids)
        # logits shape: [1, seq_len, vocab_size] — take last token
        logits = outputs.logits[0, -1, :].float()

    vocab_size = logits.shape[0]
    next_token_id = int(logits.argmax().item())
    next_token = tokenizer.decode([next_token_id])

    print(f"vocab_size={vocab_size}")
    print(f"Next token: id={next_token_id} \"{next_token}\"")
    print(f"Top-5 logits: {logits.topk(5)}")

    # Binary format:
    #   seq_len(i32), vocab_size(i32), expected_token_id(i32),
    #   token_ids(i32 * seq_len), logits(float32 * vocab_size)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(struct.pack("<iii", seq_len, vocab_size, next_token_id))
        for tid in token_ids:
            f.write(struct.pack("<i", tid))
        f.write(logits.detach().numpy().tobytes())

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
