#!/usr/bin/env python3
"""Generate reference output after layer 0 for debugging the decoder block.

Saves: hidden state after full layer 0 (attention + FFN + both residuals),
and final logits.

Usage:
    python tools/generate_layer0_reference.py
"""

import struct
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "assets/llama3"
OUTPUT_PATH = "tests/data/layer0_reference.bin"
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

    # Hook to capture hidden state after layer 0
    captured = {}

    def hook_after_layer0(module, input, output):
        # output is a tuple: (hidden_states, ...)
        captured["after_layer0"] = output[0].detach().float()

    model.model.layers[0].register_forward_hook(hook_after_layer0)

    # Also capture after attention (before FFN) in layer 0
    def hook_after_attn(module, input, output):
        # For the attention module, output is (attn_output, ...)
        captured["attn_output"] = output[0].detach().float()

    model.model.layers[0].self_attn.register_forward_hook(hook_after_attn)

    with torch.no_grad():
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :].float()

    # Extract captured tensors (handle tuple outputs from HuggingFace hooks)
    raw_layer = captured["after_layer0"]
    raw_attn = captured["attn_output"]
    if isinstance(raw_layer, tuple):
        raw_layer = raw_layer[0]
    if isinstance(raw_attn, tuple):
        raw_attn = raw_attn[0]

    # Remove batch dimension: [1, seq, hidden] -> [seq, hidden]
    after_layer0 = raw_layer.squeeze(0).float()
    attn_output = raw_attn.squeeze(0).float()

    next_token_id = int(logits.argmax().item())
    next_token = tokenizer.decode([next_token_id])
    print(f"Next token: id={next_token_id} \"{next_token}\"")
    print(f"after_layer0 shape: {after_layer0.shape}")
    print(f"after_layer0[0][:10]: {after_layer0[0][:10].tolist()}")
    print(f"attn_output[0][:10]: {attn_output[0][:10].tolist()}")

    # Binary format:
    #   seq_len(i32), hidden_size(i32),
    #   token_ids(i32 * seq_len),
    #   attn_output(float32 * seq_len * hidden_size),
    #   after_layer0(float32 * seq_len * hidden_size)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    hidden_size = after_layer0.shape[1]
    with open(OUTPUT_PATH, "wb") as f:
        f.write(struct.pack("<ii", seq_len, hidden_size))
        for tid in token_ids:
            f.write(struct.pack("<i", tid))
        f.write(attn_output.numpy().tobytes())
        f.write(after_layer0.numpy().tobytes())

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
