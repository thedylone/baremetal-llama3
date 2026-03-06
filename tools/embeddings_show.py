"""
Verify C++ embedding loader output against ground-truth embeddings.

Loads the embedding table directly from the original .safetensors weights
(not the dumped .bin files) and compares it against the float32 values
produced by the C++ loader.

Usage
-----
Basic check (prints per-token stats):
    python verify_embeddings.py \
        --model-dir  assets/llama3 \
        --cpp-output path/to/cpp_embeddings.bin \
        --token-ids  128000 9906 1917

Pass a tolerance to treat the run as pass/fail:
    python verify_embeddings.py ... --atol 1e-3 --rtol 1e-3

C++ binary format expected for --cpp-output
--------------------------------------------
The file must be a flat sequence of little-endian float32 values with no
header: N_tokens × embedding_dim floats written row-major.  In your C++
code, after calling get_embeddings(), write the buffer like:

    std::vector<float> embs = loader.get_embeddings(token_ids);
    std::ofstream out("cpp_embeddings.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(embs.data()),
              embs.size() * sizeof(float));
"""

import argparse
import sys
from pathlib import Path
from typing import List
import json
import numpy as np
import safetensors
import torch


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

EMBEDDING_TENSOR = "model.embed_tokens.weight"


def load_embedding_table(model_dir: Path) -> torch.Tensor:
    """
    Load the full embedding table from the original .safetensors weights.

    The tensor is returned as float32 on CPU regardless of how it was stored
    on disk (BF16, FP16, FP32) so comparisons are always in the same space
    as the C++ loader output.
    """
    index_path = model_dir / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_file = index["weight_map"][EMBEDDING_TENSOR]
    else:
        # Single-file checkpoint
        shard_file = "model.safetensors"

    shard_path = model_dir / shard_file
    if not shard_path.exists():
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    with safetensors.safe_open(str(shard_path), framework="pt") as f:
        table = f.get_tensor(EMBEDDING_TENSOR)

    return table.to(torch.float32).cpu()


def ground_truth_embeddings(
    table: torch.Tensor,
    token_ids: List[int],
) -> np.ndarray:
    """
    Return float32 embeddings for *token_ids* as a (N, D) numpy array.

    Out-of-range token IDs are filled with zeros, matching the C++ behaviour.
    """
    vocab_size, embed_dim = table.shape
    out = np.zeros((len(token_ids), embed_dim), dtype=np.float32)
    for i, tid in enumerate(token_ids):
        if 0 <= tid < vocab_size:
            out[i] = table[tid].numpy()
    return out


# ---------------------------------------------------------------------------
# C++ output loader
# ---------------------------------------------------------------------------


def load_cpp_embeddings(
    cpp_output_path: Path,
    n_tokens: int,
    embed_dim: int,
) -> np.ndarray:
    """
    Read the flat float32 binary written by the C++ get_embeddings() call.

    Expected layout: n_tokens × embed_dim little-endian float32 values,
    no header, row-major.
    """
    expected_bytes = n_tokens * embed_dim * 4
    actual_bytes = cpp_output_path.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError(
            f"C++ output file is {actual_bytes} bytes but expected "
            f"{expected_bytes} bytes for {n_tokens} tokens × "
            f"{embed_dim} dims × 4 bytes/float.\n"
            f"  Check that you wrote exactly the right number of tokens "
            f"and that the embedding dimension matches the model."
        )

    data = np.fromfile(str(cpp_output_path), dtype="<f4")  # little-endian f32
    return data.reshape(n_tokens, embed_dim)


# ---------------------------------------------------------------------------
# Comparison + reporting
# ---------------------------------------------------------------------------


def compare(
    ref: np.ndarray,
    got: np.ndarray,
    token_ids: List[int],
    atol: float,
    rtol: float,
) -> bool:
    """
    Compare reference and C++ embeddings.  Prints per-token statistics and
    returns True only if every token passes the tolerance check.
    """
    assert (
        ref.shape == got.shape
    ), f"Shape mismatch: reference {ref.shape} vs C++ {got.shape}"

    n_tokens, embed_dim = ref.shape
    all_pass = True

    col_w = max(len(str(tid)) for tid in token_ids)
    header = (
        f"{'token_id':>{col_w}}  "
        f"{'max_abserr':>12}  "
        f"{'mean_abserr':>12}  "
        f"{'max_relerr':>12}  "
        f"{'status':>6}"
    )
    print(header)
    print("-" * len(header))

    for i, tid in enumerate(token_ids):
        abs_err = np.abs(ref[i] - got[i])
        rel_err = abs_err / (np.abs(ref[i]) + 1e-9)

        max_abs = float(abs_err.max())
        mean_abs = float(abs_err.mean())
        max_rel = float(rel_err.max())

        # A token passes if EVERY element satisfies
        # |got - ref| <= atol + rtol*|ref|
        tol_pass = bool(np.all(abs_err <= atol + rtol * np.abs(ref[i])))
        status = "PASS" if tol_pass else "FAIL"
        if not tol_pass:
            all_pass = False

        print(
            f"{tid:>{col_w}}  "
            f"{max_abs:>12.6e}  "
            f"{mean_abs:>12.6e}  "
            f"{max_rel:>12.6e}  "
            f"{status:>6}"
        )

    print()
    print(f"Tokens : {n_tokens}")
    print(f"Dim    : {embed_dim}")
    print(f"Tol    : atol={atol:.2e}  rtol={rtol:.2e}")

    # Global stats
    abs_all = np.abs(ref - got)
    print(f"Global max |err|  : {abs_all.max():.6e}")
    print(f"Global mean |err| : {abs_all.mean():.6e}")

    # Show the worst offending position if anything failed
    if not all_pass:
        worst_token_idx = int(
            np.unravel_index(abs_all.argmax(), abs_all.shape)[0]
        )
        worst_dim = int(np.unravel_index(abs_all.argmax(), abs_all.shape)[1])
        print(
            f"\nWorst mismatch at token_id={token_ids[worst_token_idx]}, "
            f"dim={worst_dim}: "
            f"ref={ref[worst_token_idx, worst_dim]:.8f}, "
            f"got={got[worst_token_idx, worst_dim]:.8f}"
        )

    return all_pass


def print_sample(
    ref: np.ndarray,
    token_ids: List[int],
    n: int = 8,
    got: np.ndarray = None,
) -> None:
    """Print the first *n* values of each token
    for a quick visual sanity check."""
    print(f"\n--- First {n} values per token ---")
    for i, tid in enumerate(token_ids):
        ref_vals = " ".join(f"{v:+.6f}" for v in ref[i, :n])
        print(f"  token {tid}")
        print(f"    ref: {ref_vals}")
        if got is not None:
            got_vals = " ".join(f"{v:+.6f}" for v in got[i, :n])
            print(f"    cpp: {got_vals}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify C++ get_embeddings() output against ground-truth "
            "safetensors weights."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir",
        help="Directory containing config.json and .safetensors weights.",
        default="assets/llama3",
    )
    parser.add_argument(
        "--cpp-output",
        default=None,
        help=(
            "Path to the flat float32 binary written by the C++ loader. "
            "Layout: N_tokens x embed_dim little-endian float32, no header. "
            "If omitted, only the Python reference embeddings are printed."
        ),
    )
    parser.add_argument(
        "--token-ids",
        nargs="+",
        type=int,
        metavar="ID",
        help="Tokens to verify (must match exactly what was passed to C++).",
        default=[128000, 9906, 1917],
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for pass/fail (default: 1e-3).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for pass/fail (default: 1e-3).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=8,
        metavar="N",
        help="Print the first N values of each embedding."
        "(default: 8, 0 to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)

    # ------------------------------------------------------------------
    # 1. Load ground-truth embedding table from safetensors
    # ------------------------------------------------------------------
    print(f"Loading embedding table from {model_dir} ...")
    table = load_embedding_table(model_dir)
    vocab_size, embed_dim = table.shape
    print(
        f"  vocab_size={vocab_size:,}  embed_dim={embed_dim:,}  "
        f"dtype={table.dtype}\n"
    )

    # ------------------------------------------------------------------
    # 2. Compute reference embeddings
    # ------------------------------------------------------------------
    token_ids = args.token_ids
    ref = ground_truth_embeddings(table, token_ids)

    # ------------------------------------------------------------------
    # 3. Print-only mode: no C++ output provided
    # ------------------------------------------------------------------
    if args.cpp_output is None:
        if args.sample > 0:
            print_sample(ref, token_ids, n=args.sample)
        print()
        print("(No --cpp-output provided. Skipping comparison.)")
        sys.exit(0)

    # ------------------------------------------------------------------
    # 4. Load C++ output
    # ------------------------------------------------------------------
    cpp_output_path = Path(args.cpp_output)
    print(f"Loading C++ output from {cpp_output_path} ...")
    got = load_cpp_embeddings(cpp_output_path, len(token_ids), embed_dim)
    print(f"  Loaded {got.shape[0]} tokens × {got.shape[1]} dims\n")

    # ------------------------------------------------------------------
    # 5. Optional visual sample
    # ------------------------------------------------------------------
    if args.sample > 0:
        print_sample(ref, token_ids, n=args.sample, got=got)
        print()

    # ------------------------------------------------------------------
    # 6. Numerical comparison
    # ------------------------------------------------------------------
    print("--- Per-token error ---")
    passed = compare(ref, got, token_ids, atol=args.atol, rtol=args.rtol)

    # ------------------------------------------------------------------
    # 7. Exit code
    # ------------------------------------------------------------------
    if passed:
        print("\nRESULT: PASS")
        sys.exit(0)
    else:
        print("\nRESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
