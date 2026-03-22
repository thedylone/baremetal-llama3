# Your logic for dumping the weights goes here.
# You have to prepare binary files, in a format that you use for loading them
# in the C++ files.
# You may use any library listed in requirements.txt
# Read the project description for more information.

"""
Dump LLaMA model weights from .safetensors to binary format for C++ loading.

Each tensor is written to its own .bin file in this format:
    [magic number       (4 bytes, little-endian uint32)]
    [tensor name length (4 bytes, little-endian uint32)]
    [tensor name        (UTF-8, tensor name length bytes)]
    [number of dims     (4 bytes, little-endian uint32)]
    [dimension sizes    (8 bytes each, little-endian uint64)]
    [data type id       (4 bytes, little-endian uint32)]
        0 = float16
        1 = bfloat16  (raw bytes are uint16; reinterpret as bfloat16 in C++)
        2 = float32
    [tensor data        (raw bytes, row-major / C-contiguous)]

The embedding table (model.embed_tokens.weight) is written to embedding.bin
with the same format, and also included in the normal tensors/ directory so
the C++ loader can treat it uniformly if desired.

metadata.json structure:
    {
        "config":           { ... },       // original model config.json
        "dtype":            "bfloat16",    // target dtype string
        "magic":            0x32363530,    // magic number (decimal 843075888)
        "format_version":   "1.0",
        "embedding_tensor": "model.embed_tokens.weight",
        "embedding_file":   "embedding.bin",
        "tensors": {
            "<tensor_name>": {
                "file":       "<safe_name>.bin",
                "shape":      [dim0, dim1, ...],
                "dtype":      "bfloat16",
                "size_bytes": 123456
            },
            ...
        }
    }

NOTE: bfloat16 tensors are stored as raw uint16 bytes on disk. Your C++
loader must reinterpret_cast<bfloat16*> (or __nv_bfloat16*) the data — do
NOT interpret them as uint16 values.
"""

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import safetensors
import torch
import tqdm  # used for progress bars


class ModelDumper:
    """Dumps model weights from .safetensors shards
    to per-tensor binary files"""

    # Magic number for binary format validation
    MAGIC_NUMBER = 0x32363530  # 2650

    # Canonical name of the embedding tensor in LLaMA 3
    EMBEDDING_TENSOR = "model.embed_tokens.weight"

    # Data type mappings
    DTYPE_MAP = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
    }
    VALID_DTYPES = {"float32", "float16", "bfloat16"}

    def __init__(
        self, model_dir: str, output_dir: str, target_dtype: str = "bfloat16"
    ) -> None:
        """
        Initialize the dumper.

        Args:
            model_dir: Directory containing the model files
            output_dir: Directory to save dumped files
            target_dtype: Target data type for conversion
                ("float32", "float16", "bfloat16"). Default is "bfloat16".
        """
        if target_dtype not in self.VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype '{target_dtype}'. \
                Must be one of {self.VALID_DTYPES}"
            )
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.target_dtype = getattr(torch, target_dtype)

        # Validate inputs
        self._validate_inputs()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model configuration
        self.config = self._load_config()
        self.index_data = self._load_index()

    def _validate_inputs(self) -> None:
        """Check that the model directory and required files exist."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Required file not found: {self.model_dir / 'config.json'}"
            )

        has_index = (self.model_dir / "model.safetensors.index.json").exists()
        has_single = (self.model_dir / "model.safetensors").exists()
        if not has_index and not has_single:
            raise FileNotFoundError(
                f"No safetensors weights found in {self.model_dir}. "
                "Expected 'model.safetensors.index.json' (sharded) "
                "or 'model.safetensors' (single-file)."
            )

    def _load_json(self, filename: str) -> Dict:
        """Load and return a JSON file from model_dir."""
        path = self.model_dir / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            raise RuntimeError(f"Failed to load {path}: {exc}") from exc

    def _load_config(self) -> Dict:
        """Load model configuration."""
        return self._load_json("config.json")

    def _load_index(self) -> Dict:
        """Load safetensors index."""
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            return self._load_json("model.safetensors.index.json")

        # Single-file fallback: enumerate tensors directly from the file
        single = self.model_dir / "model.safetensors"
        with safetensors.safe_open(str(single), framework="pt") as f:
            keys = f.keys()
        return {"weight_map": {k: "model.safetensors" for k in keys}}

    @staticmethod
    def sanitize_filename(tensor_name: str) -> str:
        """Convert tensor name to safe filename."""
        return tensor_name.replace("/", "_").replace(".", "_")

    def _write_tensor_header(
        self, file, tensor_name: str, shape: List[int]
    ) -> None:
        """
        Write the fixed-layout binary header for one tensor.

        Layout (all little-endian):
            magic number       : uint32
            tensor name length : uint32
            tensor name        : <length> bytes, UTF-8
            number of dims     : uint32
            dimension sizes    : uint64 × ndim
            data type id       : uint32
        """
        # Magic number
        file.write(struct.pack("<I", self.MAGIC_NUMBER))

        # Tensor name
        name_bytes = tensor_name.encode("utf-8")
        file.write(struct.pack("<I", len(name_bytes)))
        file.write(name_bytes)

        # Shape
        file.write(struct.pack("<I", len(shape)))
        for dim in shape:
            # Use 8 bytes for size_t compatibility
            file.write(struct.pack("<Q", dim))

        # Data type
        dtype_id = self.DTYPE_MAP[self.target_dtype]
        file.write(struct.pack("<I", dtype_id))

    def _write_tensor_to_path(
        self,
        tensor_name: str,
        tensor: torch.Tensor,
        output_path: Path,
    ) -> Dict:
        """
        Convert *tensor* to the target dtype and write it to *output_path*.

        Returns the metadata dict for this tensor.
        """
        tensor = tensor.to(self.target_dtype).cpu()

        # Capture the logical shape BEFORE any view() reinterpretation.
        original_shape = list(tensor.shape)

        # numpy does not support bfloat16; reinterpret as uint16 for tobytes().
        # The dtype field in header/metadata still records "bfloat16" so the
        # C++ loader knows to reinterpret_cast<__nv_bfloat16*> the raw bytes.
        if self.target_dtype == torch.bfloat16:
            tensor_for_export = tensor.view(torch.uint16)
        else:
            tensor_for_export = tensor

        tensor_np = tensor_for_export.numpy()

        with open(output_path, "wb") as f:
            self._write_tensor_header(f, tensor_name, original_shape)
            f.write(tensor_np.tobytes())

        dtype_str = str(self.target_dtype).rsplit(".", maxsplit=1)[-1]
        return {
            "file": output_path.name,
            "shape": original_shape,
            "dtype": dtype_str,
            "size_bytes": int(tensor_np.nbytes),
        }

    def _group_tensors_by_shard(self) -> Dict[str, List[str]]:
        """
        Return a mapping of shard filename -> [tensor names in that shard].

        Grouping ensures each .safetensors file is opened exactly once,
        rather than once per tensor.
        """
        shards: Dict[str, List[str]] = defaultdict(list)
        for tensor_name, shard_file in self.index_data["weight_map"].items():
            shards[shard_file].append(tensor_name)
        return dict(shards)

    def _dump_shard(
        self,
        shard_file: str,
        tensor_names: List[str],
    ) -> Dict[str, Dict]:
        """
        Open one shard and write every tensor it contains to disk.

        Returns per-tensor metadata dicts keyed by tensor name.
        """
        shard_path = self.model_dir / shard_file
        tensors_meta: Dict[str, Dict] = {}

        with safetensors.safe_open(str(shard_path), framework="pt") as sf:
            progress = tqdm.tqdm(
                tensor_names,
                desc=f"  {shard_file}",
                unit="tensor",
                leave=False,
            )
            for tensor_name in progress:
                short = (
                    tensor_name[:40] + "..."
                    if len(tensor_name) > 40
                    else tensor_name
                )
                progress.set_postfix(tensor=short)
                tensors_meta[tensor_name] = self._dump_tensor(sf, tensor_name)

        return tensors_meta

    def _dump_tensor(
        self,
        sf,  # open safetensors.safe_open handle
        tensor_name: str,
    ) -> Dict:
        """
        Write a single tensor to its <safe_name>.bin output file.

        If the tensor is the embedding table it is also written to
        embedding.bin so the C++ loader can mmap it independently.
        Returns the metadata dict for the tensor's primary file entry.
        """
        tensor = sf.get_tensor(tensor_name)
        safe_name = self.sanitize_filename(tensor_name)
        primary = self.output_dir / f"{safe_name}.bin"

        meta = self._write_tensor_to_path(tensor_name, tensor, primary)

        if tensor_name == self.EMBEDDING_TENSOR:
            emb_path = self.output_dir / "embedding.bin"
            self._write_tensor_to_path(tensor_name, tensor, emb_path)

        return meta

    def _verify_output_count(
        self,
        tensors_meta: Dict[str, Dict],
        has_embedding: bool,
    ) -> None:
        """
        Warn if the number of .bin files on disk does not match expectations.

        When the embedding table is present there is one extra .bin file
        (embedding.bin) beyond the per-tensor count.
        """
        dumped_bins = len(list(self.output_dir.glob("*.bin")))
        expected_bins = len(tensors_meta) + (1 if has_embedding else 0)
        if dumped_bins != expected_bins:
            print(
                f"\nWARNING: expected {expected_bins} .bin files, "
                f"found {dumped_bins}. Some tensors may not have been written."
            )

    def _build_metadata(
        self,
        tensors_meta: Dict[str, Dict],
        embedding_file: Optional[str],
    ) -> Dict:
        """Assemble the full metadata dict that will be
        written to metadata.json."""
        dtype_str = str(self.target_dtype).rsplit(".", maxsplit=1)[-1]
        return {
            "config": self.config,
            "dtype": dtype_str,
            "magic": self.MAGIC_NUMBER,
            "format_version": "1.0",
            "embedding_tensor": self.EMBEDDING_TENSOR,
            "embedding_file": embedding_file,
            "tensors": tensors_meta,
        }

    def _write_metadata(self, metadata: Dict) -> None:
        """Serialise *metadata* to metadata.json in the output directory."""
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _print_summary(
        self,
        tensors_meta: Dict[str, Dict],
        embedding_file: Optional[str],
    ) -> None:
        """Print a human-readable summary of what was written."""
        total_mb = sum(t["size_bytes"] for t in tensors_meta.values()) / (
            1024 * 1024
        )
        print(f"\n✓ Dumped {len(tensors_meta)} tensors")
        print(f"✓ Total size : {total_mb:.1f} MB")
        if embedding_file:
            emb_shape = tensors_meta.get(self.EMBEDDING_TENSOR, {}).get(
                "shape"
            )
            print(f"✓ Embedding  : {embedding_file}  shape={emb_shape}")
        print(f"✓ Metadata   : {self.output_dir / 'metadata.json'}")
        print(f"✓ Output dir : {self.output_dir}")

    def dump_weights(self) -> None:
        """Dump all model weights to binary format."""
        print(f"Model dir  : {self.model_dir}")
        print(f"Output dir : {self.output_dir}")
        print(f"Target dtype: {self.target_dtype}")

        shards = self._group_tensors_by_shard()
        tensors_meta: Dict[str, Dict] = {}

        for shard_file, tensor_names in tqdm.tqdm(
            shards.items(), desc="Shards", unit="shard"
        ):
            tensors_meta.update(self._dump_shard(shard_file, tensor_names))

        has_embedding = self.EMBEDDING_TENSOR in tensors_meta
        embedding_file = "embedding.bin" if has_embedding else None

        self._verify_output_count(tensors_meta, has_embedding)
        self._write_metadata(
            self._build_metadata(tensors_meta, embedding_file)
        )
        self._print_summary(tensors_meta, embedding_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dump LLaMA model weights to binary format"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="assets/llama3",
        help="Input model directory (default: assets/llama3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/llama3/dumped",
        help="Output directory (default: assets/llama3/dumped)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Target data type (default: bfloat16)",
    )

    args = parser.parse_args()

    dumper = ModelDumper(args.model_dir, args.output_dir, args.dtype)
    dumper.dump_weights()


if __name__ == "__main__":
    main()
