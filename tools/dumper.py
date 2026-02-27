# Your logic for dumping the weights goes here.
# You have to prepare binary files, in a format that you use for loading them
# in the C++ files.
# You may use any library listed in requirements.txt
# Read the project description for more information.

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import safetensors
import torch
import tqdm  # used for progress bars


class ModelDumper:
    """Handles dumping of model weights from safetensors to binary format.

    Dumps each tensor in this format:
    ```
    [magic number (4 bytes)]
    [tensor name length (4 bytes)]
    [tensor name (UTF-8)]
    [number of dimensions (4 bytes)]
    [dimension sizes (8 bytes each)]
    [data type (4 bytes)]
    [tensor data (raw bytes)]
    ```
    Metadata is saved in metadata.json with the following structure:
    ```
    {
        "config": { ... },  // Original model config
        "tensors": {
            [tensor_name]: {
                "file": [sanitized_tensor_name.bin],
                "shape": [dim1, dim2, ...],
                "dtype": [target_dtype],
                "size_bytes": 123456
            },
            ...
        },
        "dtype": [target_dtype],
        "magic": [MAGIC_NUMBER],
        "format_version": "1.0"
    }
    ```
    """

    # Magic number for binary format validation
    MAGIC_NUMBER = 0x32363530  # 2650

    # Data type mappings
    DTYPE_MAP = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
    }

    def __init__(
        self, model_dir: str, output_dir: str, target_dtype: str = "bfloat16"
    ):
        """
        Initialize the dumper.

        Args:
            model_dir: Directory containing the model files
            output_dir: Directory to save dumped files
            target_dtype: Target data type for conversion
                ("float32", "float16", "bfloat16"). Default is "bfloat16".
        """
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
        """Validate input directories and files."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )

        required_files = ["config.json", "model.safetensors.index.json"]
        for file in required_files:
            if not (self.model_dir / file).exists():
                raise FileNotFoundError(f"Required file not found: {file}")

    def _load_config(self) -> Dict:
        """Load model configuration."""
        config_path = self.model_dir / "config.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise RuntimeError(f"Failed to load config: {e}") from e

    def _load_index(self) -> Dict:
        """Load safetensors index."""
        index_path = self.model_dir / "model.safetensors.index.json"
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise RuntimeError(f"Failed to load index: {e}") from e

    @staticmethod
    def sanitize_filename(tensor_name: str) -> str:
        """Convert tensor name to safe filename."""
        return tensor_name.replace("/", "_").replace(".", "_")

    def _write_tensor_header(
        self, file, tensor_name: str, shape: List[int]
    ) -> None:
        """Write binary tensor header.
        ```
        [magic number (4 bytes)]
        [tensor name length (4 bytes)]
        [tensor name (UTF-8)]
        [number of dimensions (4 bytes)]
        [dimension sizes (8 bytes each)]
        [data type (4 bytes)]
        [tensor data (raw bytes)]
        ```
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
        dtype_id = self.DTYPE_MAP.get(self.target_dtype, 1)
        file.write(struct.pack("<I", dtype_id))

    def _process_tensor(
        self, tensor_name: str, safetensor_file: str
    ) -> Tuple[str, Dict]:
        """
        Process a single tensor.

        Returns:
            Tuple of (output_filename, tensor_metadata)
        """
        safetensor_path = self.model_dir / safetensor_file

        # Load tensor
        with safetensors.safe_open(safetensor_path, framework="pt") as f:
            tensor = f.get_tensor(tensor_name)

        # Convert to target dtype
        tensor = tensor.to(self.target_dtype).cpu()
        # Handle bfloat16 (numpy does not support bfloat16)
        if self.target_dtype == torch.bfloat16:
            tensor = tensor.view(torch.uint16)
        tensor_np = tensor.numpy()

        # Generate output filename
        safe_name = self.sanitize_filename(tensor_name)
        output_filename = f"{safe_name}.bin"
        output_path = self.output_dir / output_filename

        # Write binary file
        with open(output_path, "wb") as f:
            self._write_tensor_header(f, tensor_name, list(tensor.shape))
            f.write(tensor_np.tobytes())

        # Create metadata entry
        metadata = {
            "file": output_filename,
            "shape": list(tensor.shape),
            "dtype": str(self.target_dtype).rsplit(".", maxsplit=1)[-1],
            "size_bytes": tensor_np.nbytes,
        }

        return output_filename, metadata

    def _write_metadata_files(
        self, metadata: Dict, name_mapping: Dict
    ) -> None:
        """Write metadata and mapping files."""
        # Write metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Write name mapping
        mapping_path = self.output_dir / "mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(name_mapping, f, indent=2)

        print(f"✓ Metadata saved to {metadata_path}")
        print(f"✓ Name mapping saved to {mapping_path}")

    def dump_weights(self) -> None:
        """Dump all model weights to binary format."""
        print(
            f"Dumping model weights from {self.model_dir} to {self.output_dir}"
        )
        print(f"Target dtype: {self.target_dtype}")

        # Initialize metadata
        metadata = {
            "config": self.config,
            "tensors": {},
            "dtype": str(self.target_dtype).rsplit(".", maxsplit=1)[-1],
            "magic": self.MAGIC_NUMBER,
            "format_version": "1.0",
        }

        name_mapping = {}

        # Process tensors with progress bar
        weight_map = self.index_data["weight_map"]
        progress = tqdm.tqdm(weight_map.items(), desc="Processing tensors")

        def display_name(tensor_name):
            return tensor_name[:30] + ("..." if len(tensor_name) > 30 else "")

        for tensor_name, file_name in progress:
            progress.set_postfix(tensor=display_name(tensor_name))
            _, tensor_metadata = self._process_tensor(tensor_name, file_name)
            metadata["tensors"][tensor_name] = tensor_metadata
            name_mapping[tensor_name] = self.sanitize_filename(tensor_name)

        # Write metadata files
        self._write_metadata_files(metadata, name_mapping)

        # Summary
        total_size_mb = sum(
            t["size_bytes"] for t in metadata["tensors"].values()
        ) / (1024 * 1024)

        print(f"\n✓ Successfully dumped {len(metadata['tensors'])} tensors")
        print(f"✓ Total size: {total_size_mb:.1f} MB")
        print(f"✓ Output directory: {self.output_dir}")


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
