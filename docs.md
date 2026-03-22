# CS265 LLM Inference Project Docs

# Dump & Load

## Dumper

The Python Dumper is responsible for converting model weights from the Safetensors format to a custom binary format that can be used by the C++ inference engine. The binary format is designed to be efficient and easy to parse.

### Binary Format Specification

Each tensor is dumped in the following format:

```
[magic number (4 bytes)]
[tensor name length (4 bytes)]
[tensor name (UTF-8)]
[number of dimensions (4 bytes)]
[dimension sizes (8 bytes each)]
[data type (4 bytes)]
[tensor data (raw bytes)]
```

The magic number is a fixed value to identify the file format. It helps ensure that the file is being read correctly and is not corrupted.

We save the tensor information and data in a single binary file per tensor. The file name is derived from the tensor name, sanitized to be filesystem-friendly (e.g., replacing slashes with underscores).

### Metadata

In addition to the binary files, a `metadata.json` file is created with the following structure:

```json
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

## Loader

The C++ Loader uses `MappedFile`s following RAII to manage the lifecycle of memory mapped files, releasing and unmapping when the `MappedFile` goes out of scope.

These `MappedFile`s are held by `TensorInfo` structs, which contain the extracted metadata and offset from the dumped file.

A private implementation class is used to load the tensor and populate a `TensorInfo`. Once the embeddings are loaded via `load_embeddings`, the embedding `MappedFile` is stored, allowing `vocab_size` and `embedding_dim` to be cached.