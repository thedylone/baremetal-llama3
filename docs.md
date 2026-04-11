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

## Special Tokens

> In tokenizer code, you see some special tokens such as `<|start header id|>` or `<|begin of text|>`. Check what these tokens are, why they are necessary, and how to benefit from each token.

Special tokens are used in tokenization to represent specific elements in the text that may not be directly represented by regular tokens. They serve various purposes, such as indicating the start or end of a sequence, separating different parts of the input, or representing special concepts.
For example, `<|start header id|>` might be used to indicate the beginning of a header section in a document, while `<|begin of text|>` could signify the start of the main content. These tokens help the model understand the structure of the input and can improve the quality of the generated output by providing clear markers for different sections or elements in the text.

## Granularity of Tokens

> Our tokenization determine our sequence length and hence directly affect our time and memory. You may notice that our tokenizer may even use multi-tokens for a single word. What are the benefits and drawbacks of fine-grained tokenization? Why we do not use one token id for each word?

Fine‑grained tokenization (subwords/bytes) trades vocabulary size for sequence length.

**Benefits**

- **Smaller vocab**: lower embedding/softmax memory.
- **Open‑vocabulary**: can represent any word (names, typos, new terms).
- **Better morphology**: shared subword pieces improve generalization.

**Drawbacks**

- **Longer sequences**: higher time/memory per request.
- **Fragmentation**: word boundaries are less explicit; may affect readability.

**Why not one token per word?**

- **Vocab would explode** (millions of tokens): huge embeddings and poor memory use.
- **OOV problem**: unseen words cannot be represented.
- **Poor generalization**: no sharing across related word forms.

## Load-Time Bottlenecks & Speedups

> Loading all model weights can be time-consuming. Although it is a one time job, do you know application that are bottlenecked by this and also think about ways to speed up this step.

**Bottlenecked applications**

- **Serverless / autoscaling**: cold‑start latency dominates first request.
- **Edge / mobile**: slower storage and limited RAM.
- **Batch jobs / CI**: frequent process restarts amplify load time.
- **Notebook workflows**: repeated kernel restarts.

**Ways to speed up**

- **Memory‑map weights** (mmap) to avoid full upfront read.
- **Lazy / demand loading** of layers or experts.
- **Sharded checkpoints** + **parallel I/O**.
- **Quantization / smaller dtypes** to reduce bytes read.
- **Warm pools** or **pre‑loaded workers** to amortize cold starts.
- **Faster storage** (NVMe, local cache) and **OS page cache** reuse.
- **Pack tensors contiguously** to reduce seeks.

## Embedding Table Placement (CPU vs GPU)

> Think of where is the best place for storing embedding table. CPU or GPU? What are the trade-offs?

**GPU**

- **Pros**: fastest lookup and matmul; lowest latency during inference.
- **Cons**: limited VRAM; large vocab embeddings can crowd out activations/weights.

**CPU**

- **Pros**: much larger capacity; cheaper memory; easier to host huge vocabularies.
- **Cons**: slower lookup; PCIe transfer overhead if embeddings are used on GPU.

**Practical trade‑offs**

- **Latency‑critical**: keep embeddings on GPU if they fit.
- **Memory‑constrained**: keep embeddings on CPU and **cache hot rows** on GPU.
- **Hybrid**: quantize embeddings, shard by frequency, or use **paged/managed memory**.

## Disk → CPU → GPU Transfer Speedups

> Moving weights from disk to CPU and from CPU to GPU seems suboptimal. Do you know any tricks/hack that can make this process faster?

**Practical tricks**

- **Memory‑map + prefetch**: `mmap` weights, use `madvise(MADV_SEQUENTIAL)` / `readahead` to warm pages.
- **Pinned host buffers**: stage in **page‑locked** memory to speed H2D copies.
- **Async overlap**: use CUDA streams to overlap disk I/O, H2D copy, and compute.
- **GPUDirect Storage (GDS)**: enable **disk→GPU** DMA to bypass extra CPU copies.
- **Contiguous packing**: store tensors in large, sequential blobs to reduce seeks.
- **Lower precision / compression**: fewer bytes read; decompress on GPU.
- **Parallel I/O**: shard weights and load in parallel threads/processes.
