# CS265 LLM Inference Project Docs

# Milestone 1

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


# Milestone 2

## Roofline analysis

> Before you run Nsight Compute (ncu), predict: is your RMSNorm kernel memory-bandwidth-bound or compute-bound? Estimate its arithmetic intensity in FLOPs per byte, assuming all data comes from HBM. Then profile it and compare your prediction to the measured roofline position. Do the same for your matmul kernel. Are the results what you expected?

### RMSNorm
The RMSNorm kernel computes
$$
    \mathrm{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}, \quad \mathrm{RMSNorm}(x) = \frac{x}{\mathrm{RMS}(x)} \odot \gamma
$$

The FLOP count for this kernel is:
Step | FLOPs
--- | ---
Squaring each element | $d$
Accumulating the sum  | $d$
Divide by N, add $\epsilon$, sqrt, reciprocal | ~4
Normalisation $x_i \times$ inv_rms | $d$
Scaling by $\gamma$ | $d$
**Total** | **~$\mathbf{4d}$**

Byte count (assuming float32 from HBM)
Transfer | Bytes
--- | ---
Read input | $4d$
Read $\gamma$ | $4d$
Write output | $4d$
**Total** | **$\mathbf{12d}$**

So the arithmetic intensity is approximately $\frac{4d}{12d} = \frac{1}{3}$ FLOPs/Byte in the ideal HBM scenario, which is very low (a few orders of magnitude below the ridge point a.k.a. peak FLOPS / peak HBM bandwidth), indicating that the kernel is likely memory-bound on modern GPUs.

With the current implementation, which reads the input row twice, the byte count would be $16d$, further reducing the arithmetic intensity to $\frac{4d}{16d} = \frac{1}{4}$ FLOPs/Byte, making it even more memory-bound.

Using NCU metrics for `s=4096, d=4096`:

- `dram__bytes_read.sum = 130.72 MB`
- `dram__bytes_write.sum = 66.96 MB`
- `gpu__time_duration.sum = 629.66 µs`
- `FADD = 1,048,576`, `FMUL = 33,554,432`, `FFMA = 16,797,696`

Compute:
- **Bytes** = 130.72 + 66.96 = **197.68 MB**
- **FLOPs** = FADD + FMUL + 2·FFMA  
  = 1,048,576 + 33,554,432 + 2·16,797,696  
  = **68,198,400 FLOPs**

Roofline point:
- **Arithmetic intensity** = 68,198,400 / 197.68e6 = **0.345 FLOPs/byte**
- **Performance** = 68,198,400 / 629.66e-6 = **~108 GFLOP/s**
- **Bandwidth** = 197.68e6 / 629.66e-6 = **~314 GB/s**

This uses **DRAM bytes**, so cache hits reduce the observed traffic vs the
theoretical 16d/row model.

### matmul
For the `sgemm_register_tiled_kernel`, with `BM = BN = 64` and `BK = 32`,
- FLOPs per block:
    - There are $BM \times BN \times BK$ multiplications and the same number of additions, so total FLOPs is:
    - $ 2 \times BM \times BN \times BK = 2 \times 64 \times 64 \times 32 = 262144 $ FLOPs per block
- HBM bytes per block, assuming ideal reuse inside shared/registers:
    - Read A: $4 \times BM \times BK = 8192$ bytes
    - Read B: $4 \times BK \times BN = 8192$ bytes
    - Write C: $4 \times BM \times BN = 16384$ bytes
    - Total HBM bytes: $8192 + 8192 + 16384 = 32768$ bytes per block

So the arithmetic intensity is approximately $\frac{262144}{32768} = 8$ FLOPs/Byte, which is much higher than the RMSNorm kernel, indicating that the matmul kernel is more compute-bound.

NCU metrics for 4096×4096×4096:

- `dram__bytes_read.sum = 4.50 GB`
- `dram__bytes_write.sum = 76.61 MB`
- `gpu__time_duration.sum = 27.73 ms`
- `FFMA = 68,736,253,952`

Compute:
- **Bytes** = 4.50 + 0.07661 = **4.57661 GB**
- **FLOPs** = 2·FFMA = **137,472,507,904**
- **AI** = 137,472,507,904 / 4.57661e9 = **30.05 FLOPs/byte**
- **Perf** = 137,472,507,904 / 0.02773 ≈ **4.96 TFLOP/s**
- **BW** = 4.57661e9 / 0.02773 ≈ **165 GB/s**

This exceeds the expected performance based on the theoretical arithmetic intensity and the GPU's peak capabilities, suggesting that the kernel is efficiently utilizing the GPU's compute resources and that there may be some caching effects reducing the observed HBM traffic.

## RMSNorm HBM traffic

> Your RMSNorm kernel reads each input row twice from HBM: once to accumulate the sum of squares and once to normalize and scale. What is the total HBM traffic per row in bytes?

The total HBM traffic for the RMSNorm kernel is $2 \times 4d + 4d + 4d = 16d = 65536$ bytes (reads each input twice, reads $\gamma$ once, and writes output once) per row.

> If you could restructure the kernel to keep the row in shared memory between both passes, how much HBM traffic would you save? 

This would save the second read of the input row, which is $4d$ bytes. So the HBM traffic would be reduced to $12d$ bytes, saving 25% of the original traffic.

> What prevents a naive implementation from doing this for very long rows?

Shared memory on GPUs is limited (~48 KB), so a full row may not fit.

## QKV HBM traffic

> The three Q, K, V projections read the same input matrix $X_{norm} \in \mathbb{R}^{s \times d}$ from HBM in three separate kernel launches. What is the total HBM read traffic for $X_{norm}$ across all three launches?

The total HBM read traffic for $X_{norm}$ across all three launches is $3 \times 4 \times s \times d = 12sd = 49152s$ bytes.

> If $X_{norm}$ fit entirely in the L2 cache after the first read, how would the effective memory traffic change?

If $X_{norm}$ fit entirely in the L2 cache after the first read, the effective memory traffic for $X_{norm}$ would be reduced to $4sd = 16384s$ bytes, since the subsequent reads for K and V would hit the L2 cache instead of going to HBM. This would save 66.7% of the original traffic for $X_{norm}$.

> Given the L2 size on your GPU, does $X_{norm}$ actually fit?

The A100 GPU on the cluster has an L2 cache size of 40 MB. For $X_{norm}$ to fit in the L2 cache, we need:
$$4 \times s \times d \leq 40 \text{ MB} \implies 4096 s \leq 10^7 $$
$$ s \leq \frac{10^7}{4096} \approx 2441 $$
So if the sequence length $s$ is less than or equal to 2441, $X_{norm}$ would fit in the L2 cache. For typical sequence lengths used in LLM inference (e.g., 512, 1024, 2048), $X_{norm}$ would indeed fit in the L2 cache on an A100 GPU.

On my laptop with an NVIDIA GeForce RTX 3060 Laptop GPU, which has an L2 cache size of 3 MB, the maximum sequence length for $X_{norm}$ to fit in the L2 cache would be approximately 183 tokens, which is quite small. Therefore, for longer sequences, $X_{norm}$ would not fit in the L2 cache on the laptop, and the subsequent reads for K and V would go to HBM, resulting in higher effective memory traffic.