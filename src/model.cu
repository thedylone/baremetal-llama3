#include <cstdio>
#include <memory>
#include <string>

#include "config.h"
#include "kernels.cuh"
#include "loader.h"
#include "model.h"

namespace {

// ---- CPU helpers ----

void cpu_transpose(const float* src, float* dst, int rows, int cols) {
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++) dst[c * rows + r] = src[r * cols + c];
}

// ---- Upload helpers ----

/// Load 1-D tensor from file and upload to GPU.
float* upload_1d(LlamaDumpLoader& loader, const std::string& path, size_t dim) {
  float* h = loader.load_1d(path, dim);
  float* d;
  CUDA_CHECK(cudaMalloc(&d, dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h, dim * sizeof(float), cudaMemcpyHostToDevice));
  delete[] h;
  return d;
}

/// Load 2-D tensor from file and upload to GPU (row-major, no transpose).
float* upload_2d(LlamaDumpLoader& loader, const std::string& path, size_t rows,
                 size_t cols) {
  float* h = loader.load_2d(path, rows, cols);
  float* d;
  size_t bytes = rows * cols * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d, bytes));
  CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
  delete[] h;
  return d;
}

/// Load 2-D tensor, transpose on CPU [rows,cols] -> [cols,rows], upload to GPU.
float* upload_2d_T(LlamaDumpLoader& loader, const std::string& path,
                   size_t rows, size_t cols) {
  float* h = loader.load_2d(path, rows, cols);
  auto h_T = std::make_unique<float[]>(rows * cols);
  cpu_transpose(h, h_T.get(), static_cast<int>(rows), static_cast<int>(cols));
  delete[] h;

  float* d;
  size_t bytes = rows * cols * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d, bytes));
  CUDA_CHECK(cudaMemcpy(d, h_T.get(), bytes, cudaMemcpyHostToDevice));
  return d;
}

// ---- Kernels ----

/// One block per token, threads stride over hidden_size.
__global__ void embedding_lookup_kernel(const float* __restrict__ table,
                                        const int* __restrict__ ids,
                                        float* __restrict__ out,
                                        int hidden_size) {
  const int tok = blockIdx.x;
  const float* src = table + static_cast<size_t>(ids[tok]) * hidden_size;
  float* dst = out + static_cast<size_t>(tok) * hidden_size;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) dst[i] = src[i];
}

void safe_free(float*& ptr) {
  if (ptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// LlamaModel implementation
// ---------------------------------------------------------------------------

void LlamaModel::load(const std::string& path) {
  dump_path = path;
  hidden_size = EMBEDDING_DIM;
  rms_norm_eps = RMS_NORM_EPSILON;

  LlamaDumpLoader loader(DumpFloatType::FP32);

  // --- Global weights ---
  d_embed_tokens =
      upload_2d(loader, dump_path + "/model_embed_tokens_weight.bin",
                vocab_size, hidden_size);

  d_norm_weight =
      upload_1d(loader, dump_path + "/model_norm_weight.bin", hidden_size);

  // Load lm_head weight (may differ from embed_tokens in some checkpoints)
  d_lm_head_weight = upload_2d_T(loader, dump_path + "/lm_head_weight.bin",
                                 vocab_size, hidden_size);

  // --- RoPE cos/sin tables ---
  size_t rope_bytes = max_seq_len * (head_dim / 2) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_rope_cos, rope_bytes));
  CUDA_CHECK(cudaMalloc(&d_rope_sin, rope_bytes));
  rope_precompute(d_rope_cos, d_rope_sin, max_seq_len, head_dim, rope_theta);

  // Per-layer weights are loaded lazily during forward()
  layers.resize(num_layers);
}

void LlamaModel::load_layer(int l) {
  LlamaDumpLoader loader(DumpFloatType::FP32);
  std::string p = dump_path + "/model_layers_" + std::to_string(l) + "_";
  LlamaLayer& ly = layers[l];

  ly.d_input_layernorm_weight =
      upload_1d(loader, p + "input_layernorm_weight.bin", hidden_size);
  ly.d_post_attention_layernorm_weight =
      upload_1d(loader, p + "post_attention_layernorm_weight.bin", hidden_size);

  ly.d_q_proj_weight = upload_2d_T(loader, p + "self_attn_q_proj_weight.bin",
                                   hidden_size, hidden_size);
  ly.d_k_proj_weight = upload_2d_T(loader, p + "self_attn_k_proj_weight.bin",
                                   kv_dim, hidden_size);
  ly.d_v_proj_weight = upload_2d_T(loader, p + "self_attn_v_proj_weight.bin",
                                   kv_dim, hidden_size);
  ly.d_o_proj_weight = upload_2d_T(loader, p + "self_attn_o_proj_weight.bin",
                                   hidden_size, hidden_size);

  ly.d_gate_proj_weight = upload_2d_T(loader, p + "mlp_gate_proj_weight.bin",
                                      intermediate_size, hidden_size);
  ly.d_up_proj_weight = upload_2d_T(loader, p + "mlp_up_proj_weight.bin",
                                    intermediate_size, hidden_size);
  ly.d_down_proj_weight = upload_2d_T(loader, p + "mlp_down_proj_weight.bin",
                                      hidden_size, intermediate_size);
}

void LlamaModel::free_layer(int l) {
  LlamaLayer& ly = layers[l];
  safe_free(ly.d_input_layernorm_weight);
  safe_free(ly.d_post_attention_layernorm_weight);
  safe_free(ly.d_q_proj_weight);
  safe_free(ly.d_k_proj_weight);
  safe_free(ly.d_v_proj_weight);
  safe_free(ly.d_o_proj_weight);
  safe_free(ly.d_gate_proj_weight);
  safe_free(ly.d_up_proj_weight);
  safe_free(ly.d_down_proj_weight);
}

void LlamaModel::embed(const int* d_token_ids, float* d_output, int seq_len,
                       cudaStream_t stream) {
  embedding_lookup_kernel<<<seq_len, 256, 0, stream>>>(
      d_embed_tokens, d_token_ids, d_output, hidden_size);
}

void LlamaModel::forward(const int* d_token_ids, float* d_logits, int seq_len,
                         cudaStream_t stream) {
  const int s = seq_len;
  const int d = hidden_size;

  // --- Allocate scratch buffers ---
  float *d_x, *d_normed, *d_q, *d_k, *d_v, *d_attn_out;
  float *d_gate, *d_up, *d_ffn_out;

  CUDA_CHECK(cudaMalloc(&d_x, s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normed, s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q, s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, s * kv_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, s * kv_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_out, s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gate, s * intermediate_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_up, s * intermediate_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ffn_out, s * d * sizeof(float)));

  // --- Embedding lookup ---
  embed(d_token_ids, d_x, s, stream);

  // --- 32 decoder layers (load in batches to reduce disk I/O) ---
  const int batch = num_layers / 2;  // 16 layers per batch
  for (int start = 0; start < num_layers; start += batch) {
    int end = std::min(start + batch, num_layers);

    // Load batch of layers
    for (int l = start; l < end; l++) load_layer(l);

    // Run batch
    for (int l = start; l < end; l++) {
      LlamaLayer& ly = layers[l];

      // -- Attention sub-block --
      rms_norm(d_x, ly.d_input_layernorm_weight, d_normed, s, d, stream);

      sgemm(d_normed, ly.d_q_proj_weight, d_q, s, d, d, 1.0f, 0.0f, stream);
      sgemm(d_normed, ly.d_k_proj_weight, d_k, s, kv_dim, d, 1.0f, 0.0f,
            stream);
      sgemm(d_normed, ly.d_v_proj_weight, d_v, s, kv_dim, d, 1.0f, 0.0f,
            stream);

      rope_apply(d_q, d_rope_cos, d_rope_sin, s, num_heads, head_dim, stream);
      rope_apply(d_k, d_rope_cos, d_rope_sin, s, num_kv_heads, head_dim,
                 stream);

      gqa_attention(d_q, d_k, d_v, d_attn_out, s, num_heads, num_kv_heads,
                    head_dim, stream);

      sgemm(d_attn_out, ly.d_o_proj_weight, d_normed, s, d, d, 1.0f, 0.0f,
            stream);
      residual_add(d_x, d_normed, s * d, stream);

      // -- FFN sub-block --
      rms_norm(d_x, ly.d_post_attention_layernorm_weight, d_normed, s, d,
               stream);

      sgemm(d_normed, ly.d_gate_proj_weight, d_gate, s, intermediate_size, d,
            1.0f, 0.0f, stream);
      sgemm(d_normed, ly.d_up_proj_weight, d_up, s, intermediate_size, d, 1.0f,
            0.0f, stream);

      silu(d_gate, d_up, s * intermediate_size, stream);

      sgemm(d_gate, ly.d_down_proj_weight, d_ffn_out, s, d, intermediate_size,
            1.0f, 0.0f, stream);
      residual_add(d_x, d_ffn_out, s * d, stream);
    }

    // Sync and free batch
    if (stream)
      CUDA_CHECK(cudaStreamSynchronize(stream));
    else
      CUDA_CHECK(cudaDeviceSynchronize());
    for (int l = start; l < end; l++) free_layer(l);
  }

  // --- Output layer ---

  // Final RMSNorm
  rms_norm(d_x, d_norm_weight, d_normed, s, d, stream);

  // Extract last token: d_normed + (s-1)*d points to last row [1, d]
  const float* d_last = d_normed + (s - 1) * d;

  // Logits: [1, V] = [1, d] @ [d, V]  (lm_head pre-transposed to [d, V])
  sgemm(d_last, d_lm_head_weight, d_logits, 1, vocab_size, d, 1.0f, 0.0f,
        stream);

  // --- Free scratch buffers ---
  cudaFree(d_x);
  cudaFree(d_normed);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_attn_out);
  cudaFree(d_gate);
  cudaFree(d_up);
  cudaFree(d_ffn_out);
}

void LlamaModel::free_weights() {
  safe_free(d_embed_tokens);
  safe_free(d_norm_weight);
  safe_free(d_lm_head_weight);
  safe_free(d_rope_cos);
  safe_free(d_rope_sin);
  for (auto& ly : layers) {
    safe_free(ly.d_input_layernorm_weight);
    safe_free(ly.d_post_attention_layernorm_weight);
    safe_free(ly.d_q_proj_weight);
    safe_free(ly.d_k_proj_weight);
    safe_free(ly.d_v_proj_weight);
    safe_free(ly.d_o_proj_weight);
    safe_free(ly.d_gate_proj_weight);
    safe_free(ly.d_up_proj_weight);
    safe_free(ly.d_down_proj_weight);
  }
  layers.clear();
}

LlamaModel::~LlamaModel() { free_weights(); }
