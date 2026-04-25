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

void LlamaModel::load(const std::string& dump_path) {
  hidden_size = EMBEDDING_DIM;
  rms_norm_eps = RMS_NORM_EPSILON;

  LlamaDumpLoader loader(DumpFloatType::FP32);

  // --- Global weights ---
  d_embed_tokens =
      upload_2d(loader, dump_path + "/model_embed_tokens_weight.bin",
                vocab_size, hidden_size);

  // --- Per-layer weights ---
  layers.resize(num_layers);
  for (int l = 0; l < num_layers; l++) {
    std::string p = dump_path + "/model_layers_" + std::to_string(l) + "_";
    LlamaLayer& ly = layers[l];

    // RMSNorm gamma [hidden_size]
    ly.d_input_layernorm_weight =
        upload_1d(loader, p + "input_layernorm_weight.bin", hidden_size);

    // Projection weights stored as [out_dim, in_dim] in file.
    // Pre-transpose to [in_dim, out_dim] so sgemm can compute
    //   output = normed @ weight_T  directly.
    ly.d_q_proj_weight = upload_2d_T(loader, p + "self_attn_q_proj_weight.bin",
                                     hidden_size, hidden_size);
    ly.d_k_proj_weight = upload_2d_T(loader, p + "self_attn_k_proj_weight.bin",
                                     kv_dim, hidden_size);
    ly.d_v_proj_weight = upload_2d_T(loader, p + "self_attn_v_proj_weight.bin",
                                     kv_dim, hidden_size);
    ly.d_o_proj_weight = upload_2d_T(loader, p + "self_attn_o_proj_weight.bin",
                                     hidden_size, hidden_size);

    // Post-attention RMSNorm gamma [hidden_size]
    ly.d_post_attention_layernorm_weight =
        upload_1d(loader, p + "post_attention_layernorm_weight.bin", hidden_size);

    // MLP projections (file stores [out_dim, in_dim], pre-transpose)
    ly.d_gate_proj_weight = upload_2d_T(
        loader, p + "mlp_gate_proj_weight.bin", intermediate_size, hidden_size);
    ly.d_up_proj_weight = upload_2d_T(
        loader, p + "mlp_up_proj_weight.bin", intermediate_size, hidden_size);
    ly.d_down_proj_weight = upload_2d_T(
        loader, p + "mlp_down_proj_weight.bin", hidden_size, intermediate_size);
  }

  // --- Global weights ---
  d_norm_weight =
      upload_1d(loader, dump_path + "/model_norm_weight.bin", hidden_size);

  // lm_head shares weights with embed_tokens in Llama 3
  d_lm_head_weight = d_embed_tokens;

  // --- RoPE cos/sin tables ---
  size_t rope_bytes = max_seq_len * (head_dim / 2) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_rope_cos, rope_bytes));
  CUDA_CHECK(cudaMalloc(&d_rope_sin, rope_bytes));
  rope_precompute(d_rope_cos, d_rope_sin, max_seq_len, head_dim, rope_theta);
}

void LlamaModel::embed(const int* d_token_ids, float* d_output, int seq_len,
                       cudaStream_t stream) {
  embedding_lookup_kernel<<<seq_len, 256, 0, stream>>>(
      d_embed_tokens, d_token_ids, d_output, hidden_size);
}

void LlamaModel::free_weights() {
  // lm_head shares memory with embed_tokens — avoid double free
  if (d_lm_head_weight == d_embed_tokens) d_lm_head_weight = nullptr;
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
