#include <cstdio>
#include <memory>
#include <string>
#include <vector>

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

/// One block per token, threads stride over EMBEDDING_DIM.
__global__ void embedding_lookup_kernel(const float* __restrict__ table,
                                        const int* __restrict__ ids,
                                        float* __restrict__ out,
                                        int EMBEDDING_DIM) {
  const int tok = blockIdx.x;
  const float* src = table + static_cast<size_t>(ids[tok]) * EMBEDDING_DIM;
  float* dst = out + static_cast<size_t>(tok) * EMBEDDING_DIM;
  for (int i = threadIdx.x; i < EMBEDDING_DIM; i += blockDim.x) dst[i] = src[i];
}

void safe_free(float*& ptr) {
  if (ptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

int argmax_index(const std::vector<float>& values) {
  if (values.empty()) return 0;
  int best_idx = 0;
  float best_val = values[0];
  for (int i = 1; i < static_cast<int>(values.size()); ++i) {
    if (values[i] > best_val) {
      best_val = values[i];
      best_idx = i;
    }
  }
  return best_idx;
}

}  // namespace

// ---------------------------------------------------------------------------
// LlamaModel implementation
// ---------------------------------------------------------------------------

void LlamaModel::load(const std::string& path) {
  dump_path = path;

  // Weights are dumped in BF16, so load and convert to FP32 on CPU
  LlamaDumpLoader loader(DumpFloatType::BF16);

  // --- Global weights ---
  d_embed_tokens =
      upload_2d(loader, dump_path + "/model_embed_tokens_weight.bin",
                VOCAB_SIZE, EMBEDDING_DIM);

  d_norm_weight =
      upload_1d(loader, dump_path + "/model_norm_weight.bin", EMBEDDING_DIM);

  // Load lm_head weight (may differ from embed_tokens in some checkpoints)
  d_lm_head_weight = upload_2d_T(loader, dump_path + "/lm_head_weight.bin",
                                 VOCAB_SIZE, EMBEDDING_DIM);

  // --- RoPE cos/sin tables ---
  size_t rope_bytes = MAX_SEQ_LEN * (HEAD_DIM / 2) * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_rope_cos, rope_bytes));
  CUDA_CHECK(cudaMalloc(&d_rope_sin, rope_bytes));
  rope_precompute(d_rope_cos, d_rope_sin, MAX_SEQ_LEN, HEAD_DIM,
                  ROPE_THETA_BASE);

  // Per-layer weights are loaded lazily during forward()
  layers.resize(NUM_LAYERS);
}

void LlamaModel::load_layer(int l) {
  LlamaDumpLoader loader(DumpFloatType::BF16);
  std::string p = dump_path + "/model_layers_" + std::to_string(l) + "_";
  LlamaLayer& ly = layers[l];

  ly.d_input_layernorm_weight =
      upload_1d(loader, p + "input_layernorm_weight.bin", EMBEDDING_DIM);
  ly.d_post_attention_layernorm_weight = upload_1d(
      loader, p + "post_attention_layernorm_weight.bin", EMBEDDING_DIM);

  ly.d_q_proj_weight = upload_2d_T(loader, p + "self_attn_q_proj_weight.bin",
                                   EMBEDDING_DIM, EMBEDDING_DIM);
  ly.d_k_proj_weight = upload_2d_T(loader, p + "self_attn_k_proj_weight.bin",
                                   KV_DIM, EMBEDDING_DIM);
  ly.d_v_proj_weight = upload_2d_T(loader, p + "self_attn_v_proj_weight.bin",
                                   KV_DIM, EMBEDDING_DIM);
  ly.d_o_proj_weight = upload_2d_T(loader, p + "self_attn_o_proj_weight.bin",
                                   EMBEDDING_DIM, EMBEDDING_DIM);

  ly.d_gate_proj_weight = upload_2d_T(loader, p + "mlp_gate_proj_weight.bin",
                                      INTERMEDIATE_SIZE, EMBEDDING_DIM);
  ly.d_up_proj_weight = upload_2d_T(loader, p + "mlp_up_proj_weight.bin",
                                    INTERMEDIATE_SIZE, EMBEDDING_DIM);
  ly.d_down_proj_weight = upload_2d_T(loader, p + "mlp_down_proj_weight.bin",
                                      EMBEDDING_DIM, INTERMEDIATE_SIZE);
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
      d_embed_tokens, d_token_ids, d_output, EMBEDDING_DIM);
}

void LlamaModel::swiglu_ffn_internal(int layer_idx, int seq_len,
                                     const float* d_x_norm, float* d_gate,
                                     float* d_up, float* d_out,
                                     cudaStream_t stream) {
  const int d = EMBEDDING_DIM;
  LlamaLayer& ly = layers[layer_idx];

  // Gate projection: x_norm @ W_gate -> d_gate [seq_len, intermediate_size]
  sgemm(d_x_norm, ly.d_gate_proj_weight, d_gate, seq_len, INTERMEDIATE_SIZE, d,
        1.0f, 0.0f, stream);

  // Up projection: x_norm @ W_up -> d_up [seq_len, intermediate_size]
  sgemm(d_x_norm, ly.d_up_proj_weight, d_up, seq_len, INTERMEDIATE_SIZE, d,
        1.0f, 0.0f, stream);

  // SiLU gating: d_gate = silu(d_gate) * d_up [seq_len, intermediate_size]
  silu(d_gate, d_up, seq_len * INTERMEDIATE_SIZE, stream);

  // Down projection: d_gate @ W_down -> d_out [seq_len, hidden_size]
  sgemm(d_gate, ly.d_down_proj_weight, d_out, seq_len, d, INTERMEDIATE_SIZE,
        1.0f, 0.0f, stream);
}

void LlamaModel::decoder_layer_internal(int layer_idx, int seq_len, float* d_x,
                                        float* d_normed, float* d_q, float* d_k,
                                        float* d_v, float* d_attn_out,
                                        float* d_gate, float* d_up,
                                        float* d_ffn_out, cudaStream_t stream) {
  const int d = EMBEDDING_DIM;
  LlamaLayer& ly = layers[layer_idx];

  // --- Attention sub-block ---
  // RMSNorm: x -> d_normed
  rms_norm(d_x, ly.d_input_layernorm_weight, d_normed, seq_len, d, stream);

  // QKV projections: d_normed -> d_q, d_k, d_v
  sgemm(d_normed, ly.d_q_proj_weight, d_q, seq_len, d, d, 1.0f, 0.0f, stream);
  sgemm(d_normed, ly.d_k_proj_weight, d_k, seq_len, KV_DIM, d, 1.0f, 0.0f,
        stream);
  sgemm(d_normed, ly.d_v_proj_weight, d_v, seq_len, KV_DIM, d, 1.0f, 0.0f,
        stream);

  // RoPE: apply rotary embeddings to Q and K
  rope_apply(d_q, d_rope_cos, d_rope_sin, seq_len, NUM_HEADS, HEAD_DIM, stream);
  rope_apply(d_k, d_rope_cos, d_rope_sin, seq_len, NUM_KV_HEADS, HEAD_DIM,
             stream);

  // GQA: attention -> d_attn_out
  gqa_attention(d_q, d_k, d_v, d_attn_out, seq_len, NUM_HEADS, NUM_KV_HEADS,
                HEAD_DIM, stream);

  // Output projection: d_attn_out @ W_o -> d_normed
  sgemm(d_attn_out, ly.d_o_proj_weight, d_normed, seq_len, d, d, 1.0f, 0.0f,
        stream);

  // Residual: d_x += d_normed
  residual_add(d_x, d_normed, seq_len * d, stream);

  // --- FFN sub-block ---
  // RMSNorm: x -> d_normed
  rms_norm(d_x, ly.d_post_attention_layernorm_weight, d_normed, seq_len, d,
           stream);

  // SwiGLU FFN: d_normed -> d_ffn_out
  swiglu_ffn_internal(layer_idx, seq_len, d_normed, d_gate, d_up, d_ffn_out,
                      stream);

  // Residual: d_x += d_ffn_out
  residual_add(d_x, d_ffn_out, seq_len * d, stream);
}

void LlamaModel::forward_internal(const int* d_token_ids, float* d_logits,
                                  int seq_len, float* d_x, float* d_normed,
                                  float* d_q, float* d_k, float* d_v,
                                  float* d_attn_out, float* d_gate, float* d_up,
                                  float* d_ffn_out, cudaStream_t stream) {
  const int d = EMBEDDING_DIM;

  // --- Embedding lookup ---
  embed(d_token_ids, d_x, seq_len, stream);

  // --- 32 decoder layers (load in batches to reduce disk I/O) ---
  for (int start = 0; start < NUM_LAYERS; start += LAZY_LOAD_BATCH_SIZE) {
    int end = std::min(start + LAZY_LOAD_BATCH_SIZE, NUM_LAYERS);

    // Load batch of layers
    for (int l = start; l < end; l++) load_layer(l);

    // Run batch
    for (int l = start; l < end; l++) {
      decoder_layer_internal(l, seq_len, d_x, d_normed, d_q, d_k, d_v,
                             d_attn_out, d_gate, d_up, d_ffn_out, stream);
    }

    // Sync and free batch
    if (stream)
      CUDA_CHECK(cudaStreamSynchronize(stream));
    else
      CUDA_CHECK(cudaDeviceSynchronize());
    if (!LAZY_LOAD_KEEP_LAYERS)
      for (int l = start; l < end; l++) free_layer(l);
  }

  // --- Output layer ---
  // Final RMSNorm
  rms_norm(d_x, d_norm_weight, d_normed, seq_len, d, stream);

  // Extract last token: d_normed + (seq_len-1)*d points to last row [1, d]
  const float* d_last = d_normed + (seq_len - 1) * d;

  // Logits: [1, V] = [1, d] @ [d, V]  (lm_head pre-transposed to [d, V])
  sgemm(d_last, d_lm_head_weight, d_logits, 1, VOCAB_SIZE, d, 1.0f, 0.0f,
        stream);
}

std::vector<float> LlamaModel::swiglu_ffn(const std::vector<float>& x_norm,
                                          int layer_idx, int s,
                                          cudaStream_t stream) {
  if (layer_idx < 0 || layer_idx >= static_cast<int>(layers.size())) return {};
  if (layers[layer_idx].d_gate_proj_weight == nullptr) load_layer(layer_idx);

  std::vector<float> output(static_cast<size_t>(s) * EMBEDDING_DIM, 0.0f);

  float *d_x_norm, *d_gate, *d_up, *d_out;
  CUDA_CHECK(cudaMalloc(&d_x_norm, x_norm.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &d_gate, static_cast<size_t>(s) * INTERMEDIATE_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &d_up, static_cast<size_t>(s) * INTERMEDIATE_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &d_out, static_cast<size_t>(s) * EMBEDDING_DIM * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x_norm, x_norm.data(), x_norm.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Call the internal implementation
  swiglu_ffn_internal(layer_idx, s, d_x_norm, d_gate, d_up, d_out, stream);

  if (stream)
    CUDA_CHECK(cudaStreamSynchronize(stream));
  else
    CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_out, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_x_norm);
  cudaFree(d_gate);
  cudaFree(d_up);
  cudaFree(d_out);
  return output;
}

std::vector<float> LlamaModel::decoder_block(const std::vector<float>& x,
                                             int layer_idx, int s,
                                             cudaStream_t stream) {
  if (layer_idx < 0 || layer_idx >= static_cast<int>(layers.size())) return {};
  if (layers[layer_idx].d_q_proj_weight == nullptr) load_layer(layer_idx);

  const int d = EMBEDDING_DIM;
  std::vector<float> output(x.size(), 0.0f);

  float *d_x, *d_normed, *d_q, *d_k, *d_v, *d_attn_out;
  float *d_gate, *d_up, *d_ffn_out;
  CUDA_CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normed, x.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q, static_cast<size_t>(s) * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, static_cast<size_t>(s) * KV_DIM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, static_cast<size_t>(s) * KV_DIM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_out, x.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &d_gate, static_cast<size_t>(s) * INTERMEDIATE_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &d_up, static_cast<size_t>(s) * INTERMEDIATE_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ffn_out, x.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Call the internal implementation
  decoder_layer_internal(layer_idx, s, d_x, d_normed, d_q, d_k, d_v, d_attn_out,
                         d_gate, d_up, d_ffn_out, stream);

  if (stream)
    CUDA_CHECK(cudaStreamSynchronize(stream));
  else
    CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_x, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_x);
  cudaFree(d_normed);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_attn_out);
  cudaFree(d_gate);
  cudaFree(d_up);
  cudaFree(d_ffn_out);
  return output;
}

int LlamaModel::forward_one_step(const std::vector<int>& token_ids,
                                 cudaStream_t stream) {
  if (token_ids.empty()) return 0;

  const int s = static_cast<int>(token_ids.size());
  const int d = EMBEDDING_DIM;

  // Allocate scratch buffers
  float *d_x, *d_normed, *d_q, *d_k, *d_v, *d_attn_out;
  float *d_gate, *d_up, *d_ffn_out;
  int* d_ids = nullptr;
  float* d_logits = nullptr;

  CUDA_CHECK(cudaMalloc(&d_x, s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normed, s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q, s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, s * KV_DIM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, s * KV_DIM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_out, s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gate, s * INTERMEDIATE_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_up, s * INTERMEDIATE_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ffn_out, s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ids, token_ids.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_logits, VOCAB_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_ids, token_ids.data(), token_ids.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  forward_internal(d_ids, d_logits, s, d_x, d_normed, d_q, d_k, d_v, d_attn_out,
                   d_gate, d_up, d_ffn_out, stream);

  std::vector<float> logits(VOCAB_SIZE, 0.0f);
  if (stream)
    CUDA_CHECK(cudaStreamSynchronize(stream));
  else
    CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(logits.data(), d_logits, logits.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Free scratch buffers
  cudaFree(d_x);
  cudaFree(d_normed);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_attn_out);
  cudaFree(d_gate);
  cudaFree(d_up);
  cudaFree(d_ffn_out);
  cudaFree(d_ids);
  cudaFree(d_logits);

  return argmax_index(logits);
}

std::vector<int> LlamaModel::generate(const std::vector<int>& token_ids,
                                      int n_new, cudaStream_t stream) {
  if (token_ids.empty() || n_new <= 0) return {};

  std::vector<int> prompt = token_ids;
  std::vector<int> generated;
  generated.reserve(n_new);

  // Preallocate scratch buffers for maximum sequence length
  int max_s = static_cast<int>(token_ids.size()) + n_new;
  const int d = EMBEDDING_DIM;

  float *d_x, *d_normed, *d_q, *d_k, *d_v, *d_attn_out;
  float *d_gate, *d_up, *d_ffn_out;
  int* d_ids = nullptr;
  float* d_logits = nullptr;

  CUDA_CHECK(cudaMalloc(&d_x, max_s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normed, max_s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q, max_s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, max_s * KV_DIM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, max_s * KV_DIM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_out, max_s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gate, max_s * INTERMEDIATE_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_up, max_s * INTERMEDIATE_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ffn_out, max_s * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ids, max_s * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_logits, VOCAB_SIZE * sizeof(float)));

  // Generate tokens
  for (int i = 0; i < n_new; ++i) {
    int cur_s = static_cast<int>(prompt.size());

    CUDA_CHECK(cudaMemcpy(d_ids, prompt.data(), cur_s * sizeof(int),
                          cudaMemcpyHostToDevice));

    forward_internal(d_ids, d_logits, cur_s, d_x, d_normed, d_q, d_k, d_v,
                     d_attn_out, d_gate, d_up, d_ffn_out, stream);

    std::vector<float> logits(VOCAB_SIZE, 0.0f);
    if (stream)
      CUDA_CHECK(cudaStreamSynchronize(stream));
    else
      CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(logits.data(), d_logits,
                          logits.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int next_id = argmax_index(logits);
    generated.push_back(next_id);
    prompt.push_back(next_id);
  }

  // Free scratch buffers
  cudaFree(d_x);
  cudaFree(d_normed);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_attn_out);
  cudaFree(d_gate);
  cudaFree(d_up);
  cudaFree(d_ffn_out);
  cudaFree(d_ids);
  cudaFree(d_logits);

  return generated;
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
