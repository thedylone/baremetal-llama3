#pragma once
#include <cuda_runtime.h>

#include <string>
#include <vector>

/// Per-layer weights on GPU.
/// Projection weights are stored **pre-transposed** [in_dim, out_dim] so that
/// the forward pass can use sgemm directly:
///   Q = sgemm(normed, d_q_proj_weight, seq, hidden_size, hidden_size)
struct LlamaLayer {
  // Layernorm weights [hidden_size]
  float* d_input_layernorm_weight = nullptr;
  float* d_post_attention_layernorm_weight = nullptr;

  // Attention projections (transposed: [in_dim, out_dim])
  float* d_q_proj_weight = nullptr;  // [hidden_size, hidden_size]
  float* d_k_proj_weight = nullptr;  // [hidden_size, kv_dim]
  float* d_v_proj_weight = nullptr;  // [hidden_size, kv_dim]
  float* d_o_proj_weight = nullptr;  // [hidden_size, hidden_size]

  // MLP projections (transposed: [in_dim, out_dim])
  float* d_gate_proj_weight = nullptr;  // [hidden_size, intermediate_size]
  float* d_up_proj_weight = nullptr;    // [hidden_size, intermediate_size]
  float* d_down_proj_weight = nullptr;  // [intermediate_size, hidden_size]
};

/// Llama 3 8B model with weights resident on GPU.
class LlamaModel {
 public:
  // --- Configuration (Llama 3 8B) ---
  int hidden_size = 4096;
  int num_layers = 32;
  int num_heads = 32;
  int num_kv_heads = 8;
  int head_dim = 128;  // hidden_size / num_heads
  int kv_dim = 1024;   // num_kv_heads * head_dim
  int intermediate_size = 14336;
  int vocab_size = 128256;
  float rms_norm_eps = RMS_NORM_EPSILON;
  float rope_theta = ROPE_THETA_BASE;
  int max_seq_len = 1000;

  // --- RoPE precomputed tables on GPU ---
  float* d_rope_cos = nullptr;  // [max_seq_len, head_dim/2]
  float* d_rope_sin = nullptr;  // [max_seq_len, head_dim/2]

  // --- Global weights on GPU ---
  float* d_embed_tokens = nullptr;    // [vocab_size, hidden_size]
  float* d_norm_weight = nullptr;     // [hidden_size]  (final RMSNorm)
  float* d_lm_head_weight = nullptr;  // [hidden_size, vocab_size] (transposed)

  // --- Per-layer weights ---
  std::vector<LlamaLayer> layers;

  LlamaModel() = default;
  ~LlamaModel();

  /// Load all model weights from dump directory to GPU.
  void load(const std::string& dump_path);

  /// Free all GPU memory.
  void free_weights();

  /// GPU embedding lookup.
  /// @param d_token_ids [seq_len] int array on device
  /// @param d_output    [seq_len, hidden_size] float array on device
  void embed(const int* d_token_ids, float* d_output, int seq_len,
             cudaStream_t stream = nullptr);
};
