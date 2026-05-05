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
  // --- RoPE precomputed tables on GPU ---
  float* d_rope_cos = nullptr;  // [max_seq_len, head_dim/2]
  float* d_rope_sin = nullptr;  // [max_seq_len, head_dim/2]

  // --- Global weights on GPU ---
  float* d_embed_tokens = nullptr;  // [vocab_size, hidden_size]
  float* d_norm_weight = nullptr;   // [hidden_size]  (final RMSNorm)
  float* d_lm_head_weight =
      nullptr;  // [hidden_size, vocab_size] (pre-transposed)

  // --- Per-layer weights ---
  std::vector<LlamaLayer> layers;

  // --- Weight directory (for lazy per-layer loading) ---
  std::string dump_path;

  LlamaModel() = default;
  ~LlamaModel();

  /// Load global weights (embeddings, final norm, RoPE) from dump directory.
  /// Per-layer weights are loaded lazily during forward().
  void load(const std::string& dump_path);

  /// Load a single layer's weights to GPU.
  void load_layer(int layer_idx);

  /// Free a single layer's weights from GPU.
  void free_layer(int layer_idx);

  /// Free all GPU memory.
  void free_weights();

  /// GPU embedding lookup.
  /// @param d_token_ids [seq_len] int array on device
  /// @param d_output    [seq_len, hidden_size] float array on device
  void embed(const int* d_token_ids, float* d_output, int seq_len,
             cudaStream_t stream = nullptr);

  /// Run full forward pass: embed → 32 decoder layers → final norm → logits.
  /// @param d_token_ids [seq_len] int array on device
  /// @param d_logits    [vocab_size] float array on device (output, last token)
  /// @param seq_len     number of tokens
  void forward(const int* d_token_ids, float* d_logits, int seq_len,
               cudaStream_t stream = nullptr);
};
