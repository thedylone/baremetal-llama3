#pragma once
#include <cuda_runtime.h>

#include "config.h"

/// @brief Pre-compute cos/sin tables for RoPE on the GPU.
/// @param d_cos Output cos table [max_seq_len, half_dim]
/// @param d_sin Output sin table [max_seq_len, half_dim]
/// @param max_seq_len Maximum sequence length
/// @param head_dim Dimension per head (h_d = 128)
/// @param theta_base RoPE base frequency (500000 for Llama 3)
void rope_precompute(float* d_cos, float* d_sin, int max_seq_len, int head_dim,
                     float theta_base = ROPE_THETA_BASE,
                     cudaStream_t stream = nullptr);

/// @brief Apply RoPE in-place to Q or K.
/// The tensor is in flat layout [seq_len, num_heads * head_dim].
/// For each head, pairs dim i with dim i + head_dim/2.
/// @param d_qk     Tensor to rotate in-place
/// @param d_cos    Pre-computed cos table [max_seq_len, half_dim]
/// @param d_sin    Pre-computed sin table [max_seq_len, half_dim]
/// @param seq_len  Number of token positions
/// @param num_heads Number of heads in this tensor (h=32 for Q, h_k=8 for K)
/// @param head_dim Dimension per head (128)
void rope_apply(float* d_qk, const float* d_cos, const float* d_sin,
                int seq_len, int num_heads, int head_dim,
                cudaStream_t stream = nullptr);
