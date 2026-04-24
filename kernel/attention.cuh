#pragma once
#include <cuda_runtime.h>

/// @brief Compute GQA attention with Causal Masking
/// @param d_q Pointer to Q of shape (seq_len, num_heads, head_dim)
/// @param d_k Pointer to K of shape (seq_len, num_kv_heads, head_dim)
/// @param d_v Pointer to V of shape (seq_len, num_kv_heads, head_dim)
/// @param d_out Pointer to the output of shape (seq_len, num_heads, head_dim)
/// @param seq_len Length of the sequence
/// @param num_heads Number of query heads h
/// @param num_kv_heads Number of key/value heads h_k
/// @param head_dim Dimension of each head h_d
/// @param stream CUDA stream to run the kernel
void gqa_attention(const float* __restrict__ d_q, const float* __restrict__ d_k,
                   const float* __restrict__ d_v, float* __restrict__ d_out,
                   int seq_len, int num_heads, int num_kv_heads, int head_dim,
                   cudaStream_t stream = nullptr);