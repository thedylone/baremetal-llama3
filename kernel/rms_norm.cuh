#pragma once
#include <cuda_runtime.h>

/// @brief Computes Root Mean Square Normalization (RMSNorm) on the input tensor
/// row-wise.
/// @param d_input Input tensor of shape (s, d)
/// @param d_gamma Per-element scale (input_layernorm.weight)
/// @param d_output Output tensor of shape (s, d)
/// @param s Number of rows in the input and output tensors
/// @param d Number of columns in the input and output tensors
/// @param stream CUDA stream to run the kernel
void rms_norm(const float* __restrict__ d_input,
              const float* __restrict__ d_gamma, float* __restrict__ d_output,
              int s, int d, cudaStream_t stream = nullptr);