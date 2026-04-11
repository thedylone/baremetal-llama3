#pragma once
#include <cuda_runtime.h>

/// @brief Computes Root Mean Square Normalization (RMSNorm) on the input tensor
/// row-wise.
/// @param input Input tensor of shape (s, d)
/// @param gamma Per-element scale (input_layernorm.weight)
/// @param output Output tensor of shape (s, d)
/// @param s Number of rows in the input and output tensors
/// @param d Number of columns in the input and output tensors
/// @param stream CUDA stream to run the kernel
void rms_norm(const float* __restrict__ input, const float* __restrict__ gamma, float* __restrict__ output,
              int s, int d, cudaStream_t stream = nullptr);