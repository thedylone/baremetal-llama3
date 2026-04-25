#pragma once
#include <cuda_runtime.h>

/// @brief Elementwise SiLU activation: SiLU(z) = z * sigmoid(z)
/// @param d_gate Gate tensor of shape (n,). Modified in-place
/// @param d_up Up projection output of shape (n,). Read-only.
/// @param n Number of elements
/// @param stream CUDA stream to run the kernel
void silu(float* d_gate, const float* d_up, int n, cudaStream_t stream = nullptr);