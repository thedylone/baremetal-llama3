#pragma once
#include <cuda_runtime.h>

/// @brief Elementwise residual addition: x[i] += residual[i]
/// @param d_x      input/output array (modified in-place) [n]
/// @param d_residual array to add [n]
/// @param n         total number of elements
/// @param stream    CUDA stream
void residual_add(float* d_x, const float* d_residual, int n,
                  cudaStream_t stream = nullptr);
