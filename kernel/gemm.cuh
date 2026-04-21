#pragma once
#include <cuda_runtime.h>

using SgemmLaunchFunc = void (*)(const float* d_A, const float* d_B, float* d_C,
                                 int M, int N, int K, float alpha, float beta,
                                 cudaStream_t stream);

extern SgemmLaunchFunc sgemm_naive_launch;
extern SgemmLaunchFunc sgemm_tiled_launch;
extern SgemmLaunchFunc sgemm_register_tiled_launch;

/// @brief Alias for the latest implementation of sgemm
/// Computes C := alpha * A * B + beta * C
/// @param d_A M x K matrix (device)
/// @param d_B K x N matrix (device)
/// @param d_C M x N matrix (device)
/// @param M number of rows in A and C
/// @param N number of columns in B and C
/// @param K number of columns in A and rows in B
/// @param alpha scalar multiplier for A * B
/// @param beta scalar multiplier for C
/// @param stream CUDA stream to run the kernel
inline void sgemm(const float* d_A, const float* d_B, float* d_C, int M, int N,
                  int K, float alpha = 1.0f, float beta = 0.0f,
                  cudaStream_t stream = nullptr) {
  sgemm_register_tiled_launch(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
}