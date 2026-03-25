#pragma once
#include <cuda_runtime.h>

using SgemmLaunchFunc = void (*)(const float* A, const float* B, float* C,
                                 int M, int N, int K, float alpha, float beta,
                                 cudaStream_t stream);

extern SgemmLaunchFunc sgemm_naive_launch;
extern SgemmLaunchFunc sgemm_tiled_launch;
extern SgemmLaunchFunc sgemm_register_tiled_launch;

/// @brief Alias for the latest implementation of sgemm
/// Computes C := alpha * A * B + beta * C
/// @param A M x K matrix
/// @param B K x N matrix
/// @param C M x N matrix
/// @param M number of rows in A and C
/// @param N number of columns in B and C
/// @param K number of columns in A and rows in B
/// @param alpha scalar multiplier for A * B
/// @param beta scalar multiplier for C
/// @param stream CUDA stream to run the kernel
inline void sgemm(const float* A, const float* B, float* C, int M, int N, int K,
                  float alpha = 1.0f, float beta = 0.0f,
                  cudaStream_t stream = nullptr) {
  sgemm_register_tiled_launch(A, B, C, M, N, K, alpha, beta, stream);
}