#include <cuda_runtime.h>

#include <cstdio>

#include "gemm.cuh"
#include "kernels.cuh"

__global__ void sgemm_basic(const float* A, const float* B, float* C, int M,
                            int N, int K, float alpha, float beta) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float value = 0.0f;
  for (int k = 0; k < K; ++k) {
    value += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = alpha * value + beta * C[row * N + col];
}

void sgemm_basic_launch(const float* A, const float* B, float* C, int M, int N,
                        int K, float alpha, float beta, cudaStream_t stream) {
  const dim3 blockSize(SGEMM_TM, SGEMM_TN);
  const dim3 gridSize(cdiv(N, SGEMM_TN), cdiv(M, SGEMM_TM));
  sgemm_basic<<<gridSize, blockSize, 0, stream>>>(A, B, C, M, N, K, alpha,
                                                  beta);
  CUDA_CHECK(cudaGetLastError());
}