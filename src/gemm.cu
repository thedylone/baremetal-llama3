#include <cuda_runtime.h>

#include <cstdio>

#include "gemm.cuh"
#include "kernels.cuh"

__global__ void sgemm_basic_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha,
                                   float beta) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float value = 0.0f;
  for (int k = 0; k < K; ++k) {
    value += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = alpha * value + beta * C[row * N + col];
}

SgemmLaunchFunc sgemm_basic_launch =
    [](const float* A, const float* B, float* C, int M, int N, int K,
       float alpha, float beta, cudaStream_t stream) {
      const dim3 blockSize(SGEMM_TM, SGEMM_TN);
      const dim3 gridSize(cdiv(N, SGEMM_TN), cdiv(M, SGEMM_TM));
      sgemm_basic_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, M, N, K,
                                                             alpha, beta);
      CUDA_CHECK(cudaGetLastError());
    };

__global__ void sgemm_tiled_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha,
                                   float beta) {
  __shared__ float tileA[SGEMM_BM][SGEMM_BK];  // [row][K]
  __shared__ float tileB[SGEMM_BK][SGEMM_BN];  // [K][col]

  const int row = blockIdx.y * SGEMM_BM + threadIdx.y;
  const int col = blockIdx.x * SGEMM_BN + threadIdx.x;
  float value = 0.0f;
  // Loop over tiles of K dimension
  for (int t = 0; t < cdiv(K, SGEMM_BK); ++t) {
    const int aCol = t * SGEMM_BK + threadIdx.x;
    const int bRow = t * SGEMM_BK + threadIdx.y;
    // Load tiles into shared memory
    tileA[threadIdx.y][threadIdx.x] =
        (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
    tileB[threadIdx.y][threadIdx.x] =
        (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();
// Compute partial product for this tile
#pragma unroll
    for (int k = 0; k < SGEMM_BK; ++k) {
      value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    C[row * N + col] = alpha * value + beta * C[row * N + col];
  }
}

SgemmLaunchFunc sgemm_tiled_launch = [](const float* A, const float* B,
                                        float* C, int M, int N, int K,
                                        float alpha, float beta,
                                        cudaStream_t stream) {
  const dim3 block(SGEMM_BN, SGEMM_BM);
  const dim3 grid(cdiv(N, SGEMM_BN), cdiv(M, SGEMM_BM));
  sgemm_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
  CUDA_CHECK(cudaGetLastError());
};