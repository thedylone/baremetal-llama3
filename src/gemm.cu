#include <cuda_runtime.h>

#include <cstdio>

#include "gemm.cuh"
#include "kernels.cuh"

__global__ void sgemm_basic_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C, int M, int N, int K,
                                   float alpha, float beta) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float value = 0.0f;
  for (int k = 0; k < K; ++k) {
    value += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] =
      alpha * value + (beta != 0.0f ? beta * C[row * N + col] : 0.0f);
}

SgemmLaunchFunc sgemm_basic_launch =
    [](const float* A, const float* B, float* C, int M, int N, int K,
       float alpha, float beta, cudaStream_t stream) {
      const dim3 blockSize(32, 32);
      const dim3 gridSize(cdiv(N, 32), cdiv(M, 32));
      sgemm_basic_kernel<<<gridSize, blockSize, 0, stream>>>(A, B, C, M, N, K,
                                                             alpha, beta);
      CUDA_CHECK(cudaGetLastError());
    };

__global__ void sgemm_tiled_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C, int M, int N, int K,
                                   float alpha, float beta) {
  __shared__ float tileA[SGEMM_TILED_BM][SGEMM_TILED_BK];  // [row][K]
  __shared__ float tileB[SGEMM_TILED_BK][SGEMM_TILED_BN];  // [K][col]

  const int row = blockIdx.y * SGEMM_TILED_BM + threadIdx.y;
  const int col = blockIdx.x * SGEMM_TILED_BN + threadIdx.x;
  float value = 0.0f;
  // Loop over tiles of K dimension
  for (int t = 0; t < cdiv(K, SGEMM_TILED_BK); ++t) {
    const int aCol = t * SGEMM_TILED_BK + threadIdx.x;
    const int bRow = t * SGEMM_TILED_BK + threadIdx.y;
    // Load tiles into shared memory
    tileA[threadIdx.y][threadIdx.x] =
        (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
    tileB[threadIdx.y][threadIdx.x] =
        (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();
// Compute partial product for this tile
#pragma unroll
    for (int k = 0; k < SGEMM_TILED_BK; ++k) {
      value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    C[row * N + col] =
        alpha * value + (beta != 0.0f ? beta * C[row * N + col] : 0.0f);
  }
}

SgemmLaunchFunc sgemm_tiled_launch = [](const float* A, const float* B,
                                        float* C, int M, int N, int K,
                                        float alpha, float beta,
                                        cudaStream_t stream) {
  const dim3 block(SGEMM_TILED_BN, SGEMM_TILED_BM);
  const dim3 grid(cdiv(N, SGEMM_TILED_BN), cdiv(M, SGEMM_TILED_BM));
  sgemm_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
  CUDA_CHECK(cudaGetLastError());
};

__global__ void sgemm_register_tiled_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C, int M, int N,
                                            int K, float alpha, float beta) {
  __shared__ float sA[SGEMM_REG_BM][SGEMM_REG_BK];  // [row][K]
  __shared__ float sB[SGEMM_REG_BK][SGEMM_REG_BN];  // [K][col]

  // This thread's position in the block tile
  // Threads are arranged as a (SGEMM_REG_BM/SGEMM_REG_TM) ×
  // (SGEMM_REG_BN/SGEMM_REG_TN) grid. ty is the row-group index, tx the
  // col-group index.
  const int tid = threadIdx.x;
  const int ty = tid / (SGEMM_REG_BN / SGEMM_REG_TN);
  const int tx = tid % (SGEMM_REG_BN / SGEMM_REG_TN);

  // Global top-left corner of this block's output tile
  const int block_row = blockIdx.y * SGEMM_REG_BM;
  const int block_col = blockIdx.x * SGEMM_REG_BN;

  // Per-thread accumulators (live in registers for the whole K loop)
  float acc[SGEMM_REG_TM][SGEMM_REG_TN] = {};

  // Register buffers for one k-slice of A and B
  float rA[SGEMM_REG_TM];
  float rB[SGEMM_REG_TN];

  // Precomputed smem load indices

  // A tile
  static constexpr int A_LOADS =
      (SGEMM_REG_BM * SGEMM_REG_BK) / SGEMM_REG_NT;  // 8
  const int aRow0 = tid / SGEMM_REG_BK;              // starting smem row
  const int aCol0 = tid % SGEMM_REG_BK;              // starting smem col

  // B tile
  static constexpr int B_LOADS =
      (SGEMM_REG_BK * SGEMM_REG_BN) / SGEMM_REG_NT;  // 8
  const int bRow0 = tid / SGEMM_REG_BN;
  const int bCol0 = tid % SGEMM_REG_BN;

  for (int tile = 0; tile < cdiv(K, SGEMM_REG_BK); ++tile) {
    const int tile_k = tile * SGEMM_REG_BK;

    // Load A tile
#pragma unroll
    for (int i = 0; i < A_LOADS; ++i) {
      const int row =
          aRow0 + i * (SGEMM_REG_NT / SGEMM_REG_BK);  // stride in row direction
      const int col = aCol0;
      const int gRow = block_row + row;
      const int gCol = tile_k + col;
      sA[row][col] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
    }

    // Load B tile
#pragma unroll
    for (int i = 0; i < B_LOADS; ++i) {
      const int row =
          bRow0 + i * (SGEMM_REG_NT / SGEMM_REG_BN);  // stride in row direction
      const int col = bCol0;
      const int gRow = tile_k + row;
      const int gCol = block_col + col;
      sB[row][col] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < SGEMM_REG_BK; ++k) {
      // Load this thread's slice of A and B into registers
#pragma unroll
      for (int i = 0; i < SGEMM_REG_TM; ++i)
        rA[i] = sA[ty * SGEMM_REG_TM + i][k];

#pragma unroll
      for (int j = 0; j < SGEMM_REG_TN; ++j)
        rB[j] = sB[k][tx * SGEMM_REG_TN + j];

      // Outer product accumulate
#pragma unroll
      for (int i = 0; i < SGEMM_REG_TM; ++i)
#pragma unroll
        for (int j = 0; j < SGEMM_REG_TN; ++j) acc[i][j] += rA[i] * rB[j];
    }

    __syncthreads();
  }

  // Write accumulators to C (with alpha/beta scaling)
#pragma unroll
  for (int i = 0; i < SGEMM_REG_TM; ++i) {
    const int row = block_row + ty * SGEMM_REG_TM + i;
    if (row >= M) continue;

#pragma unroll
    for (int j = 0; j < SGEMM_REG_TN; ++j) {
      const int col = block_col + tx * SGEMM_REG_TN + j;
      if (col >= N) continue;

      const int idx = row * N + col;
      C[idx] = alpha * acc[i][j] + (beta != 0.0f ? beta * C[idx] : 0.0f);
    }
  }
}

SgemmLaunchFunc sgemm_register_tiled_launch =
    [](const float* A, const float* B, float* C, int M, int N, int K,
       float alpha, float beta, cudaStream_t stream) {
      const dim3 block(SGEMM_REG_NT);
      const dim3 grid(cdiv(N, SGEMM_REG_BN), cdiv(M, SGEMM_REG_BM));
      sgemm_register_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K,
                                                              alpha, beta);
      CUDA_CHECK(cudaGetLastError());
    };
