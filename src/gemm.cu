#include <cuda_runtime.h>

#include <cstdio>

#include "gemm.cuh"
#include "kernels.cuh"

// Naive SGEMM kernel: one thread per output element, no tiling or shared memory

__global__ void sgemm_naive_kernel(const float* __restrict__ d_A,
                                   const float* __restrict__ d_B,
                                   float* __restrict__ d_C, int M, int N, int K,
                                   float alpha, float beta) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float value = 0.0f;
  for (int k = 0; k < K; ++k) {
    value += d_A[row * K + k] * d_B[k * N + col];
  }
  d_C[row * N + col] =
      alpha * value + (beta != 0.0f ? beta * d_C[row * N + col] : 0.0f);
}

SgemmLaunchFunc sgemm_naive_launch =
    [](const float* d_A, const float* d_B, float* d_C, int M, int N, int K,
       float alpha, float beta, cudaStream_t stream) {
      const dim3 blockSize(32, 32);
      const dim3 gridSize(cdiv(N, 32), cdiv(M, 32));
      sgemm_naive_kernel<<<gridSize, blockSize, 0, stream>>>(d_A, d_B, d_C, M, N, K,
                                                             alpha, beta);
      CUDA_CHECK(cudaGetLastError());
    };

// Shared memory tiling (coalesced): each block computes a BM×BN tile of d_C,
// loading BM×BK and BK×BN tiles of d_A and d_B into shared memory.

template <const int BM, const int BN, const int BK>
__global__ void sgemm_tiled_kernel(const float* __restrict__ d_A,
                                   const float* __restrict__ d_B,
                                   float* __restrict__ d_C, int M, int N, int K,
                                   float alpha, float beta) {
  __shared__ float sA[BM][BK];  // [row][K]
  __shared__ float sB[BK][BN];  // [K][col]

  const int row = blockIdx.y * BM + threadIdx.y;
  const int col = blockIdx.x * BN + threadIdx.x;
  float value = 0.0f;
  // Loop over tiles of K dimension
  for (int t = 0; t < cdiv(K, BK); ++t) {
    const int aCol = t * BK + threadIdx.x;
    const int bRow = t * BK + threadIdx.y;
    // Load tiles into shared memory
    sA[threadIdx.y][threadIdx.x] =
        (row < M && aCol < K) ? d_A[row * K + aCol] : 0.0f;
    sB[threadIdx.y][threadIdx.x] =
        (bRow < K && col < N) ? d_B[bRow * N + col] : 0.0f;
    __syncthreads();
// Compute partial product for this tile
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      value += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    d_C[row * N + col] =
        alpha * value + (beta != 0.0f ? beta * d_C[row * N + col] : 0.0f);
  }
}

SgemmLaunchFunc sgemm_tiled_launch =
    [](const float* d_A, const float* d_B, float* d_C, int M, int N, int K,
       float alpha, float beta, cudaStream_t stream) {
      const int BM = 32;  // Block rows
      const int BN = 32;  // Block columns
      const int BK = 32;  // Block depth
      const dim3 block(BN, BM);
      const dim3 grid(cdiv(N, BN), cdiv(M, BM));
      sgemm_tiled_kernel<BM, BN, BK>
          <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
      CUDA_CHECK(cudaGetLastError());
    };

// Register-tiled SGEMM (coalesced): each thread computes a TM×TN tile of d_C,
// accumulating into registers across multiple K tiles loaded into shared
// memory.

template <const int BM, const int BN, const int BK, const int TM, const int TN,
          const int NT>
__global__ void sgemm_register_tiled_kernel(const float* __restrict__ d_A,
                                            const float* __restrict__ d_B,
                                            float* __restrict__ d_C, int M, int N,
                                            int K, float alpha, float beta) {
  __shared__ float sA[BM][BK];  // [row][K]
  __shared__ float sB[BK][BN];  // [K][col]

  // This thread's position in the block tile
  // Threads are arranged as a (BM/TM) ×
  // (BN/TN) grid. ty is the row-group index, tx the
  // col-group index.
  const int tid = threadIdx.x;
  const int ty = tid / (BN / TN);
  const int tx = tid % (BN / TN);

  // Global top-left corner of this block's output tile
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  // Per-thread accumulators (live in registers for the whole K loop)
  float acc[TM][TN] = {};

  // Register buffers for one k-slice of d_A and d_B
  float rA[TM];
  float rB[TN];

  // Precomputed smem load indices

  // d_A tile
  static constexpr int A_LOADS = (BM * BK) / NT;
  const int aRow0 = tid / BK;  // starting smem row
  const int aCol0 = tid % BK;  // starting smem col

  // d_B tile
  static constexpr int B_LOADS = (BK * BN) / NT;
  const int bRow0 = tid / BN;
  const int bCol0 = tid % BN;

  for (int tile = 0; tile < cdiv(K, BK); ++tile) {
    const int tile_k = tile * BK;

    // Load d_A tile
#pragma unroll
    for (int i = 0; i < A_LOADS; ++i) {
      const int row = aRow0 + i * (NT / BK);  // stride in row direction
      const int col = aCol0;
      const int gRow = block_row + row;
      const int gCol = tile_k + col;
      sA[row][col] = (gRow < M && gCol < K) ? d_A[gRow * K + gCol] : 0.0f;
    }

    // Load d_B tile
#pragma unroll
    for (int i = 0; i < B_LOADS; ++i) {
      const int row = bRow0 + i * (NT / BN);  // stride in row direction
      const int col = bCol0;
      const int gRow = tile_k + row;
      const int gCol = block_col + col;
      sB[row][col] = (gRow < K && gCol < N) ? d_B[gRow * N + gCol] : 0.0f;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // Load this thread's slice of d_A and d_B into registers
#pragma unroll
      for (int i = 0; i < TM; ++i) rA[i] = sA[ty * TM + i][k];

#pragma unroll
      for (int j = 0; j < TN; ++j) rB[j] = sB[k][tx * TN + j];

      // Outer product accumulate
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += rA[i] * rB[j];
    }

    __syncthreads();
  }

  // Write accumulators to d_C (with alpha/beta scaling)
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = block_row + ty * TM + i;
    if (row >= M) continue;

#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = block_col + tx * TN + j;
      if (col >= N) continue;

      const int idx = row * N + col;
      d_C[idx] = alpha * acc[i][j] + (beta != 0.0f ? beta * d_C[idx] : 0.0f);
    }
  }
}

SgemmLaunchFunc sgemm_register_tiled_launch =
    [](const float* d_A, const float* d_B, float* d_C, int M, int N, int K,
       float alpha, float beta, cudaStream_t stream) {
      const int BM = 64;
      const int BN = 64;
      const int BK = 32;
      const int TM = 8;
      const int TN = 8;
      const int NT = (BM / TM) * (BN / TN);
      const dim3 block(NT);
      const dim3 grid(cdiv(N, BN), cdiv(M, BM));
      sgemm_register_tiled_kernel<BM, BN, BK, TM, TN, NT>
          <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
      CUDA_CHECK(cudaGetLastError());
    };
