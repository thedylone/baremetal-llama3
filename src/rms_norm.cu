#include <cuda_runtime.h>

#include "rms_norm.cuh"

namespace {
constexpr float kEps = 1e-5f;
constexpr int kThreads = 256;

__global__ void rms_norm_kernel(const float* __restrict__ input,
                                const float* __restrict__ gamma,
                                float* __restrict__ output, int d) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  const float* x = input + static_cast<size_t>(row) * d;
  float* y = output + static_cast<size_t>(row) * d;

  // 1) Sum of squares for this row
  float local_sum = 0.0f;
  for (int i = tid; i < d; i += blockDim.x) {
    const float v = x[i];
    local_sum += v * v;
  }

  extern __shared__ float shmem[];
  shmem[tid] = local_sum;
  __syncthreads();

  // 2) Parallel reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shmem[tid] += shmem[tid + stride];
    }
    __syncthreads();
  }

  // 3) inv_rms = 1 / sqrt(mean(x^2) + eps)
  if (tid == 0) {
    const float mean_sq = shmem[0] / static_cast<float>(d);
    shmem[0] = rsqrtf(mean_sq + kEps);
  }
  __syncthreads();

  const float inv_rms = shmem[0];

  // 4) Normalize and scale by gamma
  for (int i = tid; i < d; i += blockDim.x) {
    y[i] = x[i] * inv_rms * gamma[i];
  }
}
}  // namespace

void rms_norm(const float* __restrict__ input,
              const float* __restrict__ gamma,
              float* __restrict__ output,
              int s, int d,
              cudaStream_t stream) {
  if (!input || !gamma || !output || s <= 0 || d <= 0) return;

  const dim3 grid(s);
  const dim3 block(kThreads);
  const size_t shmem_bytes = kThreads * sizeof(float);

  rms_norm_kernel<<<grid, block, shmem_bytes, stream>>>(input, gamma, output, d);
}