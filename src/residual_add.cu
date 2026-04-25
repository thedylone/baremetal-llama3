#include "residual_add.cuh"
#include "kernels.cuh"

__global__ void residual_add_kernel(float* __restrict__ d_x,
                                    const float* __restrict__ d_residual,
                                    int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) d_x[idx] += d_residual[idx];
}

void residual_add(float* d_x, const float* d_residual, int n,
                  cudaStream_t stream) {
  const int threads = 256;
  int blocks = cdiv(n, threads);
  residual_add_kernel<<<blocks, threads, 0, stream>>>(d_x, d_residual, n);
}
