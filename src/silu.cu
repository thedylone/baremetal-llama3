#include "kernels.cuh"
#include "silu.cuh"

__global__ void silu_kernel(float* __restrict__ d_gate,
                            const float* __restrict__ d_up, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float z = d_gate[idx];
  float sigmoid = 1.0f / (1.0f + expf(-z));
  d_gate[idx] = z * sigmoid * d_up[idx];
}

void silu(float* d_gate, const float* d_up, int n, cudaStream_t stream) {
  const int threads = 256;
  int blocks = cdiv(n, threads);
  silu_kernel<<<blocks, threads, 0, stream>>>(d_gate, d_up, n);
}