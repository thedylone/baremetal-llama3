#include <cuda_runtime.h>

#include <cmath>

#include "rope.cuh"

namespace {

/// Precompute cos/sin tables.
/// Grid: (ceil(max_seq_len * half_dim / 256))
/// Each thread computes one (position, pair_index) entry.
__global__ void rope_precompute_kernel(float* __restrict__ d_cos,
                                       float* __restrict__ d_sin,
                                       int max_seq_len, int half_dim,
                                       float theta_base) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_seq_len * half_dim) return;

  const int p = idx / half_dim;  // position
  const int i = idx % half_dim;  // pair index

  // θ_i = 1 / base^(2i / head_dim)   where head_dim = 2 * half_dim
  const float theta = 1.0f / powf(theta_base, (2.0f * i) / (2.0f * half_dim));
  const float angle = p * theta;

  d_cos[idx] = cosf(angle);
  d_sin[idx] = sinf(angle);
}

/// Apply RoPE in-place.
/// Flat layout: qk[pos * (num_heads * head_dim) + head * head_dim + dim]
/// For each head vector, pairs dim i with dim i + half_dim.
///
/// Grid:  (num_heads, seq_len)
/// Block: half_dim threads (one per pair)
__global__ void rope_apply_kernel(float* __restrict__ d_qk,
                                  const float* __restrict__ d_cos,
                                  const float* __restrict__ d_sin,
                                  int num_heads, int head_dim) {
  const int half_dim = head_dim / 2;
  const int head = blockIdx.x;  // which head
  const int pos = blockIdx.y;   // which position
  const int i = threadIdx.x;    // pair index
  if (i >= half_dim) return;

  const int stride = num_heads * head_dim;
  float* base = d_qk + pos * stride + head * head_dim;

  const float q_i = base[i];
  const float q_ihalf = base[i + half_dim];

  // cos/sin table is [max_seq_len, half_dim]
  const float c = d_cos[pos * half_dim + i];
  const float s = d_sin[pos * half_dim + i];

  // Rotation: first half with second half
  base[i] = q_i * c - q_ihalf * s;
  base[i + half_dim] = q_i * s + q_ihalf * c;
}

}  // namespace

void rope_precompute(float* d_cos, float* d_sin, int max_seq_len, int head_dim,
                     float theta_base, cudaStream_t stream) {
  const int half_dim = head_dim / 2;
  const int total = max_seq_len * half_dim;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  rope_precompute_kernel<<<blocks, threads, 0, stream>>>(
      d_cos, d_sin, max_seq_len, half_dim, theta_base);
}

void rope_apply(float* d_qk, const float* d_cos, const float* d_sin,
                int seq_len, int num_heads, int head_dim, cudaStream_t stream) {
  const int half_dim = head_dim / 2;
  const dim3 grid(num_heads, seq_len);
  const dim3 block(half_dim);  // 64 threads per block (128/2)
  rope_apply_kernel<<<grid, block, 0, stream>>>(d_qk, d_cos, d_sin, num_heads,
                                                head_dim);
}
