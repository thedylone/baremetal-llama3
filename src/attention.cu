#include <cuda_runtime.h>

#include "attention.cuh"
#include "kernels.cuh"

namespace {
constexpr float kNegInf = -1e6f;

/// @brief Compute scaled dot-product attention scores for one query head
/// For each query head i, compute scores S_i = (Q_i K^T) / sqrt(head_dim)
/// @param d_q_head Pointer to Q_i of shape (seq_len, head_dim)
/// @param d_k_head Pointer to K_i of shape (seq_len, head_dim)
/// @param d_scores Pointer to the output S_i of shape (seq_len, seq_len)
/// @param seq_len Length of the sequence
/// @param head_dim Dimension of each query/key head
/// @param q_row_stride Stride for accessing rows in the query matrix
/// @param k_row_stride Stride for accessing rows in the key matrix
/// @param scale Scaling factor (1/sqrt(head_dim)) to apply to the scores
__global__ void score_kernel(const float* __restrict__ d_q_head,
                             const float* __restrict__ d_k_head,
                             float* __restrict__ d_scores, int seq_len,
                             int head_dim, int q_row_stride, int k_row_stride,
                             float scale) {
  const int k_pos = blockIdx.x * blockDim.x + threadIdx.x;  // which key
  const int q_pos = blockIdx.y * blockDim.y + threadIdx.y;  // which query
  if (q_pos >= seq_len || k_pos >= seq_len) return;

  const float* qv = d_q_head + static_cast<size_t>(q_pos) * q_row_stride;
  const float* kv = d_k_head + static_cast<size_t>(k_pos) * k_row_stride;

  float dot = 0.0f;
  for (int d = 0; d < head_dim; ++d) dot += qv[d] * kv[d];
  d_scores[static_cast<size_t>(q_pos) * seq_len + k_pos] = dot * scale;
}

/// @brief Apply causal masking to the attention scores
/// For each position pair (p, q), q > p, set S[q, p] = -inf
/// @param d_scores Pointer to the attention scores of shape (seq_len, seq_len)
/// @param seq_len Length of the sequence
/// @return
__global__ void causal_mask_kernel(float* __restrict__ d_scores, int seq_len) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;  // column
  const int p = blockIdx.y * blockDim.y + threadIdx.y;  // row
  if (p >= seq_len || q >= seq_len) return;

  const size_t idx = static_cast<size_t>(p) * seq_len + q;
  if (q > p) d_scores[idx] = kNegInf;
}

/// @brief Numerically stable row softmax kernel
/// Find the row max first and subtract it before exponentiating to prevent
/// overflow
/// @param d_scores Pointer to the attention scores of shape (seq_len, seq_len)
/// @param seq_len Length of the sequence
/// @return
__global__ void row_softmax_kernel(float* __restrict__ d_scores, int seq_len) {
  extern __shared__ float shmem[];  // shared memory per block for reduction
  // each block handles one row
  // threads within the block stride over the columns
  const int row = blockIdx.x;
  if (row >= seq_len) return;
  float* row_ptr = d_scores + static_cast<size_t>(row) * seq_len;

  // 1) find per-thread max
  float max_val = kNegInf;
  for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
    max_val = fmaxf(max_val, row_ptr[i]);
  }
  shmem[threadIdx.x] = max_val;
  __syncthreads();

  // 2) parallel reduction to find the row max
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shmem[threadIdx.x] =
          fmaxf(shmem[threadIdx.x], shmem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float row_max = shmem[0];

  // 3) compute exp and sum
  float sum_exp = 0.0f;
  for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
    const float exp_val = expf(row_ptr[i] - row_max);
    row_ptr[i] = exp_val;  // store the exp value back for now
    sum_exp += exp_val;
  }
  shmem[threadIdx.x] = sum_exp;
  __syncthreads();

  // 4) parallel reduction to find the sum
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) shmem[threadIdx.x] += shmem[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_sum_exp = 1.0f / (shmem[0] + 1e-6f);

  // 5) normalize the row
  for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
    row_ptr[i] *= inv_sum_exp;
}

__global__ void output_kernel(const float* __restrict__ d_alpha_head,
                              const float* __restrict__ d_v_head,
                              float* __restrict__ d_out_head, int seq_len,
                              int head_dim, int v_row_stride,
                              int out_row_stride) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  const int q_pos = blockIdx.y;
  if (d >= head_dim || q_pos >= seq_len) return;

  const float* alpha_row = d_alpha_head + static_cast<size_t>(q_pos) * seq_len;
  const float* v_row = d_v_head + d;
  float* out_val = d_out_head + static_cast<size_t>(q_pos) * out_row_stride + d;

  float sum = 0.0f;
  for (int k = 0; k < seq_len; ++k) {
    sum += alpha_row[k] * v_row[static_cast<size_t>(k) * v_row_stride];
  }
  *out_val = sum;
}
}  // namespace

void gqa_attention(const float* __restrict__ d_q, const float* __restrict__ d_k,
                   const float* __restrict__ d_v, float* __restrict__ d_out,
                   const int seq_len, const int num_heads,
                   const int num_kv_heads, const int head_dim,
                   cudaStream_t stream) {
  if (seq_len <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0)
    return;

  const int q_o_row_stride = num_heads * head_dim;
  const int k_v_row_stride = num_kv_heads * head_dim;
  const int heads_per_kv = num_heads / num_kv_heads;  // h / h_k
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  // Allocate temporary buffer for attention scores
  float* d_scores = nullptr;
  CUDA_CHECK(cudaMalloc(&d_scores, static_cast<size_t>(seq_len) *
                                       static_cast<size_t>(seq_len) *
                                       sizeof(float)));

  const dim3 block2d(16, 16);
  const dim3 grid2d(cdiv(seq_len, static_cast<int>(block2d.x)),
                    cdiv(seq_len, static_cast<int>(block2d.y)));

  constexpr int kSoftmaxThreads = 256;
  const int softmax_shmem = kSoftmaxThreads * sizeof(float);

  const dim3 out_block(256);
  const dim3 out_grid(cdiv(head_dim, static_cast<int>(out_block.x)), seq_len);

  // for each query head, compute attention scores
  for (int i = 0; i < num_heads; ++i) {
    const int g = i / heads_per_kv;  // corresponding KV head index

    const float* q_head = d_q + static_cast<size_t>(i) * head_dim;
    const float* k_head = d_k + static_cast<size_t>(g) * head_dim;
    const float* v_head = d_v + static_cast<size_t>(g) * head_dim;
    float* out_head = d_out + static_cast<size_t>(i) * head_dim;

    // 1) compute scores S_i = (Q_i K^T) / sqrt(head_dim)
    score_kernel<<<grid2d, block2d, 0, stream>>>(
        q_head, k_head, d_scores, seq_len, head_dim, q_o_row_stride,
        k_v_row_stride, scale);

    // 2) apply causal mask to S_i
    causal_mask_kernel<<<grid2d, block2d, 0, stream>>>(d_scores, seq_len);

    // 3) apply softmax to S_i -> a_i
    row_softmax_kernel<<<seq_len, kSoftmaxThreads, softmax_shmem, stream>>>(
        d_scores, seq_len);

    // 4) compute attended output O_i = a_i V_i and concat
    output_kernel<<<out_grid, out_block, 0, stream>>>(
        d_scores, v_head, out_head, seq_len, head_dim, k_v_row_stride,
        q_o_row_stride);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(d_scores));
}
