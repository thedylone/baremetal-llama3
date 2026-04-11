
#include <iomanip>
#include <iostream>
#include <vector>

#include "config.h"
#include "kernels.cuh"
#include "model.h"
#include "tokenizer.h"

using namespace std;

int main() {
  const string dump_path = "assets/llama3/dumped";

  // 1. Load model weights to GPU
  cout << "Loading model weights...\n";
  LlamaModel model;
  model.load(dump_path);
  cout << "Model loaded (" << model.num_layers << " layers, "
       << "hidden_size=" << model.hidden_size << ").\n";

  // 2. Tokenize input
  BPETokenizer tok(TOKENIZER_PATH);
  string input = "Hello world";
  auto ids = tok.encode(input);
  int seq_len = static_cast<int>(ids.size());
  cout << "Input: \"" << input << "\" -> " << seq_len << " tokens\n";

  // 3. Upload token ids to GPU
  int* d_ids;
  CUDA_CHECK(cudaMalloc(&d_ids, seq_len * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), seq_len * sizeof(int),
                        cudaMemcpyHostToDevice));

  // 4. Allocate working buffers
  size_t hidden_bytes = seq_len * model.hidden_size * sizeof(float);
  float *d_emb, *d_normed;
  float *d_q, *d_k, *d_v;
  CUDA_CHECK(cudaMalloc(&d_emb, hidden_bytes));
  CUDA_CHECK(cudaMalloc(&d_normed, hidden_bytes));
  CUDA_CHECK(cudaMalloc(&d_q, seq_len * model.hidden_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, seq_len * model.kv_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, seq_len * model.kv_dim * sizeof(float)));

  // 5. Embedding lookup
  model.embed(d_ids, d_emb, seq_len);

  // 6. Layer 0: RMSNorm -> Q/K/V projection
  const LlamaLayer& layer = model.layers[0];

  rms_norm(d_emb, layer.d_input_layernorm_weight, d_normed, seq_len,
           model.hidden_size);

  // Q = normed @ W_q^T   [seq, hidden] x [hidden, hidden] -> [seq, hidden]
  sgemm(d_normed, layer.d_q_proj_weight, d_q, seq_len, model.hidden_size,
        model.hidden_size);
  // K = normed @ W_k^T   [seq, hidden] x [hidden, kv_dim] -> [seq, kv_dim]
  sgemm(d_normed, layer.d_k_proj_weight, d_k, seq_len, model.kv_dim,
        model.hidden_size);
  // V = normed @ W_v^T   [seq, hidden] x [hidden, kv_dim] -> [seq, kv_dim]
  sgemm(d_normed, layer.d_v_proj_weight, d_v, seq_len, model.kv_dim,
        model.hidden_size);

  CUDA_CHECK(cudaDeviceSynchronize());

  // Print first 10 values per projection for verification
  auto print_first = [](const char* name, float* d_ptr, int total) {
    int n = min(total, 10);
    vector<float> h(n);
    cudaMemcpy(h.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    cout << name << "[0][0:" << n << "]: ";
    for (int i = 0; i < n; i++) cout << fixed << setprecision(4) << h[i] << " ";
    cout << "\n";
  };
  print_first("Q", d_q, seq_len * model.hidden_size);
  print_first("K", d_k, seq_len * model.kv_dim);
  print_first("V", d_v, seq_len * model.kv_dim);

  // Cleanup
  cudaFree(d_ids);
  cudaFree(d_emb);
  cudaFree(d_normed);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);

  return 0;
}
