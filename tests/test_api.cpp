

#include "test_api.h"

#include "config.h"
#include "kernels.cuh"
#include "loader.h"
#include "model.h"
#include "tokenizer.h"

namespace {

// Reshape (num_heads, seq_len, head_dim) row-major to (seq_len, num_heads*head_dim) row-major
vector<float> reshape_heads_last(const vector<float>& input, int num_heads,
                                  int seq_len, int head_dim) {
  vector<float> output(input.size());
  for (int h = 0; h < num_heads; ++h) {
    for (int s = 0; s < seq_len; ++s) {
      for (int d = 0; d < head_dim; ++d) {
        int src_idx = h * seq_len * head_dim + s * head_dim + d;
        int dst_idx = s * num_heads * head_dim + h * head_dim + d;
        output[dst_idx] = input[src_idx];
      }
    }
  }
  return output;
}

LlamaModel& api_model() {
  static LlamaModel model;
  static bool loaded = false;
  if (!loaded) {
    model.load("assets/llama3/dumped");
    loaded = true;
  }
  return model;
}

}  // namespace

vector<int> TestAPI::tokenize(string input) {
  // Your code for testing the tokenizer goes here.
  // You can use the tokenizer you implemented in src/tokenizer_bpe.cpp and
  // include the header file here. Read the project description for more
  // information.
  BPETokenizer tok(TOKENIZER_PATH);
  return tok.encode(input);
}

string TestAPI::detokenize(vector<int> token_ids) {
  BPETokenizer tok(TOKENIZER_PATH);
  return tok.decode(token_ids);
}

vector<float> TestAPI::get_embeddings(vector<int> token_ids) {
  LlamaDumpLoader loader(DumpFloatType::FP32);
  if (!loader.load_embeddings("assets/llama3/dumped", 4096)) {
    throw runtime_error("Failed to load embeddings");
  }
  float_t* embeddings = loader.get_embeddings(token_ids);
  vector<float> result(token_ids.size() * 4096);
  std::copy(embeddings, embeddings + token_ids.size() * 4096, result.data());
  delete[] embeddings;  // Free the allocated buffer
  return result;
}

vector<float> TestAPI::matmul(const vector<float>& A, const vector<float>& B,
                              int M, int K, int N) {
  vector<float> C(M * N, 0.0f);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  sgemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, nullptr);
  cudaDeviceSynchronize();

  cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return C;
}

vector<float> TestAPI::rmsnorm(const vector<float>& x,
                               const vector<float>& gamma, int s, int d) {
  vector<float> output(s * d, 0.0f);

  float *d_x, *d_gamma, *d_output;
  cudaMalloc(&d_x, s * d * sizeof(float));
  cudaMalloc(&d_gamma, d * sizeof(float));
  cudaMalloc(&d_output, s * d * sizeof(float));

  cudaMemcpy(d_x, x.data(), s * d * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, gamma.data(), d * sizeof(float), cudaMemcpyHostToDevice);

  rms_norm(d_x, d_gamma, d_output, s, d);
  cudaDeviceSynchronize();

  cudaMemcpy(output.data(), d_output, s * d * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_gamma);
  cudaFree(d_output);
  return output;
}

vector<float> TestAPI::rope(const vector<float>& qk, int n_heads, int s,
                            int h_d) {
  // Reshape input from (n_heads, s, h_d) to (s, n_heads*h_d)
  auto qk_reshaped = reshape_heads_last(qk, n_heads, s, h_d);
  
  vector<float> output_reshaped(qk_reshaped.size(), 0.0f);

  float* d_qk;
  cudaMalloc(&d_qk, qk_reshaped.size() * sizeof(float));

  cudaMemcpy(d_qk, qk_reshaped.data(), qk_reshaped.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  float *d_cos, *d_sin;
  cudaMalloc(&d_cos, s * (h_d / 2) * sizeof(float));
  cudaMalloc(&d_sin, s * (h_d / 2) * sizeof(float));
  rope_precompute(d_cos, d_sin, s, h_d, ROPE_THETA_BASE);
  rope_apply(d_qk, d_cos, d_sin, s, n_heads, h_d);

  cudaMemcpy(output_reshaped.data(), d_qk, qk_reshaped.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  
  // Reshape output back from (s, n_heads*h_d) to (n_heads, s, h_d)
  vector<float> output(qk.size());
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < s; ++i) {
      for (int d = 0; d < h_d; ++d) {
        int src_idx = i * n_heads * h_d + h * h_d + d;
        int dst_idx = h * s * h_d + i * h_d + d;
        output[dst_idx] = output_reshaped[src_idx];
      }
    }
  }
  
  cudaFree(d_qk);
  cudaFree(d_cos);
  cudaFree(d_sin);
  return output;
}

vector<float> TestAPI::gqa_attention(const vector<float>& Q,
                                     const vector<float>& K,
                                     const vector<float>& V, int s) {
  // Reshape inputs from (num_heads, s, head_dim) to (s, num_heads*head_dim)
  auto Q_reshaped = reshape_heads_last(Q, NUM_HEADS, s, HEAD_DIM);
  auto K_reshaped = reshape_heads_last(K, NUM_KV_HEADS, s, HEAD_DIM);
  auto V_reshaped = reshape_heads_last(V, NUM_KV_HEADS, s, HEAD_DIM);
  
  vector<float> output_reshaped(s * NUM_HEADS * HEAD_DIM, 0.0f);

  float *d_Q, *d_K, *d_V, *d_output;
  cudaMalloc(&d_Q, Q_reshaped.size() * sizeof(float));
  cudaMalloc(&d_K, K_reshaped.size() * sizeof(float));
  cudaMalloc(&d_V, V_reshaped.size() * sizeof(float));
  cudaMalloc(&d_output, output_reshaped.size() * sizeof(float));

  cudaMemcpy(d_Q, Q_reshaped.data(), Q_reshaped.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, K_reshaped.data(), K_reshaped.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, V_reshaped.data(), V_reshaped.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  ::gqa_attention(d_Q, d_K, d_V, d_output, s, NUM_HEADS, NUM_KV_HEADS,
                  HEAD_DIM);
  cudaDeviceSynchronize();

  cudaMemcpy(output_reshaped.data(), d_output, output_reshaped.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  
  // Output is already in (s, num_heads*head_dim) format, reshape to (s, num_heads*head_dim) which is what test expects
  // Actually, the test expects (s, num_heads*head_dim), so just return as-is
  vector<float> output = output_reshaped;
  
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_output);
  return output;
}

vector<float> TestAPI::residual_add(const vector<float>& a,
                                    const vector<float>& b) {
  vector<float> output(a.size(), 0.0f);
  float *d_a, *d_b;
  cudaMalloc(&d_a, a.size() * sizeof(float));
  cudaMalloc(&d_b, b.size() * sizeof(float));

  cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

  ::residual_add(d_a, d_b, a.size());
  cudaDeviceSynchronize();

  cudaMemcpy(output.data(), d_a, a.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  return output;
}

vector<float> TestAPI::silu_mul(const vector<float>& gate,
                                const vector<float>& up) {
  vector<float> output(gate.size(), 0.0f);
  float *d_gate, *d_up;

  cudaMalloc(&d_gate, gate.size() * sizeof(float));
  cudaMalloc(&d_up, up.size() * sizeof(float));

  cudaMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_up, up.data(), up.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  silu(d_gate, d_up, gate.size());
  cudaDeviceSynchronize();

  cudaMemcpy(output.data(), d_gate, gate.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(d_gate);
  cudaFree(d_up);
  return output;
}

vector<float> TestAPI::swiglu_ffn(const vector<float>& x_norm, int layer_idx,
                                  int s) {
  return api_model().swiglu_ffn(x_norm, layer_idx, s);
}

vector<float> TestAPI::decoder_block(const vector<float>& x, int layer_idx,
                                     int s) {
  return api_model().decoder_block(x, layer_idx, s);
}

int TestAPI::forward_one_step(const vector<int>& token_ids) {
  return api_model().forward_one_step(token_ids);
}

vector<int> TestAPI::generate(const vector<int>& token_ids, int n_new) {
  return api_model().generate(token_ids, n_new);
}
