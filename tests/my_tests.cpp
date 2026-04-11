
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "config.h"
#include "kernels.cuh"
#include "loader.h"
#include "model.h"
#include "rms_norm.cuh"
#include "tokenizer.h"

using namespace std;

// -----------------------------------------------------------------------------
// Tokenizer tests
// -----------------------------------------------------------------------------

void show_tokenize(string input = "Hello world") {
  BPETokenizer tok(TOKENIZER_PATH);

  auto ids = tok.encode(input);
  for (int &id : ids) {
    cout << id << " ";
  }

  cout << "Token IDs: ";
  for (int id : ids) {
    cout << id << " ";
  }
  cout << "\n";
}

// -----------------------------------------------------------------------------
// Matmul test
// -----------------------------------------------------------------------------

void test_matrix_multiply() {
  int M = 2, K = 3, N = 4;
  vector<float> A = {1, 2, 3, 4, 5, 6};
  vector<float> B = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  vector<float> C(M * N);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  sgemm(d_A, d_B, d_C, M, N, K);
  cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cout << "Result of A * B:\n";
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      cout << fixed << setprecision(2) << C[i * N + j] << " ";
    }
    cout << "\n";
  }
}

// -----------------------------------------------------------------------------
// Embedding test
// -----------------------------------------------------------------------------

void show_embeddings(string input = "Hello world") {
  BPETokenizer tok(TOKENIZER_PATH);
  auto ids = tok.encode(input);

  LlamaDumpLoader loader(DumpFloatType::FP32);
  if (!loader.load_embeddings("assets/llama3/dumped", 4096)) {
    throw runtime_error("Failed to load embeddings");
  }
  float_t *embeddings = loader.get_embeddings(ids);

  cout << "Embeddings for input: " << input << "\n";
  for (size_t i = 0; i < ids.size(); ++i) {
    cout << "Token ID: " << ids[i] << " Embedding: [";
    for (int j = 0; j < 10; ++j) {
      cout << fixed << setprecision(4) << embeddings[i * 4096 + j] << " ";
    }
    cout << "...]\n";
  }

  std::ofstream out("cpp_embeddings.bin", ios::binary);
  if (!out) {
    cerr << "Failed to open output file for embeddings\n";
    return;
  }
  out.write(reinterpret_cast<char *>(embeddings),
            ids.size() * 4096 * sizeof(float_t));

  delete[] embeddings;
}

// -----------------------------------------------------------------------------
// RMSNorm test
// -----------------------------------------------------------------------------

bool test_rms_norm() {
  const float EPS = 1e-5f;
  const float TOL = 1e-4f;

  // Test case 1: small known input (2 rows x 4 cols)
  {
    const int s = 2, d = 4;
    vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f,
                           0.5f, -1.0f, 1.5f, -2.0f};
    vector<float> gamma = {1.0f, 1.0f, 1.0f, 1.0f};
    vector<float> expected(s * d);

    for (int r = 0; r < s; r++) {
      float sum_sq = 0.0f;
      for (int c = 0; c < d; c++) {
        float v = input[r * d + c];
        sum_sq += v * v;
      }
      float inv_rms = 1.0f / sqrtf(sum_sq / d + EPS);
      for (int c = 0; c < d; c++) {
        expected[r * d + c] = input[r * d + c] * inv_rms * gamma[c];
      }
    }

    float *d_in, *d_gamma, *d_out;
    cudaMalloc(&d_in, s * d * sizeof(float));
    cudaMalloc(&d_gamma, d * sizeof(float));
    cudaMalloc(&d_out, s * d * sizeof(float));
    cudaMemcpy(d_in, input.data(), s * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), d * sizeof(float), cudaMemcpyHostToDevice);

    rms_norm(d_in, d_gamma, d_out, s, d);
    cudaDeviceSynchronize();

    vector<float> got(s * d);
    cudaMemcpy(got.data(), d_out, s * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_gamma);
    cudaFree(d_out);

    for (int i = 0; i < s * d; i++) {
      if (fabs(got[i] - expected[i]) > TOL) {
        cout << "FAILED test_rms_norm case 1: index " << i
             << " got=" << got[i] << " expected=" << expected[i] << "\n";
        return false;
      }
    }
    cout << "test_rms_norm case 1 (small, gamma=1): PASSED\n";
  }

  // Test case 2: non-trivial gamma
  {
    const int s = 1, d = 4;
    vector<float> input = {2.0f, -3.0f, 1.0f, 4.0f};
    vector<float> gamma = {0.5f, 2.0f, 1.5f, 0.1f};
    vector<float> expected(d);

    float sum_sq = 0.0f;
    for (int c = 0; c < d; c++) sum_sq += input[c] * input[c];
    float inv_rms = 1.0f / sqrtf(sum_sq / d + EPS);
    for (int c = 0; c < d; c++) expected[c] = input[c] * inv_rms * gamma[c];

    float *d_in, *d_gamma, *d_out;
    cudaMalloc(&d_in, d * sizeof(float));
    cudaMalloc(&d_gamma, d * sizeof(float));
    cudaMalloc(&d_out, d * sizeof(float));
    cudaMemcpy(d_in, input.data(), d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), d * sizeof(float), cudaMemcpyHostToDevice);

    rms_norm(d_in, d_gamma, d_out, s, d);
    cudaDeviceSynchronize();

    vector<float> got(d);
    cudaMemcpy(got.data(), d_out, d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_gamma);
    cudaFree(d_out);

    for (int i = 0; i < d; i++) {
      if (fabs(got[i] - expected[i]) > TOL) {
        cout << "FAILED test_rms_norm case 2: index " << i
             << " got=" << got[i] << " expected=" << expected[i] << "\n";
        return false;
      }
    }
    cout << "test_rms_norm case 2 (non-trivial gamma): PASSED\n";
  }

  // Test case 3: d > blockDim (d=512, exercises the stride loop)
  {
    const int s = 3, d = 512;
    vector<float> input(s * d);
    vector<float> gamma(d);
    srand(42);
    for (int i = 0; i < s * d; i++) input[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < d; i++) gamma[i] = (rand() / (float)RAND_MAX) * 2.0f;

    vector<float> expected(s * d);
    for (int r = 0; r < s; r++) {
      float sum_sq = 0.0f;
      for (int c = 0; c < d; c++) {
        float v = input[r * d + c];
        sum_sq += v * v;
      }
      float inv_rms = 1.0f / sqrtf(sum_sq / d + EPS);
      for (int c = 0; c < d; c++) {
        expected[r * d + c] = input[r * d + c] * inv_rms * gamma[c];
      }
    }

    float *d_in, *d_gamma, *d_out;
    cudaMalloc(&d_in, s * d * sizeof(float));
    cudaMalloc(&d_gamma, d * sizeof(float));
    cudaMalloc(&d_out, s * d * sizeof(float));
    cudaMemcpy(d_in, input.data(), s * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), d * sizeof(float), cudaMemcpyHostToDevice);

    rms_norm(d_in, d_gamma, d_out, s, d);
    cudaDeviceSynchronize();

    vector<float> got(s * d);
    cudaMemcpy(got.data(), d_out, s * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_gamma);
    cudaFree(d_out);

    float max_err = 0.0f;
    for (int i = 0; i < s * d; i++) {
      float err = fabs(got[i] - expected[i]);
      if (err > max_err) max_err = err;
    }
    if (max_err > TOL) {
      cout << "FAILED test_rms_norm case 3: max_err=" << max_err << "\n";
      return false;
    }
    cout << "test_rms_norm case 3 (d=512, stride loop): PASSED (max_err=" << max_err << ")\n";
  }

  // Test case 4: d=4096 (typical LLM hidden dim)
  {
    const int s = 5, d = 4096;
    vector<float> input(s * d);
    vector<float> gamma(d);
    srand(123);
    for (int i = 0; i < s * d; i++) input[i] = (rand() / (float)RAND_MAX) * 4.0f - 2.0f;
    for (int i = 0; i < d; i++) gamma[i] = (rand() / (float)RAND_MAX) + 0.5f;

    vector<float> expected(s * d);
    for (int r = 0; r < s; r++) {
      float sum_sq = 0.0f;
      for (int c = 0; c < d; c++) {
        float v = input[r * d + c];
        sum_sq += v * v;
      }
      float inv_rms = 1.0f / sqrtf(sum_sq / d + EPS);
      for (int c = 0; c < d; c++) {
        expected[r * d + c] = input[r * d + c] * inv_rms * gamma[c];
      }
    }

    float *d_in, *d_gamma, *d_out;
    cudaMalloc(&d_in, s * d * sizeof(float));
    cudaMalloc(&d_gamma, d * sizeof(float));
    cudaMalloc(&d_out, s * d * sizeof(float));
    cudaMemcpy(d_in, input.data(), s * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), d * sizeof(float), cudaMemcpyHostToDevice);

    rms_norm(d_in, d_gamma, d_out, s, d);
    cudaDeviceSynchronize();

    vector<float> got(s * d);
    cudaMemcpy(got.data(), d_out, s * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_gamma);
    cudaFree(d_out);

    float max_err = 0.0f;
    for (int i = 0; i < s * d; i++) {
      float err = fabs(got[i] - expected[i]);
      if (err > max_err) max_err = err;
    }
    if (max_err > TOL) {
      cout << "FAILED test_rms_norm case 4: max_err=" << max_err << "\n";
      return false;
    }
    cout << "test_rms_norm case 4 (d=4096, LLM dim): PASSED (max_err=" << max_err << ")\n";
  }

  cout << "All RMSNorm tests PASSED\n";
  return true;
}

// -----------------------------------------------------------------------------
// QKV projection test (embed -> RMSNorm -> Q/K/V on real weights)
// -----------------------------------------------------------------------------

void test_qkv_projection() {
  cout << "\n=== Loading model weights to GPU ===\n";
  LlamaModel model;
  model.load("assets/llama3/dumped");
  cout << "Model loaded (" << model.num_layers << " layers).\n";

  BPETokenizer tok(TOKENIZER_PATH);
  auto ids = tok.encode("Hello world");
  int seq_len = static_cast<int>(ids.size());
  cout << "Tokens (" << seq_len << "): ";
  for (int id : ids) cout << id << " ";
  cout << "\n";

  // Verify GPU embedding lookup matches CPU loader
  {
    LlamaDumpLoader loader(DumpFloatType::FP32);
    loader.load_embeddings("assets/llama3/dumped", model.hidden_size);
    float* cpu_emb = loader.get_embeddings(ids);

    int* d_ids;
    float* d_emb;
    cudaMalloc(&d_ids, seq_len * sizeof(int));
    cudaMalloc(&d_emb, seq_len * model.hidden_size * sizeof(float));
    cudaMemcpy(d_ids, ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);

    model.embed(d_ids, d_emb, seq_len);
    cudaDeviceSynchronize();

    vector<float> gpu_emb(seq_len * model.hidden_size);
    cudaMemcpy(gpu_emb.data(), d_emb, gpu_emb.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < seq_len * model.hidden_size; i++) {
      float err = fabs(gpu_emb[i] - cpu_emb[i]);
      if (err > max_err) max_err = err;
    }
    cout << "Embedding lookup GPU vs CPU max_err=" << max_err
         << (max_err < 1e-4f ? " PASSED" : " FAILED") << "\n";
    delete[] cpu_emb;
    cudaFree(d_ids);
    cudaFree(d_emb);
  }

  // Forward pass: embed -> RMSNorm -> Q/K/V projection (layer 0)
  int* d_ids;
  float *d_emb, *d_normed, *d_q, *d_k, *d_v;
  cudaMalloc(&d_ids, seq_len * sizeof(int));
  cudaMalloc(&d_emb, seq_len * model.hidden_size * sizeof(float));
  cudaMalloc(&d_normed, seq_len * model.hidden_size * sizeof(float));
  cudaMalloc(&d_q, seq_len * model.hidden_size * sizeof(float));
  cudaMalloc(&d_k, seq_len * model.kv_dim * sizeof(float));
  cudaMalloc(&d_v, seq_len * model.kv_dim * sizeof(float));

  cudaMemcpy(d_ids, ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);

  model.embed(d_ids, d_emb, seq_len);
  rms_norm(d_emb, model.layers[0].d_input_layernorm_weight, d_normed,
           seq_len, model.hidden_size);
  sgemm(d_normed, model.layers[0].d_q_proj_weight, d_q,
        seq_len, model.hidden_size, model.hidden_size);
  sgemm(d_normed, model.layers[0].d_k_proj_weight, d_k,
        seq_len, model.kv_dim, model.hidden_size);
  sgemm(d_normed, model.layers[0].d_v_proj_weight, d_v,
        seq_len, model.kv_dim, model.hidden_size);
  cudaDeviceSynchronize();

  auto print_first = [](const char* name, float* d_ptr, int n) {
    vector<float> h(n);
    cudaMemcpy(h.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    cout << name << "[0][0:10]: ";
    for (int i = 0; i < 10 && i < n; i++)
      cout << fixed << setprecision(4) << h[i] << " ";
    cout << "\n";
  };
  print_first("Q", d_q, seq_len * model.hidden_size);
  print_first("K", d_k, seq_len * model.kv_dim);
  print_first("V", d_v, seq_len * model.kv_dim);

  cudaFree(d_ids);
  cudaFree(d_emb);
  cudaFree(d_normed);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
  show_tokenize("hello world");
  test_matrix_multiply();
  show_tokenize("Hello world");
  show_embeddings("Hello world");
  test_rms_norm();
  test_qkv_projection();
  return 0;
}
