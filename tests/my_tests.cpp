
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "config.h"
#include "kernels.cuh"
#include "model.h"
#include "rms_norm.cuh"
#include "tokenizer.h"

using namespace std;

// -----------------------------------------------------------------------------
// Comparison helper
// -----------------------------------------------------------------------------

template <typename T>
static bool check_max_abs(const char* label, const vector<T>& got,
                          const vector<T>& expected, float epsilon) {
  if (got.size() != expected.size()) {
    cout << "  " << label << " size mismatch: got " << got.size()
         << " expected " << expected.size() << "\n";
    return false;
  }
  float max_err = 0.0f;
  int worst = 0;
  for (int i = 0; i < (int)got.size(); i++) {
    float err = fabs((float)got[i] - (float)expected[i]);
    if (err > max_err) {
      max_err = err;
      worst = i;
    }
  }
  bool ok = max_err <= epsilon;
  cout << "  " << label << ": max_err=" << max_err << " at index " << worst
       << (ok ? " PASSED" : " FAILED") << "\n";
  if (!ok) {
    cout << "    got=" << got[worst] << " expected=" << expected[worst]
         << " eps=" << epsilon << "\n";
  }
  return ok;
}

// -----------------------------------------------------------------------------
// Tokenizer tests
// -----------------------------------------------------------------------------

bool test_tokenizer() {
  cout << "\n=== Tokenizer test ===\n";
  BPETokenizer tok(TOKENIZER_PATH);

  // Test case 1: "Hello world"
  {
    string input = "Hello world";
    vector<int> expected = {128000, 9906, 1917};
    auto token_ids = tok.encode(input);
    if (!check_max_abs("\"Hello world\"", token_ids, expected,
                       0)) {  // exact match
      return false;
    }
  }

  // Test case 2: long sentence
  {
    string input = "The quick brown fox jumps over the lazy dog.";
    vector<int> expected = {128000, 791, 4062,  14198, 39935, 35308,
                            927,    279, 16053, 5679,  13};
    auto token_ids = tok.encode(input);
    if (!check_max_abs("long sentence", token_ids, expected,
                       0)) {  // exact match
      return false;
    }
  }

  cout << "All Tokenizer tests PASSED\n";
  return true;
}

// -----------------------------------------------------------------------------
// Embeddings vs Python reference test
// (requires: python tools/generate_embedding_reference.py)
// -----------------------------------------------------------------------------

bool test_embeddings_vs_python() {
  const char* path = "tests/data/embedding_reference.bin";
  const float EPSILON = 1e-2f;

  cout << "\n=== Embeddings vs Python reference ===\n";

  ifstream f(path, ios::binary);
  if (!f) {
    cout << "Reference file not found: " << path << "\n"
         << "Run: python tools/generate_embedding_reference.py\n";
    return false;
  }

  int seq_len, hidden_size;
  f.read(reinterpret_cast<char*>(&seq_len), sizeof(int));
  f.read(reinterpret_cast<char*>(&hidden_size), sizeof(int));

  vector<int> token_ids(seq_len);
  f.read(reinterpret_cast<char*>(token_ids.data()), seq_len * sizeof(int));

  vector<float> ref_emb(seq_len * hidden_size);
  f.read(reinterpret_cast<char*>(ref_emb.data()),
         seq_len * hidden_size * sizeof(float));
  f.close();

  cout << "Reference: seq_len=" << seq_len << " hidden_size=" << hidden_size
       << " token_ids: ";
  for (int id : token_ids) cout << id << " ";
  cout << "\n";

  // GPU embedding lookup
  LlamaModel model;
  model.load("assets/llama3/dumped");

  int* d_ids;
  float* d_emb;
  cudaMalloc(&d_ids, seq_len * sizeof(int));
  cudaMalloc(&d_emb, seq_len * hidden_size * sizeof(float));
  cudaMemcpy(d_ids, token_ids.data(), seq_len * sizeof(int),
             cudaMemcpyHostToDevice);

  model.embed(d_ids, d_emb, seq_len);
  cudaDeviceSynchronize();

  vector<float> got_emb(seq_len * hidden_size);
  cudaMemcpy(got_emb.data(), d_emb, got_emb.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_ids);
  cudaFree(d_emb);

  bool ok = check_max_abs("embeddings", got_emb, ref_emb, EPSILON);
  cout << (ok ? "Embeddings vs Python PASSED\n"
              : "Embeddings vs Python FAILED\n");
  return ok;
}

// -----------------------------------------------------------------------------
// Matmul test
// -----------------------------------------------------------------------------

bool test_matrix_multiply() {
  cout << "\n=== Matmul test ===\n";
  const float EPSILON = 1e-5f;

  // Test case: small 2x3 @ 3x4
  {
    int M = 2, K = 3, N = 4;
    vector<float> A = {1, 2, 3, 4, 5, 6};
    vector<float> B = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    vector<float> C(M * N);
    vector<float> C_expected = {74, 80, 86, 92, 173, 188, 203, 218};

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

    if (!check_max_abs("small 2x3 @ 3x4", C, C_expected, EPSILON)) {
      return false;
    }
  }

  // Test case: negative values
  {
    int M = 3, K = 2, N = 3;
    vector<float> A = {1, -1, 0, 2, -2, 3};
    vector<float> B = {-1, 0, 1, 2, -2, 0};
    vector<float> C(M * N);
    vector<float> C_expected = {-3, 2, 1, 4, -4, 0, 8, -6, -2};

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

    if (!check_max_abs("negative values", C, C_expected, EPSILON)) {
      return false;
    }
  }

  cout << "All Matmul tests PASSED\n";
  return true;
}

// -----------------------------------------------------------------------------
// RMSNorm test
// -----------------------------------------------------------------------------

bool test_rms_norm() {
  cout << "\n=== RMSNorm test ===\n";
  const float EPS = 1e-5f;
  const float TOL = 1e-4f;

  // Test case 1: small known input (2 rows x 4 cols)
  {
    const int s = 2, d = 4;
    vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -1.0f, 1.5f, -2.0f};
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
    cudaMemcpy(d_in, input.data(), s * d * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), d * sizeof(float),
               cudaMemcpyHostToDevice);

    rms_norm(d_in, d_gamma, d_out, s, d);
    cudaDeviceSynchronize();

    vector<float> got(s * d);
    cudaMemcpy(got.data(), d_out, s * d * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_gamma);
    cudaFree(d_out);

    if (!check_max_abs("small known input", got, expected, TOL)) {
      return false;
    }
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
    cudaMemcpy(d_gamma, gamma.data(), d * sizeof(float),
               cudaMemcpyHostToDevice);

    rms_norm(d_in, d_gamma, d_out, s, d);
    cudaDeviceSynchronize();

    vector<float> got(d);
    cudaMemcpy(got.data(), d_out, d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_gamma);
    cudaFree(d_out);

    if (!check_max_abs("non-trivial gamma", got, expected, TOL)) {
      return false;
    }
  }

  // Test case 3: d > blockDim (d=512, exercises the stride loop)
  {
    const int s = 3, d = 512;
    vector<float> input(s * d);
    vector<float> gamma(d);
    srand(42);
    for (int i = 0; i < s * d; i++)
      input[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
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
    cudaMemcpy(d_in, input.data(), s * d * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), d * sizeof(float),
               cudaMemcpyHostToDevice);

    rms_norm(d_in, d_gamma, d_out, s, d);
    cudaDeviceSynchronize();

    vector<float> got(s * d);
    cudaMemcpy(got.data(), d_out, s * d * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_gamma);
    cudaFree(d_out);

    if (!check_max_abs("d > blockDim", got, expected, TOL)) {
      return false;
    }
  }

  // Test case 4: d=4096 (typical LLM hidden dim)
  {
    const int s = 5, d = 4096;
    vector<float> input(s * d);
    vector<float> gamma(d);
    srand(123);
    for (int i = 0; i < s * d; i++)
      input[i] = (rand() / (float)RAND_MAX) * 4.0f - 2.0f;
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
    cudaMemcpy(d_in, input.data(), s * d * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), d * sizeof(float),
               cudaMemcpyHostToDevice);

    rms_norm(d_in, d_gamma, d_out, s, d);
    cudaDeviceSynchronize();

    vector<float> got(s * d);
    cudaMemcpy(got.data(), d_out, s * d * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_gamma);
    cudaFree(d_out);

    if (!check_max_abs("d=4096", got, expected, TOL)) {
      return false;
    }
  }

  cout << "All RMSNorm tests PASSED\n";
  return true;
}

// -----------------------------------------------------------------------------
// QKV vs Python reference test
// (requires: python tools/generate_qkv_reference.py)
// -----------------------------------------------------------------------------

bool test_qkv_vs_python() {
  const char* path = "tests/data/qkv_reference.bin";
  const float EPSILON = 1e-2f;

  cout << "\n=== QKV vs Python reference ===\n";

  ifstream f(path, ios::binary);
  if (!f) {
    cout << "Reference file not found: " << path << "\n"
         << "Run: python tools/generate_qkv_reference.py\n";
    return false;
  }

  int seq_len, hidden_size, kv_dim;
  f.read(reinterpret_cast<char*>(&seq_len), sizeof(int));
  f.read(reinterpret_cast<char*>(&hidden_size), sizeof(int));
  f.read(reinterpret_cast<char*>(&kv_dim), sizeof(int));

  vector<int> token_ids(seq_len);
  f.read(reinterpret_cast<char*>(token_ids.data()), seq_len * sizeof(int));

  auto read_floats = [&](int n) {
    vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return v;
  };
  vector<float> ref_normed = read_floats(seq_len * hidden_size);
  vector<float> ref_q = read_floats(seq_len * hidden_size);
  vector<float> ref_k = read_floats(seq_len * kv_dim);
  vector<float> ref_v = read_floats(seq_len * kv_dim);
  f.close();

  cout << "Reference: seq_len=" << seq_len << " hidden_size=" << hidden_size
       << " kv_dim=" << kv_dim << " token_ids: ";
  for (int id : token_ids) cout << id << " ";
  cout << "\n";

  // Load model and run forward pass
  LlamaModel model;
  model.load("assets/llama3/dumped");

  int* d_ids;
  float *d_emb, *d_normed, *d_q, *d_k, *d_v;
  cudaMalloc(&d_ids, seq_len * sizeof(int));
  cudaMalloc(&d_emb, seq_len * hidden_size * sizeof(float));
  cudaMalloc(&d_normed, seq_len * hidden_size * sizeof(float));
  cudaMalloc(&d_q, seq_len * hidden_size * sizeof(float));
  cudaMalloc(&d_k, seq_len * kv_dim * sizeof(float));
  cudaMalloc(&d_v, seq_len * kv_dim * sizeof(float));

  cudaMemcpy(d_ids, token_ids.data(), seq_len * sizeof(int),
             cudaMemcpyHostToDevice);

  model.embed(d_ids, d_emb, seq_len);
  rms_norm(d_emb, model.layers[0].d_input_layernorm_weight, d_normed, seq_len,
           hidden_size);
  sgemm(d_normed, model.layers[0].d_q_proj_weight, d_q, seq_len, hidden_size,
        hidden_size);
  sgemm(d_normed, model.layers[0].d_k_proj_weight, d_k, seq_len, kv_dim,
        hidden_size);
  sgemm(d_normed, model.layers[0].d_v_proj_weight, d_v, seq_len, kv_dim,
        hidden_size);
  cudaDeviceSynchronize();

  auto download = [](float* d_ptr, int n) {
    vector<float> h(n);
    cudaMemcpy(h.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
  };
  vector<float> got_normed = download(d_normed, seq_len * hidden_size);
  vector<float> got_q = download(d_q, seq_len * hidden_size);
  vector<float> got_k = download(d_k, seq_len * kv_dim);
  vector<float> got_v = download(d_v, seq_len * kv_dim);

  cudaFree(d_ids);
  cudaFree(d_emb);
  cudaFree(d_normed);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);

  bool all_ok = true;
  all_ok &= check_max_abs("RMSNorm", got_normed, ref_normed, EPSILON);
  all_ok &= check_max_abs("Q", got_q, ref_q, EPSILON);
  all_ok &= check_max_abs("K", got_k, ref_k, EPSILON);
  all_ok &= check_max_abs("V", got_v, ref_v, EPSILON);

  cout << (all_ok ? "All QKV vs Python checks PASSED\n"
                  : "Some QKV vs Python checks FAILED\n");
  return all_ok;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
  test_tokenizer();
  test_embeddings_vs_python();
  test_matrix_multiply();
  test_rms_norm();
  test_qkv_vs_python();
  return 0;
}
