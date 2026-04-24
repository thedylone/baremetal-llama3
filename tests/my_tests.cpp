#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "attention.cuh"
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
// RoPE vs Python reference test
// (requires: python tools/generate_rope_reference.py)
// -----------------------------------------------------------------------------

bool test_rope_vs_python() {
  const char* path = "tests/data/rope_reference.bin";
  const float EPSILON = 1e-2f;

  cout << "\n=== RoPE vs Python reference ===\n";

  ifstream f(path, ios::binary);
  if (!f) {
    cout << "Reference file not found: " << path << "\n"
         << "Run: python tools/generate_rope_reference.py\n";
    return false;
  }

  int seq_len, hidden_size, kv_dim;
  f.read(reinterpret_cast<char*>(&seq_len), sizeof(int));
  f.read(reinterpret_cast<char*>(&hidden_size), sizeof(int));
  f.read(reinterpret_cast<char*>(&kv_dim), sizeof(int));

  vector<int> token_ids(seq_len);
  f.read(reinterpret_cast<char*>(token_ids.data()), seq_len * sizeof(int));

  vector<float> ref_q(seq_len * hidden_size);
  vector<float> ref_k(seq_len * kv_dim);
  f.read(reinterpret_cast<char*>(ref_q.data()),
         seq_len * hidden_size * sizeof(float));
  f.read(reinterpret_cast<char*>(ref_k.data()),
         seq_len * kv_dim * sizeof(float));
  f.close();

  cout << "Reference: seq_len=" << seq_len << " hidden_size=" << hidden_size
       << " kv_dim=" << kv_dim << " token_ids: ";
  for (int id : token_ids) cout << id << " ";
  cout << "\n";

  // Load model and run: embed -> RMSNorm -> Q/K -> RoPE
  LlamaModel model;
  model.load("assets/llama3/dumped");

  int* d_ids;
  float *d_emb, *d_normed, *d_q, *d_k;
  cudaMalloc(&d_ids, seq_len * sizeof(int));
  cudaMalloc(&d_emb, seq_len * hidden_size * sizeof(float));
  cudaMalloc(&d_normed, seq_len * hidden_size * sizeof(float));
  cudaMalloc(&d_q, seq_len * hidden_size * sizeof(float));
  cudaMalloc(&d_k, seq_len * kv_dim * sizeof(float));

  cudaMemcpy(d_ids, token_ids.data(), seq_len * sizeof(int),
             cudaMemcpyHostToDevice);

  model.embed(d_ids, d_emb, seq_len);
  rms_norm(d_emb, model.layers[0].d_input_layernorm_weight, d_normed, seq_len,
           hidden_size);
  sgemm(d_normed, model.layers[0].d_q_proj_weight, d_q, seq_len, hidden_size,
        hidden_size);
  sgemm(d_normed, model.layers[0].d_k_proj_weight, d_k, seq_len, kv_dim,
        hidden_size);

  // Apply RoPE in-place
  rope_apply(d_q, model.d_rope_cos, model.d_rope_sin, seq_len, model.num_heads,
             model.head_dim);
  rope_apply(d_k, model.d_rope_cos, model.d_rope_sin, seq_len,
             model.num_kv_heads, model.head_dim);
  cudaDeviceSynchronize();

  vector<float> got_q(seq_len * hidden_size);
  vector<float> got_k(seq_len * kv_dim);
  cudaMemcpy(got_q.data(), d_q, got_q.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(got_k.data(), d_k, got_k.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_ids);
  cudaFree(d_emb);
  cudaFree(d_normed);
  cudaFree(d_q);
  cudaFree(d_k);

  bool all_ok = true;
  all_ok &= check_max_abs("Q_rope", got_q, ref_q, EPSILON);
  all_ok &= check_max_abs("K_rope", got_k, ref_k, EPSILON);

  cout << (all_ok ? "All RoPE vs Python checks PASSED\n"
                  : "Some RoPE vs Python checks FAILED\n");
  return all_ok;
}

// -----------------------------------------------------------------------------
// CPU GQA reference (self-contained, no model weights needed)
// -----------------------------------------------------------------------------

namespace {

/// CPU causal-masked, numerically-stable softmax matching the CUDA kernel:
///   mask value = -1e6, denominator += 1e-6
void cpu_causal_softmax(vector<float>& scores, int seq_len) {
  for (int p = 0; p < seq_len; p++) {
    float* row = scores.data() + p * seq_len;
    // causal mask
    for (int q = p + 1; q < seq_len; q++) row[q] = -1e6f;
    // row max
    float mx = row[0];
    for (int q = 1; q < seq_len; q++) mx = fmaxf(mx, row[q]);
    // exp and sum
    float sum = 0.0f;
    for (int q = 0; q < seq_len; q++) {
      row[q] = expf(row[q] - mx);
      sum += row[q];
    }
    float inv = 1.0f / (sum + 1e-6f);
    for (int q = 0; q < seq_len; q++) row[q] *= inv;
  }
}

/// CPU GQA reference.
/// Layouts: Q [S, H, D], K [S, H_kv, D], V [S, H_kv, D], out [S, H, D]
vector<float> cpu_gqa(const vector<float>& Q, const vector<float>& K,
                      const vector<float>& V, int seq_len, int num_heads,
                      int num_kv_heads, int head_dim) {
  const int heads_per_kv = num_heads / num_kv_heads;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const int q_row = num_heads * head_dim;
  const int kv_row = num_kv_heads * head_dim;

  vector<float> out(seq_len * q_row, 0.0f);

  for (int h = 0; h < num_heads; h++) {
    int g = h / heads_per_kv;
    // scores [S, S]
    vector<float> scores(seq_len * seq_len, 0.0f);
    for (int p = 0; p < seq_len; p++) {
      for (int q = 0; q < seq_len; q++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
          dot += Q[p * q_row + h * head_dim + d] *
                 K[q * kv_row + g * head_dim + d];
        scores[p * seq_len + q] = dot * scale;
      }
    }
    cpu_causal_softmax(scores, seq_len);
    // output = alpha @ V_g
    for (int p = 0; p < seq_len; p++) {
      for (int d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (int k = 0; k < seq_len; k++)
          sum += scores[p * seq_len + k] *
                 V[k * kv_row + g * head_dim + d];
        out[p * q_row + h * head_dim + d] = sum;
      }
    }
  }
  return out;
}

}  // namespace

bool test_gqa_cpu() {
  cout << "\n=== GQA self-contained CPU reference tests ===\n";
  const float TOL = 1e-4f;
  bool all_ok = true;

  // Helper: deterministic pseudo-random fill
  auto fill_rand = [](vector<float>& v, unsigned seed) {
    srand(seed);
    for (auto& x : v) x = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
  };

  // ---- Test 1: basic GQA (4 query heads, 2 KV heads) ----
  {
    const int S = 4, H = 4, H_kv = 2, D = 8;
    vector<float> Q(S * H * D), K(S * H_kv * D), V(S * H_kv * D);
    fill_rand(Q, 42);
    fill_rand(K, 43);
    fill_rand(V, 44);

    vector<float> expected = cpu_gqa(Q, K, V, S, H, H_kv, D);

    float *d_q, *d_k, *d_v, *d_out;
    cudaMalloc(&d_q, Q.size() * sizeof(float));
    cudaMalloc(&d_k, K.size() * sizeof(float));
    cudaMalloc(&d_v, V.size() * sizeof(float));
    cudaMalloc(&d_out, expected.size() * sizeof(float));
    cudaMemcpy(d_q, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);

    gqa_attention(d_q, d_k, d_v, d_out, S, H, H_kv, D);
    cudaDeviceSynchronize();

    vector<float> got(expected.size());
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);

    all_ok &= check_max_abs("basic S=4 H=4 H_kv=2 D=8", got, expected, TOL);
  }

  // ---- Test 2: MHA special case (num_heads == num_kv_heads) ----
  {
    const int S = 3, H = 2, H_kv = 2, D = 4;
    vector<float> Q(S * H * D), K(S * H_kv * D), V(S * H_kv * D);
    fill_rand(Q, 100);
    fill_rand(K, 101);
    fill_rand(V, 102);

    vector<float> expected = cpu_gqa(Q, K, V, S, H, H_kv, D);

    float *d_q, *d_k, *d_v, *d_out;
    cudaMalloc(&d_q, Q.size() * sizeof(float));
    cudaMalloc(&d_k, K.size() * sizeof(float));
    cudaMalloc(&d_v, V.size() * sizeof(float));
    cudaMalloc(&d_out, expected.size() * sizeof(float));
    cudaMemcpy(d_q, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);

    gqa_attention(d_q, d_k, d_v, d_out, S, H, H_kv, D);
    cudaDeviceSynchronize();

    vector<float> got(expected.size());
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);

    all_ok &= check_max_abs("MHA S=3 H=2 H_kv=2 D=4", got, expected, TOL);
  }

  // ---- Test 3: seq_len=1 (single token, no masking needed) ----
  {
    const int S = 1, H = 4, H_kv = 2, D = 8;
    vector<float> Q(S * H * D), K(S * H_kv * D), V(S * H_kv * D);
    fill_rand(Q, 200);
    fill_rand(K, 201);
    fill_rand(V, 202);

    vector<float> expected = cpu_gqa(Q, K, V, S, H, H_kv, D);

    float *d_q, *d_k, *d_v, *d_out;
    cudaMalloc(&d_q, Q.size() * sizeof(float));
    cudaMalloc(&d_k, K.size() * sizeof(float));
    cudaMalloc(&d_v, V.size() * sizeof(float));
    cudaMalloc(&d_out, expected.size() * sizeof(float));
    cudaMemcpy(d_q, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);

    gqa_attention(d_q, d_k, d_v, d_out, S, H, H_kv, D);
    cudaDeviceSynchronize();

    vector<float> got(expected.size());
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);

    all_ok &= check_max_abs("single token S=1", got, expected, TOL);
  }

  // ---- Test 4: larger, Llama-like ratio (H=32, H_kv=8, D=16) ----
  {
    const int S = 6, H = 32, H_kv = 8, D = 16;
    vector<float> Q(S * H * D), K(S * H_kv * D), V(S * H_kv * D);
    fill_rand(Q, 300);
    fill_rand(K, 301);
    fill_rand(V, 302);

    vector<float> expected = cpu_gqa(Q, K, V, S, H, H_kv, D);

    float *d_q, *d_k, *d_v, *d_out;
    cudaMalloc(&d_q, Q.size() * sizeof(float));
    cudaMalloc(&d_k, K.size() * sizeof(float));
    cudaMalloc(&d_v, V.size() * sizeof(float));
    cudaMalloc(&d_out, expected.size() * sizeof(float));
    cudaMemcpy(d_q, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);

    gqa_attention(d_q, d_k, d_v, d_out, S, H, H_kv, D);
    cudaDeviceSynchronize();

    vector<float> got(expected.size());
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);

    all_ok &= check_max_abs("Llama-like S=6 H=32 H_kv=8 D=16", got, expected,
                            TOL);
  }

  // ---- Test 5: verify causal masking (token 0 output must not depend on
  //      later tokens) ----
  {
    const int S = 4, H = 2, H_kv = 1, D = 4;
    vector<float> Q(S * H * D), K(S * H_kv * D), V(S * H_kv * D);
    fill_rand(Q, 500);
    fill_rand(K, 501);
    fill_rand(V, 502);

    // Run once with full sequence
    float *d_q, *d_k, *d_v, *d_out;
    cudaMalloc(&d_q, Q.size() * sizeof(float));
    cudaMalloc(&d_k, K.size() * sizeof(float));
    cudaMalloc(&d_v, V.size() * sizeof(float));
    cudaMalloc(&d_out, S * H * D * sizeof(float));
    cudaMemcpy(d_q, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);

    gqa_attention(d_q, d_k, d_v, d_out, S, H, H_kv, D);
    cudaDeviceSynchronize();

    vector<float> full_out(S * H * D);
    cudaMemcpy(full_out.data(), d_out, full_out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);

    // Run with only token 0
    const int S1 = 1;
    vector<float> Q1(Q.begin(), Q.begin() + S1 * H * D);
    vector<float> K1(K.begin(), K.begin() + S1 * H_kv * D);
    vector<float> V1(V.begin(), V.begin() + S1 * H_kv * D);

    cudaMalloc(&d_q, Q1.size() * sizeof(float));
    cudaMalloc(&d_k, K1.size() * sizeof(float));
    cudaMalloc(&d_v, V1.size() * sizeof(float));
    cudaMalloc(&d_out, S1 * H * D * sizeof(float));
    cudaMemcpy(d_q, Q1.data(), Q1.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K1.data(), K1.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V1.data(), V1.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    gqa_attention(d_q, d_k, d_v, d_out, S1, H, H_kv, D);
    cudaDeviceSynchronize();

    vector<float> single_out(S1 * H * D);
    cudaMemcpy(single_out.data(), d_out, single_out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);

    // Token 0's output row from the full run must match the single-token run
    vector<float> row0(full_out.begin(), full_out.begin() + H * D);
    all_ok &= check_max_abs("causal: token 0 unchanged", row0, single_out, TOL);
  }

  cout << (all_ok ? "All GQA CPU-reference tests PASSED\n"
                  : "Some GQA CPU-reference tests FAILED\n");
  return all_ok;
}

// -----------------------------------------------------------------------------
// GQA vs Python reference test
// (requires: python tools/generate_gqa_reference.py)
// -----------------------------------------------------------------------------

bool test_gqa_vs_python() {
  const char* path = "tests/data/gqa_reference.bin";
  const float EPSILON = 2e-2f;

  cout << "\n=== GQA vs Python reference ===\n";

  ifstream f(path, ios::binary);
  if (!f) {
    cout << "Reference file not found: " << path << "\n"
         << "Run: python tools/generate_gqa_reference.py\n";
    return false;
  }

  int seq_len, hidden_size, kv_dim, num_heads, num_kv_heads, head_dim;
  f.read(reinterpret_cast<char*>(&seq_len), sizeof(int));
  f.read(reinterpret_cast<char*>(&hidden_size), sizeof(int));
  f.read(reinterpret_cast<char*>(&kv_dim), sizeof(int));
  f.read(reinterpret_cast<char*>(&num_heads), sizeof(int));
  f.read(reinterpret_cast<char*>(&num_kv_heads), sizeof(int));
  f.read(reinterpret_cast<char*>(&head_dim), sizeof(int));

  vector<int> token_ids(seq_len);
  f.read(reinterpret_cast<char*>(token_ids.data()), seq_len * sizeof(int));

  vector<float> ref_out(seq_len * hidden_size);
  f.read(reinterpret_cast<char*>(ref_out.data()),
         ref_out.size() * sizeof(float));
  f.close();

  cout << "Reference: seq_len=" << seq_len << " hidden_size=" << hidden_size
       << " kv_dim=" << kv_dim << " num_heads=" << num_heads
       << " num_kv_heads=" << num_kv_heads << " head_dim=" << head_dim
       << " token_ids: ";
  for (int id : token_ids) cout << id << " ";
  cout << "\n";

  if (hidden_size != num_heads * head_dim) {
    cout << "Invalid reference: hidden_size != num_heads * head_dim\n";
    return false;
  }
  if (kv_dim != num_kv_heads * head_dim) {
    cout << "Invalid reference: kv_dim != num_kv_heads * head_dim\n";
    return false;
  }

  // Load model and run: embed -> RMSNorm -> QKV -> RoPE(Q,K) -> GQA
  LlamaModel model;
  model.load("assets/llama3/dumped");

  int* d_ids = nullptr;
  float *d_emb = nullptr, *d_normed = nullptr, *d_q = nullptr, *d_k = nullptr,
        *d_v = nullptr, *d_out = nullptr;

  cudaMalloc(&d_ids, seq_len * sizeof(int));
  cudaMalloc(&d_emb, seq_len * hidden_size * sizeof(float));
  cudaMalloc(&d_normed, seq_len * hidden_size * sizeof(float));
  cudaMalloc(&d_q, seq_len * hidden_size * sizeof(float));
  cudaMalloc(&d_k, seq_len * kv_dim * sizeof(float));
  cudaMalloc(&d_v, seq_len * kv_dim * sizeof(float));
  cudaMalloc(&d_out, seq_len * hidden_size * sizeof(float));

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

  rope_apply(d_q, model.d_rope_cos, model.d_rope_sin, seq_len, num_heads,
             head_dim);
  rope_apply(d_k, model.d_rope_cos, model.d_rope_sin, seq_len, num_kv_heads,
             head_dim);

  gqa_attention(d_q, d_k, d_v, d_out, seq_len, num_heads, num_kv_heads,
                head_dim, nullptr);
  cudaDeviceSynchronize();

  vector<float> got_out(seq_len * hidden_size);
  cudaMemcpy(got_out.data(), d_out, got_out.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_ids);
  cudaFree(d_emb);
  cudaFree(d_normed);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_out);

  bool ok = check_max_abs("GQA output", got_out, ref_out, EPSILON);
  cout << (ok ? "GQA vs Python PASSED\n" : "GQA vs Python FAILED\n");
  return ok;
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
  test_rope_vs_python();
  test_gqa_cpu();
  test_gqa_vs_python();
  return 0;
}
