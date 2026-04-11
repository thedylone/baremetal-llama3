
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "config.h"
#include "kernels.cuh"
#include "loader.h"
#include "rms_norm.cuh"
#include "tokenizer.h"

using namespace std;

const int MAX_BATCH_SIZE = 16;
const int MAX_SEQUENCE_SIZE = 64;

void show_tokenize(string input);
void show_embeddings(string input);
void test_matrix_multiply();
bool test_rms_norm();

int main() {
  show_tokenize("hello world");

  test_matrix_multiply();

  show_tokenize("Hello world");

  show_embeddings("Hello world");

  test_rms_norm();

  return 0;
}

// if you implement tokenizer_bpe you should see token ids here.
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

void test_matrix_multiply() {
  int M = 2, K = 3, N = 4;
  vector<float> A = {1, 2, 3, 4, 5, 6};
  vector<float> B = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  vector<float> C(M * N);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));
  // Copy data to device
  cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

  sgemm(d_A, d_B, d_C, M, N, K);  // alpha=1, beta=0, stream=nullptr
  // Copy result back to host
  cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
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
    for (int j = 0; j < 10; ++j) {  // Print first 10 dimensions for brevity
      cout << fixed << setprecision(4) << embeddings[i * 4096 + j] << " ";
    }
    cout << "...]\n";
  }

  // Save the embeddings to a file *cpp_embeddings.bin*
  std::ofstream out("cpp_embeddings.bin", ios::binary);
  if (!out) {
    cerr << "Failed to open output file for embeddings\n";
    return;
  }
  out.write(reinterpret_cast<char *>(embeddings),
            ids.size() * 4096 * sizeof(float_t));

  delete[] embeddings;  // Free the allocated buffer
}

bool test_rms_norm() {
  const float EPS = 1e-5f;
  const float TOL = 1e-4f;

  // Test case 1: small known input (2 rows x 4 cols)
  {
    const int s = 2, d = 4;
    vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f,
                           0.5f, -1.0f, 1.5f, -2.0f};
    vector<float> gamma = {1.0f, 1.0f, 1.0f, 1.0f};  // identity scale
    vector<float> expected(s * d);

    // CPU reference
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

    // GPU
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