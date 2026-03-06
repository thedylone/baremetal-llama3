
#include <iomanip>
#include <iostream>

#include "config.h"
#include "kernels.cuh"
#include "tokenizer.h"

using namespace std;

const int MAX_BATCH_SIZE = 16;
const int MAX_SEQUENCE_SIZE = 64;

void show_tokenize(string input);
void test_matrix_multiply();

int main() {
  show_tokenize("hello world");

  test_matrix_multiply();

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