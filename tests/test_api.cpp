

#include "test_api.h"
#include "config.h"
#include "tokenizer.h"
#include "loader.h"
#include "kernels.cuh"

vector<int> TestAPI::tokenize(string input) {
    // Your code for testing the tokenizer goes here.
    // You can use the tokenizer you implemented in src/tokenizer_bpe.cpp and
    // include the header file here. Read the project description for more
    // information.
    BPETokenizer tok(TOKENIZER_PATH);
    return tok.encode(input);
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
