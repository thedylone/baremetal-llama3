#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "config.h"
#include "kernels.cuh"
#include "model.h"
#include "tokenizer.h"

using namespace std;

int main() {
  auto full_t0 = chrono::high_resolution_clock::now();

  const string dump_path = "assets/llama3/dumped";

  // 1. Load model
  cout << "Loading model weights...\n";
  auto load_t0 = chrono::high_resolution_clock::now();
  LlamaModel model;
  model.load(dump_path);
  cout << "  Model loaded (" << NUM_LAYERS << " layers, "
       << "embedding_dim=" << EMBEDDING_DIM << ").\n";
  auto load_t1 = chrono::high_resolution_clock::now();
  double load_ms = chrono::duration<double, milli>(load_t1 - load_t0).count();
  cout << "  Model loading time: " << load_ms << " ms\n";

  // 2. Tokenize input
  cout << "Tokenizing input...\n";
  auto tokenize_t0 = chrono::high_resolution_clock::now();
  BPETokenizer tok(TOKENIZER_PATH);
  string input = "Yellow Fruit Like a";
  auto ids = tok.encode(input);
  int seq_len = static_cast<int>(ids.size());
  cout << "  Input: \"" << input << "\" -> " << seq_len << " tokens: ";
  for (int id : ids) cout << id << " ";
  cout << "\n";
  auto tokenize_t1 = chrono::high_resolution_clock::now();
  double tokenize_ms =
      chrono::duration<double, milli>(tokenize_t1 - tokenize_t0).count();
  cout << "  Tokenization time: " << tokenize_ms << " ms\n";

  // 3. Upload token ids to GPU
  int* d_ids;
  CUDA_CHECK(cudaMalloc(&d_ids, seq_len * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), seq_len * sizeof(int),
                        cudaMemcpyHostToDevice));

  // 4. Forward pass → logits for next token
  float* d_logits;
  CUDA_CHECK(cudaMalloc(&d_logits, VOCAB_SIZE * sizeof(float)));

  cout << "Running forward pass...\n";
  auto forward_t0 = chrono::high_resolution_clock::now();
  model.forward(d_ids, d_logits, seq_len);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto forward_t1 = chrono::high_resolution_clock::now();
  double forward_ms =
      chrono::duration<double, milli>(forward_t1 - forward_t0).count();
  cout << "  Forward pass: " << forward_ms << " ms\n";

  // 5. Argmax → next token id
  vector<float> logits(VOCAB_SIZE);
  CUDA_CHECK(cudaMemcpy(logits.data(), d_logits,
                        VOCAB_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int next_id = static_cast<int>(max_element(logits.begin(), logits.end()) -
                                 logits.begin());

  // 6. Decode and print
  cout << "Decoding next token...\n";
  auto decode_t0 = chrono::high_resolution_clock::now();
  string next_token = tok.decode({next_id});
  cout << "  Next token id: " << next_id << " -> \"" << next_token << "\"\n";
  cout << "  Full output: \"" << input << next_token << "\"\n";
  auto decode_t1 = chrono::high_resolution_clock::now();
  double decode_ms =
      chrono::duration<double, milli>(decode_t1 - decode_t0).count();
  cout << "  Decoding time: " << decode_ms << " ms\n";

  // Cleanup
  cudaFree(d_ids);
  cudaFree(d_logits);

  auto full_t1 = chrono::high_resolution_clock::now();
  double full_ms = chrono::duration<double, milli>(full_t1 - full_t0).count();
  cout << "Total execution time: " << full_ms << " ms\n";

  return 0;
}
