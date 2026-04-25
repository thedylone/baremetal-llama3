#include <algorithm>
#include <iostream>
#include <vector>

#include "config.h"
#include "kernels.cuh"
#include "model.h"
#include "tokenizer.h"

using namespace std;

int main() {
  const string dump_path = "assets/llama3/dumped";

  // 1. Load model
  cout << "Loading model weights...\n";
  LlamaModel model;
  model.load(dump_path);
  cout << "Model loaded (" << model.num_layers << " layers, "
       << "hidden_size=" << model.hidden_size << ").\n";

  // 2. Tokenize input
  BPETokenizer tok(TOKENIZER_PATH);
  string input = "Yellow Fruit Like a";
  auto ids = tok.encode(input);
  int seq_len = static_cast<int>(ids.size());
  cout << "Input: \"" << input << "\" -> " << seq_len << " tokens: ";
  for (int id : ids) cout << id << " ";
  cout << "\n";

  // 3. Upload token ids to GPU
  int* d_ids;
  CUDA_CHECK(cudaMalloc(&d_ids, seq_len * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), seq_len * sizeof(int),
                        cudaMemcpyHostToDevice));

  // 4. Forward pass → logits for next token
  float* d_logits;
  CUDA_CHECK(cudaMalloc(&d_logits, model.vocab_size * sizeof(float)));

  cout << "Running forward pass...\n";
  model.forward(d_ids, d_logits, seq_len);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 5. Argmax → next token id
  vector<float> logits(model.vocab_size);
  CUDA_CHECK(cudaMemcpy(logits.data(), d_logits,
                        model.vocab_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int next_id = static_cast<int>(
      max_element(logits.begin(), logits.end()) - logits.begin());

  // 6. Decode and print
  string next_token = tok.decode({next_id});
  cout << "Next token id: " << next_id << " -> \"" << next_token << "\"\n";
  cout << "Full output: \"" << input << next_token << "\"\n";

  // Cleanup
  cudaFree(d_ids);
  cudaFree(d_logits);

  return 0;
}
