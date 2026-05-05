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

  // 3. Generate a few next tokens autoregressively.
  const int n_new = 8;
  cout << "Generating " << n_new << " tokens...\n";
  auto forward_t0 = chrono::high_resolution_clock::now();
  auto new_ids = model.generate(ids, n_new);
  auto forward_t1 = chrono::high_resolution_clock::now();
  double forward_ms =
      chrono::duration<double, milli>(forward_t1 - forward_t0).count();
  cout << "  Generation time: " << forward_ms << " ms\n";

  // 4. Decode and print tokens as they are generated.
  cout << "Decoding generated tokens...\n";
  auto decode_t0 = chrono::high_resolution_clock::now();
  string output = input;
  cout << "  Next token ids: ";
  for (int next_id : new_ids) {
    string next_token = tok.decode({next_id});
    cout << next_id << " ";
    output += next_token;
  }
  cout << "\n";
  cout << "  Full output: \"" << output << "\"\n";
  auto decode_t1 = chrono::high_resolution_clock::now();
  double decode_ms =
      chrono::duration<double, milli>(decode_t1 - decode_t0).count();
  cout << "  Decoding time: " << decode_ms << " ms\n";

  auto full_t1 = chrono::high_resolution_clock::now();
  double full_ms = chrono::duration<double, milli>(full_t1 - full_t0).count();
  cout << "Total execution time: " << full_ms << " ms\n";

  return 0;
}
