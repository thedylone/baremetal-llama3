

#include "test_api.h"
#include "config.h"
#include "tokenizer.h"
#include "loader.h"

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

vector<float> TestAPI::matmul(const vector<float> &A, const vector<float> &B,
                              int M, int K, int N) {
    throw runtime_error("Not implemented: you need to implement the "
                        "matmul function here");
}
