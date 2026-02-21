

#include "test_api.h"
#include "config.h"
#include "tokenizer.h"

vector<int> TestAPI::tokenize(string input) {
    // Your code for testing the tokenizer goes here.
    // You can use the tokenizer you implemented in src/tokenizer_bpe.cpp and
    // include the header file here. Read the project description for more
    // information.
    BPETokenizer tok(TOKENIZER_PATH);
    return tok.encode(input);
}

vector<float> TestAPI::get_embeddings(vector<int> token_ids) {
    throw runtime_error("Not implemented: you need to implement the "
                        "get_embeddings function here");
}

vector<float> TestAPI::matmul(const vector<float> &A, const vector<float> &B,
                              int M, int K, int N) {
    throw runtime_error("Not implemented: you need to implement the "
                        "matmul function here");
}
