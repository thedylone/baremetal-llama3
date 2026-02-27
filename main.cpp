
#include "config.h"
#include "tokenizer.h"
#include "loader.h"
#include <iomanip>
#include <iostream>

using namespace std;

const int MAX_BATCH_SIZE = 16;
const int MAX_SEQUENCE_SIZE = 64;

void show_tokenize(string input);
void show_embeddings(string input);

int main() {


    show_tokenize("Hello world");

    show_embeddings("Hello world");

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

void show_embeddings(string input = "Hello world") {
    BPETokenizer tok(TOKENIZER_PATH);
    auto ids = tok.encode(input);

    LlamaDumpLoader loader(DumpFloatType::FP32);
    if (!loader.load_embeddings("assets/llama3/dumped", 4096)) {
        throw runtime_error("Failed to load embeddings");
    }
    float_t* embeddings = loader.get_embeddings(ids);

    cout << "Embeddings for input: " << input << "\n";
    for (size_t i = 0; i < ids.size(); ++i) {
        cout << "Token ID: " << ids[i] << " Embedding: [";
        for (int j = 0; j < 10; ++j) { // Print first 10 dimensions for brevity
            cout << fixed << setprecision(4) << embeddings[i * 4096 + j] << " ";
        }
        cout << "...]\n";
    }

    delete[] embeddings; // Free the allocated buffer
}