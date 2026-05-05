// You can set your configuration here...
// Read the manual for our assumption for directory of model weights and
// tokenizer files.
#pragma once

#include <string>

const std::string TOKENIZER_PATH = "assets/llama3/token.model";

const int EMBEDDING_DIM = 4096;
const float RMS_NORM_EPSILON = 1e-5f;
const float ROPE_THETA_BASE = 500000.0f;
const int NUM_LAYERS = 32;
const int NUM_HEADS = 32;
const int NUM_KV_HEADS = 8;
const int HEAD_DIM = 128;  // HIDDEN_SIZE / NUM_HEADS
const int KV_DIM = 1024;   // NUM_KV_HEADS * HEAD_DIM
const int INTERMEDIATE_SIZE = 14336;
const int VOCAB_SIZE = 128256;
const int MAX_SEQ_LEN = 1000;