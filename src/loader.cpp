#include "loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <unordered_map>

/// @brief Memory-mapped file representation.
/// This struct manages the lifecycle of a memory-mapped file, ensuring that
/// resources are properly released when the struct goes out of scope.
struct MappedFile {
  // Pointer to the mapped data in memory.
  void* data;
  // Size of the mapped file in bytes.
  size_t size;
  // File descriptor for the mapped file.
  int fd;

  MappedFile() : data(nullptr), size(0), fd(-1) {}
  ~MappedFile() {
    if (data) munmap(data, size);
    if (fd != -1) close(fd);
  }
};

/// @brief Information about a tensor stored in the dump file.
struct TensorInfo {
  // Name of the tensor
  std::string name;
  // Shape of the tensor (e.g., [vocab_size, embedding_dim] for embeddings)
  std::vector<size_t> shape;
  // Data type of the tensor
  DumpFloatType dtype;
  // Offset in the dump file where the tensor data starts
  size_t offset;
  // Size of the tensor data in bytes
  size_t size;
  // Memory-mapped file for the tensor data
  std::shared_ptr<MappedFile> mapped_file;
};

/// @brief Implementation details for the LlamaDumpLoader class.
class LlamaDumpLoader::Impl {
 public:
  DumpFloatType float_type;
  // Map tensor name to its info
  std::unordered_map<std::string, TensorInfo> tensor_map;
  // Shared pointer to the memory-mapped file for the embeddings
  std::shared_ptr<MappedFile> embedding_file;

  size_t vocab_size;
  int embedding_dim;

  explicit Impl(DumpFloatType float_type) : float_type(float_type) {};
  ~Impl() {};

  /// @brief Load a tensor from the specified file path and populate the
  /// provided TensorInfo struct with its metadata and memory-mapped data.
  /// @param file_path The path to the tensor file.
  /// @param tensor_info The TensorInfo struct to populate.
  /// @return True if the tensor was loaded successfully, false otherwise.
  bool load_tensor(const std::string& file_path, TensorInfo& tensor_info) {
    auto mapped_file = std::make_shared<MappedFile>();

    // Open file
    mapped_file->fd = open(file_path.c_str(), O_RDONLY);
    if (mapped_file->fd == -1) return false;

    // Get file size
    struct stat st;
    if (fstat(mapped_file->fd, &st) == -1) return false;
    mapped_file->size = st.st_size;

    // Map file into memory
    mapped_file->data = mmap(nullptr, mapped_file->size, PROT_READ, MAP_PRIVATE,
                             mapped_file->fd, 0);
    if (mapped_file->data == MAP_FAILED) return false;

    // Parse header
    const char* ptr = static_cast<const char*>(mapped_file->data);
    size_t offset = 0;

    // Magic number (4 bytes)
    uint32_t magic;
    std::memcpy(&magic, ptr + offset, sizeof(magic));
    offset += sizeof(magic);
    if (magic != 0x32363530) return false;  // "2650" in hex
    // Tensor name length (4 bytes) and name
    uint32_t name_len;
    std::memcpy(&name_len, ptr + offset, sizeof(name_len));
    offset += sizeof(name_len);
    tensor_info.name = std::string(ptr + offset, name_len);
    offset += name_len;
    // Number of dimensions (4 bytes)
    uint32_t num_dims;
    std::memcpy(&num_dims, ptr + offset, sizeof(num_dims));
    offset += sizeof(num_dims);
    // Shape (8 bytes per dimension)
    tensor_info.shape.resize(num_dims);
    tensor_info.size = 1;
    for (uint32_t i = 0; i < num_dims; ++i) {
      std::memcpy(&tensor_info.shape[i], ptr + offset, sizeof(size_t));
      offset += sizeof(size_t);
      tensor_info.size *= tensor_info.shape[i];
    }
    // Data type
    uint32_t dtype;
    std::memcpy(&dtype, ptr + offset, sizeof(dtype));
    offset += sizeof(dtype);
    tensor_info.dtype = static_cast<DumpFloatType>(dtype);
    switch (tensor_info.dtype) {
      case DumpFloatType::FP32:
        tensor_info.size *= sizeof(float);
        break;
      case DumpFloatType::FP16:
        tensor_info.size *= sizeof(uint16_t);
        break;
      case DumpFloatType::BF16:
        tensor_info.size *= sizeof(uint16_t);
        break;
      default:
        return false;
    }

    tensor_info.offset = offset;
    tensor_info.mapped_file = std::move(mapped_file);
    return true;
  }
};

LlamaDumpLoader::LlamaDumpLoader(DumpFloatType float_type)
    : float_type(float_type), pImpl_(std::make_unique<Impl>(float_type)) {}

LlamaDumpLoader::~LlamaDumpLoader() = default;

size_t LlamaDumpLoader::vocab_size(const std::string& dump_path,
                                   int embedding_dim) {
  if (pImpl_->vocab_size > 0) return pImpl_->vocab_size;
  // Embeddings not loaded yet, just retrieve vocab size and embedding dim
  TensorInfo tensor_info;
  std::string embedding_file = dump_path + "/model_embed_tokens_weight.bin";
  if (!pImpl_->load_tensor(embedding_file, tensor_info))
    throw std::runtime_error("Failed to load embedding tensor");
  if (tensor_info.shape.size() != 2)
    throw std::runtime_error("Expected 2D embedding tensor");
  pImpl_->vocab_size = tensor_info.shape[0];
  pImpl_->embedding_dim = tensor_info.shape[1];
  if (pImpl_->embedding_dim != embedding_dim)
    throw std::runtime_error("Embedding dimension mismatch");
  return pImpl_->vocab_size;
}

bool LlamaDumpLoader::load_embeddings(const std::string& dump_path,
                                      int embedding_dim) {
  std::string embedding_file = dump_path + "/model_embed_tokens_weight.bin";
  TensorInfo tensor_info;
  if (!pImpl_->load_tensor(embedding_file, tensor_info)) return false;
  if (tensor_info.shape.size() != 2) return false;
  if (tensor_info.shape[1] != static_cast<size_t>(embedding_dim)) return false;
  // Store the memory-mapped file and tensor info for later use
  pImpl_->embedding_file = tensor_info.mapped_file;
  pImpl_->vocab_size = tensor_info.shape[0];
  pImpl_->embedding_dim = tensor_info.shape[1];
  pImpl_->tensor_map["model.embed_tokens.weight"] = tensor_info;
  return true;
}

// Helper function for bfloat16 conversion
inline float bfloat16_to_float32(uint16_t bf16_bits) {
  uint32_t float32_bits = static_cast<uint32_t>(bf16_bits) << 16;
  // return *reinterpret_cast<float*>(&float32_bits);
  // gives compiler warning about strict aliasing
  float result;
  std::memcpy(&result, &float32_bits, sizeof(float));
  return result;
}

void LlamaDumpLoader::convert_and_write(float_t* output_ptr,
                                        const char* data_ptr, const int size,
                                        const DumpFloatType float_type) {
  // Handle type conversion
  if (float_type == DumpFloatType::FP32) {
    const float* float_data = reinterpret_cast<const float*>(data_ptr);
    for (int i = 0; i < size; ++i) {
      output_ptr[i] = static_cast<float_t>(float_data[i]);
    }
  } else if (float_type == DumpFloatType::BF16) {
    const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(data_ptr);
    for (int i = 0; i < size; ++i) {
      output_ptr[i] = static_cast<float_t>(bfloat16_to_float32(bf16_data[i]));
    }
  } else if (float_type == DumpFloatType::FP16) {
    const uint16_t* f16_data = reinterpret_cast<const uint16_t*>(data_ptr);
    for (int i = 0; i < size; ++i) {
      // Simplified FP16 conversion
      output_ptr[i] = static_cast<float_t>(f16_data[i]);  // Placeholder
    }
  }
}

float_t* LlamaDumpLoader::get_embeddings(const std::vector<int>& token_ids) {
  if (!pImpl_->embedding_file || token_ids.empty()) return nullptr;

  // Retrieve tensor info and data pointer for the embeddings
  const TensorInfo& tensor_info =
      pImpl_->tensor_map["model.embed_tokens.weight"];
  const char* base_ptr =
      static_cast<const char*>(tensor_info.mapped_file->data) +
      tensor_info.offset;

  // Calculate size of each element based on the data type
  size_t element_size;
  switch (tensor_info.dtype) {
    case DumpFloatType::FP32:
      element_size = sizeof(float);
      break;
    case DumpFloatType::FP16:
    case DumpFloatType::BF16:
      element_size = sizeof(uint16_t);
      break;
    default:
      return nullptr;
  }

  // Allocate output buffer for the requested token embeddings
  size_t output_size = token_ids.size() * pImpl_->embedding_dim;
  float_t* output = new float_t[output_size];

  // Iterate over token IDs and copy their embeddings to the output buffer
  for (size_t i = 0; i < token_ids.size(); ++i) {
    int token_id = token_ids[i];
    // Pointer to the output buffer for this token
    float_t* output_ptr = output + i * pImpl_->embedding_dim;

    // Handle out-of-bounds token ID
    if (token_id < 0 || static_cast<size_t>(token_id) >= pImpl_->vocab_size) {
      // Fill with zeros for out-of-bounds token IDs
      std::memset(output_ptr, 0, pImpl_->embedding_dim * sizeof(float_t));
      continue;
    }

    // Pointer to the start of the token's embedding vector
    const char* token_ptr =
        base_ptr + token_id * pImpl_->embedding_dim * element_size;

    // Handle type conversion
    convert_and_write(output_ptr, token_ptr, pImpl_->embedding_dim,
                      tensor_info.dtype);
  }

  return output;
}

float_t* LlamaDumpLoader::load_1d(const std::string& dump_file,
                                  const size_t& dim0) {
  TensorInfo tensor_info;
  if (!pImpl_->load_tensor(dump_file, tensor_info)) return nullptr;
  // Validate dimensions
  if (tensor_info.shape.size() != 1 || tensor_info.shape[0] != dim0)
    return nullptr;

  const char* data_ptr =
      static_cast<const char*>(tensor_info.mapped_file->data) +
      tensor_info.offset;
  float_t* output = new float_t[dim0];
  convert_and_write(output, data_ptr, dim0, tensor_info.dtype);
  return output;
}

float_t* LlamaDumpLoader::load_2d(const std::string& dump_file,
                                  const size_t& dim0, const size_t& dim1) {
  TensorInfo tensor_info;
  if (!pImpl_->load_tensor(dump_file, tensor_info)) return nullptr;
  // Validate dimensions
  if (tensor_info.shape.size() != 2 || tensor_info.shape[0] != dim0 ||
      tensor_info.shape[1] != dim1)
    return nullptr;

  const char* data_ptr =
      static_cast<const char*>(tensor_info.mapped_file->data) +
      tensor_info.offset;
  size_t total_size = dim0 * dim1;
  float_t* output = new float_t[total_size];
  convert_and_write(output, data_ptr, total_size, tensor_info.dtype);
  return output;
}
