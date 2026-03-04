#include "loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <memory>
#include <unordered_map>
#include <utility>

namespace {

constexpr uint32_t kMagic = 0x32363530;  // "2650" in hex
class LoaderException : public std::runtime_error {
 public:
  explicit LoaderException(const std::string& message)
      : std::runtime_error(message) {}
};

/// @brief Memory-mapped file representation.
/// This class manages the lifecycle of a memory-mapped file, ensuring that
/// resources are properly released when the class goes out of scope.
class MappedFile {
 private:
  // Pointer to the mapped data in memory.
  void* data_{nullptr};
  // Size of the mapped file in bytes.
  size_t size_{0};
  // File descriptor for the mapped file.
  int fd_{-1};

 public:
  explicit MappedFile(const std::string& file_path) {
    // Open the file for reading
    fd_ = ::open(file_path.c_str(), O_RDONLY);
    if (fd_ == -1) throw LoaderException("Failed to open file: " + file_path);
    struct stat st{};
    // Get the size of the file
    if (::fstat(fd_, &st) == -1) {
      ::close(fd_);
      throw LoaderException("Failed to get file size: " + file_path);
    }
    size_ = st.st_size;
    if (size_ == 0) {
      ::close(fd_);
      throw LoaderException("File is empty: " + file_path);
    }
    // Map the file into memory
    data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      ::close(fd_);
      throw LoaderException("Failed to map file: " + file_path);
    }
  }

  ~MappedFile() {
    if (data_) munmap(data_, size_);
    if (fd_ != -1) close(fd_);
  }

  const char* data() const noexcept { return static_cast<const char*>(data_); }
  size_t size() const noexcept { return size_; }

  // Make non-copyable
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        fd_(std::exchange(other.fd_, -1)) {}
};

/// @brief Safe reader over mapped memory
class BufferReader {
 private:
  const char* data_;
  size_t size_;
  size_t offset_{0};

 public:
  explicit BufferReader(const char* data, size_t size)
      : data_(data), size_(size) {}

  void validate(size_t bytes) const {
    if (bytes > size_ - offset_)
      throw LoaderException("Unexpected EOF while reading");
  }

  template <typename T>
  T read() {
    validate(sizeof(T));
    T value;
    std::memcpy(&value, data_ + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;
  }

  std::string read_string(size_t length) {
    validate(length);
    std::string str(data_ + offset_, length);
    offset_ += length;
    return str;
  }

  size_t offset() const noexcept { return offset_; }
};

/// @brief Information about a tensor stored in the dump file.
struct TensorInfo {
  // Name of the tensor
  std::string name;
  // Shape of the tensor (e.g., [vocab_size, embedding_dim] for embeddings)
  std::vector<size_t> shape;
  // Data type of the tensor
  DumpFloatType dtype{};
  // Offset in the dump file where the tensor data starts
  size_t offset{0};
  // Size of the tensor data in bytes
  size_t size{0};
  // Memory-mapped file for the tensor data
  std::unique_ptr<MappedFile> mapped_file;
};

// Helper function to get the size of each element based on the data type
inline size_t get_element_size(DumpFloatType dtype) {
  switch (dtype) {
    case DumpFloatType::FP32:
      return sizeof(float);
    case DumpFloatType::FP16:
    case DumpFloatType::BF16:
      return sizeof(uint16_t);
    default:
      throw LoaderException("Unsupported data type");
  }
}

// Helper function for bfloat16 conversion
inline float bfloat16_to_float32(uint16_t bf16_bits) {
  uint32_t float32_bits = static_cast<uint32_t>(bf16_bits) << 16;
  float result;
  std::memcpy(&result, &float32_bits, sizeof(float));
  return result;
}

// Helper function for float16 conversion
inline float float16_to_float32(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;  // bit 15 → bit 31
  const uint32_t exp = (h & 0x7C00u) >> 10;   // bits 10–14
  const uint32_t mant = (h & 0x03FFu);        // bits 0–9

  uint32_t f_bits = 0;

  if (exp == 0) {
    if (mant == 0) {
      // ±0
      f_bits = sign;
    } else {
      // Subnormal
      // Normalize the mantissa
      uint32_t m = mant;
      int shift = 0;

      while ((m & 0x0400u) == 0) {
        m <<= 1;
        ++shift;
      }

      m &= 0x03FFu;

      const uint32_t exp32 = 127 - 15 - shift;
      const uint32_t mant32 = m << 13;

      f_bits = sign | (exp32 << 23) | mant32;
    }
  } else if (exp == 0x1F) {
    // Inf or NaN
    f_bits = sign | 0x7F800000u | (mant << 13);
  } else {
    // Normalized number
    const uint32_t exp32 = exp - 15 + 127;
    const uint32_t mant32 = mant << 13;

    f_bits = sign | (exp32 << 23) | mant32;
  }

  float result;
  std::memcpy(&result, &f_bits, sizeof(result));
  return result;
}

}  // anonymous namespace

/// @brief Implementation details for the LlamaDumpLoader class.
class LlamaDumpLoader::Impl {
 public:
  // Map tensor name to its info
  std::unordered_map<std::string, TensorInfo> tensor_map;

  size_t vocab_size{0};
  size_t embedding_dim{0};

  Impl() = default;
  ~Impl() = default;

  /// @brief Load a tensor from the specified file path and populate the
  /// provided TensorInfo struct with its metadata and memory-mapped data.
  /// @param file_path The path to the tensor file.
  /// @return The loaded TensorInfo struct.
  TensorInfo load_tensor(const std::string& file_path) {
    auto mapped_file = std::make_unique<MappedFile>(file_path);
    BufferReader reader(mapped_file->data(), mapped_file->size());

    // Parse header
    // Magic number (4 bytes)
    if (reader.read<uint32_t>() != kMagic)
      throw LoaderException("Invalid magic number in file: " + file_path);

    TensorInfo tensor_info;
    tensor_info.mapped_file = std::move(mapped_file);

    // Tensor name length (4 bytes) and name
    const uint32_t name_len = reader.read<uint32_t>();
    tensor_info.name = reader.read_string(name_len);
    // Number of dimensions (4 bytes)
    const uint32_t num_dims = reader.read<uint32_t>();
    // Shape (8 bytes per dimension)
    tensor_info.shape.resize(num_dims);
    size_t num_elements = 1;
    for (uint32_t i = 0; i < num_dims; ++i) {
      size_t dim = reader.read<size_t>();
      if (dim == 0 || num_elements > SIZE_MAX / dim)
        throw LoaderException("Invalid tensor shape in file: " + file_path);
      tensor_info.shape[i] = dim;
      num_elements *= dim;
    }
    // Data type
    tensor_info.dtype = static_cast<DumpFloatType>(reader.read<uint32_t>());
    size_t element_size = get_element_size(tensor_info.dtype);

    if (num_elements > SIZE_MAX / element_size)
      throw LoaderException("Tensor size overflow in file: " + file_path);
    tensor_info.size = num_elements * element_size;
    tensor_info.offset = reader.offset();
    if (tensor_info.offset + tensor_info.size > tensor_info.mapped_file->size())
      throw LoaderException("Tensor data exceeds file size: " + file_path);

    return tensor_info;
  }
};

//
// LlamaDumpLoader public interface implementation
//

LlamaDumpLoader::LlamaDumpLoader(DumpFloatType float_type)
    : float_type(float_type), pImpl_(std::make_unique<Impl>()) {}

LlamaDumpLoader::~LlamaDumpLoader() = default;

size_t LlamaDumpLoader::vocab_size(const std::string& dump_path,
                                   int embedding_dim) {
  if (pImpl_->vocab_size == 0) load_embeddings(dump_path, embedding_dim);
  return pImpl_->vocab_size;
}

bool LlamaDumpLoader::load_embeddings(const std::string& dump_path,
                                      int embedding_dim) {
  const std::string embedding_file =
      dump_path + "/model_embed_tokens_weight.bin";
  TensorInfo tensor_info = pImpl_->load_tensor(embedding_file);
  if (tensor_info.shape.size() != 2)
    throw LoaderException("Expected 2D tensor for embeddings in file: " +
                          embedding_file);
  if (tensor_info.shape[1] != static_cast<size_t>(embedding_dim))
    throw LoaderException("Embedding dimension mismatch in file: " +
                          embedding_file);
  // Store the memory-mapped file and tensor info for later use
  pImpl_->vocab_size = tensor_info.shape[0];
  pImpl_->embedding_dim = tensor_info.shape[1];
  pImpl_->tensor_map["model.embed_tokens.weight"] = std::move(tensor_info);
  return true;
}

void LlamaDumpLoader::convert_and_write(float_t* output_ptr,
                                        const char* data_ptr, const size_t size,
                                        const DumpFloatType float_type) {
  // Handle type conversion
  switch (float_type) {
    case DumpFloatType::FP32:
      std::memcpy(output_ptr, data_ptr, size * sizeof(float_t));
      break;

    case DumpFloatType::BF16:
      // Convert BF16 to FP32
      {
        const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(data_ptr);
        for (size_t i = 0; i < size; ++i) {
          output_ptr[i] =
              static_cast<float_t>(bfloat16_to_float32(bf16_data[i]));
        }
      }
      break;

    case DumpFloatType::FP16:
      // Convert FP16 to FP32
      {
        const uint16_t* f16_data = reinterpret_cast<const uint16_t*>(data_ptr);
        for (size_t i = 0; i < size; ++i) {
          output_ptr[i] = static_cast<float_t>(float16_to_float32(f16_data[i]));
        }
      }
      break;

    default:
      throw LoaderException("Unsupported float type for conversion");
  }
}

float_t* LlamaDumpLoader::get_embeddings(const std::vector<int>& token_ids) {
  if (token_ids.empty()) return nullptr;

  auto it = pImpl_->tensor_map.find("model.embed_tokens.weight");
  if (it == pImpl_->tensor_map.end())
    throw LoaderException(
        "Embeddings not loaded. Call load_embeddings() first.");

  // Retrieve tensor info and data pointer for the embeddings
  const TensorInfo& tensor_info = it->second;
  const char* base_ptr = tensor_info.mapped_file->data() + tensor_info.offset;

  // Calculate size of each element based on the data type
  size_t element_size = get_element_size(tensor_info.dtype);

  // Allocate output buffer for the requested token embeddings
  size_t output_size = token_ids.size() * pImpl_->embedding_dim;
  // Use unique_ptr for automatic memory management
  auto output = std::make_unique<float_t[]>(output_size);

  // Iterate over token IDs and copy their embeddings to the output buffer
  for (size_t i = 0; i < token_ids.size(); ++i) {
    int token_id = token_ids[i];
    // Pointer to the output buffer for this token
    float_t* output_ptr = output.get() + i * pImpl_->embedding_dim;

    // Handle out-of-bounds token ID
    if (token_id < 0 || static_cast<size_t>(token_id) >= pImpl_->vocab_size) {
      // Fill with zeros for out-of-bounds token IDs
      std::memset(output_ptr, 0, pImpl_->embedding_dim * sizeof(float_t));
      continue;
    }

    // Pointer to the start of the token's embedding vector
    const char* token_ptr = base_ptr + static_cast<size_t>(token_id) *
                                           pImpl_->embedding_dim * element_size;

    // Handle type conversion
    convert_and_write(output_ptr, token_ptr, pImpl_->embedding_dim,
                      tensor_info.dtype);
  }

  return output.release();
}

float_t* LlamaDumpLoader::load_1d(const std::string& dump_file,
                                  const size_t& dim0) {
  TensorInfo tensor_info = pImpl_->load_tensor(dump_file);
  // Validate dimensions
  if (tensor_info.shape.size() != 1 || tensor_info.shape[0] != dim0)
    throw LoaderException("Tensor shape mismatch for file: " + dump_file);

  const char* data_ptr = tensor_info.mapped_file->data() + tensor_info.offset;
  auto output = std::make_unique<float_t[]>(dim0);
  convert_and_write(output.get(), data_ptr, dim0, tensor_info.dtype);
  return output.release();
}

float_t* LlamaDumpLoader::load_2d(const std::string& dump_file,
                                  const size_t& dim0, const size_t& dim1) {
  TensorInfo tensor_info = pImpl_->load_tensor(dump_file);
  // Validate dimensions
  if (tensor_info.shape.size() != 2 || tensor_info.shape[0] != dim0 ||
      tensor_info.shape[1] != dim1)
    throw LoaderException("Tensor shape mismatch for file: " + dump_file);

  const char* data_ptr = tensor_info.mapped_file->data() + tensor_info.offset;
  size_t total_size = dim0 * dim1;
  auto output = std::make_unique<float_t[]>(total_size);
  convert_and_write(output.get(), data_ptr, total_size, tensor_info.dtype);
  return output.release();
}
