// Your kernels signature functions go here...
// It is a suggested structure but you can proceed with your own structure.

// All compute kernels must be implemented by you in CUDA. While matching
// state-of-the-art performance is not required, we expect few optimization
// efforts: shared-memory tiling, double buffering, and memory coalescing. Extra
// optimizations like tensore core usage and vectorization will give you bonus
// points. You are encouraged to study existing high-performance
// implementations, such as GEMM kernels, to enhance your kernels.

#pragma once
#include <cuda_runtime.h>

#include "gemm.cuh"
#include "rms_norm.cuh"
#include "rope.cuh"
#include "attention.cuh"
#include "residual_add.cuh"

#define CUDA_CHECK(expr)                                              \
  do {                                                                \
    cudaError_t _e = (expr);                                          \
    if (_e != cudaSuccess) {                                          \
      fprintf(stderr, "[CUDA error] %s:%d  %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(_e));                                \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

/// @brief Utility function to compute the ceiling of integer division
/// @tparam T integer type
/// @param a numerator
/// @param b denominator
/// @return ceiling of a / b
template <typename T>
__host__ __device__ __forceinline__ T cdiv(T a, T b) {
  return (a + b - 1) / b;
}
