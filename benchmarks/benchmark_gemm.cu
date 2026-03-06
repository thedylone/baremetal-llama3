// benchmark_gemm.cu
//
// Correctness tests followed by performance benchmarks for all gemm variants.
// A variant that fails any correctness test is skipped in the benchmark.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "kernel/gemm.cuh"
#include "kernel/kernels.cuh"


// ─── Variants
// ─────────────────────────────────────────────────────────────────

struct Variant {
  const char* name;
  SgemmLaunchFunc fn;
};

static const std::vector<Variant> VARIANTS = {
    {"naive", sgemm_basic_launch},
    {"tiled", sgemm_tiled_launch},
};

// ═════════════════════════════════════════════════════════════════════════════
// Part 1 — Correctness
// ═════════════════════════════════════════════════════════════════════════════

static void sgemm_cpu(const std::vector<float>& A, const std::vector<float>& B,
                      std::vector<float>& C, int M, int N, int K, float alpha,
                      float beta) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) sum += A[m * K + k] * B[k * N + n];
      C[m * N + n] = alpha * sum + beta * C[m * N + n];
    }
}

struct TestCase {
  const char* name;
  int M, K, N;
  float alpha, beta;
};

static const std::vector<TestCase> TESTS = {
    {"square small", 64, 64, 64, 1.0f, 0.0f},
    {"square medium", 512, 512, 512, 1.0f, 0.0f},
    {"non-square", 128, 256, 512, 1.0f, 0.0f},
    {"alpha scale", 128, 128, 128, 2.5f, 0.0f},
    {"beta blend", 128, 128, 128, 1.0f, 0.5f},
    {"odd dims", 127, 63, 65, 1.0f, 0.0f},
    {"skinny", 32, 1024, 32, 1.0f, 0.0f},
    {"fat", 1024, 32, 1024, 1.0f, 0.0f},
};

// Returns true if the variant passes all correctness tests.
static bool run_correctness(const Variant& v) {
  static constexpr float TOL = 1e-3f;

  printf("  %-24s", v.name);

  for (const auto& tc : TESTS) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_A(tc.M * tc.K);
    std::vector<float> h_B(tc.K * tc.N);
    for (float& x : h_A) x = dist(rng);
    for (float& x : h_B) x = dist(rng);

    // CPU reference — initialise C to 1 to exercise the beta path
    std::vector<float> ref_C(tc.M * tc.N, 1.0f);
    sgemm_cpu(h_A, h_B, ref_C, tc.M, tc.N, tc.K, tc.alpha, tc.beta);

    // GPU run
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, tc.M * tc.K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, tc.K * tc.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, tc.M * tc.N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), tc.M * tc.K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), tc.K * tc.N * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::vector<float> init_C(tc.M * tc.N, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_C, init_C.data(), tc.M * tc.N * sizeof(float),
                          cudaMemcpyHostToDevice));

    v.fn(d_A, d_B, d_C, tc.M, tc.N, tc.K, tc.alpha, tc.beta, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_C(tc.M * tc.N);
    CUDA_CHECK(cudaMemcpy(gpu_C.data(), d_C, tc.M * tc.N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    for (int i = 0; i < (int)ref_C.size(); ++i) {
      if (std::fabs(ref_C[i] - gpu_C[i]) > TOL) {
        printf("FAIL (%s: first mismatch at [%d][%d] ref=%.4f got=%.4f)\n",
               tc.name, i / tc.N, i % tc.N, ref_C[i], gpu_C[i]);
        return false;
      }
    }
  }

  printf("PASS\n");
  return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Part 2 — Benchmark
// ═════════════════════════════════════════════════════════════════════════════

static constexpr int WARMUP_RUNS = 5;
static constexpr int TIMED_RUNS = 20;

static const std::vector<int> SIZES = {128, 256, 512, 1024, 2048, 4096};

static float time_kernel(SgemmLaunchFunc fn, const float* d_A, const float* d_B,
                         float* d_C, int M, int N, int K, cudaStream_t stream) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < WARMUP_RUNS; ++i)
    fn(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < TIMED_RUNS; ++i)
    fn(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, stream);
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return ms / TIMED_RUNS;
}

static double gflops(int M, int N, int K, float ms) {
  return (2.0 * M * N * K) / (ms * 1e-3) / 1e9;
}

static void run_benchmark(const std::vector<Variant>& variants,
                          cudaStream_t stream) {
  printf("%-24s  %6s  %10s  %10s\n", "variant", "size", "ms", "GFLOPS");
  printf("%s\n", std::string(56, '-').c_str());

  for (int size : SIZES) {
    const int M = size, N = size, K = size;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_A, 1, M * K * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_B, 1, K * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    for (const auto& v : variants) {
      const float ms = time_kernel(v.fn, d_A, d_B, d_C, M, N, K, stream);
      const double gf = gflops(M, N, K, ms);
      printf("%-24s  %6d  %10.3f  %10.1f\n", v.name, size, ms, gf);
    }

    printf("\n");
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("Device: %s\n\n", prop.name);

  // ── Correctness ───────────────────────────────────────────────────────
  printf("Correctness\n");
  printf("%s\n", std::string(56, '-').c_str());

  std::vector<Variant> passing;
  for (const auto& v : VARIANTS) {
    if (run_correctness(v)) passing.push_back(v);
  }

  printf("\n%zu / %zu variants passed\n\n", passing.size(), VARIANTS.size());

  if (passing.empty()) {
    printf("No variants to benchmark.\n");
    return 1;
  }

  // ── Benchmark (only passing variants) ────────────────────────────────
  printf("Benchmark\n");
  printf("%s\n", std::string(56, '-').c_str());

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  run_benchmark(passing, stream);
  CUDA_CHECK(cudaStreamDestroy(stream));

  return passing.size() < VARIANTS.size() ? 1 : 0;
}