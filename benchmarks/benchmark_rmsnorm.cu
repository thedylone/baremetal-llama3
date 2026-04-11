#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "config.h"
#include "rms_norm.cuh"

using namespace std;

static int parse_int(const char* s, int def) {
  if (!s) return def;
  char* end = nullptr;
  long v = strtol(s, &end, 10);
  if (end == s || v <= 0) return def;
  return static_cast<int>(v);
}

static void benchmark_rms_norm(int s, int d, int iters) {
  cout << "\n=== RMSNorm benchmark ===\n";
  cout << "s=" << s << " d=" << d << " iters=" << iters << "\n";

  vector<float> h_in(static_cast<size_t>(s) * d);
  vector<float> h_gamma(d);

  for (size_t i = 0; i < h_in.size(); ++i)
    h_in[i] = (static_cast<float>(i % 1024) * 0.001f) - 0.5f;
  for (int i = 0; i < d; ++i)
    h_gamma[i] = 1.0f + (i % 7) * 0.01f;

  float *d_in = nullptr, *d_gamma = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, h_in.size() * sizeof(float));
  cudaMalloc(&d_gamma, d * sizeof(float));
  cudaMalloc(&d_out, h_in.size() * sizeof(float));

  cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, h_gamma.data(), d * sizeof(float),
             cudaMemcpyHostToDevice);

  // Warm-up
  for (int i = 0; i < 10; ++i) rms_norm(d_in, d_gamma, d_out, s, d);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) rms_norm(d_in, d_gamma, d_out, s, d);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  double avg_ms = ms / iters;

  // Model bytes (current kernel): input read twice + gamma read + output write
  const double bytes_per_row = 16.0 * d;
  const double total_bytes = bytes_per_row * s;
  const double gbps = total_bytes / (avg_ms * 1e-3) / 1e9;

  // Approx FLOPs: 2d (sumsq) + 2d (scale) = 4d per row
  const double flops_per_row = 4.0 * d;
  const double gflops = (flops_per_row * s) / (avg_ms * 1e-3) / 1e9;

  cout << fixed << setprecision(3);
  cout << "avg_ms=" << avg_ms << "  approx_GB/s=" << gbps
       << "  approx_GFLOPs=" << gflops << "\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_gamma);
  cudaFree(d_out);
}

int main(int argc, char** argv) {
  int s = (argc > 1) ? parse_int(argv[1], 4096) : 4096;
  int d = (argc > 2) ? parse_int(argv[2], 4096) : 4096;
  int iters = (argc > 3) ? parse_int(argv[3], 100) : 100;

  benchmark_rms_norm(s, d, iters);
  return 0;
}