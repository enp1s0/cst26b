#include <cstdint>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                         \
    do {                                                         \
      cudaError_t err = (call);                                  \
      if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)   \
        << " (" << __FILE__ << ":" << __LINE__ << ")\n";         \
        std::exit(EXIT_FAILURE);                                 \
      }                                                          \
    } while (0)

__global__
void add_kernel(const float* a,
                const float* b,
                float* c,
                unsigned n) {
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}

int main() {
  constexpr std::uint32_t N = 1 << 26;

  std::vector<float> hx(N);
  std::vector<float> hy(N);
  std::vector<float> hz(N);

#pragma omp parallel for
  for (std::uint32_t i = 0; i < N; ++i) {
    hx[i] = 1.0f;
    hy[i] = 2.0f;
  }

  float *dx = nullptr;
  float *dy = nullptr;
  float *dz = nullptr;

  CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dz, N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dx, hx.data(),
                        N * sizeof(float),
                        cudaMemcpyDefault));

  CUDA_CHECK(cudaMemcpy(dy, hy.data(),
                        N * sizeof(float),
                        cudaMemcpyDefault));

  constexpr std::uint32_t block_size = 256;
  constexpr std::uint32_t grid_size  = (N + block_size - 1) / block_size;

  add_kernel<<<grid_size, block_size>>>(dx, dy, dz, N);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(hz.data(), dz,
                        N * sizeof(float),
                        cudaMemcpyDefault));

  std::printf("x  : ");
  for (std::uint32_t i = 0; i < 10; i++) std::printf("%+e ", hx[i]);
  std::printf("...\n");
  std::printf("y  : ");
  for (std::uint32_t i = 0; i < 10; i++) std::printf("%+e ", hy[i]);
  std::printf("...\n");
  std::printf("x+y: ");
  for (std::uint32_t i = 0; i < 10; i++) std::printf("%+e ", hz[i]);
  std::printf("...\n");

  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(dz));
}
