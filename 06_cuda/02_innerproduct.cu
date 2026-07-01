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

__device__ float warpReduceSum(float value) {
  for (std::uint32_t offset = warpSize / 2; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);

  return value;
}

__global__
void dot_kernel(const float* x,
               const float* y,
               float* result,
               const std::uint32_t n) {
  float sum = 0.0f;

  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  const std::uint32_t stride = blockDim.x * gridDim.x;

  for (std::uint32_t i = tid; i < n; i += stride)
    sum += x[i] * y[i];

  sum = warpReduceSum(sum);

  constexpr std::uint32_t WarpSize = 32;
  __shared__ float warpSum[32];

  const std::uint32_t lane = threadIdx.x & (WarpSize - 1);
  const std::uint32_t warp = threadIdx.x / WarpSize;

  if (lane == 0)
    warpSum[warp] = sum;

  __syncthreads();

  if (warp == 0) {
    const std::uint32_t numWarps = (blockDim.x + WarpSize - 1) / WarpSize;

    sum = (lane < numWarps) ? warpSum[lane] : 0.0f;
    sum = warpReduceSum(sum);

    if (lane == 0)
      atomicAdd(result, sum);
  }
}

int main() {
  constexpr std::uint32_t N = 1 << 26;

  std::vector<float> hx(N);
  std::vector<float> hy(N);

#pragma omp parallel for
  for (std::uint32_t i = 0; i < N; ++i) {
    hx[i] = 1.0f;
    hy[i] = 2.0f;
  }

  float *dx = nullptr;
  float *dy = nullptr;
  float *d_result = nullptr;

  CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dx, hx.data(),
                        N * sizeof(float),
                        cudaMemcpyDefault));

  CUDA_CHECK(cudaMemcpy(dy, hy.data(),
                        N * sizeof(float),
                        cudaMemcpyDefault));

  CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

  constexpr std::uint32_t block_size = 256;
  constexpr std::uint32_t grid_size  = 1024;

  dot_kernel<<<grid_size, block_size>>>(dx, dy, d_result, N);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float gpu_result = 0.0f;

  CUDA_CHECK(cudaMemcpy(&gpu_result,
                        d_result,
                        sizeof(float),
                        cudaMemcpyDefault));

  float cpu_result = 0.0f;

#pragma omp parallel for reduction(+: cpu_result)
  for (std::uint32_t i = 0; i < N; ++i)
    cpu_result += hx[i] * hy[i];

  std::printf("GPU : %e\n", gpu_result);
  std::printf("CPU : %e\n", cpu_result);

  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(d_result));
}
