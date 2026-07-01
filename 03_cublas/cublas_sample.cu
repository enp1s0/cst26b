#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
      cudaError_t error = (call);                                            \
      if (error != cudaSuccess) {                                            \
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE);                                             \
      }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
      cublasStatus_t status = (call);                                         \
      if (status != CUBLAS_STATUS_SUCCESS) {                                  \
        std::fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status)); \
        std::exit(EXIT_FAILURE);                                              \
      }                                                                       \
    } while (0)

int main() {
  constexpr std::uint32_t n = 8192;
  float x[n];
  float y[n];
  float *d_x = nullptr;
  float *d_y = nullptr;

  for (std::uint32_t i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i);
    y[i] = 1.f;
  }

  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // y = alpha * x + y
  const float alpha = 1.0f;
  CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1));

  // dot = x^T * y
  float dot = 0.0f;
  CUBLAS_CHECK(cublasSdot(handle, n, d_x, 1, d_y, 1, &dot));

  CUDA_CHECK(cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("y[0:5] =");
  for (std::uint32_t i = 0; i < 5; ++i) std::printf(" %.1f", y[i]);
  std::printf("\ny[%d] = %.1f\n", n - 1, y[n - 1]);
  std::printf("x dot y = %.1f\n", dot);

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
}
