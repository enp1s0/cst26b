#include <iostream>
#include <cstdint>

int main() {
  constexpr std::uint32_t n = 1u << 15;

  auto *a = (float*)malloc(n * sizeof(float));
  auto *b = (float*)malloc(n * sizeof(float));
  auto *c = (float*)malloc(n * sizeof(float));
  for (std::uint32_t i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = 1.0f;
  }

  // Add one directive before the CPU loop
#pragma acc parallel loop
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }

  float sum = 0.0f;
#pragma acc parallel loop reduction(+:sum)
  for (int i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }

  std::printf("c[0] = %f\n", c[0]);
  std::printf("sum = %f\n", sum);

  free(a);
  free(b);
  free(c);
}
