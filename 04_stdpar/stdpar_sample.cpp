#include <vector>
#include <iostream>
#include <execution>
#include <algorithm>
#include <numeric>

int main() {
  constexpr std::size_t N = 1lu << 14;

  std::vector<float> a(N), b(N), c(N);

  std::fill(
    std::execution::par_unseq,
    a.begin(), a.end(),
    1.f
    );
  std::fill(
    std::execution::par_unseq,
    b.begin(), b.end(),
    2.f
    );

  // vectoradd
  std::transform(
    std::execution::par_unseq,
    a.begin(), a.end(), b.begin(), c.begin(),
    [=] (const float a, const float b) {
      return a + b;
    });

  // Inner-product
  const auto ip = std::transform_reduce(
    std::execution::par,
    a.begin(), a.end(), b.begin(),
    0.f,
    std::plus<>{},
    std::multiplies<>{}
    );


  std::printf("a: ");
  for (std::uint32_t i = 0; i < 10; i++) std::printf("%.f ", a[i]);
  std::printf("...\n");
  std::printf("b: ");
  for (std::uint32_t i = 0; i < 10; i++) std::printf("%.f ", b[i]);
  std::printf("...\n");
  std::printf("c: ");
  for (std::uint32_t i = 0; i < 10; i++) std::printf("%.f ", c[i]);
  std::printf("...\n");
  std::printf("ip = %e, expected = %e\n", ip, 2. * N);
}
