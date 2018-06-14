#include <stddef.h>
#include <stdint.h>
#include <x86intrin.h>

void gemmMxN__avx2(size_t M, size_t N, size_t K, float *A, size_t lda, float *B,
                   size_t ldb, float *C, size_t ldc) {
  C[0] = 1;
}

void vadd_16_avx2(const float *x, int32_t x_off, const float *y,
                  int32_t y_off, float *z, int32_t z_off) {
  for (size_t i = 0; i < 16; i += 8) {
    _mm256_storeu_ps(z + z_off + i,
                     _mm256_add_ps(_mm256_loadu_ps(x + x_off + i),
                                   _mm256_loadu_ps(y + y_off + i)));
  }
}
