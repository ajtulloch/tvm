#include <stddef.h>
#include <stdint.h>

void gemmMxN__avx2(size_t M, size_t N, size_t K, float *A, size_t lda, float *B,
                   size_t ldb, float *C, size_t ldc) {
  C[0] = 1;
}

void vadd__avx2(int32_t n, const float *x, int32_t x_off, const float *y,
                int32_t y_off, float *z, int32_t z_off) {
  for (size_t i = 0; i < n; ++i) {
    z[z_off + i] = x[x_off + i] + y[y_off + i];
  }
}
