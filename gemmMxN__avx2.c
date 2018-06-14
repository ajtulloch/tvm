#include <stddef.h>
#include <stdint.h>

void gemmMxN__avx2(size_t M, size_t N, size_t K, float *A, size_t lda, float *B,
                   size_t ldb, float *C, size_t ldc) {
  C[0] = 1;
}
