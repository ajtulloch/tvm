#include <stddef.h>
#include <stdint.h>
#include <x86intrin.h>
#include <stdio.h>
#include <string.h>

void gemmMxN__avx2(size_t M, size_t N, size_t K, float *A, size_t lda, float *B,
                   size_t ldb, float *C, size_t ldc) {
  C[0] = 1;
}

void vadd_16_avx2(const float *x, int32_t x_off, const float *y,
                  int32_t y_off, float *z, int32_t z_off) {
  /* __builtin_prefetch(x + x_off + 32); */
  /* __builtin_prefetch(y + y_off + 32); */
  for (size_t i = 0; i < 16; i += 8) {
    _mm256_storeu_ps(z + z_off + i,
                     _mm256_add_ps(_mm256_loadu_ps(x + x_off + i),
                                   _mm256_loadu_ps(y + y_off + i)));
  }
}

// State is 3 rows, 32 cols in 3 vectors.

// Load 1 vector of A (1 load)
// Load 4 vectors of B (4 loads)
// Shuffle A 3 times (3 shuffles)

// Do FMA (12 times).


// Throughput is 3 * 32 (96) FMAs in 5 loads + 3 shuffles + 12 FMAs

// Compute intensity is 1.5
// Once this is done.

void sgemm_only_3x32__avx2(int32_t k, const float *a, int32_t a_off,
                           const float *b, int32_t b_off, float *c,
                           int32_t c_off, int32_t ldc) {
  __m256 acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13, acc20, acc21,
      acc22, acc23;
  acc00 = _mm256_setzero_ps();
  acc01 = _mm256_setzero_ps();
  acc02 = _mm256_setzero_ps();
  acc03 = _mm256_setzero_ps();
  acc10 = _mm256_setzero_ps();
  acc11 = _mm256_setzero_ps();
  acc12 = _mm256_setzero_ps();
  acc13 = _mm256_setzero_ps();
  acc20 = _mm256_setzero_ps();
  acc21 = _mm256_setzero_ps();
  acc22 = _mm256_setzero_ps();
  acc23 = _mm256_setzero_ps();

  a += a_off;
  b += b_off;
  c += c_off;
  for (int32_t kk = 0; kk < k; ++kk) {
    const __m256 vb0 = _mm256_loadu_ps(b + 0);
    const __m256 vb1 = _mm256_loadu_ps(b + 8);
    const __m256 vb2 = _mm256_loadu_ps(b + 16);
    const __m256 vb3 = _mm256_loadu_ps(b + 24);

    const __m256 a0 = _mm256_set1_ps(a[0]);

    acc00 = _mm256_fmadd_ps(a0, vb0, acc00);
    acc01 = _mm256_fmadd_ps(a0, vb1, acc01);
    acc02 = _mm256_fmadd_ps(a0, vb2, acc02);
    acc03 = _mm256_fmadd_ps(a0, vb3, acc03);

    const __m256 a1 = _mm256_set1_ps(a[1]);
    acc10 = _mm256_fmadd_ps(a1, vb0, acc10);
    acc11 = _mm256_fmadd_ps(a1, vb1, acc11);
    acc12 = _mm256_fmadd_ps(a1, vb2, acc12);
    acc13 = _mm256_fmadd_ps(a1, vb3, acc13);

    const __m256 a2 = _mm256_set1_ps(a[2]);
    acc20 = _mm256_fmadd_ps(a2, vb0, acc20);
    acc21 = _mm256_fmadd_ps(a2, vb1, acc21);
    acc22 = _mm256_fmadd_ps(a2, vb2, acc22);
    acc23 = _mm256_fmadd_ps(a2, vb3, acc23);


    a += 3;
    b += 32;
  }

  _mm256_storeu_ps(c + 0, acc00);
  _mm256_storeu_ps(c + 8, acc01);
  _mm256_storeu_ps(c + 16, acc02);
  _mm256_storeu_ps(c + 24, acc03);
  c += ldc;

  _mm256_storeu_ps(c + 0, acc10);
  _mm256_storeu_ps(c + 8, acc11);
  _mm256_storeu_ps(c + 16, acc12);
  _mm256_storeu_ps(c + 24, acc13);
  c += ldc;

  _mm256_storeu_ps(c + 0, acc20);
  _mm256_storeu_ps(c + 8, acc21);
  _mm256_storeu_ps(c + 16, acc22);
  _mm256_storeu_ps(c + 24, acc23);
}

void sgemm_only_4x24__avx2(int32_t k, const float *a, int32_t a_off,
                           const float *b, int32_t b_off, float *c,
                           int32_t c_off, int32_t ldc) {
  a = a + a_off;
  b = b + b_off;
  c = c + c_off;
  size_t k_size_t = k;
  size_t ldc_size_t = ldc;
  asm volatile("shl    $0x2,%[ldc_size_t]\n\t"
               "prefetcht0 (%[c])\n\t"
               "add    %[ldc_size_t],%[c]\n\t"
               "prefetcht0 (%[c])\n\t"
               "add    %[ldc_size_t],%[c]\n\t"
               "prefetcht0 (%[c])\n\t"
               "add    %[ldc_size_t],%[c]\n\t"
               "prefetcht0 (%[c])\n\t"
               "vzeroall\n\t"
               "LOOP_START%=:\n\t"
               "vmovaps (%[b]),%%ymm3\n\t"
               "vmovaps 0x20(%[b]),%%ymm2\n\t"
               "vmovaps 0x40(%[b]),%%ymm1\n\t"
               "add    $0x60,%[b]\n\t"
               "vbroadcastss (%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm8\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm9\n\t"
               "vfmadd231ps %%ymm1,%%ymm0,%%ymm10\n\t"
               "vbroadcastss 0x4(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm11\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm12\n\t"
               "vfmadd231ps %%ymm1,%%ymm0,%%ymm13\n\t"
               "vbroadcastss 0x8(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm14\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm15\n\t"
               "vfmadd231ps %%ymm1,%%ymm0,%%ymm7\n\t"
               "vbroadcastss 0xc(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm6\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm5\n\t"
               "vfmadd231ps %%ymm1,%%ymm0,%%ymm4\n\t"
               "add    $0x10,%[a]\n\t"
               "dec    %[k_size_t]\n\t"
               "jne    LOOP_START%=\n\t"
               "vmovups %%ymm6,(%[c])\n\t"
               "vmovups %%ymm5,0x20(%[c])\n\t"
               "vmovups %%ymm4,0x40(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm14,(%[c])\n\t"
               "vmovups %%ymm15,0x20(%[c])\n\t"
               "vmovups %%ymm7,0x40(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm11,(%[c])\n\t"
               "vmovups %%ymm12,0x20(%[c])\n\t"
               "vmovups %%ymm13,0x40(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm8,(%[c])\n\t"
               "vmovups %%ymm9,0x20(%[c])\n\t"
               "vmovups %%ymm10,0x40(%[c])\n\t"
               "vzeroupper\n\t"
               : [c] "+r"(c), [b] "+r"(b), [a] "+r"(a),
                 [k_size_t] "+r"(k_size_t), [ldc_size_t] "+r"(ldc_size_t)
               :
               : "cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                 "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
                 "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}
