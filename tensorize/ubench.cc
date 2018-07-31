#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

void sgemm_compute_6x8__neon(int32_t k, const float *a, int32_t a_off,
                             const float *b, int32_t b_off, float *c,
                             int32_t c_off, int32_t ldc) {
  a = a + a_off;
  b = b + b_off;
  c = c + c_off;
  size_t k_size_t = k;
  size_t ldc_size_t = ldc;
  asm volatile("	VMOV.I32  q4, #0\n\t"
               "	VMOV.I32  q5, #0\n\t"
               "	VMOV.I32  q6, #0\n\t"
               "	VMOV.I32  q7, #0\n\t"
               "	VMOV.I32  q8, #0\n\t"
               "	VMOV.I32  q9, #0\n\t"
               "	VMOV.I32 q10, #0\n\t"
               "	VMOV.I32 q11, #0\n\t"
               "	VMOV.I32 q12, #0\n\t"
               "	VMOV.I32 q13, #0\n\t"
               "	VMOV.I32 q14, #0\n\t"
               "	VMOV.I32 q15, #0\n\t"
               "0:\n\t"
               "	VLD1.32 {d4-d7}, [%[b]]!\n\t"
               "	VLD1.32 {d0-d2}, [%[a]]!\n\t"
               "	VMLA.F32 q4, q2, d0[0]\n\t"
               "	VMLA.F32 q5, q3, d0[0]\n\t"
               "	VMLA.F32 q6, q2, d0[1]\n\t"
               "	VMLA.F32 q7, q3, d0[1]\n\t"
               "	VMLA.F32  q8, q2, d1[0]\n\t"
               "	VMLA.F32  q9, q3, d1[0]\n\t"
               "	VMLA.F32 q10, q2, d1[1]\n\t"
               "	VMLA.F32 q11, q3, d1[1]\n\t"
               "	VMLA.F32 q12, q2, d2[0]\n\t"
               "	VMLA.F32 q13, q3, d2[0]\n\t"
               "	VMLA.F32 q14, q2, d2[1]\n\t"
               "	VMLA.F32 q15, q3, d2[1]\n\t"
               "	SUBS %[k_size_t], %[k_size_t], #1\n\t"
               "	BNE 0b\n\t"
               "	LSL %[ldc_size_t], %[ldc_size_t], #2\n\t"
               "	VST1.32 {d8-d11}, [%[c]], %[ldc_size_t]\n\t"
               "	VST1.32 {d12-d15}, [%[c]], %[ldc_size_t]\n\t"
               "	VST1.32 {d16-d19}, [%[c]], %[ldc_size_t]\n\t"
               "	VST1.32 {d20-d23}, [%[c]], %[ldc_size_t]\n\t"
               "	VST1.32 {d24-d27}, [%[c]], %[ldc_size_t]\n\t"
               "	VST1.32 {d28-d31}, [%[c]]\n\t"
               : [c] "+r"(c), [b] "+r"(b), [a] "+r"(a),
                 [k_size_t] "+r"(k_size_t), [ldc_size_t] "+r"(ldc_size_t)
               :
               : "cc", "memory",
                 // note: someone on internet says that quad registers are
                 // unsupported in the clobber list!
                 "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
                 "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18",
                 "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
                 "d28", "d29", "d30", "d31");
}

double currentTime() {
  struct timespec tend;
  clock_gettime(CLOCK_MONOTONIC, &tend);
  return (double)tend.tv_sec + 1.0e-9*(double)tend.tv_nsec;
}

int main(int argc, char** argv) {
  int64_t K = 256;
  int64_t M = 6;
  int64_t N = 8;
  int64_t iters = atoi(argv[1]);
  float* A = (float*)calloc(K * M, sizeof(float));
  float* B = (float*)calloc(K * N, sizeof(float));
  float* C = (float*)calloc(M * N, sizeof(float));
  double start = currentTime();
  for (int i = 0; i < iters; ++i) {
    sgemm_compute_6x8__neon(K, A, 0, B, 0, C, 0, N);
  }
  double end = currentTime();
  double flops = 2 * K * M * N * iters;
  printf("%.2f", flops / (end - start) / 1.0E9);
  return 0;
}
