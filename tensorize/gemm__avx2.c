#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>


void sgemm_compute_4x24__avx2(int32_t k, const float *a, int32_t a_off,
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

void sgemm_compute_5x16__avx2(int32_t k, const float *a, int32_t a_off,
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
               "add    %[ldc_size_t],%[c]\n\t"
               "prefetcht0 (%[c])\n\t"
               "vzeroall\n\t"
               "LOOP_START%=:\n\t"
               "vmovaps (%[b]),%%ymm3\n\t"
               "vmovaps 0x20(%[b]),%%ymm2\n\t"
               "add    $0x40,%[b]\n\t"
               "vbroadcastss (%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm8\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm9\n\t"
               "vbroadcastss 0x4(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm11\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm12\n\t"
               "vbroadcastss 0x8(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm14\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm15\n\t"
               "vbroadcastss 0xc(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm6\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm5\n\t"
               "vbroadcastss 0x10(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm13\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm10\n\t"
               "add    $0x14,%[a]\n\t"
               "dec    %[k_size_t]\n\t"
               "jne    LOOP_START%=\n\t"
               "vmovups %%ymm13,(%[c])\n\t"
               "vmovups %%ymm10,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm6,(%[c])\n\t"
               "vmovups %%ymm5,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm14,(%[c])\n\t"
               "vmovups %%ymm15,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm11,(%[c])\n\t"
               "vmovups %%ymm12,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm8,(%[c])\n\t"
               "vmovups %%ymm9,0x20(%[c])\n\t"
               "vzeroupper\n\t"
               : [c] "+r"(c), [b] "+r"(b), [a] "+r"(a),
                 [k_size_t] "+r"(k_size_t), [ldc_size_t] "+r"(ldc_size_t)
               :
               : "cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                 "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
                 "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void sgemm_compute_6x16__avx2(int32_t k, const float *a, int32_t a_off,
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
               "add    %[ldc_size_t],%[c]\n\t"
               "prefetcht0 (%[c])\n\t"
               "add    %[ldc_size_t],%[c]\n\t"
               "prefetcht0 (%[c])\n\t"
               "vzeroall\n\t"
               "LOOP_START%=:\n\t"
               "vmovaps (%[b]),%%ymm3\n\t"
               "vmovaps 0x20(%[b]),%%ymm2\n\t"
               "add    $0x40,%[b]\n\t"
               "vbroadcastss (%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm8\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm9\n\t"
               "vbroadcastss 0x4(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm11\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm12\n\t"
               "vbroadcastss 0x8(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm14\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm15\n\t"
               "vbroadcastss 0xc(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm6\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm5\n\t"
               "vbroadcastss 0x10(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm13\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm10\n\t"
               "vbroadcastss 0x14(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm1\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm4\n\t"
               "add    $0x18,%[a]\n\t"
               "dec    %[k_size_t]\n\t"
               "jne    LOOP_START%=\n\t"
               "vmovups %%ymm1,(%[c])\n\t"
               "vmovups %%ymm4,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm13,(%[c])\n\t"
               "vmovups %%ymm10,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm6,(%[c])\n\t"
               "vmovups %%ymm5,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm14,(%[c])\n\t"
               "vmovups %%ymm15,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm11,(%[c])\n\t"
               "vmovups %%ymm12,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm8,(%[c])\n\t"
               "vmovups %%ymm9,0x20(%[c])\n\t"
               "vzeroupper\n\t"
               : [c] "+r"(c), [b] "+r"(b), [a] "+r"(a),
                 [k_size_t] "+r"(k_size_t), [ldc_size_t] "+r"(ldc_size_t)
               :
               : "cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                 "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
                 "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void sgemm_compute_4x16__avx2(int32_t k, const float *a, int32_t a_off,
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
               "add    $0x40,%[b]\n\t"
               "vbroadcastss (%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm8\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm9\n\t"
               "vbroadcastss 0x4(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm11\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm12\n\t"
               "vbroadcastss 0x8(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm14\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm15\n\t"
               "vbroadcastss 0xc(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm6\n\t"
               "vfmadd231ps %%ymm2,%%ymm0,%%ymm5\n\t"
               "add    $0x10,%[a]\n\t"
               "dec    %[k_size_t]\n\t"
               "jne    LOOP_START%=\n\t"
               "vmovups %%ymm6,(%[c])\n\t"
               "vmovups %%ymm5,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm14,(%[c])\n\t"
               "vmovups %%ymm15,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm11,(%[c])\n\t"
               "vmovups %%ymm12,0x20(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm8,(%[c])\n\t"
               "vmovups %%ymm9,0x20(%[c])\n\t"
               "vzeroupper\n\t"
               : [c] "+r"(c), [b] "+r"(b), [a] "+r"(a),
                 [k_size_t] "+r"(k_size_t), [ldc_size_t] "+r"(ldc_size_t)
               :
               : "cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                 "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
                 "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void sgemm_compute_4x8__avx2(int32_t k, const float *a, int32_t a_off,
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
               "add    $0x20,%[b]\n\t"
               "vbroadcastss (%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm8\n\t"
               "vbroadcastss 0x4(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm11\n\t"
               "vbroadcastss 0x8(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm14\n\t"
               "vbroadcastss 0xc(%[a]),%%ymm0\n\t"
               "vfmadd231ps %%ymm3,%%ymm0,%%ymm6\n\t"
               "add    $0x10,%[a]\n\t"
               "dec    %[k_size_t]\n\t"
               "jne    LOOP_START%=\n\t"
               "vmovups %%ymm6,(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm14,(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm11,(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vmovups %%ymm8,(%[c])\n\t"
               "vzeroupper\n\t"
               : [c] "+r"(c), [b] "+r"(b), [a] "+r"(a),
                 [k_size_t] "+r"(k_size_t), [ldc_size_t] "+r"(ldc_size_t)
               :
               : "cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                 "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
                 "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void sgemm_reset_4x24__avx2(float *c, int32_t c_off, int32_t ldc) {
  c = c + c_off;
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
               : [c] "+r"(c), [ldc_size_t] "+r"(ldc_size_t)
               :
               : "cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                 "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
                 "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void sgemm_update_4x24__avx2(int32_t k, const float *a, int32_t a_off,
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
               "vaddps (%[c]),%%ymm6,%%ymm6\n\t"
               "vmovups %%ymm6,(%[c])\n\t"
               "vaddps 0x20(%[c]),%%ymm5,%%ymm5\n\t"
               "vmovups %%ymm5,0x20(%[c])\n\t"
               "vaddps 0x40(%[c]),%%ymm4,%%ymm4\n\t"
               "vmovups %%ymm4,0x40(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vaddps (%[c]),%%ymm14,%%ymm14\n\t"
               "vmovups %%ymm14,(%[c])\n\t"
               "vaddps 0x20(%[c]),%%ymm15,%%ymm15\n\t"
               "vmovups %%ymm15,0x20(%[c])\n\t"
               "vaddps 0x40(%[c]),%%ymm7,%%ymm7\n\t"
               "vmovups %%ymm7,0x40(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vaddps (%[c]),%%ymm11,%%ymm11\n\t"
               "vmovups %%ymm11,(%[c])\n\t"
               "vaddps 0x20(%[c]),%%ymm12,%%ymm12\n\t"
               "vmovups %%ymm12,0x20(%[c])\n\t"
               "vaddps 0x40(%[c]),%%ymm13,%%ymm13\n\t"
               "vmovups %%ymm13,0x40(%[c])\n\t"
               "sub    %[ldc_size_t],%[c]\n\t"
               "vaddps (%[c]),%%ymm8,%%ymm8\n\t"
               "vmovups %%ymm8,(%[c])\n\t"
               "vaddps 0x20(%[c]),%%ymm9,%%ymm9\n\t"
               "vmovups %%ymm9,0x20(%[c])\n\t"
               "vaddps 0x40(%[c]),%%ymm10,%%ymm10\n\t"
               "vmovups %%ymm10,0x40(%[c])\n\t"
               "vzeroupper\n\t"
               : [c] "+r"(c), [b] "+r"(b), [a] "+r"(a),
                 [k_size_t] "+r"(k_size_t), [ldc_size_t] "+r"(ldc_size_t)
               :
               : "cc", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
                 "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
                 "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}

void qgemm_compute_8x8x4__avx2(int32_t K, const uint8_t *A, int32_t a_off,
                               const int8_t *B, int32_t b_off, int32_t *C,
                               int32_t c_off, int32_t ldc) {
  A += a_off;
  B += b_off;
  C += c_off;

  uint64_t ldc_ = ldc * sizeof(C[0]);
  uint64_t k_updated = 8 * K;
  int16_t vone = 1;
  int16_t *vonep = &vone;
  asm volatile(
#if !defined(__clang__)
      "\t mov r8, %[k_updated]\n"
      "\t mov r9, %[A]\n"
      "\t mov rax, %[vonep]\n"
      "\t mov r10, %[B]\n"
      "\t mov rsi, %[B]\n"
      "\t mov r12, %[C]\n"
      "\t mov r13, %[ldc]\n"
#else
      "\t mov %[k_updated], %%r8\n"
      "\t mov %[A], %%r9\n"
      "\t mov %[vonep], %%rax\n"
      "\t mov %[B], %%r10\n"
      "\t mov %[B], %%rsi\n"
      "\t mov %[C], %%r12\n"
      "\t mov %[ldc], %%r13\n"
      "\t .intel_syntax noprefix\n"
#endif

      // initialize all Ctile registers to zeros
      "\t vzeroall\n"

      // load constant "1"
      "\t vpbroadcastw ymm15, WORD PTR [rax]\n"

      "\t mov r14, 0\n"
      "\t loop_inner%=:\n"

      // <Main micro-kernel>
      // load all B's
      "\t mov rbx, r10\n"
      "\t vmovaps ymm9,YMMWORD PTR [rbx]\n"
      "\t vpbroadcastd ymm8,DWORD PTR [r9+r14 + 0]\n"
      "\t vpmaddubsw ymm14, ymm8, ymm9\n"
      "\t vpmaddwd ymm14, ymm15, ymm14\n"
      "\t vpaddd ymm0, ymm0, ymm14\n"
      "\t vpbroadcastd ymm10,DWORD PTR [r9+r14 + 4]\n"
      "\t vpmaddubsw ymm14, ymm10, ymm9\n"
      "\t vpmaddwd ymm14, ymm15, ymm14\n"
      "\t vpaddd ymm1, ymm1, ymm14\n"
      "\t vpbroadcastd ymm11,DWORD PTR [r9+r14 + 8]\n"
      "\t vpmaddubsw ymm14, ymm11, ymm9\n"
      "\t vpmaddwd ymm14, ymm15, ymm14\n"
      "\t vpaddd ymm2, ymm2, ymm14\n"
      "\t vpbroadcastd ymm12,DWORD PTR [r9+r14 + 12]\n"
      "\t vpmaddubsw ymm14, ymm12, ymm9\n"
      "\t vpmaddwd ymm14, ymm15, ymm14\n"
      "\t vpaddd ymm3, ymm3, ymm14\n"
      "\t vpbroadcastd ymm13,DWORD PTR [r9+r14 + 16]\n"
      "\t vpmaddubsw ymm14, ymm13, ymm9\n"
      "\t vpmaddwd ymm14, ymm15, ymm14\n"
      "\t vpaddd ymm4, ymm4, ymm14\n"
      "\t vpbroadcastd ymm8,DWORD PTR [r9+r14 + 20]\n"
      "\t vpmaddubsw ymm14, ymm8, ymm9\n"
      "\t vpmaddwd ymm14, ymm15, ymm14\n"
      "\t vpaddd ymm5, ymm5, ymm14\n"
      "\t vpbroadcastd ymm10,DWORD PTR [r9+r14 + 24]\n"
      "\t vpmaddubsw ymm14, ymm10, ymm9\n"
      "\t vpmaddwd ymm14, ymm15, ymm14\n"
      "\t vpaddd ymm6, ymm6, ymm14\n"
      "\t vpbroadcastd ymm11,DWORD PTR [r9+r14 + 28]\n"
      "\t vpmaddubsw ymm14, ymm11, ymm9\n"
      "\t vpmaddwd ymm14, ymm15, ymm14\n"
      "\t vpaddd ymm7, ymm7, ymm14\n"

      // loop increment
      "\t add r10, 32\n"
      "\t add r14, 4 * 8\n"
      "\t cmp r14, r8\n"
      "\t jl loop_inner%=\n"

      "\t vmovups YMMWORD PTR [r12 + 0], ymm0\n"
      "\t add r12, r13\n"
      "\t vmovups YMMWORD PTR [r12 + 0], ymm1\n"
      "\t add r12, r13\n"
      "\t vmovups YMMWORD PTR [r12 + 0], ymm2\n"
      "\t add r12, r13\n"
      "\t vmovups YMMWORD PTR [r12 + 0], ymm3\n"
      "\t add r12, r13\n"
      "\t vmovups YMMWORD PTR [r12 + 0], ymm4\n"
      "\t add r12, r13\n"
      "\t vmovups YMMWORD PTR [r12 + 0], ymm5\n"
      "\t add r12, r13\n"
      "\t vmovups YMMWORD PTR [r12 + 0], ymm6\n"
      "\t add r12, r13\n"
      "\t vmovups YMMWORD PTR [r12 + 0], ymm7\n"

      // register list
      :
      : [k_updated] "rm"(k_updated), [A] "rm"(A), [B] "rm"(B), [C] "rm"(C),
        [vonep] "rm"(vonep), [ldc] "rm"(ldc_)
      : "cc", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rax",
        "rbx", "rsi", "memory", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4",
        "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11",
        "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}
