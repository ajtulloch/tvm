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
