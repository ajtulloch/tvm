#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <arm_neon.h>

static inline float32x4_t vld1q_f32_aligned(const float *address) {
  return vld1q_f32(
      (const float *)__builtin_assume_aligned(address, sizeof(float32x4_t)));
}

static inline void vst1q_f32_aligned(float* address, float32x4_t vector) {
  vst1q_f32((float*) __builtin_assume_aligned(address, sizeof(float32x4_t)), vector);
}

static inline float32x4_t vmuladdq_lane0_f32(float32x4_t acc, float32x4_t a,
                                             float32x2_t b) {
#if defined(__aarch64__)
  return vfmaq_lane_f32(acc, a, b, 0);
#else
  return vmlaq_lane_f32(acc, a, b, 0);
#endif
}

static inline float32x4_t vmuladdq_lane1_f32(float32x4_t acc, float32x4_t a,
                                             float32x2_t b) {
#if defined(__aarch64__)
  return vfmaq_lane_f32(acc, a, b, 1);
#else
  return vmlaq_lane_f32(acc, a, b, 1);
#endif
}

void sgemm_compute_6x8__neon(int32_t k, const float *a, int32_t a_off,
                            const float *b, int32_t b_off, float *c,
                            int32_t c_off, int32_t ldc) {
  a = a + a_off;
  b = b + b_off;
  c = c + c_off;

  float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f);
  float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f);
  float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f);
  float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f);
  float32x4_t vc40 = vdupq_n_f32(0.0f), vc41 = vdupq_n_f32(0.0f);
  float32x4_t vc50 = vdupq_n_f32(0.0f), vc51 = vdupq_n_f32(0.0f);
  do {
    const float32x4_t va0123 = vld1q_f32(a);
    const float32x2_t va45 = vld1_f32(a + 4);
    a += 6;

    const float32x4_t vb0 = vld1q_f32_aligned(b + 0);
    const float32x4_t vb1 = vld1q_f32_aligned(b + 4);
    b += 8;

    vc00 = vmuladdq_lane0_f32(vc00, vb0, vget_low_f32(va0123));
    vc10 = vmuladdq_lane1_f32(vc10, vb0, vget_low_f32(va0123));
    vc20 = vmuladdq_lane0_f32(vc20, vb0, vget_high_f32(va0123));
    vc30 = vmuladdq_lane1_f32(vc30, vb0, vget_high_f32(va0123));
    vc40 = vmuladdq_lane0_f32(vc40, vb0, va45);
    vc50 = vmuladdq_lane1_f32(vc50, vb0, va45);

    vc01 = vmuladdq_lane0_f32(vc01, vb1, vget_low_f32(va0123));
    vc11 = vmuladdq_lane1_f32(vc11, vb1, vget_low_f32(va0123));
    vc21 = vmuladdq_lane0_f32(vc21, vb1, vget_high_f32(va0123));
    vc31 = vmuladdq_lane1_f32(vc31, vb1, vget_high_f32(va0123));
    vc41 = vmuladdq_lane0_f32(vc41, vb1, va45);
    vc51 = vmuladdq_lane1_f32(vc51, vb1, va45);
  } while (--k);

  vst1q_f32(c + 0, vc00);
  vst1q_f32(c + 4, vc01);
  c += ldc;
  vst1q_f32(c + 0, vc10);
  vst1q_f32(c + 4, vc11);
  c += ldc;
  vst1q_f32(c + 0, vc20);
  vst1q_f32(c + 4, vc21);
  c += ldc;
  vst1q_f32(c + 0, vc30);
  vst1q_f32(c + 4, vc31);
  c += ldc;
  vst1q_f32(c + 0, vc40);
  vst1q_f32(c + 4, vc41);
  c += ldc;
  vst1q_f32(c + 0, vc50);
  vst1q_f32(c + 4, vc51);
}

void sgemm_reset_6x8__neon(float *c, int32_t c_off, int32_t ldc) {
  c = c + c_off;
  const float32x4_t vzero = vdupq_n_f32(0.0);
  vst1q_f32(c + 0, vzero);
  vst1q_f32(c + 4, vzero);
  c += ldc;
  vst1q_f32(c + 0, vzero);
  vst1q_f32(c + 4, vzero);
  c += ldc;
  vst1q_f32(c + 0, vzero);
  vst1q_f32(c + 4, vzero);
  c += ldc;
  vst1q_f32(c + 0, vzero);
  vst1q_f32(c + 4, vzero);
  c += ldc;
  vst1q_f32(c + 0, vzero);
  vst1q_f32(c + 4, vzero);
  c += ldc;
  vst1q_f32(c + 0, vzero);
  vst1q_f32(c + 4, vzero);
}

void sgemm_update_6x8__neon(int32_t k, const float *a, int32_t a_off,
                            const float *b, int32_t b_off, float *c,
                            int32_t c_off, int32_t ldc) {
  a = a + a_off;
  b = b + b_off;
  c = c + c_off;

  float32x4_t vc00 = vdupq_n_f32(0.0f), vc01 = vdupq_n_f32(0.0f);
  float32x4_t vc10 = vdupq_n_f32(0.0f), vc11 = vdupq_n_f32(0.0f);
  float32x4_t vc20 = vdupq_n_f32(0.0f), vc21 = vdupq_n_f32(0.0f);
  float32x4_t vc30 = vdupq_n_f32(0.0f), vc31 = vdupq_n_f32(0.0f);
  float32x4_t vc40 = vdupq_n_f32(0.0f), vc41 = vdupq_n_f32(0.0f);
  float32x4_t vc50 = vdupq_n_f32(0.0f), vc51 = vdupq_n_f32(0.0f);
  do {
    const float32x4_t va0123 = vld1q_f32(a);
    const float32x2_t va45 = vld1_f32(a + 4);
    a += 6;

    const float32x4_t vb0 = vld1q_f32_aligned(b + 0);
    const float32x4_t vb1 = vld1q_f32_aligned(b + 4);
    b += 8;

    vc00 = vmuladdq_lane0_f32(vc00, vb0, vget_low_f32(va0123));
    vc10 = vmuladdq_lane1_f32(vc10, vb0, vget_low_f32(va0123));
    vc20 = vmuladdq_lane0_f32(vc20, vb0, vget_high_f32(va0123));
    vc30 = vmuladdq_lane1_f32(vc30, vb0, vget_high_f32(va0123));
    vc40 = vmuladdq_lane0_f32(vc40, vb0, va45);
    vc50 = vmuladdq_lane1_f32(vc50, vb0, va45);

    vc01 = vmuladdq_lane0_f32(vc01, vb1, vget_low_f32(va0123));
    vc11 = vmuladdq_lane1_f32(vc11, vb1, vget_low_f32(va0123));
    vc21 = vmuladdq_lane0_f32(vc21, vb1, vget_high_f32(va0123));
    vc31 = vmuladdq_lane1_f32(vc31, vb1, vget_high_f32(va0123));
    vc41 = vmuladdq_lane0_f32(vc41, vb1, va45);
    vc51 = vmuladdq_lane1_f32(vc51, vb1, va45);
  } while (--k);

  vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc00));
  vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc01));
  c += ldc;
  vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc10));
  vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc11));
  c += ldc;
  vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc20));
  vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc21));
  c += ldc;
  vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc30));
  vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc31));
  c += ldc;
  vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc40));
  vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc41));
  c += ldc;
  vst1q_f32(c + 0, vaddq_f32(vld1q_f32(c + 0), vc50));
  vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4), vc51));
}
