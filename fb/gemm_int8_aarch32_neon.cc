#include <arm_neon.h>

#include <stddef.h>
#include <stdint.h>
#include <cstdint>

void gemm_ukernel_4x8__neon(int32_t k, const int8_t* a, size_t a_stride,
                            const int8_t* w, int8_t* c, size_t c_stride) {
  int32x4_t vacc0x0123 = vdupq_n_s32(0);
  int32x4_t vacc0x4567 = vdupq_n_s32(0);
  int32x4_t vacc1x0123 = vacc0x0123;
  int32x4_t vacc1x4567 = vacc0x4567;
  int32x4_t vacc2x0123 = vacc0x0123;
  int32x4_t vacc2x4567 = vacc0x4567;
  int32x4_t vacc3x0123 = vacc0x0123;
  int32x4_t vacc3x4567 = vacc0x4567;

  const int8_t* a0 = a;
  const int8_t* a1 = a0 + a_stride;
  const int8_t* a2 = a1 + a_stride;
  const int8_t* a3 = a2 + a_stride;

  for (; k >= 8; k -= 8) {
    const int8x8_t va0 = vld1_s8(a0);
    a0 += 8;
    const int16x8_t vxa0 = (vmovl_s8(va0));
    const int8x8_t va1 = vld1_s8(a1);
    a1 += 8;
    const int16x8_t vxa1 = (vmovl_s8(va1));
    const int8x8_t va2 = vld1_s8(a2);
    a2 += 8;
    const int16x8_t vxa2 = (vmovl_s8(va2));
    const int8x8_t va3 = vld1_s8(a3);
    a3 += 8;
    const int16x8_t vxa3 = (vmovl_s8(va3));

    const int8x8_t vb01234567c0 = vld1_s8(w);
    w += 8;
    const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);

    const int8x8_t vb01234567c1 = vld1_s8(w);
    w += 8;
    const int16x8_t vxb01234567c1 = (vmovl_s8(vb01234567c1));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);

    const int8x8_t vb01234567c2 = vld1_s8(w);
    w += 8;
    const int16x8_t vxb01234567c2 = (vmovl_s8(vb01234567c2));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);

    const int8x8_t vb01234567c3 = vld1_s8(w);
    w += 8;
    const int16x8_t vxb01234567c3 = (vmovl_s8(vb01234567c3));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);

    const int8x8_t vb01234567c4 = vld1_s8(w);
    w += 8;
    const int16x8_t vxb01234567c4 = (vmovl_s8(vb01234567c4));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);

    const int8x8_t vb01234567c5 = vld1_s8(w);
    w += 8;
    const int16x8_t vxb01234567c5 = (vmovl_s8(vb01234567c5));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);

    const int8x8_t vb01234567c6 = vld1_s8(w);
    w += 8;
    const int16x8_t vxb01234567c6 = (vmovl_s8(vb01234567c6));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);

    const int8x8_t vb01234567c7 = vld1_s8(w);
    w += 8;
    const int16x8_t vxb01234567c7 = (vmovl_s8(vb01234567c7));

    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
  }

  int32_t* c0 = c;
  int32_t* c1 = c0 + c_stride;
  int32_t* c2 = c1 + c_stride;
  int32_t* c3 = c2 + c_stride;

  vst1q_s32(c0 + 0, vacc0x0123);
  vst1q_s32(c0 + 4, vacc0x4567);
  vst1q_s32(c1 + 0, vacc1x0123);
  vst1q_s32(c1 + 4, vacc1x4567);
  vst1q_s32(c2 + 0, vacc2x0123);
  vst1q_s32(c2 + 4, vacc2x4567);
  vst1q_s32(c3 + 0, vacc3x0123);
  vst1q_s32(c3 + 4, vacc3x4567);
}
