/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "assembly.h"

.syntax unified

# void q8gemm_ukernel_4x8__aarch32_neon(
#     size_t mr,
#     size_t nr,
#     size_t k,
#     const uint8_t*restrict a,
#     size_t a_stride,
#     const void*restrict w,
#     uint8_t*restrict c,
#     size_t c_stride,
#     const union qnnp_conv_quantization_params quantization_params[restrict static 1])
BEGIN_FUNCTION q8gemm_ukernel_4x8__aarch32_neon
  .arm
#ifndef __APPLE__
  .arch armv7-a
  .fpu neon
#endif
  # Load w
  # - ip = w
  LDR ip, [sp, 4]
  PUSH {r4, r5, r6, r7}

  VPUSH {d8-d15}
  # Load quantization params
  # - r7 = quantization_params
  ;; LDR r7, [sp, 96]

  ;; # Load bias0123, bias4567
  ;; VLDM ip!, {d16-d19}

  # Load a_stride
  # - r6 = a_stride
  LDR r6, [sp, 80]
  CMP r0, 2

  ADD r4, r3, r6

  # Load b_zero_point:
  # - d15 = b_zero_point
  VLD1.8 {d15[]}, [r7]
  MOVLO r4, r3

  ADD r7, r7, 4
  ADD r5, r4, r6

  # q10 := vacc1x0123
  VMOV.I32 q10, q8
  MOVLS r5, r4
  # q11 := vacc1x4567
  VMOV.I32 q11, q9
  ADD r6, r5, r6
  # q12 := vacc2x0123
  VMOV.I32 q12, q8
  CMP r0, 4
  # q13 := vacc2x4567
  VMOV.I32 q13, q9
  MOVNE r6, r5
  # q14 := vacc3x0123
  VMOV.I32 q14, q8
  SUBS r2, r2, 8
  # q15 := vacc3x4567
  VMOV.I32 q15, q9
  # Load multiplier:
  # - d12 = vmultiplier
  VLD1.32 {d12[]}, [r7]!
  BLO 1f

  .p2align 5
0:
  # Load a0
  # - d1 = a0
  VLD1.8 {d1}, [r3]!

  # Load a1
  # - d3 = a1
  VLD1.8 {d3}, [r4]!

  # Load b0-b7 (channel 0)
  # - d9 = b0-b7
  VLD1.8 {d9}, [ip:64]!

  # Load a2
  # - d5 = a2
  VLD1.8 {d5}, [r5]!

  # q0 = va0 = a0
  VMOVL.U8 q0, d1

  # Load a3
  # - d7 = a3
  VLD1.8 {d7}, [r6]!

  # q1 = va1 = a1
  VMOVL.U8 q1, d3

  # q4 = b0:7 - b_zero_point
  # - d8 = vb0123 (channel 0)
  # - d9 = vb4567 (channel 0)
  VSUBL.U8 q4, d9, d15

  # q2 = va2 = a2
  VMOVL.U8 q2, d5
  # q3 = va3 = a3
  VMOVL.U8 q3, d7

  ### Channel 0 ###

  # Load b0-b7 (channel 1)
  # - d11 = b0-b7
  VLD1.8 {d11}, [ip:64]!

  # vacc0x0123 += vb0123 * va0[0]
  VMLAL.S16 q8, d8, d0[0]
  # vacc0x4567 += vb4567 * va0[0]
  VMLAL.S16 q9, d9, d0[0]

  # vacc1x0123 += vb0123 * va1[0]
  VMLAL.S16 q10, d8, d2[0]
  # vacc1x4567 += vb4567 * va1[0]
  VMLAL.S16 q11, d9, d2[0]

  # vacc2x0123 += vb0123 * va2[0]
  VMLAL.S16 q12, d8, d4[0]
  # vacc2x4567 += vb4567 * va2[0]
  VMLAL.S16 q13, d9, d4[0]

  # q5 = b0:7 - b_zero_point
  # - d10 = vb0123 (channel 1)
  # - d11 = vb4567 (channel 1)
  VSUBL.U8 q5, d11, d15

  # vacc3x0123 += vb0123 * va3[0]
  VMLAL.S16 q14, d8, d6[0]
  # vacc3x4567 += vb4567 * va3[0]
  VMLAL.S16 q15, d9, d6[0]

  ### Channel 1 ###

  # Load b0-b7 (channel 2)
  # - d9 = b0-b7
  VLD1.8 {d9}, [ip:64]!

  # vacc0x0123 += vb0123 * va0[1]
  VMLAL.S16 q8, d10, d0[1]
  # vacc0x4567 += vb4567 * va0[1]
  VMLAL.S16 q9, d11, d0[1]

  # vacc1x0123 += vb0123 * va1[1]
  VMLAL.S16 q10, d10, d2[1]
  # vacc1x4567 += vb4567 * va1[1]
  VMLAL.S16 q11, d11, d2[1]

  # vacc2x0123 += vb0123 * va2[1]
  VMLAL.S16 q12, d10, d4[1]
  # vacc2x4567 += vb4567 * va2[1]
  VMLAL.S16 q13, d11, d4[1]

  # q4 = b0:7 - b_zero_point
  # - d8 = vb0123 (channel 2)
  # - d9 = vb4567 (channel 2)
  VSUBL.U8 q4, d9, d15

  # vacc3x0123 += vb0123 * va3[1]
  VMLAL.S16 q14, d10, d6[1]
  # vacc3x4567 += vb4567 * va3[1]
  VMLAL.S16 q15, d11, d6[1]

  ### Channel 2 ###

  # Load b0-b7 (channel 3)
  # - d11 = b0-b7
  VLD1.8 {d11}, [ip:64]!

  # vacc0x0123 += vb0123 * va0[2]
  VMLAL.S16 q8, d8, d0[2]
  # vacc0x4567 += vb4567 * va0[2]
  VMLAL.S16 q9, d9, d0[2]

  # vacc1x0123 += vb0123 * va1[2]
  VMLAL.S16 q10, d8, d2[2]
  # vacc1x4567 += vb4567 * va1[2]
  VMLAL.S16 q11, d9, d2[2]

  # vacc2x0123 += vb0123 * va2[2]
  VMLAL.S16 q12, d8, d4[2]
  # vacc2x4567 += vb4567 * va2[2]
  VMLAL.S16 q13, d9, d4[2]

  # q5 = b0:7 - b_zero_point
  # - d10 = vb0123 (channel 3)
  # - d11 = vb4567 (channel 3)
  VSUBL.U8 q5, d11, d15

  # vacc3x0123 += vb0123 * va3[2]
  VMLAL.S16 q14, d8, d6[2]
  # vacc3x4567 += vb4567 * va3[2]
  VMLAL.S16 q15, d9, d6[2]

  ### Channel 3 ###

  # Load b0-b7 (channel 4)
  # - d9 = b0-b7
  VLD1.8 {d9}, [ip:64]!

  # vacc0x0123 += vb0123 * va0[3]
  VMLAL.S16 q8, d10, d0[3]
  # vacc0x4567 += vb4567 * va0[3]
  VMLAL.S16 q9, d11, d0[3]

  # vacc1x0123 += vb0123 * va1[3]
  VMLAL.S16 q10, d10, d2[3]
  # vacc1x4567 += vb4567 * va1[3]
  VMLAL.S16 q11, d11, d2[3]

  # vacc2x0123 += vb0123 * va2[3]
  VMLAL.S16 q12, d10, d4[3]
  # vacc2x4567 += vb4567 * va2[3]
  VMLAL.S16 q13, d11, d4[3]

  # q5 = b0:7 - b_zero_point
  # - d10 = vb0123 (channel 4)
  # - d11 = vb4567 (channel 4)
  VSUBL.U8 q4, d9, d15

  # vacc3x0123 += vb0123 * va3[3]
  VMLAL.S16 q14, d10, d6[3]
  # vacc3x4567 += vb4567 * va3[3]
  VMLAL.S16 q15, d11, d6[3]

  ### Channel 4 ###

  # Load b0-b7 (channel 5)
  # - d11 = b0-b7
  VLD1.8 {d11}, [ip:64]!

  # vacc0x0123 += vb0123 * va0[4]
  VMLAL.S16 q8, d8, d1[0]
  # vacc0x4567 += vb4567 * va0[4]
  VMLAL.S16 q9, d9, d1[0]

  # vacc1x0123 += vb0123 * va1[4]
  VMLAL.S16 q10, d8, d3[0]
  # vacc1x4567 += vb4567 * va1[4]
  VMLAL.S16 q11, d9, d3[0]

  # vacc2x0123 += vb0123 * va2[4]
  VMLAL.S16 q12, d8, d5[0]
  # vacc2x4567 += vb4567 * va2[4]
  VMLAL.S16 q13, d9, d5[0]

  # q4 = b0:7 - b_zero_point
  # - d8 = vb0123 (channel 5)
  # - d9 = vb4567 (channel 5)
  VSUBL.U8 q5, d11, d15

  # vacc3x0123 += vb0123 * va3[4]
  VMLAL.S16 q14, d8, d7[0]
  # vacc3x4567 += vb4567 * va3[4]
  VMLAL.S16 q15, d9, d7[0]

  ### Channel 5 ###

  # Load b0-b7 (channel 6)
  # - d9 = b0-b7
  VLD1.8 {d9}, [ip:64]!

  # vacc0x0123 += vb0123 * va0[5]
  VMLAL.S16 q8, d10, d1[1]
  # vacc0x4567 += vb4567 * va0[5]
  VMLAL.S16 q9, d11, d1[1]

  # vacc1x0123 += vb0123 * va1[5]
  VMLAL.S16 q10, d10, d3[1]
  # vacc1x4567 += vb4567 * va1[5]
  VMLAL.S16 q11, d11, d3[1]

  # vacc2x0123 += vb0123 * va2[5]
  VMLAL.S16 q12, d10, d5[1]
  # vacc2x4567 += vb4567 * va2[5]
  VMLAL.S16 q13, d11, d5[1]

  # q4 = b0:7 - b_zero_point
  # - d8 = vb0123 (channel 6)
  # - d9 = vb4567 (channel 6)
  VSUBL.U8 q4, d9, d15

  # vacc3x0123 += vb0123 * va3[5]
  VMLAL.S16 q14, d10, d7[1]
  # vacc3x4567 += vb4567 * va3[5]
  VMLAL.S16 q15, d11, d7[1]

  ### Channel 6 ###

  # Load b0-b7 (channel 7)
  # - d11 = b0-b7
  VLD1.8 {d11}, [ip:64]!

  # vacc0x0123 += vb0123 * va0[6]
  VMLAL.S16 q8, d8, d1[2]
  # vacc0x4567 += vb4567 * va0[6]
  VMLAL.S16 q9, d9, d1[2]

  # vacc1x0123 += vb0123 * va1[6]
  VMLAL.S16 q10, d8, d3[2]
  # vacc1x4567 += vb4567 * va1[6]
  VMLAL.S16 q11, d9, d3[2]

  # vacc2x0123 += vb0123 * va2[6]
  VMLAL.S16 q12, d8, d5[2]

  # q5 = b0:7 - b_zero_point
  # - d10 = vb0123 (channel 7)
  # - d11 = vb4567 (channel 7)
  VSUBL.U8 q5, d11, d15

  # vacc2x4567 += vb4567 * va2[6]
  VMLAL.S16 q13, d9, d5[2]

  # vacc3x0123 += vb0123 * va3[6]
  VMLAL.S16 q14, d8, d7[2]
  # vacc3x4567 += vb4567 * va3[6]
  VMLAL.S16 q15, d9, d7[2]

  ### Channel 8 ###
  SUBS r2, r2, 8

  # vacc0x0123 += vb0123 * va0[7]
  VMLAL.S16 q8, d10, d1[3]
  # vacc0x4567 += vb4567 * va0[7]
  VMLAL.S16 q9, d11, d1[3]

  # vacc1x0123 += vb0123 * va1[7]
  VMLAL.S16 q10, d10, d3[3]
  # vacc1x4567 += vb4567 * va1[7]
  VMLAL.S16 q11, d11, d3[3]

  # vacc2x0123 += vb0123 * va2[7]
  VMLAL.S16 q12, d10, d5[3]
  # vacc2x4567 += vb4567 * va2[7]
  VMLAL.S16 q13, d11, d5[3]

  # vacc3x0123 += vb0123 * va3[7]
  VMLAL.S16 q14, d10, d7[3]
  # vacc3x4567 += vb4567 * va3[7]
  VMLAL.S16 q15, d11, d7[3]

  BHS 0b

1:
  CMP r2, -8
  BEQ 2f
  .p2align 4
2:

  # Load c, c_stride:
  # - r2 = c
  # - r2 = c_stride
  LDRD r2, r3, [sp, 88]
  ADD r4, r2, r3
  ADD r5, r4, r3
  ADD r3, r5, r3

  VST1.32 {q8}, [r2]!
  VST1.32 {q9}, [r2]!
  VST1.32 {q10}, [r4]!
  VST1.32 {q11}, [r4]!
  VST1.8 {q12}, [r5]!
  VST1.8 {q13}, [r5]!
  VST1.8 {q14}, [r3]!
  VST1.8 {q14}, [r3]!

  VPOP {d8-d15}
  POP {r4, r5, r6, r7}
  BX lr

  .p2align 3
END_FUNCTION q8gemm_ukernel_4x8__aarch32_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
