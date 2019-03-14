
// avoid using sysroot.

#ifndef __arm__
#error Only valid on ARMv7
#else
using int32_t = int;
using int8_t = char;
#endif

extern "C" void gemm_ukernel_4x8__neon_asm(
    int32_t k, const int8_t* a, int32_t a_off, int32_t a_stride, const int8_t* b, int32_t b_off,
    int32_t b_stride, int32_t* c, int32_t c_off, int32_t c_stride) {
  a = a + a_off;
  b = b + b_off;
  c = c + c_off;

  const int8_t* a0 = a;
  const int8_t* a1 = a0 + a_stride;
  const int8_t* a2 = a1 + a_stride;
  const int8_t* a3 = a2 + a_stride;

  const int32_t* c0 = c;

  int32_t k_size_t = k;
  int32_t c_stride_bytes_size_t = c_stride * 4;
  asm volatile(
      " VMOV.I32  q8, #0\n\t"
      " VMOV.I32  q9, #0\n\t"
      " VMOV.I32 q10, #0\n\t"
      " VMOV.I32 q11, #0\n\t"
      " VMOV.I32 q12, #0\n\t"
      " VMOV.I32 q13, #0\n\t"
      " VMOV.I32 q14, #0\n\t"
      " VMOV.I32 q15, #0\n\t"

      "0:\n\t"
      " VLD1.8 {d1}, [%[a0]]!\n\t"
      " VLD1.8 {d3}, [%[a1]]!\n\t"
      " VLD1.8 {d5}, [%[a2]]!\n\t"
      " VLD1.8 {d7}, [%[a3]]!\n\t"
      // Load b0
      " VLD1.8 {d9}, [%[b]]!\n\t"

      " VMOVL.S8 q0, d1\n\t"
      " VMOVL.S8 q1, d3\n\t"
      " VMOVL.S8 q2, d5\n\t"
      " VMOVL.S8 q3, d7\n\t"

      " VMOVL.S8 q4, d9\n\t"

      // Load b1
      " VLD1.8 {d11}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d8, d0[0]\n\t"
      " VMLAL.S16 q9, d9, d0[0]\n\t"
      " VMLAL.S16 q10, d8, d2[0]\n\t"
      " VMLAL.S16 q11, d9, d2[0]\n\t"
      " VMLAL.S16 q12, d8, d4[0]\n\t"
      " VMLAL.S16 q13, d9, d4[0]\n\t"
      " VMOVL.S8 q5, d11\n\t"
      " VMLAL.S16 q14, d8, d6[0]\n\t"
      " VMLAL.S16 q15, d9, d6[0]\n\t"

      // Load b2
      " VLD1.8 {d9}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d10, d0[1]\n\t"
      " VMLAL.S16 q9, d11, d0[1]\n\t"
      " VMLAL.S16 q10, d10, d2[1]\n\t"
      " VMLAL.S16 q11, d11, d2[1]\n\t"
      " VMLAL.S16 q12, d10, d4[1]\n\t"
      " VMLAL.S16 q13, d11, d4[1]\n\t"
      " VMOVL.S8 q4, d9\n\t"
      " VMLAL.S16 q14, d10, d6[1]\n\t"
      " VMLAL.S16 q15, d11, d6[1]\n\t"

      // Load b3
      " VLD1.8 {d11}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d8, d0[2]\n\t"
      " VMLAL.S16 q9, d9, d0[2]\n\t"
      " VMLAL.S16 q10, d8, d2[2]\n\t"
      " VMLAL.S16 q11, d9, d2[2]\n\t"
      " VMLAL.S16 q12, d8, d4[2]\n\t"
      " VMLAL.S16 q13, d9, d4[2]\n\t"
      " VMOVL.S8 q5, d11\n\t"
      " VMLAL.S16 q14, d8, d6[2]\n\t"
      " VMLAL.S16 q15, d9, d6[2]\n\t"

      // Load b4
      " VLD1.8 {d9}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d10, d0[3]\n\t"
      " VMLAL.S16 q9, d11, d0[3]\n\t"
      " VMLAL.S16 q10, d10, d2[3]\n\t"
      " VMLAL.S16 q11, d11, d2[3]\n\t"
      " VMLAL.S16 q12, d10, d4[3]\n\t"
      " VMLAL.S16 q13, d11, d4[3]\n\t"
      " VMOVL.S8 q4, d9\n\t"
      " VMLAL.S16 q14, d10, d6[3]\n\t"
      " VMLAL.S16 q15, d11, d6[3]\n\t"

      // Load b5
      " VLD1.8 {d11}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d8, d1[0]\n\t"
      " VMLAL.S16 q9, d9, d1[0]\n\t"
      " VMLAL.S16 q10, d8, d3[0]\n\t"
      " VMLAL.S16 q11, d9, d3[0]\n\t"
      " VMLAL.S16 q12, d8, d5[0]\n\t"
      " VMLAL.S16 q13, d9, d5[0]\n\t"
      " VMOVL.S8 q5, d11\n\t"
      " VMLAL.S16 q14, d8, d7[0]\n\t"
      " VMLAL.S16 q15, d9, d7[0]\n\t"

      // Load b6
      " VLD1.8 {d9}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d10, d1[1]\n\t"
      " VMLAL.S16 q9, d11, d1[1]\n\t"
      " VMLAL.S16 q10, d10, d3[1]\n\t"
      " VMLAL.S16 q11, d11, d3[1]\n\t"
      " VMLAL.S16 q12, d10, d5[1]\n\t"
      " VMLAL.S16 q13, d11, d5[1]\n\t"
      " VMOVL.S8 q4, d9\n\t"
      " VMLAL.S16 q14, d10, d7[1]\n\t"
      " VMLAL.S16 q15, d11, d7[1]\n\t"

      // Load b7
      " VLD1.8 {d11}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d8, d1[2]\n\t"
      " VMLAL.S16 q9, d9, d1[2]\n\t"
      " VMLAL.S16 q10, d8, d3[2]\n\t"
      " VMLAL.S16 q11, d9, d3[2]\n\t"
      " VMLAL.S16 q12, d8, d5[2]\n\t"
      " VMLAL.S16 q13, d9, d5[2]\n\t"
      " VMOVL.S8 q5, d11\n\t"
      " VMLAL.S16 q14, d8, d7[2]\n\t"
      " VMLAL.S16 q15, d9, d7[2]\n\t"

      " VMLAL.S16 q8, d10, d1[3]\n\t"
      " VMLAL.S16 q9, d11, d1[3]\n\t"
      " VMLAL.S16 q10, d10, d3[3]\n\t"
      " VMLAL.S16 q11, d11, d3[3]\n\t"
      " VMLAL.S16 q12, d10, d5[3]\n\t"
      " VMLAL.S16 q13, d11, d5[3]\n\t"
      " VMLAL.S16 q14, d10, d7[3]\n\t"
      " VMLAL.S16 q15, d11, d7[3]\n\t"

      "	SUBS %[k_size_t], %[k_size_t], #8\n\t"
      "	BNE 0b\n\t"

      "	VST1.32 {d16-d19}, [%[c0]], %[c_stride_bytes_size_t]\n\t"
      "	VST1.32 {d20-d23}, [%[c0]], %[c_stride_bytes_size_t]\n\t"
      "	VST1.32 {d24-d27}, [%[c0]], %[c_stride_bytes_size_t]\n\t"
      "	VST1.32 {d28-d31}, [%[c0]], %[c_stride_bytes_size_t]\n\t"
      : [c0] "+r"(c0), [b] "+r"(b), [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2), [a3] "+r"(a3),
        [k_size_t] "+r"(k_size_t), [c_stride_bytes_size_t] "+r"(c_stride_bytes_size_t)
      :
      : "cc", "memory",
        // note: someone on internet says that quad registers are
        // unsupported in the clobber list!
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13",
        "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "d28", "d29", "d30", "d31");
}
