from ..util import traverse_inline, get_const_tuple, const_matrix
from ..nn.util import get_const_int, get_pad_tuple
from ..nn import pad, conv2d, conv2d_NCHWc, conv2d_alter_layout
from ..generic import schedule_conv2d_nhwc
import tvm
from tvm import autotvm


def intrin_3x8_gemm_neon_ir():
    import os
    src = r"""

// avoid using sysroot.

#ifndef __arm__
#error Only valid on ARMv7
#else
using int32_t = int;
using int8_t = char;
#endif

extern "C" void gemm_ukernel_3x8__neon_asm(
    int32_t k, const int8_t* a, int32_t a_off, int32_t a_stride, const int8_t* b, int32_t b_off,
    int32_t b_stride, int32_t* c, int32_t c_off, int32_t c_stride) {
  a = a + a_off;
  b = b + b_off;
  c = c + c_off;

  const int8_t* a0 = a;
  const int8_t* a1 = a0 + a_stride;
  const int8_t* a2 = a1 + a_stride;

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

      "0:\n\t"
      " VLD1.8 {d1}, [%[a0]]!\n\t"
      " VLD1.8 {d3}, [%[a1]]!\n\t"
      " VLD1.8 {d5}, [%[a2]]!\n\t"
      // Load b0
      " VLD1.8 {d9}, [%[b]]!\n\t"

      " VMOVL.S8 q0, d1\n\t"
      " VMOVL.S8 q1, d3\n\t"
      " VMOVL.S8 q2, d5\n\t"

      " VMOVL.S8 q4, d9\n\t"

      // Load b1
      " VLD1.8 {d11}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d8, d0[0]\n\t"
      " VMLAL.S16 q9, d9, d0[0]\n\t"
      " VMLAL.S16 q10, d8, d2[0]\n\t"
      " VMLAL.S16 q11, d9, d2[0]\n\t"
      " VMOVL.S8 q5, d11\n\t"
      " VMLAL.S16 q12, d8, d4[0]\n\t"
      " VMLAL.S16 q13, d9, d4[0]\n\t"


      // Load b2
      " VLD1.8 {d9}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d10, d0[1]\n\t"
      " VMLAL.S16 q9, d11, d0[1]\n\t"
      " VMLAL.S16 q10, d10, d2[1]\n\t"
      " VMLAL.S16 q11, d11, d2[1]\n\t"
      " VMOVL.S8 q4, d9\n\t"
      " VMLAL.S16 q12, d10, d4[1]\n\t"
      " VMLAL.S16 q13, d11, d4[1]\n\t"

      // Load b3
      " VLD1.8 {d11}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d8, d0[2]\n\t"
      " VMLAL.S16 q9, d9, d0[2]\n\t"
      " VMLAL.S16 q10, d8, d2[2]\n\t"
      " VMLAL.S16 q11, d9, d2[2]\n\t"
      " VMOVL.S8 q5, d11\n\t"
      " VMLAL.S16 q12, d8, d4[2]\n\t"
      " VMLAL.S16 q13, d9, d4[2]\n\t"


      // Load b4
      " VLD1.8 {d9}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d10, d0[3]\n\t"
      " VMLAL.S16 q9, d11, d0[3]\n\t"
      " VMLAL.S16 q10, d10, d2[3]\n\t"
      " VMLAL.S16 q11, d11, d2[3]\n\t"
      " VMOVL.S8 q4, d9\n\t"
      " VMLAL.S16 q12, d10, d4[3]\n\t"
      " VMLAL.S16 q13, d11, d4[3]\n\t"

      // Load b5
      " VLD1.8 {d11}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d8, d1[0]\n\t"
      " VMLAL.S16 q9, d9, d1[0]\n\t"
      " VMLAL.S16 q10, d8, d3[0]\n\t"
      " VMLAL.S16 q11, d9, d3[0]\n\t"
      " VMOVL.S8 q5, d11\n\t"
      " VMLAL.S16 q12, d8, d5[0]\n\t"
      " VMLAL.S16 q13, d9, d5[0]\n\t"


      // Load b6
      " VLD1.8 {d9}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d10, d1[1]\n\t"
      " VMLAL.S16 q9, d11, d1[1]\n\t"
      " VMLAL.S16 q10, d10, d3[1]\n\t"
      " VMLAL.S16 q11, d11, d3[1]\n\t"
      " VMOVL.S8 q4, d9\n\t"
      " VMLAL.S16 q12, d10, d5[1]\n\t"
      " VMLAL.S16 q13, d11, d5[1]\n\t"


      // Load b7
      " VLD1.8 {d11}, [%[b]]!\n\t"

      " VMLAL.S16 q8, d8, d1[2]\n\t"
      " VMLAL.S16 q9, d9, d1[2]\n\t"
      " VMLAL.S16 q10, d8, d3[2]\n\t"
      " VMLAL.S16 q11, d9, d3[2]\n\t"
      " VMOVL.S8 q5, d11\n\t"
      " VMLAL.S16 q12, d8, d5[2]\n\t"
      " VMLAL.S16 q13, d9, d5[2]\n\t"


      " VMLAL.S16 q8, d10, d1[3]\n\t"
      " VMLAL.S16 q9, d11, d1[3]\n\t"
      " VMLAL.S16 q10, d10, d3[3]\n\t"
      " VMLAL.S16 q11, d11, d3[3]\n\t"
      " VMLAL.S16 q12, d10, d5[3]\n\t"
      " VMLAL.S16 q13, d11, d5[3]\n\t"

      "	SUBS %[k_size_t], %[k_size_t], #8\n\t"
      "	BNE 0b\n\t"

      "	VST1.32 {d16-d19}, [%[c0]], %[c_stride_bytes_size_t]\n\t"
      "	VST1.32 {d20-d23}, [%[c0]], %[c_stride_bytes_size_t]\n\t"
      "	VST1.32 {d24-d27}, [%[c0]], %[c_stride_bytes_size_t]\n\t"
      : [c0] "+r"(c0), [b] "+r"(b), [a0] "+r"(a0), [a1] "+r"(a1), [a2] "+r"(a2),
        [k_size_t] "+r"(k_size_t), [c_stride_bytes_size_t] "+r"(c_stride_bytes_size_t)
      :
      : "cc", "memory",
        // note: someone on internet says that quad registers are
        // unsupported in the clobber list!
        "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13",
        "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26",
        "d27", "d28", "d29", "d30", "d31");
}
    """
    from tvm.contrib import clang
    z = clang.create_llvm(src, options=["-O3", "--target=armv7-none-linux-gnueabihf"], output="/tmp/x.ll")
    return "/tmp/x.ll"

def intrin_4x8_gemm_neon_ir():
    import os
    src = r"""

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
    """
    from tvm.contrib import clang
    z = clang.create_llvm(src, options=["-O3", "--target=armv7-none-linux-gnueabihf"], output="/tmp/x.ll")
    return "/tmp/x.ll"

# M == 4 input width
# N == 8 output channels.
# Computes a multiplication of

def _intrin_Kx3xint8_Kx8xint8_3x8_int32(K):
    X = tvm.placeholder((3, K), dtype="int8", name='X')
    W = tvm.placeholder((K, 8), dtype="int8", name='X')
    k = tvm.reduce_axis((0, K), name='k')
    Z = tvm.compute((3, 8), lambda i, j: tvm.sum(X[i, k].astype("int32") * W[k, j].astype("int32"), axis=[k]), name="Z")

    Xb = tvm.decl_buffer(X.shape, X.dtype,
                         name="Xb",
                         offset_factor=K,
                         strides=[tvm.var('ldX'), 1])
    Wb = tvm.decl_buffer(W.shape, W.dtype,
                         name="Wb",
                         offset_factor=K * 8,
                         strides=[8, 1])
    Zb = tvm.decl_buffer(Z.shape, Z.dtype,
                         name="Zb",
                         offset_factor=8,
                         strides=[tvm.var('ldZ'), 1])

    def _intrin_func(ins, outs):
        xx, ww = ins
        zz = outs[0]

        def _instr(index):
            irb = tvm.ir_builder.create()
            irb.scope_attr(tvm.const(1, dtype="int32"), "pragma_import_llvm", intrin_3x8_gemm_neon_ir())
            extern_call = tvm.call_extern(
                "int32",
                "gemm_ukernel_3x8__neon_asm",
                K,
                irb.buffer_ptr(xx),
                xx.elem_offset,
                xx.strides[0],
                irb.buffer_ptr(ww),
                ww.elem_offset,
                ww.strides[0],
                irb.buffer_ptr(zz),
                zz.elem_offset,
                zz.strides[0])
            irb.emit(extern_call)
            return irb.get()
        # body, reset, update
        return _instr(0)
    with tvm.build_config():
        return tvm.decl_tensor_intrin(Z.op, _intrin_func, binds={X: Xb, W:Wb, Z: Zb})

def _intrin_Kx4xint8_Kx8xint8_4x8_int32(K):
    X = tvm.placeholder((4, K), dtype="int8", name='X')
    W = tvm.placeholder((K, 8), dtype="int8", name='X')
    k = tvm.reduce_axis((0, K), name='k')
    Z = tvm.compute((4, 8), lambda i, j: tvm.sum(X[i, k].astype("int32") * W[k, j].astype("int32"), axis=[k]), name="Z")

    Xb = tvm.decl_buffer(X.shape, X.dtype,
                         name="Xb",
                         offset_factor=K,
                         strides=[tvm.var('ldX'), 1])
    Wb = tvm.decl_buffer(W.shape, W.dtype,
                         name="Wb",
                         offset_factor=K * 8,
                         strides=[8, 1])
    Zb = tvm.decl_buffer(Z.shape, Z.dtype,
                         name="Zb",
                         offset_factor=8,
                         strides=[tvm.var('ldZ'), 1])

    def _intrin_func(ins, outs):
        xx, ww = ins
        zz = outs[0]

        def _instr(index):
            irb = tvm.ir_builder.create()
            irb.scope_attr(tvm.const(1, dtype="int32"), "pragma_import_llvm", intrin_4x8_gemm_neon_ir())
            extern_call = tvm.call_extern(
                "int32",
                "gemm_ukernel_4x8__neon_asm",
                K,
                irb.buffer_ptr(xx),
                xx.elem_offset,
                xx.strides[0],
                irb.buffer_ptr(ww),
                ww.elem_offset,
                ww.strides[0],
                irb.buffer_ptr(zz),
                zz.elem_offset,
                zz.strides[0])
            irb.emit(extern_call)
            return irb.get()
        # body, reset, update
        return _instr(0)
    with tvm.build_config():
        return tvm.decl_tensor_intrin(Z.op, _intrin_func, binds={X: Xb, W:Wb, Z: Zb})


def decl_spatial_pack(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, num_tile):
    wkl = None
    out_dtype = out_dtype or data.dtype
    N, IH, IW, CI = get_const_tuple(data.shape)
    KH, KW, CI_, CO = get_const_tuple(kernel.shape)
    assert CI_ == CI

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (KH, KW))
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    OH = (IH + pad_top + pad_bottom - KH) // HSTR + 1
    OW = (IW + pad_left + pad_right - KW) // WSTR + 1
    data_pad = pad(data, [0, pad_top, pad_left, 0], [0, pad_bottom, pad_right, 0], name="data_pad")

    # ==================== define configuration space ====================
    n, co, oh, ow = cfg.axis(N), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    ow, vw = cfg.define_split(
        'tile_ow', cfg.axis(OW), num_outputs=2,
        filter=lambda x: x.size[-1] % 4 == 0 if OW % 4 == 0 else True)
    oh, vh = cfg.define_split('tile_oh', cfg.axis(OH), num_outputs=2)
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]
    VC = 8
    vw = cfg.axis(VW)
    vc = cfg.axis(VC)

    # cfg.define_reorder("reorder_0",
    #                    [n, coo, cii, oh, ow, kh, kw, vc, ciii, vh, vw],
    #                    policy='candidate', candidate=[
    #                        # [n, coo, cii, oh, ow, kh, kw, vc, ciii, vh, vw],
    #                        [n, coo, cii, oh, ow, kh, kw, ciii, vh, vw, vc],
    #                        [n, coo, cii, oh, ow, kh, kw, vh, ciii, vw, vc],
    #                        [n, coo, oh, ow, cii, kh, kw, ciii, vh, vw, vc],
    #                        # [n, coo, cii, oh, ow, kh, kw, vc, vh, ciii, vw],
    #                        # [n, coo, cii, oh, ow, kh, kw, ciii, vh, vc, vw],
    #                        # [n, coo, oh, cii, ow, kh, kw, ciii, vh, vw, vc],
    #                    ])

    # cfg.define_reorder("reorder_1",
    #                    [n, coo, oh, ow, vh, vw, vc],
    #                    policy='candidate', candidate=[
    #                        [n, coo, oh, ow, vh, vw, vc],
    #                        [n, coo, oh, ow, vc, vh, vw],
    #                        [n, coo, oh, ow, vh, vc, vw]
    #                    ])

    # cfg.define_annotate("ann_reduce", [kh, kw, ciii], policy='try_unroll')
    # cfg.define_annotate("ann_spatial", [vc], policy='try_vec')
    # cfg.define_annotate("ann_spatial", [vh, vw, vc], policy='try_unroll_vec')

    # fallback support
    # if cfg.is_fallback:
    #     if num_tile == 2:     # arm cpu
    #         ref_log = autotvm.tophub.load_reference_log('cpu', 'rk3399', 'conv2d', 'direct')
    #         cfg.fallback_with_reference_log(ref_log)
    #     elif num_tile == 3:  # mali gpu
    #         ref_log = autotvm.tophub.load_reference_log('mali', 'rk3399', 'conv2d', 'direct')
    #         cfg.fallback_with_reference_log(ref_log)
    # ====================================================================

    # assert VW == 4
    assert VC == 8
    assert CO % VC == 0
    # input            = (N, CII, IH, IW, CIII)
    # -> transpose
    ############################################################
    # input_tile_shape = (N, CII, OH // VH, OH // VH, VH + KH, VW + KW, CIII)
    # oshape           = (N, COO, OH // VH, OW // VH, VH, VW, COOO)
    ############################################################
    # -> transpose
    # O_shape          = (N, COO, OH, OW, COOO)


    # dvshape = (N, CII, OH // VH, OW // VW, VH*HSTR + KH-1, VW*WSTR + KW-1, CIII)
    # ovshape = (N, COO, OH // VH, OW // VW, VH, VW, VC)
    # oshape = (N, COO, OH, OW, VC)

    dvshape = (N, OH // VH, OW // VW, VH, VW, KH * KW * CI)
    kvshape = (CO // VC, KH * KW * CI, VC)
    oshape = (N, OH, OW, CO)

    def data_vec_compute(n, h, w, vh, vw, kh_kw_ci):
        ci = kh_kw_ci % CI
        kh = (kh_kw_ci // CI) % KW
        kw = (kh_kw_ci // CI) // KW
        kh = ((kh_kw_ci // CI) // KW) % KH
        return data_pad[n][h * VH * HSTR + vh + kh][w * VW * WSTR + vw + kw][ci]
    data_vec = tvm.compute(dvshape, data_vec_compute, name='data_vec')

    def kernel_vec_compute(coo, kh_kw_ci, vc):
        ci = kh_kw_ci % CI
        kh = (kh_kw_ci // CI) % KW
        kw = (kh_kw_ci // CI) // KW
        kh = ((kh_kw_ci // CI) // KW) % KH
        return kernel[kh][kw][ci][coo * VC + vc]
    kernel_vec = tvm.compute(kvshape, kernel_vec_compute, name='kernel_vec')

    kh_kw_ci = tvm.reduce_axis((0, KH * KW * CI), name='kh_kw_ci')

    output = tvm.compute(
        oshape,
        lambda n, h, w, co: tvm.sum(
            data_vec[n, h // VH, w // VW, h % VH, w % VW, kh_kw_ci].astype("int32") *
            kernel_vec[co // VC, kh_kw_ci, co % VC].astype("int32"),
            axis=[kh_kw_ci]
        ),
        name='conv',
        tag='spatial_conv2d_NHWC_output')
    assert output.dtype == "int32"
    return output

@autotvm.register_topi_schedule(schedule_conv2d_nhwc, 'arm_cpu', ['direct', 'winograd'])
def schedule_conv2d_nhwc_arm_cpu(cfg, outs):
    """TOPI schedule callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if 'spatial_conv2d_NHWC_output' in op.tag:
            output = op.output(0)
            _schedule_spatial_pack_NHWC(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s

def _schedule_spatial_pack_NHWC(cfg, s, output, last):
    """schedule implementation"""
    data_vec = output.op.input_tensors[0]
    data_pad = data_vec.op.input_tensors[0]
    kernel_vec = output.op.input_tensors[1]

    (n, h, w, ci) = s[data_pad].op.axis
    s[data_pad].vectorize(ci)

    CI = get_const_int(ci.dom.extent)
    (n, h, w, vh, vw, kh_kw_ci) = s[data_vec].op.axis
    VH = get_const_int(vh.dom.extent)
    VW = get_const_int(vw.dom.extent)

    (kh_kw, ci) = s[data_vec].split(kh_kw_ci, CI)
    (kh, kw) = s[data_vec].split(kh_kw, 3)
    s[data_vec].vectorize(ci)
    s[data_vec].unroll(kh)
    s[data_vec].unroll(kw)

    cfg.define_knob('data_pad_compute_at', [0, 1, 2])
    if cfg['data_pad_compute_at'].val == 2:
        s[data_pad].compute_inline()
    else:
        s[data_pad].compute_at(s[data_vec], [h, vh][cfg['data_pad_compute_at'].val])

    (n, h, w, co) = s[output].op.axis
    (h, vh) = s[output].split(h, VH)
    (w, vw) = s[output].split(w, VW)

    (co, vc) = s[output].split(co, 8)
    (kh_kw_ci, ) = s[output].op.reduce_axis
    s[output].reorder(n, h, w, co, vh, vw, kh_kw_ci, vc)
    K = get_const_int(kh_kw_ci.dom.extent)

    if VW % 4 == 0:
        (vwo, vwi) = s[output].split(vw, 4)
        s[output].tensorize(vwi, _intrin_Kx4xint8_Kx8xint8_4x8_int32(K))
        s[output].unroll(vwo)
        s[output].unroll(vh)

    cfg.define_knob('data_vec_compute_at', [0, 1, 2])
    s[data_vec].compute_at(s[output], [n, h, w][cfg['data_vec_compute_at'].val])

    if kernel_vec.op.name == 'kernel_vec':
        if autotvm.GLOBAL_SCOPE.in_tuning:
            s[kernel_vec].pragma(s[kernel_vec].op.axis[0], 'debug_skip_region')
        else:
            pass
    return s
