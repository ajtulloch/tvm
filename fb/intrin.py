import tvm


def intrin_4x8_gemm_neon_ir():
    import os
    src = open(os.path.join(os.path.dirname(__file__), "gemm_int8_aarch32_asm.cc")).read()
    from tvm.contrib import clang
    return clang.create_llvm(src, options=["-O3", "--target=armv7-none-linux-gnueabihf", "-mcpu=cortex-a53", "--sysroot=~/src/panda/usr/include/"])

# print(intrin_4x8_gemm_neon_ir())

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
                         offset_factor=8,
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

K = 32

X = tvm.placeholder((4, K), dtype="int8", name='X')
W = tvm.placeholder((K, 8), dtype="int8", name='X')
k = tvm.reduce_axis((0, K), name='k')
Z = tvm.compute((4, 8), lambda i, j: tvm.sum(X[i, k].astype("int32") * W[k, j].astype("int32"), axis=[k]), name="Z")
s = tvm.create_schedule(Z.op)
print(tvm.lower(s, [X, W, Z], simple_mode=True))
s[Z].vectorize(s[Z].op.axis[1])
# s[Z].tensorize(s[Z].op.axis[0], _intrin_Kx4xint8_Kx8xint8_4x8_int32(K))

target = tvm.target.arm_cpu("rasp3b")
with target:
    with tvm.build_config(dump_pass_ir=True):
        lib = tvm.build(s, [X, W, Z])
# print(lib.get_source("asm"))
lib.save("model2.o")
print(lib.get_source("ll"))
