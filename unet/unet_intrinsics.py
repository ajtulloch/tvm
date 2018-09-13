import tvm
import tvm.contrib.clang


temp = tvm.contrib.util.tempdir()
ll_path = temp.relpath("unet_intrinsics.ll")
ll_code = tvm.contrib.clang.create_llvm(open("unet/unet_intrinsics.cc").read(), output=ll_path)

# Tensorized
def gemm(M, N, K):
    MNTiles = [
        (4, 24),
        (6, 16),
        # (5, 16),
    ]
    ARCH = "avx2"

    assert (M, N) in MNTiles, (M, N)
    dtype = 'float32'
    A = tvm.placeholder((K, M), dtype=dtype, name='A')
    B = tvm.placeholder((K, N), dtype=dtype, name='B')
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((M, N), lambda m, n:
                    tvm.sum(A[k, m] * B[k, n], axis=[k]), name='C')

    Ab = tvm.decl_buffer(A.shape, A.dtype,
                        name="A",
                        offset_factor=M,
                        strides=[M, 1])
    Bb = tvm.decl_buffer(B.shape, B.dtype,
                        name="B",
                        offset_factor=N,
                        strides=[N, 1])
    Cb = tvm.decl_buffer(C.shape, C.dtype,
                        name="C",
                        offset_factor=1,
                        strides=[tvm.var('ldc'), 1])

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def body():
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "sgemm_compute_{M}x{N}__{ARCH}".format(M=M, N=N, ARCH=ARCH),
                K,
                irb.buffer_ptr(aa),
                aa.elem_offset,
                irb.buffer_ptr(bb),
                bb.elem_offset,
                irb.buffer_ptr(cc),
                cc.elem_offset,
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()

        def reset():
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "sgemm_reset_{M}x{N}__{ARCH}".format(M=M, N=N, ARCH=ARCH),
                irb.buffer_ptr(cc),
                cc.elem_offset,
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()

        def update():
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "sgemm_update_{M}x{N}__{ARCH}".format(M=M, N=N, ARCH=ARCH),
                K,
                irb.buffer_ptr(aa),
                aa.elem_offset,
                irb.buffer_ptr(bb),
                bb.elem_offset,
                irb.buffer_ptr(cc),
                cc.elem_offset,
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()
        return body(), reset(), update()


    with tvm.build_config():
        return (tvm.decl_tensor_intrin(C.op,
                                       intrin_func,
                                       binds={A: Ab, B: Bb, C: Cb}), ll_path)
