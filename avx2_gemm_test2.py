import tvm
import numpy as np
from tvm.contrib import cblas
import os
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

target = 'llvm -mcpu=core-avx2'
ctx = tvm.context(target, 0)

REPEATS = 100

BITCODE_PATHS = [
    "gemmMxN__avx2.bc"
]

M = 512 / 24 * 24
N = 1024 / 24 * 24
K = 128


@tvm.register_func("tvm_callback_llvm_bitcode_path")
def bitcode_paths():
    global BITCODE_PATHS
    return BITCODE_PATHS


def test_gemm_blas():
    A = tvm.placeholder((M, K), name='A', dtype='float32')
    B = tvm.placeholder((N, K), name='B', dtype='float32')
    C = cblas.matmul(A, B, transb=True)
    s = tvm.create_schedule(C.op)

    def verify(target="llvm"):
        ctx = tvm.cpu(0)
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(size=(M, K)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(N, K)).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros((M, N), dtype=C.dtype), ctx)
        f(a, b, c)
        ftimer = f.time_evaluator(f.entry_name, ctx, number=REPEATS)
        tcost = ftimer(a, b, c).mean
        print("MATMUL %s: exec=%g GFLOPS" % (ctx, 2 * M * N * K / tcost / 10 ** 9))
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy().T), rtol=1e-5)
    verify()

def test_gemm_tensor_tensorize_extern_asm():
    def intrin_gemm(M, N, K):
        assert M == 4
        assert N == 24
        dtype = 'float32'
        A = tvm.placeholder((K, M), dtype=dtype, name='A')
        B = tvm.placeholder((K, N), dtype=dtype, name='B')
        k = tvm.reduce_axis((0, K), name='k')
        C = tvm.compute((M, N), lambda m, n:
                        tvm.sum(A[k, m] * B[k, n], axis=[k]), name='C')

        Ab = tvm.decl_buffer(A.shape, A.dtype,
                            name="A",
                            offset_factor=4,
                            strides=[M, 1])
        Bb = tvm.decl_buffer(B.shape, B.dtype,
                            name="B",
                            offset_factor=24,
                            strides=[N, 1])
        Cb = tvm.decl_buffer(C.shape, C.dtype,
                            name="C",
                            offset_factor=1,
                            strides=[tvm.var('ldc'), 1])

        def intrin_func(ins, outs):
            aa, bb = ins
            cc = outs[0]
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "sgemm_only_4x24__avx2",
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

        with tvm.build_config():
            return tvm.decl_tensor_intrin(C.op,
                                          intrin_func,
                                          binds={A: Ab, B: Bb, C: Cb})

    MTile = 4
    NTile = 24

    assert M % MTile == 0
    assert N % NTile == 0
    A = tvm.placeholder((M, K), name='A', dtype='float32')
    B = tvm.placeholder((N, K), name='B', dtype='float32')

    APanel = tvm.compute(
        (M / MTile, K, MTile), lambda mtile, k, m: A[m + mtile * MTile, k], name='APanel')
    BPanel = tvm.compute(
        (N / NTile, K, NTile), lambda ntile, k, n: B[n + ntile * NTile, k], name='BPanel')

    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute(
        (M, N),
        lambda m, n: tvm.sum(APanel[m / MTile, k, m % MTile] * BPanel[n / NTile, k, n % NTile], axis=[k]),
        name='C')
    s = tvm.create_schedule(C.op)

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], MTile, NTile)
    s[C].reorder(xo, yo, xi, yi)
    gemm_intrinsic_function = intrin_gemm(M=MTile, N=NTile, K=K)
    s[C].tensorize(xi, gemm_intrinsic_function)


    # one line to build the function.
    def check_device(device):
        ctx = tvm.context(device, 0)
        with tvm.target.create(device):
            print(tvm.lower(s, [A, B, C], simple_mode=True))

            f = tvm.build(s, [A, B, C])
            f.save(os.path.join(os.getcwd(), 'gemm_tensor_extern.asm'))
            f.save(os.path.join(os.getcwd(), 'gemm_tensor_extern.ll'))
            f.save(os.path.join(os.getcwd(), 'gemm_tensor_extern.o'))
        # launch the kernel.
        a_np = np.random.randn(M, K).astype(A.dtype)
        b_np = np.random.randn(N, K).astype(B.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros((M, N), dtype=C.dtype), ctx)
        f(a, b, c)
        ftimer = f.time_evaluator(f.entry_name, ctx, number=REPEATS)
        tcost = ftimer(a, b, c).mean
        print("TVM Tensorize %s: exec=%g GFLOPS" % (ctx, 2 * M * N * K / tcost / 10 ** 9))
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a_np, b_np.T), rtol=1e-4, atol=1e-4)

    check_device(target)


test_gemm_tensor_tensorize_extern_asm()
test_gemm_blas()
