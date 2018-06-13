import tvm
import numpy as np
from tvm.contrib import cblas
import os
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
REPEATS = 10
RTOL = 1.0e-5
AVX2_WIDTH = 256
target = 'llvm -mcpu=core-avx2 -mattr=+popcnt'
ctx = tvm.context(target, 0)

SIZE = 256

def test_matmul_add():
    n = SIZE
    l = n
    m = n
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((m, l), name='B')
    C = cblas.matmul(A, B, transb=True)
    D = tvm.compute(C.shape, lambda i, j: C[i,j], name="D")
    s = tvm.create_schedule(D.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.cblas.matmul", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.cpu(0)
        print(tvm.lower(s, [A, B, D], simple_mode=True))
        f = tvm.build(s, [A, B, D], target)
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(m, l)).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), ctx)
        f(a, b, d)
        ftimer = f.time_evaluator(f.entry_name, ctx, number=REPEATS)
        tcost = ftimer(a, b, d).mean
        print("MATMUL %s: exec=%g GFLOPS" % (ctx, 2 * n * m * l / tcost / 10 ** 9))
        np.testing.assert_allclose(
            d.asnumpy(), np.dot(a.asnumpy(), b.asnumpy().T), rtol=1e-5)
    verify()


def intrin_gemm(M, N, K_8):
    print(M, N, K_8)
    dtype = 'float32'
    print(K_8)
    A = tvm.placeholder((K_8, M, 8), dtype=dtype, name='A')
    B = tvm.placeholder((K_8, N, 8), dtype=dtype, name='B')
    k = tvm.reduce_axis((0, K_8), name='k')
    C = tvm.compute((M, N, 8), lambda m, n, i:
                    tvm.sum(A[k, m, i] * B[k, n, i], axis=k), name='C')

    Ab = tvm.decl_buffer(A.shape, A.dtype,
                        name="A",
                        offset_factor=8,
                        strides=[tvm.var('lda'), tvm.var('ldaa'), 1])
    Bb = tvm.decl_buffer(B.shape, B.dtype,
                        name="B",
                        offset_factor=8,
                        strides=[tvm.var('ldb'), tvm.var('ldbb'), 1])
    Cb = tvm.decl_buffer(C.shape, C.dtype,
                        name="C",
                        offset_factor=8,
                        strides=[tvm.var('ldc'), tvm.var('ldcc'), 1])

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]
        args_1 = tvm.const(1, 'uint32')

        irb = tvm.ir_builder.create()
        print(K_8)
        ABregisters = [[tvm.const(0, 'float32x8') for n in range(N)] for m in range(M)]
        for k_8 in range(K_8):
            Aregisters = [Ab.vload([k_8, m, 0], 'float32x8') for m in range(M)]
            Bregisters = [Bb.vload([k_8, n, 0], 'float32x8') for n in range(N)]
            ABregisters = [[ABregisters[m][n] + Aregisters[m] * Bregisters[n] for n in range(N)] for m in range(M)]

        for m in range(M):
            for n in range(N):
                irb.emit(cc.vstore([m, n, 0], ABregisters[m][n]))
        result = irb.get()
        print(result)
        return result

    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: Ab, B: Bb, C: Cb})


def test_gemm():
    # graph
    nn = SIZE
    # n = tvm.var('n')
    n = tvm.convert(nn)
    m = n
    ll = n
    A = tvm.placeholder((n, ll), name='A', dtype='float32')
    B = tvm.placeholder((m, ll), name='B', dtype='float32')
    k = tvm.reduce_axis((0, ll), name='k')

    bn = 32
    packedB = tvm.compute((n / bn, ll, bn), lambda x, y, z: B[x * bn + z, y], name='packedB')
    C = tvm.compute((m, n),
                    lambda x, y: tvm.sum(A[x, k] * packedB[y / bn, k, y % bn], axis=k),
                    name='C')

    s = tvm.create_schedule(C.op)

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=AVX2_WIDTH / 32)

    print(xi, xo, yi, yo, ki, ko)
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)
    # s[C].prefetch(A, ki, tvm.convert(1))

    # s[packedB].compute_at(s[C], yo)
    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    # s[packedB].parallel(x)

    # lowering test
    s = s.normalize()


    # one line to build the function.
    def check_device(device):
        ctx = tvm.context(device, 0)
        with tvm.target.create(device):
            print(tvm.lower(s, [A, B, C], simple_mode=True))
            f = tvm.build(s, [A, B, C])
            f.save(os.path.join(os.getcwd(), 'gemm.asm'))
            f.save(os.path.join(os.getcwd(), 'gemm.ll'))

        # launch the kernel.
        n = nn
        m = n
        ll = n
        a_np = np.random.uniform(size=(n, ll)).astype(A.dtype)
        b_np = np.random.uniform(size=(m, ll)).astype(B.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        ftimer = f.time_evaluator(f.entry_name, ctx, number=REPEATS)
        tcost = ftimer(a, b, c).mean
        print("TVM %s: exec=%g GFLOPS" % (ctx, 2 * n * m * ll / tcost / 10 ** 9))
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a_np, b_np.T), rtol=1e-5)
    check_device(target)

def test_gemm_tensor():
    # graph
    nn = SIZE
    # n = tvm.var('n')
    n = tvm.convert(nn)
    m = n
    ll = n
    A = tvm.placeholder((n, ll), name='A', dtype='float32')
    B = tvm.placeholder((m, ll), name='B', dtype='float32')

    Apacked = tvm.compute(
        (ll / 8, m, 8), lambda k_8, m, i: A[m, k_8 * 8 + i], name='Apacked')
    Bpacked = tvm.compute(
        (ll / 8, n, 8), lambda k_8, n, i: B[n, k_8 * 8 + i], name='Bpacked')
    k_8 = tvm.reduce_axis(
        (0, ll / 8), name='k_8')
    Cpacked = tvm.compute(
        (m, n, 8),
        lambda x, y, i: tvm.sum(Apacked[k_8, x, i] * Bpacked[k_8, y, i], axis=k_8),
        name='Cpacked')
    k_inner_8 = tvm.reduce_axis((0, 8), name='k_inner_8')
    C = tvm.compute((m, n), lambda x, y: tvm.sum(Cpacked[x, y, k_inner_8], axis=k_inner_8), name="C")
    s = tvm.create_schedule(C.op)


    xo, yo, xi, yi = s[Cpacked].tile(Cpacked.op.axis[0], Cpacked.op.axis[1], 4, 4)
    # k_8, = s[Cpacked].op.reduce_axis
    # ko, ki = s[C].split(k, factor=8)

    s[Apacked].compute_at(s[Cpacked], xo)
    s[Apacked].vectorize(Apacked.op.axis[2])
    s[Bpacked].compute_at(s[Cpacked], yo)
    s[Bpacked].vectorize(Bpacked.op.axis[2])
    # # print(xi, xo, yi, yo, ki, ko)
    s[Cpacked].reorder(xo, yo, xi, yi)
    # # s[C].prefetch(A, ki, tvm.convert(1))

    gemm_intrinsic_function = intrin_gemm(M=4, N=4, K_8=SIZE / 8)
    s[Cpacked].tensorize(xi, gemm_intrinsic_function)
    # s[Cpacked].compute_at(s[C], xo)
    # lowering test
    s = s.normalize()


    # one line to build the function.
    def check_device(device):
        ctx = tvm.context(device, 0)
        with tvm.target.create(device):
            print(tvm.lower(s, [A, B, C], simple_mode=True))
            f = tvm.build(s, [A, B, C])
            f.save(os.path.join(os.getcwd(), 'gemm_tensor.asm'))
            f.save(os.path.join(os.getcwd(), 'gemm_tensor.ll'))

        # launch the kernel.
        n = nn
        m = n
        ll = n
        a_np = np.random.uniform(size=(n, ll)).astype(A.dtype)
        b_np = np.random.uniform(size=(m, ll)).astype(B.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        ftimer = f.time_evaluator(f.entry_name, ctx, number=REPEATS)
        tcost = ftimer(a, b, c).mean
        print("TVM Tensor %s: exec=%g GFLOPS" % (ctx, 2 * n * m * ll / tcost / 10 ** 9))
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a_np, b_np.T), rtol=1e-5)
    check_device(target)

if __name__ == "__main__":
    test_gemm_tensor()
    test_gemm()


    test_matmul_add()
