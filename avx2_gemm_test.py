import tvm
import numpy as np
from tvm.contrib import cblas
import os
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
REPEATS = 2000
RTOL = 1.0e-5
AVX2_WIDTH = 256
target = 'llvm -mcpu=core-avx2 -mattr=+popcnt'
ctx = tvm.context(target, 0)

SIZE = 256

def test_matmul_add():
    n = SIZE
    l = n
    m = n
    A = tvm.placeholder((n, l), name='A', dtype='float32')
    B = tvm.placeholder((m, l), name='B', dtype='float32')
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

    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)

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

if __name__ == "__main__":
    test_gemm()
    test_matmul_add()
