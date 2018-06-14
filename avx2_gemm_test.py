import tvm
import numpy as np
from tvm.contrib import cblas
import os
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
REPEATS = 1000
RTOL = 1.0e-5
AVX2_WIDTH = 256
target = 'llvm -mcpu=core-avx2 -mattr=+popcnt'
ctx = tvm.context(target, 0)

SIZE = 256

BITCODE_PATHS = [
    "gemmMxN__avx2.bc"
]
@tvm.register_func("tvm_callback_llvm_bitcode_path")
def bitcode_paths():
    global BITCODE_PATHS
    return BITCODE_PATHS

def test_matmul_gemm():
    n = SIZE
    l = n
    m = n
    A = tvm.placeholder((n, l), name='A', dtype='float32')
    B = tvm.placeholder((m, l), name='B', dtype='float32')
    bn = 32
    k = tvm.reduce_axis((0, l), 'k')
    packedB = tvm.compute((n / bn, l, bn), lambda x, y, z: B[x * bn + z, y], name='packedB')

    C = tvm.compute((m, n),
                    lambda x, y: tvm.sum(A[x, k] * packedB[y / bn, k, y % bn], axis=k),
                    name = 'C')
    s = tvm.create_schedule(C.op)

    # Allocate write cache
    CC = s.cache_write(C, 'global')

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

    # Write cache is computed at yo
    s[CC].compute_at(s[C], yo)

    # New inner axes
    xc, yc = s[CC].op.axis

    k, = s[CC].op.reduce_axis
    ko, ki = s[CC].split(k, factor=8)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)

    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(y)


    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.cblas.matmul", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.cpu(0)
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(m, l)).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        f(a, b, c)
        ftimer = f.time_evaluator(f.entry_name, ctx, number=REPEATS)
        tcost = ftimer(a, b, c).mean
        print("MATMUL %s: exec=%g GFLOPS" % (ctx, 2 * n * m * l / tcost / 10 ** 9))
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy().T), rtol=1e-5)
    verify()

def test_matmul_add():
    n = SIZE
    l = n
    m = n
    A = tvm.placeholder((n, l), name='A', dtype='float32')
    B = tvm.placeholder((m, l), name='B', dtype='float32')
    C = cblas.matmul(A, B, transb=True)
    s = tvm.create_schedule(C.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.cblas.matmul", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.cpu(0)
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(m, l)).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        f(a, b, c)
        ftimer = f.time_evaluator(f.entry_name, ctx, number=REPEATS)
        tcost = ftimer(a, b, c).mean
        print("MATMUL %s: exec=%g GFLOPS" % (ctx, 2 * n * m * l / tcost / 10 ** 9))
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy().T), rtol=1e-5)
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
            for m in range(M):
                for n in range(N):
                    ABregisters[m][n] = ABregisters[m][n] + Aregisters[m] * Bregisters[n]

        for m in range(M):
            for n in range(N):
                irb.emit(cc.vstore([m, n, 0], ABregisters[m][n]))
        result = irb.get()
        print("IR result: ")
        print(result)
        # print(result.dtype)
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
    n = nn # tvm.convert(nn)
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


    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 4, 2)
    # k_8, = s[Cpacked].op.reduce_axis
    # ko, ki = s[C].split(k, factor=8)

    # s[Apacked].compute_at(s[C], xo)
    # s[Apacked].vectorize(Apacked.op.axis[2])
    # s[Bpacked].compute_at(s[C], yo)
    # s[Bpacked].vectorize(Bpacked.op.axis[2])
    # # print(xi, xo, yi, yo, ki, ko)
    s[C].reorder(xo, yo, xi, yi)
    s[C].vectorize(yi)
    s[Apacked].compute_at(s[C], xo)
    s[Bpacked].compute_at(s[C], yo)
    s[Apacked].vectorize(Apacked.op.axis[2])
    s[Bpacked].vectorize(Bpacked.op.axis[2])
    s[Bpacked].unroll(Bpacked.op.axis[1])
    s[Apacked].unroll(Apacked.op.axis[1])
    # # s[C].prefetch(A, ki, tvm.convert(1))
    s[Cpacked].compute_at(s[C], yo)
    gemm_intrinsic_function = intrin_gemm(M=4, N=2, K_8=SIZE / 8)
    # s[Cpacked].compute_at(s[C], xo)
    xo, yo, xi, yi = s[Cpacked].tile(Cpacked.op.axis[0], Cpacked.op.axis[1], 4, 2)
    s[Cpacked].tensorize(yo, gemm_intrinsic_function)

    # lowering test
    s = s.normalize()


    # one line to build the function.
    def check_device(device):
        ctx = tvm.context(device, 0)
        with tvm.target.create(device):
            print(tvm.lower(s, [A, B, C], simple_mode=True))
            print(tvm.lower(s, [A, B, C], simple_mode=False))
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
        print("TVM Tensorize %s: exec=%g GFLOPS" % (ctx, 2 * n * m * ll / tcost / 10 ** 9))
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a_np, b_np.T), rtol=1e-5)
    check_device(target)

def test_gemm_tensor_no_tensorize():
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
    # s[Cpacked].compute_at(s[C], xo)
    # lowering test
    s = s.normalize()


    # one line to build the function.
    def check_device(device):
        ctx = tvm.context(device, 0)
        with tvm.target.create(device):
            print(tvm.lower(s, [A, B, C], simple_mode=True))
            f = tvm.build(s, [A, B, C])
            f.save(os.path.join(os.getcwd(), 'gemm_notensor.asm'))
            f.save(os.path.join(os.getcwd(), 'gemm_notensor.ll'))

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
        print("TVM Tensor No Tensorize %s: exec=%g GFLOPS" % (ctx, 2 * n * m * ll / tcost / 10 ** 9))
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a_np, b_np.T), rtol=1e-5)
    check_device(target)

def test_gemm_tensor_tensorize_extern():
    def intrin_gemm(M, N, K_8):
        print(M, N, K_8)
        dtype = 'float32'
        print(K_8)
        A = tvm.placeholder((K_8, M, 8), dtype=dtype, name='A')
        B = tvm.placeholder((K_8, N, 8), dtype=dtype, name='B')
        k = tvm.reduce_axis((0, K_8), name='k')
        k_inner_8 = tvm.reduce_axis((0, 8), name='k_inner_8')
        C = tvm.compute((M, N), lambda m, n:
                        tvm.sum(A[k, m, k_inner_8] * B[k, n, k_inner_8], axis=[k, k_inner_8]), name='C')

        Ab = tvm.decl_buffer(A.shape, A.dtype,
                            name="A",
                            offset_factor=1,
                            strides=[tvm.var('lda'), 8, 1])
        Bb = tvm.decl_buffer(B.shape, B.dtype,
                            name="B",
                            offset_factor=1,
                            strides=[tvm.var('ldb'), 8, 1])
        Cb = tvm.decl_buffer(C.shape, C.dtype,
                            name="C",
                            offset_factor=1,
                            strides=[tvm.var('ldc'), 1])

        def intrin_func(ins, outs):
            aa, bb = ins
            cc = outs[0]
            irb = tvm.ir_builder.create()
            print(bb.shape)
            print(bb.strides)
            extern_call = tvm.call_extern(
                "int32",
                "gemmMxN__avx2",
                aa.shape[1],
                bb.shape[1],
                aa.shape[0],
                irb.buffer_ptr(aa),
                aa.strides[0],
                irb.buffer_ptr(bb),
                bb.strides[0],
                irb.buffer_ptr(cc),
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()

        with tvm.build_config(offset_factor=16, data_alignment=16):
            return tvm.decl_tensor_intrin(C.op, intrin_func, binds={A: Ab, B: Bb, C: Cb})

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
    k_inner_8 = tvm.reduce_axis(
        (0, 8), name='k_inner_8')
    C = tvm.compute(
        (m, n),
        lambda x, y: tvm.sum(Apacked[k_8, x, k_inner_8] * Bpacked[k_8, y, k_inner_8], axis=[k_8, k_inner_8]),
        name='Cpacked')
    s = tvm.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 4, 4)
    s[C].reorder(xo, yo, xi, yi)
    gemm_intrinsic_function = intrin_gemm(M=4, N=4, K_8=SIZE / 8)
    s[C].tensorize(xi, gemm_intrinsic_function)
    # s[C].vectorize(yi)
    # s[Apacked].compute_at(s[C], xo)
    # s[Bpacked].compute_at(s[C], yo)
    # s[Apacked].vectorize(Apacked.op.axis[2])
    # s[Bpacked].vectorize(Bpacked.op.axis[2])
    # s[Bpacked].unroll(Bpacked.op.axis[1])
    # s[Apacked].unroll(Apacked.op.axis[1])
    # # s[C].prefetch(A, ki, tvm.convert(1))

    # s[Cpacked].compute_at(s[C], xo)
    # s = s.normalize()


    # one line to build the function.
    def check_device(device):
        ctx = tvm.context(device, 0)
        with tvm.target.create(device):
            print("simple_mode=True")
            print(tvm.lower(s, [A, B, C], simple_mode=True))
            print("simple_mode=False")
            print(tvm.lower(s, [A, B, C], simple_mode=False))
            print("Buildling the kernel")
            f = tvm.build(s, [A, B, C])
            f.save(os.path.join(os.getcwd(), 'gemm_tensor_extern.asm'))
            f.save(os.path.join(os.getcwd(), 'gemm_tensor_extern.ll'))
            f.save(os.path.join(os.getcwd(), 'gemm_tensor_extern.o'))
        print("Build the kernel")
        # launch the kernel.
        n = nn
        m = n
        ll = n
        a_np = np.random.uniform(size=(n, ll)).astype(A.dtype)
        b_np = np.random.uniform(size=(m, ll)).astype(B.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        print("Trying to invoke F")
        f(a, b, c)

        ftimer = f.time_evaluator(f.entry_name, ctx, number=REPEATS)
        print("ftimer worked")
        tcost = ftimer(a, b, c).mean
        print("TVM Tensor No Tensorize %s: exec=%g GFLOPS" % (ctx, 2 * n * m * ll / tcost / 10 ** 9))
        print(c.asnumpy().sum())
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a_np, b_np.T), rtol=1e-5)
    check_device(target)


def intrin_vadd_16():
    x = tvm.placeholder((16,), name='vx')
    y = tvm.placeholder((16,), name='vy')
    z = tvm.compute(x.shape, lambda i: x[i] + y[i], name='z')
    def intrin_func(ins, outs):
        xx, yy = ins
        zz = outs[0]
        irb = tvm.ir_builder.create()
        extern_call = tvm.call_extern(
            "int32",
            "vadd_16_avx2",
            irb.buffer_ptr(xx),
            xx.elem_offset,
            irb.buffer_ptr(yy),
            yy.elem_offset,
            irb.buffer_ptr(zz),
            zz.elem_offset)
        irb.emit(extern_call)
        return irb.get()
        return irbb

    with tvm.build_config(offset_factor=16):
        return tvm.decl_tensor_intrin(z.op, intrin_func)

def test_tensorize_vadd():
    m = 2048
    x = tvm.placeholder((m,), name='x')
    y = tvm.placeholder((m,), name='y')
    z = tvm.compute(x.shape, lambda i: x[i] + y[i], name='z')

    def check():
        s = tvm.create_schedule(z.op)
        xo, xi = s[z].split(z.op.axis[0], factor=16)
        vadd = intrin_vadd_16()
        s[z].tensorize(xi, vadd)
        s = s.normalize()
        dom_map = tvm.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[z], dom_map)
        assert tvm.ir_pass.Equal(out_dom[z.op.axis[0]].extent, 16)
        assert tvm.ir_pass.Equal(out_dom[z.op.axis[0]].min, xo * 16)
        assert tvm.ir_pass.Equal(in_dom.items()[0][1][0].extent, 16)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[z], out_dom, in_dom, vadd)
        assert tvm.ir_pass.Equal(tvm.ir_pass.CanonicalSimplify(body[0]),
                                 tvm.ir_pass.CanonicalSimplify(vadd.op.body[0]))
        stmt = tvm.schedule.ScheduleOps(s, dom_map)
        print(tvm.lower(s, [x, y, z], simple_mode=True))
        f = tvm.build(s, [x, y, z], target)
        f.save("fadd.o")
        f.save("fadd.ll")
        f.save("fadd.asm")
        xx = tvm.nd.array(np.random.uniform(size=(m,)).astype('float32'), ctx)
        yy = tvm.nd.array(np.random.uniform(size=(m,)).astype('float32'), ctx)
        zz = tvm.nd.array(np.zeros((m,), dtype='float32'), ctx)
        f(xx, yy, zz)
        ftimer = f.time_evaluator(f.entry_name, ctx, number=REPEATS)
        tcost = ftimer(xx, yy, zz).mean
        print("TVM VADD %s: exec=%g GFLOPS" % (ctx, 2 * m / tcost / 10 ** 9))

        np.testing.assert_allclose(zz.asnumpy(), xx.asnumpy() + yy.asnumpy(), rtol=1e-5)

    check()

def test_no_tensorize_vadd():
    m = 2048
    x = tvm.placeholder((m,), name='x')
    y = tvm.placeholder((m,), name='y')
    z = tvm.compute(x.shape, lambda i: x[i] + y[i], name='z')

    def check():
        s = tvm.create_schedule(z.op)
        xo, xi = s[z].split(z.op.axis[0], factor=16)
        s[z].vectorize(xi)
        s = s.normalize()

        print(tvm.lower(s, [x, y, z], simple_mode=True))
        f = tvm.build(s, [x, y, z], target)
        f.save("fadd_notensor.o")
        f.save("fadd_notensor.ll")
        f.save("fadd_notensor.asm")
        xx = tvm.nd.array(np.random.uniform(size=(m,)).astype('float32'), ctx)
        yy = tvm.nd.array(np.random.uniform(size=(m,)).astype('float32'), ctx)
        zz = tvm.nd.array(np.zeros((m,), dtype='float32'), ctx)
        f(xx, yy, zz)
        ftimer = f.time_evaluator(f.entry_name, ctx, number=REPEATS)
        tcost = ftimer(xx, yy, zz).mean
        print("TVM VADD %s: exec=%g GFLOPS" % (ctx, 2 * m / tcost / 10 ** 9))

        np.testing.assert_allclose(zz.asnumpy(), xx.asnumpy() + yy.asnumpy(), rtol=1e-5)

    check()


if __name__ == "__main__":
    test_matmul_add()
    test_matmul_gemm()

    # test_tensorize_vadd()
    # test_no_tensorize_vadd()
    # test_gemm_tensor()
    # test_gemm_tensor_tensorize_extern()
    # test_gemm_tensor_no_tensorize()
    # test_gemm()
    # test_matmul_add()
