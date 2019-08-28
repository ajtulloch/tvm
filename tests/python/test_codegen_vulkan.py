import tvm
import numpy as np

tx = tvm.thread_axis("threadIdx.x")
bx = tvm.thread_axis("blockIdx.x")


def test_vulkan_copy():
    num_thread = 8
    def check_vulkan(dtype, n):
        if not tvm.vulkan(0).exist or not tvm.module.enabled("vulkan"):
            print("skip because vulkan is not enabled..")
            return
        A = tvm.placeholder((n,), name='A', dtype=dtype)
        B = tvm.compute((n,), lambda i: A[i]+tvm.const(1, A.dtype), name='B')
        s = tvm.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(xo, bx)
        s[B].bind(xi, tx)
        fun = tvm.build(s, [A, B], "vulkan")
        ctx = tvm.vulkan(0)
        a_np = np.random.uniform(size=(n,)).astype(A.dtype)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(a_np)
        b_np = a.asnumpy()
        tvm.testing.assert_allclose(a_np, b_np)
        tvm.testing.assert_allclose(a_np, a.asnumpy())

    for _ in range(100):
        dtype = np.random.choice(["float32", "float16", "int8", "int32"])
        logN = np.random.randint(1, 15)
        peturb = np.random.uniform(low=0.5, high=1.5)
        check_vulkan(dtype, int(peturb * (2 ** logN)))


def test_vulkan_vectorize_add():
    num_thread = 8
    def check_vulkan(dtype, n, lanes):
        if not tvm.vulkan(0).exist or not tvm.module.enabled("vulkan"):
            print("skip because vulkan is not enabled..")
            return
        A = tvm.placeholder((n,), name='A', dtype="%sx%d" % (dtype, lanes))
        B = tvm.compute((n,), lambda i: A[i]+tvm.const(1, A.dtype), name='B')
        s = tvm.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(xo, bx)
        s[B].bind(xi, tx)
        fun = tvm.build(s, [A, B], "vulkan")
        ctx = tvm.vulkan(0)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(
            np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), B.dtype, ctx)
        fun(a, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1)

    check_vulkan("float32", 64, 2)
    check_vulkan("float16", 64, 2)

def test_vulkan_performance():
    num_thread = 32
    def check_vulkan(dtype, n, lanes):
        if not tvm.vulkan(0).exist or not tvm.module.enabled("vulkan"):
            print("skip because vulkan is not enabled..")
            return
        A = tvm.placeholder((n,), name='A', dtype="%sx%d" % (dtype, lanes))
        B = tvm.compute((n,), lambda i: A[i]+tvm.const(1, A.dtype), name='B')
        s = tvm.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(xo, bx)
        s[B].bind(xi, tx)
        fun = tvm.build(s, [A, B], "vulkan")
        ctx = tvm.vulkan(0)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(
            np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), B.dtype, ctx)
        fun(a, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1)
        te = fun.time_evaluator(fun.entry_name, ctx=ctx, min_repeat_ms=500, number=5)
        for _ in range(3):
            print(f"Time: {te(a, c).mean * 1.0e6:.2f}us")

    check_vulkan("float32", 64, 2)
    check_vulkan("float32", 1024, 2)
    check_vulkan("float32", 2048, 2)
    check_vulkan("float32", 1024 * 1024, 4)


def test_vulkan_stress():
    """
    Launch a randomized test with multiple kernels per stream, multiple uses of
    kernels per stream, over multiple threads.
    """
    import random
    n = 1024
    num_thread = 64
    def run():
        if not tvm.vulkan(0).exist or not tvm.module.enabled("vulkan"):
            print("skip because vulkan is not enabled..")
            return
        A = tvm.placeholder((n,), name='A', dtype="float32")
        B = tvm.placeholder((n,), name='B', dtype="float32")
        functions = [
            (lambda: tvm.compute((n,), lambda i: 2 * A[i] + 3 * B[i], name='B'), lambda a, b: 2 * a + 3 * b),
            (lambda: tvm.compute((n,), lambda i: A[i]+B[i], name='B'), lambda a, b: a + b),
            (lambda: tvm.compute((n,), lambda i: A[i]+2 * B[i], name='B'), lambda a, b: a + 2 * b),
        ]

        def build_f(f_ref):
            (C_f, ref) = f_ref
            C = C_f()
            s = tvm.create_schedule(C.op)
            xo, xi = s[C].split(C.op.axis[0], factor=num_thread)
            s[C].bind(xo, bx)
            s[C].bind(xi, tx)
            fun = tvm.build(s, [A, B, C], "vulkan")
            return (fun, ref)

        fs = [build_f(random.choice(functions)) for _ in range(np.random.randint(low=1, high=10))]
        ctx = tvm.vulkan(0)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(
            np.random.uniform(size=(n,)))
        b = tvm.nd.empty((n,), B.dtype, ctx).copyfrom(
            np.random.uniform(size=(n,)))
        cs = [tvm.nd.empty((n,), A.dtype, ctx) for _ in fs]
        for ((f, _), c) in zip(fs, cs):
            f(a, b, c)

        for ((_, ref), c) in zip(fs, cs):
            tvm.testing.assert_allclose(c.asnumpy(), ref(a.asnumpy(), b.asnumpy()))
    run()

    import threading
    ts = [threading.Thread(target=run) for _ in range(np.random.randint(1, 10))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
