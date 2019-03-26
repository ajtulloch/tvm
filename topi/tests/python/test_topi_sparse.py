"""Test code for sparse operator"""
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_tuple
import tvm.contrib.sparse as tvmsp
from collections import namedtuple
import time

def verify_dynamic_csrmv(batch, in_dim, out_dim, use_bias=True):
    nr, nc, n = tvm.var("nr"), tvm.var("nc"), tvm.var("n")
    dtype = 'float32'
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, dtype=dtype, name='A')
    B = tvm.placeholder((in_dim, 1), name='B')
    C = tvm.placeholder((nr,), name='C')
    D = topi.sparse.csrmv(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)
    dtype = A.dtype

    # get the test data
    def get_ref_data():
        a_np = np.maximum(np.random.uniform(size=(batch, in_dim)).astype(dtype)-0.5, 0.)
        b_np = np.random.uniform(size=(in_dim, 1)).astype(dtype)-0.5
        c_np = np.random.uniform(size=(batch, )).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np) + c_np.reshape((batch, 1))
        else:
            d_np = np.dot(a_np, b_np)
        return (a_np, b_np, c_np, d_np)
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvmsp.array(a_np, ctx)
        _nr, _nc, _n = a.shape[0], a.shape[1], a.data.shape[0]
        assert a.shape[0] == a.indptr.shape[0]-1
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros((_nr, 1), dtype=dtype), ctx)
        assert a.data.dtype == A.data.dtype
        assert a.indices.dtype == A.indices.dtype
        assert a.indptr.dtype == A.indptr.dtype
        f = tvm.build(s, [nr, A.data, A.indices, A.indptr, B, C, D], device, name="csrmv")
        f(_nr, a.data, a.indices, a.indptr, b, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-4, atol=1e-4)

    for device in ["llvm"]:
        check_device(device)

def verify_dynamic_csrmm(batch, in_dim, out_dim, use_bias=True):
    nr, nc, n = tvm.var("nr"), tvm.var("nc"), tvm.var("n")
    dtype = 'float32'
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, dtype=dtype, name='A')
    B = tvm.placeholder((in_dim, out_dim), name='B')
    C = tvm.placeholder((nr,), name='C')
    D = topi.sparse.csrmm(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)
    dtype = A.dtype

    # get the test data
    def get_ref_data():
        a_np = np.maximum(np.random.uniform(size=(batch, in_dim)).astype(dtype)-0.5, 0.)
        b_np = np.random.uniform(size=(in_dim, out_dim)).astype(dtype)-0.5
        c_np = np.random.uniform(size=(batch, )).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np) + c_np.reshape((batch, 1))
        else:
            d_np = np.dot(a_np, b_np)
        return (a_np, b_np, c_np, d_np)
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvmsp.array(a_np, ctx)
        _nr, _nc, _n = a.shape[0], a.shape[1], a.data.shape[0]
        assert a.shape[0] == a.indptr.shape[0]-1
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros((_nr, out_dim), dtype=dtype), ctx)
        f = tvm.build(s, [nr, A.data, A.indices, A.indptr, B, C, D], device, name="csrmm")

        f(_nr, a.data, a.indices, a.indptr, b, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-2, atol=1e-2)

    for device in ["llvm"]:
        check_device(device)

def verify_dense_si(batch, in_dim, out_dim, use_bias=True, dtype='float32'):
    nonzeros = tvm.var('nonzeros')
    A = tvmsp.placeholder(shape=(batch, in_dim), nonzeros=nonzeros, dtype=dtype, name='A')
    B = tvm.placeholder((out_dim, in_dim), dtype=dtype, name='B')
    C = tvm.placeholder((out_dim,), dtype=dtype, name='C')
    D = topi.sparse.dense(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)

    # get the test data
    def get_ref_data():
        mag = 10.
        a_np = np.maximum(mag*(np.random.uniform(size=(batch, in_dim)).astype('float32')-0.5), 0.).astype(dtype)
        b_np = (mag*(np.random.uniform(size=(out_dim, in_dim)).astype('float32')-.5)).astype(dtype)
        c_np = (mag*(np.random.uniform(size=(out_dim,)).astype('float32')-.5)).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np.T) + c_np
        else:
            d_np = np.dot(a_np, b_np.T)
        return (a_np, b_np, c_np, d_np)
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvmsp.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A.data, A.indices, A.indptr, B, C, D], device, name="dense")
        f(a.data, a.indices, a.indptr, b, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-4, atol=1e-4)

    check_device('llvm')

def verify_dense_sw(batch, in_dim, out_dim, use_bias=True, dtype='float32'):
    nonzeros = tvm.var('nonzeros')
    A = tvm.placeholder((batch, in_dim), dtype=dtype, name='A')
    B = tvmsp.placeholder(shape=(out_dim, in_dim), nonzeros=nonzeros, dtype=dtype, name='B')
    C = tvm.placeholder((out_dim,), dtype=dtype, name='C')
    D = topi.sparse.dense(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)

    # get the test data
    def get_ref_data():
        mag = 10.
        a_np = (mag*(np.random.uniform(size=(batch, in_dim)).astype('float32')-.5)).astype(dtype)
        b_np = np.maximum(mag*(np.random.uniform(size=(out_dim, in_dim)).astype('float32')-0.5), 0.).astype(dtype)
        c_np = (mag*(np.random.uniform(size=(out_dim,)).astype('float32')-.5)).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np.T) + c_np
        else:
            d_np = np.dot(a_np, b_np.T)
        return (a_np, b_np, c_np, d_np)
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvm.nd.array(a_np, ctx)
        b = tvmsp.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B.data, B.indices, B.indptr, C, D], device, name="dense")
        f(a, b.data, b.indices, b.indptr, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-4, atol=1e-4)

    check_device('llvm')

def test_csrmv():
    verify_dynamic_csrmv(batch=5, in_dim=7, out_dim=1, use_bias=False)
    verify_dynamic_csrmv(batch=5, in_dim=7, out_dim=1, use_bias=True)

def test_csrmm():
    M, K, N = 5, 7, 2
    verify_dynamic_csrmm(batch=M, in_dim=K, out_dim=N, use_bias=False)
    verify_dynamic_csrmm(batch=M, in_dim=K, out_dim=N, use_bias=True)

def test_dense_si():
    M, K, N = 3, 5, 2
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='float32')
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='float32')
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='int32')
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='int32')
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='int16')
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='int16')

def test_dense_sw():
    M, K, N = 3, 5, 2
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='float32')
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='float32')
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='int32')
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='int32')
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='int16')
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='int16')

def test_dense():
    test_dense_si()
    test_dense_sw()


def test_sparse_dense_csr():
    import scipy.sparse as sp
    M, N, K, density = 1, 17, 47, 0.2
    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = sp.random(N, K, density=density, format='csr', dtype="float32")
    W_np = W_sp_np.todense()
    Y_np = X_np.dot(W_np.T)

    W_data = tvm.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
    W_indices = tvm.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
    W_indptr = tvm.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
    X = tvm.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))
    Y = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr)
    s = tvm.create_schedule(Y.op)
    func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
    Y_tvm = tvm.ndarray.array(np.zeros(Y_np.shape, dtype=Y_np.dtype))
    func(tvm.ndarray.array(X_np), tvm.ndarray.array(W_sp_np.data), tvm.ndarray.array(W_sp_np.indices), tvm.ndarray.array(W_sp_np.indptr), Y_tvm)
    tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-4, rtol=1e-4)


def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype):
    import scipy.sparse as sp
    import itertools
    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r:r + BS_R, c:c + BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.indices.shape == (num_blocks, )
    assert s.indptr.shape == (M // BS_R + 1, )
    return s

def to_bf16(x):
    assert x.dtype == np.float32
    return ((x.view('<u4') + 2 ** 15) >> 16).astype("uint16")

def from_bf16(x):
    assert x.dtype == np.uint16
    return (x.astype("uint32") << 16).view('<f4')


def test_sparse_dense_bsr():
    M, N, K, BS_R, BS_C, density = 1, 64, 128, 8, 16, 0.9
    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
    W_np = W_sp_np.todense()
    Y_np = X_np.dot(W_np.T)

    W_data = tvm.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
    W_indices = tvm.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
    W_indptr = tvm.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
    X = tvm.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))
    Y = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr)
    s = tvm.create_schedule(Y.op)
    func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
    Y_tvm = tvm.ndarray.array(np.zeros(Y_np.shape, dtype=Y_np.dtype))
    func(tvm.ndarray.array(X_np), tvm.ndarray.array(W_sp_np.data), tvm.ndarray.array(W_sp_np.indices), tvm.ndarray.array(W_sp_np.indptr), Y_tvm)
    tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-4, rtol=1e-4)

def test_bf16():
    np.random.seed(42)

    x = np.zeros((10,)).astype(np.float32)
    x_rt = from_bf16(to_bf16(x))
    np.testing.assert_equal(x, x_rt)

    x = np.random.randn(50000).astype(np.float32)
    x_rt = from_bf16(to_bf16(x))
    abs_x = np.max(np.abs(x - x_rt))
    rel_x = np.max(np.abs((x - x_rt) / x))
    print(abs_x, rel_x, x[np.argmax(np.abs(x - x_rt))], x_rt[np.argmax(np.abs(x - x_rt))])
    np.testing.assert_allclose(x, x_rt, rtol=1e-2, atol=1e-2)

def test_sparse_dense_bsr_bf16():
    M, N, K, BS_R, BS_C, density = 1, 64, 128, 2, 2, 0.99
    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
    W_np = W_sp_np.todense()
    W_sp_bf16 = to_bf16(W_sp_np.data)
    W_sp_np.data = from_bf16(to_bf16(W_sp_np.data))

    Y_np = X_np.dot(W_np.T)
    Y_np_bf16 = X_np.dot(W_sp_np.todense().T)

    def at_argmax(x, cond):
        return x[np.unravel_index(np.argmax(cond, axis=None), x.shape)]

    tvm.testing.assert_allclose(from_bf16(to_bf16(W_sp_np.data)), W_sp_np.data, atol=1e-2, rtol=1e-2)
    print(at_argmax(Y_np, np.abs(Y_np - Y_np_bf16)), at_argmax(Y_np_bf16, np.abs(Y_np - Y_np_bf16)))
    print(at_argmax(Y_np, np.abs((Y_np - Y_np_bf16) / Y_np)), at_argmax(Y_np_bf16, np.abs((Y_np - Y_np_bf16) / Y_np)))
    tvm.testing.assert_allclose(Y_np_bf16, Y_np, atol=1e-1, rtol=1e-2)

    W_data = tvm.placeholder(shape=W_sp_bf16.shape, dtype=str(W_sp_bf16.dtype))
    W_indices = tvm.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
    W_indptr = tvm.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
    X = tvm.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))
    Y = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr)
    s = tvm.create_schedule(Y.op)
    func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
    Y_tvm = tvm.ndarray.array(np.zeros(Y_np.shape, dtype=Y_np.dtype))
    func(tvm.ndarray.array(X_np), tvm.ndarray.array(W_sp_bf16), tvm.ndarray.array(W_sp_np.indices), tvm.ndarray.array(W_sp_np.indptr), Y_tvm)

    tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np_bf16, atol=1e-2, rtol=1e-2)


def test_sparse_dense():
    test_sparse_dense_csr()
    test_sparse_dense_bsr()
    test_sparse_dense_bsr_bf16()
    test_bf16()

if __name__ == "__main__":
    test_csrmv()
    test_csrmm()
    test_dense()
    test_sparse_dense()
