"""Test code for batch_matmul operator"""
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize

from common import get_all_backend

def verify_batch_matmul(ashape, bshape, trans_a=False, trans_b=False):
    A = tvm.placeholder(ashape, name='A')
    B = tvm.placeholder(bshape, name='B')
    batch = np.prod(ashape[0:-2])
    N = ashape[-1] if trans_a else ashape[-2]
    K = ashape[-2] if trans_a else ashape[-1]
    M = bshape[-2] if trans_b else bshape[-1]
    C = tvm.placeholder(ashape[0:-2] + [N, M], name='C')
    dtype = A.dtype

    # use memoize to pickle the test data for next time use
    #@memoize("topi.tests.test_topi_batch_matmul")
    def get_ref_data():
        a_np = np.random.uniform(size=ashape).astype(dtype)
        b_np = np.random.uniform(size=bshape).astype(dtype)
        c_np = np.random.uniform(size=(batch, N, M)).astype(dtype)
        a_np_reshape = np.reshape(a_np, [batch] + ashape[-2:])
        b_np_reshape = np.reshape(b_np, [batch] + bshape[-2:])
        if trans_a:
            a_np_reshape = np.transpose(a_np_reshape, [0, 2, 1])
        if trans_b:
            b_np_reshape = np.transpose(b_np_reshape, [0, 2, 1])
        for i in range(batch):
            c_np[i, :, :] = np.matmul(a_np_reshape[i, :, :], b_np_reshape[i, :, :])
        return (a_np, b_np, np.reshape(c_np, ashape[0:-2] + [N, M]))
    # get the test data
    a_np, b_np, c_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            C = topi.nn.batch_matmul(A, B, trans_a, trans_b)
            s = topi.generic.schedule_dense(C)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B, C], device, name="batch_matmul")
        f(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)

def test_batch_matmul():
    verify_batch_matmul([4, 1024, 1000], [4, 1000, 256])
    verify_batch_matmul([4, 1000, 1024], [4, 1000, 256], trans_a=True)
    verify_batch_matmul([4, 1024, 1000], [4, 256, 1000], trans_b=True)
    verify_batch_matmul([4, 1000, 1024], [4, 256, 1000], trans_a=True, trans_b=True)
    verify_batch_matmul([2, 3, 1024, 1000], [2, 3, 1000, 256])
    verify_batch_matmul([2, 3, 1000, 1024], [2, 3, 1000, 256], trans_a=True)
    verify_batch_matmul([2, 3, 1024, 1000], [2, 3, 256, 1000], trans_b=True)
    verify_batch_matmul([2, 3, 1000, 1024], [2, 3, 256, 1000], trans_a=True, trans_b=True)

if __name__ == "__main__":
    test_batch_matmul()
