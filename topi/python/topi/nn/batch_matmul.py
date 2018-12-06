"""TVM operator batch matmul compute."""
from __future__ import absolute_import
import topi
import tvm
from .. import tag


def product(l):
    p = 1
    for i in l:
        p *= i
    return p

def batch_matmul_default(A, B, trans_a, trans_b):
    """The default implementation of batched_matmul in topi.

    Parameters
    ----------
    A: tvm.Tensor
        n-D with shape [b0, ... bn, N, K]

    B: tvm.Tensor
        n-D with shape [b0, ... bn, K, M]

    trans_a: bool
        Whether transpose A[-2:] or not

    trans_b: bool
        Whether transpose B[-2:] or not

    Returns
    -------
    output: tvm.Tensor
        n-D with shape [[b0, ... bn, N, M]
    """
    assert len(A.shape) == len(B.shape), \
        "Shape mismatch between inputs"
    batch_a = A.shape[0]
    batch_b = B.shape[0]
    N = A.shape[-1] if trans_a else A.shape[-2]
    K = A.shape[-2] if trans_a else A.shape[-1]
    M = B.shape[-2] if trans_b else B.shape[-1]
    oshape = A.shape[:-2] + [N, M]
    if A.shape > 3:
        batch_a = product(A.shape[:-2])
        batch_b = product(B.shape[:-2])
        A = topi.reshape(A, [batch_a] + A.shape[-2:])
        B = topi.reshape(B, [batch_b] + B.shape[-2:])
    if trans_a:
        A = topi.transpose(A, [0, 2, 1])
    if trans_b:
        B = topi.transpose(B, [0, 2, 1])

    k = tvm.reduce_axis((0, K), name='k')
    bmm = tvm.compute((batch_a, N, M),
                      lambda b, y, x: tvm.sum(A[b, y, k] * B[b, k, x], axis=k),
                      tag='batch_matmul')
    return topi.reshape(bmm, oshape) if bmm.shape != oshape else bmm

@tvm.target.override_native_generic_func("batch_matmul")
def batch_matmul(A, B, trans_a=False, trans_b=False):
    """Applies batched matmul.

    Parameters
    ----------
    A: tvm.Tensor
        n-D with shape [b0, ... bn, N, K]

    B: tvm.Tensor
        n-D with shape [b0, ... bn, K, M]

    trans_a: bool
        Whether transpose A[-2:] or not

    trans_b: bool
        Whether transpose B[-2:] or not

    Returns
    -------
    output: tvm.Tensor
        n-D with shape [[b0, ... bn, N, M]
    """
    return batch_matmul_default(A, B, trans_a, trans_b)
