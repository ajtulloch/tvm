"""External function interface to BLAS libraries."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin

def matmul(lhs, rhs, transa=False, transb=False, **kwargs):
    """Create an extern op that compute matrix mult of A and rhs with CrhsLAS

    This function serves as an example on how to call external libraries.

    Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs

    Returns
    -------
    C : Tensor
        The result tensor.
    """
    n = lhs.shape[1] if transa else lhs.shape[0]
    m = rhs.shape[0] if transb else rhs.shape[1]
    return _api.extern(
        (n, m), [lhs, rhs],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.cblas.matmul",
            ins[0], ins[1], outs[0], transa, transb), name="C", **kwargs)

def batch_matmul(lhs, rhs, transa=False, transb=False, iterative=False, **kwargs):
    """Create an extern op that compute batched matrix mult of A and rhs with CBLAS
     This function serves as an example on how to call external libraries.
     Parameters
    ----------
    lhs : Tensor
        The left matrix operand
    rhs : Tensor
        The right matrix operand
    transa : bool
        Whether transpose lhs
    transb : bool
        Whether transpose rhs
     Returns
    -------
    C : Tensor
        The result tensor.
    """
    b = lhs.shape[0]
    n = lhs.shape[2] if transa else lhs.shape[1]
    m = rhs.shape[1] if transb else rhs.shape[2]
    return _api.extern(
        (b, n, m), [lhs, rhs],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.cblas.batch_matmul" if not iterative
            else "tvm.contrib.cblas.batch_matmul_iterative",
            ins[0], ins[1], outs[0], transa, transb), name="C", **kwargs)
