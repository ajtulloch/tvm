"""Elementwise operators"""
from __future__ import absolute_import
import tvm
from .. import tag

@tvm.target.generic_func
def sparse_dense(data, weight_data, weight_indices, weight_indptr):
    import topi
    assert topi.util.get_const_tuple(data.shape)[0] == 1
    oshape = (topi.util.get_const_tuple(data.shape)[0], topi.util.get_const_tuple(weight_indptr.shape)[0] - 1)
    assert weight_indices.dtype == "int32", weight_indices.dtype
    assert weight_indptr.dtype == "int32", weight_indptr.dtype
    def f(i, row):
        assert row.dtype == "int32"
        row_start = weight_indptr[row]
        row_end = weight_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem].astype("float32")
        weight_val = data[i, weight_indices[elem]]
        return tvm.sum(a_val * weight_val, axis=elem_idx)
    return tvm.compute(oshape, f, name="sparse_dense", tag="sparse_dense")
