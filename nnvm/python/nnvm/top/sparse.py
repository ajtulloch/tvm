# pylint: disable=invalid-name, unused-argument
"""Definition of sparse ops"""
from __future__ import absolute_import

import tvm
import topi
from topi.util import get_const_int, get_const_tuple
from .tensor import _fschedule_broadcast, _fschedule_injective
from . import registry as reg
from .registry import OpPattern

# dense
@reg.register_compute("sparse_lengths_sum")
def compute_sparse_lengths_sum(_, inputs, __):
    return topi.sparse.sparse_lengths_sum(inputs[0], inputs[1], inputs[2])

@reg.register_schedule("sparse_lengths_sum")
def schedule_sparse_lengths_sum(_, outs, target):
    """Schedule definition of dense"""
    with tvm.target.create(target):
        return topi.generic.schedule_sparse_lengths_sum(outs)

reg.register_pattern("sparse_lengths_sum", OpPattern.OUT_ELEMWISE_FUSABLE)

# dense
@reg.register_compute("sparse_lengths_sum_fused_8bit_rowwise")
def compute_sparse_lengths_sum_fused_8bit_rowwise(_, inputs, __):
    return topi.sparse.sparse_lengths_sum_fused_8_bit_rowwise(inputs[0], inputs[1], inputs[2])

@reg.register_schedule("sparse_lengths_sum_fused_8bit_rowwise")
def schedule_sparse_lengths_sum_fused_8bit_rowwise(_, outs, target):
    """Schedule definition of dense"""
    with tvm.target.create(target):
        return topi.generic.schedule_extern(outs)

reg.register_pattern("sparse_lengths_sum_fused_8bit_rowwise", OpPattern.OUT_ELEMWISE_FUSABLE)

@reg.register_compute("batch_matmul")
def compute_batch_matmul(attrs, inputs, _):
    """Compute definition of dense"""
    return topi.nn.batch_matmul(
        inputs[0], inputs[1],
        trans_a=attrs.get_bool("trans_a"),
        trans_b=attrs.get_bool("trans_b"))

@reg.register_schedule("batch_matmul")
def schedule_batch_matmul(_, outs, target):
    """Schedule definition of dense"""
    with tvm.target.create(target):
        return topi.generic.schedule_batch_matmul(outs)

reg.register_pattern("batch_matmul", OpPattern.OUT_ELEMWISE_FUSABLE)

# gather_nd
reg.register_pattern("batch_gather_nd", OpPattern.INJECTIVE)
reg.register_schedule("batch_gather_nd", _fschedule_injective)
