# pylint: disable=invalid-name
"""generic declaration and schedules."""
from __future__ import absolute_import as _abs

import tvm

@tvm.target.generic_func
def schedule_sparse_lengths_sum(outs):
    """Schedule for sparse length sum op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    s = tvm.create_schedule([x.op for x in outs])
    return s
