# pylint: disable=invalid-name,unused-variable
"""dense schedule on ARM Mali GPU"""

from __future__ import absolute_import as _abs

import tvm
from tvm import autotvm
import copy
from .. import generic, nn, tag
from ..util import traverse_inline, get_const_tuple, get_const_int
import numpy as np

from tvm.contrib import cblas
from .check_targets import fp32_vector_width

@autotvm.register_topi_compute(nn.dense, 'cpu', ['direct'])
def dense(cfg, data, weight, bias=None, data_layout="NI", kernel_layout="OI", out_layout=""):
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1

    assert data_layout == "NI"
    batch, in_dim = get_const_tuple(data.shape)
    if kernel_layout == "OI":
        out_dim, _ = get_const_tuple(weight.shape)
    else:
        assert kernel_layout == "IO"
        _, out_dim = get_const_tuple(weight.shape)


    cfg.define_knob('pretranspose', [1])
    if kernel_layout == "OI" and cfg['pretranspose'].val:
        import topi
        weight = topi.transpose(weight, [1, 0])
        kernel_layout = "IO"
    if cfg['pretranspose'].val:
        assert kernel_layout == "IO"

    k = tvm.reduce_axis((0, in_dim), name='k')
    cfg.define_split("tile_y", cfg.axis(out_dim), num_outputs=2, filter=lambda x: x.size[-1] % 16 == 0)
    matmul = tvm.compute(
        (batch, out_dim),
        lambda i, j: tvm.sum(
            data[i, k] * (
                weight[k, j] if kernel_layout == "IO" else weight[j, k]),
            axis=k),
        tag='dense',
        name="matmul",
    )

    if bias is not None:
        matmul = tvm.compute(
            (batch, out_dim),
            lambda i, j: matmul[i, j] + bias[j],
            name="bias_add",
            tag=tag.BROADCAST
        )
    return matmul


def schedule_dense_tvm(s, cfg, op, out):
    C = op.output(0)
    A, B = op.input_tensors
    if isinstance(B.op, tvm.tensor.ComputeOp) and autotvm.GLOBAL_SCOPE.in_tuning:
        assert cfg['pretranspose'].val == 1
        s[B].pragma(s[B].op.axis[0], "debug_skip_region")

    x, y = s[C].op.axis
    k = s[C].op.reduce_axis[0]
    (M, N) = get_const_tuple(C.shape)
    K = get_const_int(k.dom.extent)
    xa = cfg.axis(M)
    ya = cfg.axis(N)
    ka = cfg.axis(K)

    yo, yi = cfg["tile_y"].apply(s, C, y)
    s[C].reorder(x, yo, k, yi)
    s[C].vectorize(yi)
    if op != out:
        (x, y) = s[out].op.axis
        yo, yi = cfg["tile_y"].apply(s, out, y)
        s[out].vectorize(yi)
        s[C].compute_at(s[out], yo)


@autotvm.register_topi_schedule(generic.schedule_dense, 'cpu', ['direct'])
def schedule_dense(cfg, outs):
    """Schedule for dense operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config entity for this template
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'dense':
            schedule_dense_tvm(s, cfg, op, outs[0].op)
    traverse_inline(s, outs[0].op, _callback)
    return s

@nn.dense_alter_layout.register("cpu")
def dense_alter_layout(attrs, inputs, tinfo, F):
    dispatch_ctx = autotvm.task.DispatchContext.current
    target = tvm.target.current_target()
    # query schedule and fallback if necessary
    workload = autotvm.task.args_to_workload(
        [tinfo[0], tinfo[1], None, attrs['data_layout'], attrs['kernel_layout'], attrs['out_layout']], nn.dense)

    cfg = dispatch_ctx.query(target, workload)
    if cfg.is_fallback:
        return None
    pretranspose = cfg['pretranspose'].val
    if not pretranspose:
        return None

    new_attrs = {k: attrs[k] for k in attrs.keys()}
    new_attrs['kernel_layout'] = "IO"

    weights = tinfo[1]
    transposed_weights_placeholder = tvm.placeholder(
        (weights.shape[0], weights.shape[1]), dtype=weights.dtype)
    transposed_workload = autotvm.task.args_to_workload(
        [tinfo[0], transposed_weights_placeholder, None, new_attrs['data_layout'], new_attrs['kernel_layout'], new_attrs['out_layout']], nn.dense)
    transposed_cfg = copy.deepcopy(cfg)
    dispatch_ctx.update(target, transposed_workload, transposed_cfg)
    ret = F.nn.dense(*inputs, **new_attrs)
    return ret
