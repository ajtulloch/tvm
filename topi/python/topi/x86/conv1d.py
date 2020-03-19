# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,unused-argument,invalid-name
"""Conv1D schedule on for Intel CPU"""
from tvm import te
from .. import tag


def schedule_conv1d_ncw(outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])
    output_op = outs[0].op
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            else: # inject custom schedule
                if len(op.axis) == 3: # schedule bias + bn + relu
                    n, c, w = op.axis
                    fused = s[op].fuse(n, c)
                    s[op].parallel(fused)
                    s[op].vectorize(w)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv1d_ncw' in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            if isinstance(kernel.op, te.tensor.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, te.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            n_pad, c_pad, w_pad = data_pad.op.axis
            pad_fused = s[data_pad].fuse(n_pad, c_pad)
            s[data_pad].parallel(pad_fused)
            C = conv
            n, c, w = C.op.axis
            rc, rw = C.op.reduce_axis
            n_out, c_out, w_out = output_op.axis
            s[C].vectorize(w)
            if op != output_op: # fuse bias + bn + relu into conv
                s[C].compute_at(s[output_op], w_out)
            else:
                fused = s[C].fuse(n, c)
                s[C].parallel(fused)

        scheduled_ops.append(op)

    traverse(output_op)
    return s


import tvm
from tvm import te
from tvm import autotvm
from topi.nn.dilate import dilate
from topi.nn.pad import pad
from topi.util import get_const_int, get_const_tuple, simplify, traverse_inline
from topi.nn.util import get_pad_tuple1d
from topi import tag

@autotvm.register_topi_compute('conv1d_nwc.x86')
def conv1d_nwc(cfg,
               data,
               kernel,
               strides=1,
               padding='VALID',
               dilation=1,
               out_dtype=None):
    """ 1D convolution forward operator for NWC layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_width, in_channel]

    kernel : tvm.te.Tensor
        3-D with shape [filter_size, in_channel, num_filter]

    strides : int or tuple
        The spatial stride along width

    padding : int, tuple, or str
        Padding size can be an integer for equal padding,
        a tuple of (left, right) or a string in ['VALID', 'SAME'].

    dilation : int or tuple
        Dilation rate if convolution should be dilated.

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    print("Invoking conv1d_nwc compute")
    if out_dtype is None:
        out_dtype = data.dtype
    if isinstance(strides, (tuple, list)):
        strides = strides[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]

    batch, data_width, in_channels = data.shape
    kernel_size, _, out_channels = kernel.shape

    # Compute the output shape
    dilated_kernel_size = (kernel_size - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_kernel_size, ))
    out_channels = simplify(out_channels)
    out_width = simplify(
        (data_width - dilated_kernel_size + pad_left + pad_right) // strides + 1)

    # Apply padding
    pad_before = [0, pad_left, 0]
    pad_after = [0, pad_right, 0]
    temp = pad(data, pad_before, pad_after, name='data_pad')

    # Compute graph
    rc = te.reduce_axis((0, in_channels), name='rc')
    rw = te.reduce_axis((0, kernel_size), name='rw')

    return te.compute(
        (batch, out_width, out_channels),
        lambda b, w, c: te.sum(
            temp[b, w * strides + rw * dilation, rc].astype(out_dtype)
            * kernel[rw, rc, c].astype(out_dtype),
            axis=[rw, rc]),
        tag="conv1d_nwc")


@autotvm.register_topi_schedule('conv1d_nwc.x86')
def schedule_conv1d_nwc(cfg, outs):
    print("Invoking conv1d_nwc schedule")
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(cfg, s, output, data, kernel):
        n, w, c = s[output].op.axis
        rw, rc = s[output].op.reduce_axis
        cfg.define_split("tile_c", c, num_outputs=2,
                         filter=lambda x: x.size[-1] in (8, 16) or x.size[-1] == 1)
        cfg.define_split("tile_w", w, num_outputs=2, filter=lambda x: x.size[-1] <= 8)
        cfg.define_split("tile_rc", rc, num_outputs=2, filter=lambda x: x.size[-1] <= 8)

        # schedule according to config
        co, ci = cfg["tile_c"].apply(s, output, c)
        rco, rci = cfg["tile_rc"].apply(s, output, rc)
        wo, wi = cfg["tile_w"].apply(s, output, w)

        cfg.define_annotate('ann_reduce', [rci], policy='try_unroll')
        cfg.define_annotate('ann_spatial', [wi], policy='unroll')
        s[output].vectorize(ci)
        cfg['ann_reduce'].apply(s, output, [rci])
        cfg['ann_spatial'].apply(s, output, [wi])
        # s[output].reorder(n, wo, co, rco, rw, rci, wi, ci)
        s[output].reorder(n, wo, co, rw, rco, rci, wi, ci)


        if "data_pad" in data.op.name:
            data_pad = data.op
            cfg.define_knob("data_pad_compute_at", [1, 2, 3])
            (nd, wd, cd) = s[data_pad].op.axis
            cdo, cdi = cfg["tile_rc"].apply(s, data_pad, cd)
            # s[data.op].reorder(nd, wd, cdo, cdi)

            if cfg['data_pad_compute_at'].val == 0:
                s[data_pad].compute_inline()
            if cfg['data_pad_compute_at'].val == 1:
                s[data_pad].compute_at(s[output], rw)
                s[data_pad].vectorize(cdi)
                # s[data_pad].unroll(wd)
            if cfg['data_pad_compute_at'].val == 2:
                s[data_pad].compute_at(s[output], rci)
                s[data_pad].vectorize(cdi)
                # s[data_pad].unroll(wd)
            if cfg['data_pad_compute_at'].val == 3:
                s[data_pad].compute_at(s[output], co)
                s[data_pad].vectorize(cdi)
                # s[data_pad].unroll(wd)
        if output.op != outs[0].op:
            (n, w, c) = s[outs[0].op].op.axis
            co, ci = cfg["tile_c"].apply(s, outs[0].op, c)
            wo, wi = cfg["tile_w"].apply(s, outs[0].op, w)
            s[outs[0].op].reorder(n, wo, co, wi, ci)
            s[outs[0].op].vectorize(ci)
            # s[outs[0].op].unroll(wi)
            s[output.op].compute_at(s[outs[0].op], co)
        print("Scheduling conv1d_nwc_schedule properly")
        # s[data].compute_at(s[output], wo)

    def _callback(op):
        if op.tag == 'conv1d_nwc':
            output = op.output(0)
            data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            _schedule(cfg, s, output, data, kernel)

    traverse_inline(s, outs[0].op, _callback)
    return s
