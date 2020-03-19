"""Schedule for bitserial dense operator."""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from tvm import autotvm
from topi.nn.dilate import dilate
from topi.nn.pad import pad
from topi.util import get_const_int, get_const_tuple, simplify, traverse_inline
from topi.nn.util import get_pad_tuple1d
from topi import tag, generic


@autotvm.register_topi_compute('conv1d_transpose_nwc.x86')
def conv1d_transpose_nwc(cfg, data, kernel, stride, padding, out_dtype, out_padding=0):
    print("Invoking conv1d_transpose_nwc compute - {data.shape}, {kernel.shape}, {stride}, {padding}, {out_padding}".format(**locals()))

    # dilate and pad
    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(padding, (tuple, list)):
        padding = padding[0]
    if isinstance(out_padding, (tuple, list)):
        out_padding = out_padding[0]

    batch, data_width, channels_in = get_const_tuple(data.shape)
    channels_out, kernel_width, channels_in_  = get_const_tuple(kernel.shape)
    assert channels_in == channels_in_, (channels_in, channels_in_)
    out_width = (data_width - 1) * stride - 2 * padding + (kernel_width - 1) + out_padding + 1
    cfg.add_flop(2 * batch * out_width * channels_out * channels_in * kernel_width)
    return te.extern((batch, out_width, channels_out),
                    [data, kernel],
                    lambda ins, outs: tvm.tir.call_packed(
                        "tvm.contrib.xnnpack.conv1d_transpose", 
                        ins[0], ins[1], outs[0], stride, padding, out_padding),
                    tag="conv1d_transpose_nwc",
                    name="conv1d_transpose_xnnpack",
                    dtype=out_dtype)        


@autotvm.register_topi_schedule('conv1d_transpose_nwc.x86')
def schedule_conv1d_transpose_nwc(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    te.schedule.AutoInlineInjective(s)
    if outs[0].op.tag != "conv1d_transpose_nwc":
        last = list(s[outs[0].op].op.axis)[-1]
        (lo, li) = s[outs[0].op].split(last, factor=16)
        s[outs[0].op].vectorize(li)
    return s
