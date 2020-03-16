"""Schedule for bitserial dense operator."""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from tvm import autotvm
from topi.nn.dilate import dilate
from topi.nn.pad import pad
from topi.util import get_const_int, get_const_tuple, simplify, traverse_inline
from topi.nn.util import get_pad_tuple1d
from topi import tag


@autotvm.register_topi_compute('conv1d_transpose_nwc.x86')
def conv1d_transpose_nwc(cfg, data, kernel, stride, padding, out_dtype):
    print("Invoking conv1d_transpose_nwc compute - {data.shape}, {kernel.shape}, {stride}, {padding}".format(**locals()))

    # dilate and pad
    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    batch, data_width, channels_in = get_const_tuple(data.shape)
    kernel_width, channels_in_, channels_out = get_const_tuple(kernel.shape)
    assert channels_in == channels_in_, (channels_in, channels_in_)
    channels_out = simplify(channels_out)
    data = dilate(data, [1, stride, 1], name='data_dilate')
    pad_left, pad_right = get_pad_tuple1d(padding, (kernel_width,))
    pad_left = kernel_width - 1 - pad_left
    pad_right = kernel_width - 1 - pad_right
    data = pad(data, [0, pad_left, 0], [0, pad_right, 0], name='data_pad')

    _, data_width, _ = data.shape
    out_w = simplify(data_width - kernel_width + 1)
    rc = te.reduce_axis((0, channels_in), name='rc')
    rw = te.reduce_axis((0, kernel_width), name='rw')
    output = te.compute(
        (batch, out_w, channels_out),
        lambda b, w, c: te.sum(
            data[b, w+rw, rc].astype(out_dtype) *
            kernel[kernel_width - 1 - rw, rc, c].astype(out_dtype),
            axis=[rw, rc]), tag="conv1d_transpose_nwc")
    return output

@autotvm.register_topi_schedule('conv1d_transpose_nwc.x86')
def schedule_conv1d_transpose_nwc(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(cfg, s, output, data, kernel):
        # if "data_pad" in data.op.name:
        #     print("Has padding")
        #     s[data.op].compute_inline()

        if "data_dilate" in data.op.input_tensors[0].op.name:
            s[data.op.input_tensors[0].op].compute_inline()

        n, w, c = s[output].op.axis
        rw, rc = s[output].op.reduce_axis
        cfg.define_split("tile_c", c, num_outputs=2,
                         filter=lambda x: x.size[-1] in (8, 16))
        cfg.define_split("tile_w", w, num_outputs=2, filter=lambda x: x.size[-1] in (4,))
        cfg.define_split("tile_rc", rc, num_outputs=2, filter=lambda x: x.size[-1] in (2, 4, 8, 16))

        # schedule according to config
        co, ci = cfg["tile_c"].apply(s, output, c)
        rco, rci = cfg["tile_rc"].apply(s, output, rc)
        wo, wi = cfg["tile_w"].apply(s, output, w)

        cfg.define_annotate('ann_reduce', [rci], policy='try_unroll')
        cfg.define_annotate('ann_spatial', [wi], policy='try_unroll')
        s[output].vectorize(ci)
        # cfg['ann_reduce'].apply(s, output, [rci])
        # cfg['ann_spatial'].apply(s, output, [wi])
        # s[output].reorder(n, wo, co, rco, rw, rci, wi, ci)
        s[output].reorder(n, wo, co, rw, rco, rci, wi, ci)
        (nd, wd, cd) = s[data].op.axis

        if "data_pad" in data.op.name:
            data_pad = data.op
            cfg.define_knob("data_pad_compute_at", [1])
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

        else:
            assert False
        if output.op != outs[0].op:
            (n, w, c) = s[outs[0].op].op.axis
            co, ci = cfg["tile_c"].apply(s, outs[0].op, c)
            wo, wi = cfg["tile_w"].apply(s, outs[0].op, w)
            s[outs[0].op].reorder(n, wo, co, wi, ci)
            s[outs[0].op].vectorize(ci)
            s[outs[0].op].unroll(wi)
            s[output.op].compute_at(s[outs[0].op], co)

    def _callback(op):
        if op.tag == 'conv1d_transpose_nwc':
            output = op.output(0)
            data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            _schedule(cfg, s, output, data, kernel)

    traverse_inline(s, outs[0].op, _callback)
    return s
