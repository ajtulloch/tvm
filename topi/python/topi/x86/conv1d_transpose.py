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
    kernel_width, channels_in_, channels_out = get_const_tuple(kernel.shape)
    assert channels_in == channels_in_, (channels_in, channels_in_)
    out_width = (data_width - 1) * stride - 2 * padding + (kernel_width - 1) + out_padding + 1
    try:
        cfg.define_split('tile_w', cfg.axis(out_width), num_outputs=2)
        (_, W_inner) = cfg['tile_w'].size
    except IndexError:
        print("Failed AutoTVM!!!")
        W_inner = out_width
    assert out_width % W_inner == 0
    W_outer = out_width // W_inner

    try:
        cfg.define_split('tile_c', cfg.axis(channels_out), num_outputs=2, filter=lambda x: x.size[-1] in (8, 16, 32) if channels_out % 8 == 0 else True)
        (_, C_inner) = cfg['tile_c'].size
    except IndexError:
        print("Failed AutoTVM!!!")
        C_inner = channels_out
    assert channels_out % C_inner == 0
    cfg.define_knob("w_inner_for_type", ["serial", "unroll"])
    w_inner_for_type = cfg["w_inner_for_type"].val
    C_outer = channels_out // C_inner
    N = batch
    RW = kernel_width
    RC = channels_in
    pad_left = padding
    stride = stride
    IW = data_width
    def conv_transpose_1d_ir_builder(X, W, Y):
        ib = tvm.tir.ir_builder.create()
        with ib.for_range(0, N, 'n') as n:
            with ib.for_range(0, W_outer, 'w.outer') as w_outer:
                with ib.for_range(0, C_outer, 'c.outer') as c_outer:
                    with ib.for_range(0, W_inner, 'w.inner', for_type=w_inner_for_type) as w_inner:
                        print("C_inner", C_inner)
                        if C_inner == 1:
                            acc = ib.allocate(f'float32', (1,), 'acc', scope='local')
                            acc[0] = tvm.tir.const(0, 'float32')
                        else:
                            acc = ib.allocate(f'float32x{C_inner}', (1,), 'acc', scope='local')
                            acc[0] = tvm.tir.const(0, 'float32').astype(f'float32x{C_inner}')

                        with ib.for_range(0, RW, 'rw') as rw:
                            ow = (w_outer * W_inner + w_inner).astype('uint32')
                            w = ow + tvm.tir.const(pad_left, 'uint32') - rw.astype('uint32')
                            iw = tvm.te.floordiv(w, tvm.tir.const(stride, 'uint32'))
                            with ib.if_scope((iw * stride == w)):
                                with ib.if_scope(iw < tvm.tir.const(IW, 'uint32')):
                                    # with ib.if_scope(0 <= iw):
                                    with ib.for_range(0, RC, 'rc') as rc:
                                        acc[0] += X.vload([n, iw, rc], 'float32').astype(f'float32x{C_inner}') * W.vload([rw, rc, c_outer * C_inner], f'float32x{C_inner}')
                        ib.emit(Y.vstore([n, ow, c_outer * C_inner], acc[0]))

        return ib.get()
    if cfg.flop == 0:
        cfg.add_flop(2 * batch * out_width * channels_out * kernel_width * channels_in)
    print("In shape: ", data.shape, ", kernel shape: ", kernel.shape, ", Padding: ", padding, ", stride: ", stride,  ", Output shape: ", (batch, out_width, channels_out))
    return te.extern((batch, out_width, channels_out),
                     [data, kernel],
                     lambda ins, outs: conv_transpose_1d_ir_builder(ins[0], ins[1], outs[0]),
                     tag="conv1d_transpose_nwc",
                     name="conv1d_transpose",
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
