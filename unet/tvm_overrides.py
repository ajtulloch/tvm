import tvm
import topi.nn.util
from topi.nn.conv2d import conv2d_alter_layout, _get_workload
from topi.nn.upsampling import upsampling_alter_layout
import nnvm.top.registry
from topi import generic, tag
from tvm import autotvm

@conv2d_alter_layout.register("cpu", override=True)
def _alter_conv2d_layout(attrs, inputs, tinfos):
    import nnvm.symbol as sym
    copy_inputs = [s for s in inputs]
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    # only optimize for NCHW, groups=1 conv
    if attrs['layout'] != 'NCHW' or attrs.get_int("groups") != 1:
        return None

    data = tinfos[0]
    kernel = tinfos[1]

    import ast
    padding = ast.literal_eval(attrs['padding'])
    stride = ast.literal_eval(attrs['strides'])

    wkl = _get_workload(data, kernel, stride, padding, data.dtype)
    # cfg = autotvm.DispatchContext.current.query(tvm.target.current_target(), wkl)

    # if cfg.is_fallback:  # if is fallback, clear query cache and return None
    #     autotvm.task.clear_fallback_cache(tvm.target.current_target(), workload)
    #     return None

    # if cfg.template_key == 'direct' and 'tile_co' in cfg:  # packing weight tensor
    #     new_attrs['kernel_layout'] = 'OIHW%do' % (cfg['tile_co'].size[-1])
    #     return sym.conv2d(*copy_inputs, **new_attrs)
    # print(wkl)
    if wkl.in_filter % 16 == 0 and wkl.out_filter % 16 == 0:
        print("Altering layout to NCHW16c")
        new_attrs['layout'] = 'NCHW16c'
        new_attrs['out_layout'] = 'NCHW16c'
        new_attrs['kernel_layout'] = 'OIHW16i16o'
        return sym.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)
    # if wkl.in_filter % 8 == 0 and wkl.out_filter % 8 == 0:
    #     new_attrs['layout'] = 'NCHW8c'
    #     new_attrs['out_layout'] = 'NCHW8c'
    #     new_attrs['kernel_layout'] = 'OIHW8i8o'
    #     return sym.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)

# # data = sym.max_pool2d(data=data, pool_size=(2, 2), strides=(2, 2), layout=layout)
@nnvm.top.registry.register_alter_op_layout("max_pool2d")
def alter_max_pool2d_layout(attrs, inputs, tinfos):
    import nnvm.symbol as sym
    copy_inputs = [s for s in inputs]
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    if attrs['layout'] != 'NCHW':
        return None

    data = tinfos[0]
    _, CI, _, _ = [x.value for x in data.shape]
    if CI % 16 == 0:
        new_attrs['layout'] = 'NCHW16c'
        return sym.max_pool2d(*copy_inputs, **new_attrs)
    # if CI % 8 == 0:
    #     new_attrs['layout'] = 'NCHW8c'
    #     return sym.max_pool2d(*copy_inputs, **new_attrs)

@upsampling_alter_layout.register("cpu", override=True)
def _upsampling_alter_layout(attrs, inputs, tinfos):
    import nnvm.symbol as sym
    copy_inputs = [s for s in inputs]
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    print(new_attrs)
    if attrs['layout'] != 'NCHW':
        return None

    data = tinfos[0]
    _, CI, _, _ = [x.value for x in data.shape]
    if CI % 16 == 0:
        new_attrs['layout'] = 'NCHW16c'
        return sym.upsampling(*copy_inputs, **new_attrs)

@generic.schedule_pool.register(["cpu"], override=True)
def schedule_pool(outs, layout):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []
    output_op = outs[0].op

    def traverse(op):
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        elif op.tag.startswith('pool'):
            # schedule the pooling op.
            kw, kh = op.reduce_axis
            s[op].unroll(kw)
            s[op].unroll(kh)
            s[op].vectorize(list(op.axis)[-1])
            s[op].compute_at(s[outs[0].op], list(outs[0].op.axis)[-2])

            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

        scheduled_ops.append(op)

    traverse(output_op)
    s[output_op].vectorize(list(output_op.axis)[-1])
    s[output_op].fuse(output_op.axis[0], output_op.axis[1])
    return s

def _default_declaration_conv_NCHWc(data, kernel, num_filter, kernel_size, stride,
                                    padding, layout, out_layout, out_dtype='float32'):
    HPAD, WPAD, _, _ = topi.nn.get_pad_tuple(padding, kernel)
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    batch_size = data.shape[0]
    _, CII, IH, IW, CIII = [x.value for x in data.shape]
    COO, CII, KH, KW, CIII_, COOO = [x.value for x in kernel.shape]
    assert CIII == CIII_
    out_height = (IH + 2 * HPAD - KH) // HSTR + 1

    out_width = (IW + 2 * WPAD - KW) // WSTR + 1

    # pack data
    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    # convolution
    oshape = (batch_size, COO, out_height, out_width, COOO)

    ic = tvm.reduce_axis((0, CII * CIII), name='ic')
    ciii = tvm.reduce_axis((0, CIII), name='ciii')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_pad[n, ic // CIII, oh*HSTR+kh, ow*WSTR+kw, ic % CIII]
                               .astype(out_dtype) *
                               kernel[oc_chunk, ic // CIII, kh, kw, ic % CIII, oc_block],
                               axis=[ic, kh, kw]), name='conv2d_NCHWc', tag="conv2d_NCHWc")
    return conv

topi.nn.conv2d_NCHWc.fdefault = _default_declaration_conv_NCHWc
