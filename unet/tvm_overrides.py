import tvm
import topi.nn.util
from topi.nn.conv2d import conv2d_alter_layout, _get_workload
import nnvm.top.registry
from topi import generic, tag

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
    print(wkl)
    if wkl.in_filter % 8 == 0 and wkl.out_filter % 8 == 0:
        new_attrs['layout'] = 'NCHW8c'
        new_attrs['out_layout'] = 'NCHW8c'
        new_attrs['kernel_layout'] = 'OIHW8i8o'
        return sym.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)

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
    if CI % 8 == 0:
        new_attrs['layout'] = 'NCHW8c'
        return sym.max_pool2d(*copy_inputs, **new_attrs)

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
