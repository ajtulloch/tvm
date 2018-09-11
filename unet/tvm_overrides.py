import tvm
import topi.nn.util
from topi.nn.conv2d import conv2d_alter_layout, _get_workload
import nnvm.top.registry

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
