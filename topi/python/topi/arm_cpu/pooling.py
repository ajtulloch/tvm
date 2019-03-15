from ..nn import max_pool2d_alter_layout

@max_pool2d_alter_layout.register(["arm_cpu"])
def _alter_max_pool2d_layout_arm(attrs, inputs, tinfos, F):
    """Alter op layout for pre-computing kernel transformation

    Parameters
    ----------
    attrs : nnvm.top.AttrDict or tvm.attrs.Attrs
        Attributes of current convolution
    inputs : nnvm.symbol or tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    F: symbol
        The context, can be either nnvm.sym or relay.op

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level,
    so we have to pass 'F' to make it support our two versions of graph IR, NNVM and Relay.
    """
    copy_inputs = [s for s in inputs]
    new_attrs = {k: attrs[k] for k in attrs.keys()}
    data_layout_key = "data_layout" if "data_layout" in new_attrs else "layout"
    (data, ) = tinfos
    if data.dtype == "int8":
        new_attrs[data_layout_key] = 'NHWC'
        return F.nn.max_pool2d(*copy_inputs, **new_attrs)
