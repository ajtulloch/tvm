# pylint: disable=invalid-name,unused-variable,no-else-return
"""Conv2D schedule for ARM CPU"""
from __future__ import absolute_import as _abs

import numpy as np

import tvm
from tvm import autotvm
import nnvm.symbol as sym
from topi.generic import schedule_conv2d_nchw, schedule_conv2d_NCHWc_, schedule_conv2d_NCHWc_winograd_without_weight_transform
from topi.util import traverse_inline, get_const_tuple, const_matrix
from topi.nn import pad, conv2d, conv2d_NCHWc, conv2d_alter_layout, conv2d_NCHWc_winograd_without_weight_transform
from topi.nn.conv2d import _get_workload
from topi.nn.util import get_const_int, get_pad_tuple
import topi.nn
import logging

import unet_conv2d_NCHWc_direct
import unet_conv2d_winograd

def _conv_arg_to_workload(data, kernel, strides, padding, layout, out_dtype):
    """convert argument to workload"""
    if len(kernel.shape) == 4:
        raw_kernel = kernel
    else:  # the input kernel is transformed by alter_op_layout
        shape = get_const_tuple(kernel.shape)
        raw_kernel = tvm.placeholder((shape[0] * shape[4], shape[1], shape[2], shape[3]),
                                     dtype=kernel.dtype)
    return ('conv2d', ) + autotvm.task.args_to_workload(
        [data, raw_kernel, strides, padding, layout, out_dtype])

def _conv_inferred_winograd_NCHWc_arg_to_workload(data, kernel, strides, padding, layout, out_dtype):
    N, CI, IH, IW = get_const_tuple(data.shape)
    assert CI % 16 == 0
    data_NCHWc = tvm.placeholder((N, CI // 16, IH, IW, 16), dtype="float32")
    CO, CI, KH, KW = get_const_tuple(kernel.shape)
    assert CO % 16 == 0
    assert CI % 16 == 0
    kernel_NCHWc = tvm.placeholder((CO // 16, CI // 16, KH, KW, 16, 16), dtype="float32")
    return unet_conv2d_NCHWc_direct._conv_NCHWc_arg_to_workload(data_NCHWc, kernel_NCHWc, CO, (KH, KW), strides, padding, layout="NCHW16c", out_layout="NCHW16c", out_dtype="float32")

def winograd_alter_layout_transform(attrs, inputs, tinfos):
    copy_inputs = [s for s in inputs]
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    data = tinfos[0]
    kernel = tinfos[1]

    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.get_int('groups')
    layout = attrs["layout"]
    out_dtype = attrs["out_dtype"]
    out_dtype = tinfos[0].dtype if out_dtype == "same" else out_dtype
    workload = _conv_inferred_winograd_NCHWc_arg_to_workload(tinfos[0], tinfos[1], strides, padding,
                                                             layout, out_dtype)
    cfg = autotvm.DispatchContext.current.query(
        tvm.target.current_target(), workload)
    if cfg.is_fallback:  # if is fallback, clear query cache and return None
        autotvm.task.clear_fallback_cache(
            tvm.target.current_target(), workload)
        raise RuntimeError("Unknown configuration for Winograd in AutoTVM, need a fallback")

    if cfg.template_key != 'winograd':
        raise RuntimeError("Winograd is not the optimal configuration for AutoTVM")
    tile_size = 4
    transformed_weight = sym.contrib.conv2d_NCHWc_winograd_weight_transform(
        copy_inputs[1],
        kernel_layout="OIHW16i16o",
        tile_size=tile_size)
    copy_inputs[1] = transformed_weight
    new_attrs['tile_size'] = tile_size
    new_attrs['layout'] = 'NCHW16c'
    new_attrs['out_layout'] = 'NCHW16c'
    new_attrs['kernel_layout'] = 'OIHW16i16o'
    # import ipdb; ipdb.set_trace()
    logging.info("Winograd selected for %s, %s, %s", attrs, inputs, tinfos)
    logging.info("Workload: %s", workload)
    return sym.contrib.conv2d_NCHWc_winograd_without_weight_transform(*copy_inputs, **new_attrs)


@conv2d_alter_layout.register("cpu", override=True)
def _alter_conv2d_layout(attrs, inputs, tinfos):
    copy_inputs = [s for s in inputs]
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    # only optimize for NCHW, groups=1 conv
    if attrs['layout'] != 'NCHW' or attrs.get_int("groups") != 1:
        return None

    data = tinfos[0]
    kernel = tinfos[1]

    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.get_int('groups')
    layout = attrs["layout"]
    out_dtype = attrs["out_dtype"]
    out_dtype = tinfos[0].dtype if out_dtype == "same" else out_dtype

    # workload = _conv_arg_to_workload(data, kernel, strides, padding,
    #                                  layout, out_dtype)

    try:
        return winograd_alter_layout_transform(attrs, inputs, tinfos)
    except:
        logging.debug("Failed to find Winograd transform for %s, %s, %s", attrs, inputs, tinfos)
    wkl = _get_workload(data, kernel, strides, padding, data.dtype)
    if wkl.in_filter % 16 == 0 and wkl.out_filter % 16 == 0:
        logging.debug("Altering layout to NCHW16c")
        new_attrs['layout'] = 'NCHW16c'
        new_attrs['out_layout'] = 'NCHW16c'
        new_attrs['kernel_layout'] = 'OIHW16i16o'
        return sym.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)
    if wkl.in_filter % 3 == 0 and wkl.out_filter % 16 == 0:
        new_attrs['layout'] = 'NCHW3c'
        new_attrs['out_layout'] = 'NCHW16c'
        new_attrs['kernel_layout'] = 'OIHW3i16o'
        return sym.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)


@conv2d.register('cpu', override=True)
@autotvm.task.dispatcher
def conv2d_cpu(data, kernel, strides, padding, layout, out_dtype):
    return _conv_arg_to_workload(data, kernel, strides, padding, layout, out_dtype)


@conv2d_NCHWc.register('cpu', override=True)
@autotvm.task.dispatcher
def conv2d_NCHWc_cpu(data, kernel, num_filter, kernel_size, stride, padding, layout, out_layout, out_dtype="float32"):
    """TOPI compute callback. Mark this function as a dispatcher, so
    this template can assign config according to workload

    Returns
    -------
    workload: Tuple
        Dispatcher will use this workload to query corresponding config.
        Then use cfg.template_key to call a registered template.
    """
    return unet_conv2d_NCHWc_direct._conv_NCHWc_arg_to_workload(data, kernel, num_filter, kernel_size, stride, padding, layout, out_layout, out_dtype)

@conv2d_cpu.register(['direct'])
def decl_spatial_pack(cfg, data, kernel, strides, padding, layout, out_dtype):
    """spatial packing template"""
    return _decl_spatial_pack(cfg, data, kernel, strides, padding, layout, out_dtype, num_tile=2)


@conv2d_NCHWc_cpu.register(['direct'])
def decl_spatial_pack_NCHWc(cfg, data, kernel, num_filter, kernel_size, stride, padding, layout, out_layout, out_dtype):
    return unet_conv2d_NCHWc_direct._decl_spatial_pack_NCHWc(cfg, data, kernel, num_filter, kernel_size, stride, padding, layout, out_layout, out_dtype)

@conv2d_NCHWc_cpu.register(['winograd'])
def decl_winograd_NCHWc(cfg, data, kernel, num_filter, kernel_size, stride, padding, layout, out_layout, out_dtype):
    return unet_conv2d_winograd._decl_winograd_NCHWc(cfg, data, kernel, num_filter, kernel_size, stride, padding, layout, out_layout, out_dtype, m=4)

# @conv2d_NCHWc_winograd_without_weight_transform.register('cpu', override=True)
# def conv2d_NCHWc_winograd_without_weight_transform(input, filter, strides, padding,
#                                                    layout, out_dtype, tile_size):


@autotvm.register_topi_schedule(
    schedule_conv2d_NCHWc_, 'cpu', ['direct', 'winograd'], override=True)
def schedule_conv2d_NCHWc_cpu(cfg, outs):
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if 'spatial_conv2d_output' in op.tag:
            output = op.output(0)
            unet_conv2d_NCHWc_direct._schedule_spatial_pack_NCHWc(cfg, s, output, outs[0])
        if 'winograd_conv2d_output' in op.tag:
            output = op.output(0)
            unet_conv2d_winograd._schedule_winograd_NCHWc(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s

# @autotvm.register_topi_schedule(schedule_conv2d_winograd_without_weight_transform,
#                                 'cpu', ['winograd'])
# def schedule_conv2d_winograd_without_weight_transform_(cfg, outs):
#     """TOPI schedule callback"""
#     s = tvm.create_schedule([x.op for x in outs])

#     def _callback(op):
#         if 'winograd_conv2d_output' in op.tag:
#             output = op.output(0)
#             _schedule_winograd(cfg, s, output, outs[0])

#     traverse_inline(s, outs[0].op, _callback)
#     return s


@autotvm.register_topi_schedule(
    schedule_conv2d_nchw, 'cpu', ['direct'], override=True)
def schedule_conv2d_nchw_cpu(cfg, outs):
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d

        if 'spatial_conv2d_output' in op.tag:
            output = op.output(0)
            conv = op.input_tensors[0]

            data_vec = conv.op.input_tensors[0]
            data_pad = data_vec.op.input_tensors[0]
            s[data_pad].compute_inline()

            kernel_vec = conv.op.input_tensors[1]
            if kernel_vec.op.name == 'kernel_vec':
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            _schedule_spatial_pack(cfg, s, data_vec, kernel_vec, conv, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def _decl_spatial_pack(cfg, data, kernel, strides, padding, layout, out_dtype, num_tile):
    assert layout == "NCHW", "Only support NCHW"
    # create workload according to raw arguments
    wkl = _conv_arg_to_workload(data, kernel, strides, padding, layout, out_dtype)

    out_dtype = out_dtype or data.dtype
    N, CI, IH, IW = get_const_tuple(data.shape)
    if len(kernel.shape) == 4:
        pre_packed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        pre_packed = True
        CO, _, KH, KW, VC = get_const_tuple(kernel.shape)
        CO = CO * VC

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (KH, KW))
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    OH = (IH + pad_top + pad_bottom - KH) // HSTR + 1
    OW = (IW + pad_left + pad_right - KW) // WSTR + 1
    data_pad = pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_bottom, pad_right], name="data_pad")

    # ==================== define configuration space ====================
    n, co, oh, ow = cfg.axis(N), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    if num_tile == 2:     # for arm cpu
        co, vc = cfg.define_split('tile_co', co, num_outputs=2)
        oh, vh = cfg.define_split('tile_oh', oh, num_outputs=2)
        ow, vw = cfg.define_split('tile_ow', ow, num_outputs=2)
    elif num_tile == 3:   # for mali gpu
        co, _, vc = cfg.define_split('tile_co', co, num_outputs=3)
        oh, _, vh = cfg.define_split('tile_oh', oh, num_outputs=3)
        ow, _, vw = cfg.define_split('tile_ow', ow, num_outputs=3)
    else:
        raise RuntimeError("Invalid num_tile")

    cfg.define_reorder("reorder_0",
                       [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
                       policy='candidate', candidate=[
                           [n, co, oh, ow, ci, kh, kw, vh, vw, vc],
                           [n, co, oh, ow, ci, kh, kw, vc, vh, vw]])

    cfg.define_reorder("reorder_1",
                       [n, co, oh, ow, vh, vw, vc],
                       policy='candidate', candidate=[
                           [n, co, oh, ow, vh, vw, vc],
                           [n, co, oh, ow, vc, vh, vw],
                           [n, co, oh, ow, vh, vc, vw]])

    cfg.define_annotate("ann_reduce", [kh, kw], policy='try_unroll')
    cfg.define_annotate("ann_spatial", [vh, vw, vc], policy='try_unroll_vec')

    # fallback support
    if cfg.is_fallback:
        if num_tile == 2:     # arm cpu
            ref_log = autotvm.tophub.load_reference_log('cpu', 'rk3399', 'conv2d', 'direct')
            cfg.fallback_with_reference_log(ref_log)
        elif num_tile == 3:  # mali gpu
            ref_log = autotvm.tophub.load_reference_log('mali', 'rk3399', 'conv2d', 'direct')
            cfg.fallback_with_reference_log(ref_log)
    # ====================================================================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    dvshape = (N, OH // VH, OW // VW, CI, VH*HSTR + KH-1, VW*WSTR + KW-1)
    kvshape = (CO // VC, CI, KH, KW, VC)
    ovshape = (N, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (N, CO, OH, OW)

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw:
                           data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw],
                           name='data_vec')

    if pre_packed:
        kernel_vec = kernel
    else:
        kernel_vec = tvm.compute(kvshape, lambda co, ci, kh, kw, vc:
                                 kernel[co*VC+vc][ci][kh][kw],
                                 name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
        tvm.sum(data_vec[n, h, w, ci, vh*HSTR+kh, vw*WSTR+kw].astype(out_dtype) *
                kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                axis=[ci, kh, kw]), name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w:
                         conv[n][co//VC][h//VH][w//VW][h%VH][w%VW][co%VC],
                         name='output_unpack', tag='spatial_conv2d_output',
                         attrs={'workload': wkl})
    return output

def _schedule_spatial_pack(cfg, s, data_vec, kernel_vec,
                           conv, output, last):
    """schedule implementation"""
    # import ipdb
    # ipdb.set_trace()

    n, co, oh, ow, vh, vw, vc = s[conv].op.axis
    ci, kh, kw = s[conv].op.reduce_axis
    data_pad = data_vec.op.input_tensors[0]
    if data_pad.op.name == "data_pad":
        assert type(data_pad.op) == tvm.tensor.ComputeOp
        has_padding = True
    else:
        import ipdb; ipdb.set_trace()
        assert type(data_pad.op) == tvm.tensor.PlaceholderOp
        has_padding = False
    cfg.define_knob('data_pad_inline', [0, 1, 2])

    if cfg['data_pad_inline'].val == 1 and has_padding:
        s[data_pad].compute_inline()
    if cfg['data_pad_inline'].val == 2 and has_padding:
        s[data_pad].vectorize(list(s[data_pad].op.axis)[-1])
    # schedule conv
    cfg["reorder_0"].apply(s, conv, [n, co, oh, ow, ci, kh, kw, vh, vw, vc])
    cfg["ann_reduce"].apply(s, conv, [kh, kw],
                            axis_lens=[get_const_int(kh.dom.extent),
                                       get_const_int(kw.dom.extent)],
                            max_unroll=16,
                            cfg=cfg)
    cfg["ann_spatial"].apply(s, conv, [vh, vw, vc],
                             axis_lens=[cfg['tile_oh'].size[-1],
                                        cfg['tile_ow'].size[-1],
                                        cfg['tile_co'].size[-1]],
                             max_unroll=16,
                             cfg=cfg)

    # schedule fusion
    n, co, h, w = s[last].op.axis
    co, vc = cfg['tile_co'].apply(s, last, co)
    oh, vh = cfg['tile_oh'].apply(s, last, h)
    ow, vw = cfg['tile_ow'].apply(s, last, w)
    cfg["reorder_1"].apply(s, last, [n, co, oh, ow, vh, vw, vc])
    if last != output:
        s[output].compute_inline()
        cfg["ann_spatial"].apply(s, last, [vh, vw, vc],
                                 axis_lens=[cfg['tile_oh'].size[-1],
                                            cfg['tile_ow'].size[-1],
                                            cfg['tile_co'].size[-1]],
                                 max_unroll=16,
                                 cfg=cfg)
    else:
        s[last].vectorize(vw)
    s[conv].compute_at(s[last], ow)

    # mark parallel
    # s[last].parallel(co)

    _, h, _, _, _, _ = s[data_vec].op.axis
    # s[data_vec].parallel(h)

    if kernel_vec.op.name == 'kernel_vec':
        co, _, _, _, _ = s[kernel_vec].op.axis
        s[kernel_vec].pragma(co, 'debug_skip_region')
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel packing will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[kernel_vec].pragma(co, 'debug_skip_region')
        else:
            pass
            # s[kernel_vec].parallel(co)
    return s

@conv2d_NCHWc_winograd_without_weight_transform.register('cpu')
@autotvm.task.dispatcher
def _conv2d_NCHWc_winograd_without_weight_transform_config_dispatcher(data, kernel, strides, padding, layout, out_dtype, tile_size):
    # import ipdb; ipdb.set_trace()
    KH = 3
    KW = 3
    (COO, CII, CIII, alpha_h, alpha_w, COOO) = get_const_tuple(kernel.shape)
    assert alpha_h == tile_size + KH - 1
    assert alpha_w == tile_size + KW - 1

    kernel_NCHWc = tvm.placeholder((COO, CII, KH, KW, CIII, COOO), dtype=kernel.dtype)
    workload = unet_conv2d_NCHWc_direct._conv_NCHWc_arg_to_workload(data, kernel_NCHWc, COO * COOO, (KH, KW), strides, padding, layout="NCHW16c", out_layout="NCHW16c", out_dtype="float32")
    print(workload)
    return workload


@_conv2d_NCHWc_winograd_without_weight_transform_config_dispatcher.register(['winograd'])
def _decl_conv2d_NCHWc_winograd_without_weight_transform(cfg, data, kernel, strides, padding, layout, out_dtype, tile_size):
    # import ipdb; ipdb.set_trace()
    (COO, CII, CIII, alpha_h, alpha_w, COOO) = get_const_tuple(kernel.shape)
    KH = 3
    KW = 3
    assert alpha_h == tile_size + KH - 1, get_const_tuple(kernel.shape)
    assert alpha_w == tile_size + KW - 1, get_const_tuple(kernel.shape)
    num_filter = COO * COOO
    kernel_size = (3, 3)
    return unet_conv2d_winograd._decl_winograd_NCHWc(cfg, data, kernel, num_filter, kernel_size, strides, padding, layout=layout, out_layout=layout, out_dtype=out_dtype, m=tile_size)


@autotvm.register_topi_schedule(schedule_conv2d_NCHWc_winograd_without_weight_transform,
                                'cpu', ['winograd'])
def _schedule_conv2d_NCHWc_winograd_without_weight_transform(cfg, outs):
    """TOPI schedule callback"""
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'winograd_conv2d_output' in op.tag:
            output = op.output(0)
            unet_conv2d_winograd._schedule_winograd_NCHWc(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s
