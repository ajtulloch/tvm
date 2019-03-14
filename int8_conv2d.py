from __future__ import division

import click
from topi.util import get_const_int, const_matrix
from topi.nn.conv2d import Workload
import numpy as np
import tvm
import tvm.rpc
from tvm import autotvm
import collections
import logging
import sys
from topi.util import traverse_inline, get_const_tuple, const_matrix
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn import pad, conv2d, conv2d_NCHWc, conv2d_alter_layout


# V
# Storage - N x H x W x C
#
def _intrin_Kx4xint8_Kx8xint8_4x8_int32(K):
    dtype = "int8"
    X = tvm.placeholder((4, K), dtype="int8", name='X')
    W = tvm.placeholder((K, 8), dtype=dtype, name='X')
    k = tvm.reduce_axis((0, K), name='k')
    Z = tvm.compute((4, 8), lambda i, j: tvm.sum(X[i, k].astype("int32") * W[k, j].astype("int32"), axis=[k]), name="Z")

    Xb = tvm.decl_buffer(X.shape, X.dtype,
                         name="Xb",
                         offset_factor=K,
                         strides=[tvm.var('ldX'), 1])
    Wb = tvm.decl_buffer(W.shape, W.dtype,
                         name="Wb",
                         offset_factor=K,
                         strides=[tvm.var('ldW'), 1])

    def _intrin_func(ins, outs):
        xx, ww = ins
        zz = outs[0]
        # vpadd = "llvm.arm.neon.vpadd.v8u8"
        # vpadalu = "llvm.arm.neon.vmlal.v4i32.v8i16"
        # args_1 = tvm.const(1, 'uint32')
        # args_2 = tvm.const(2, 'uint32')

        def _instr(index):
            irb = tvm.ir_builder.create()
            if index == 1:
                irb.emit(zz.vstore([0, 0], tvm.const(0, 'int8x8')))
                irb.emit(zz.vstore([1, 0], tvm.const(0, 'int8x8')))
                irb.emit(zz.vstore([2, 0], tvm.const(0, 'int8x8')))
                irb.emit(zz.vstore([3, 0], tvm.const(0, 'int8x8')))
                return irb.get()


            accs = [(tvm.const(0, 'int32x4'), (tvm.const(0, 'int32x4')))] * 4
            assert K % 8 == 0
            for k in range(0, K, 8):
                va0 = xx.vload([0, k], 'int8x8').astype('int16x8')
                va1 = xx.vload([1, k], 'int8x8').astype('int16x8')
                va2 = xx.vload([2, k], 'int8x8').astype('int16x8')
                va3 = xx.vload([3, k], 'int8x8').astype('int16x8')

                vxb = ww.vload([k, 0], 'int8x8').astype('int16x8')

                def vl(x):
                    return tvm.call_pure_intrin('int16x4', 'vectorlow', x)

                def vh(x):
                    return tvm.call_pure_intrin('int16x4', 'vectorhigh', x)


                accs[0][0] = accs[0][0] + vl(vxb).astype('int32x4') * (vl(va0)[0]).astype("int32x4")
                accs[0][1] = accs[0][1] + vh(vxb).astype('int32x4') * (vl(va0)[0]).astype("int32x4")
                accs[1][0] = accs[1][0] + vl(vxb).astype('int32x4') * (vl(va1)[0]).astype("int32x4")
                accs[1][1] = accs[1][1] + vh(vxb).astype('int32x4') * (vl(va1)[0]).astype("int32x4")
                accs[2][0] = accs[2][0] + vl(vxb).astype('int32x4') * (vl(va2)[0]).astype("int32x4")
                accs[2][1] = accs[2][1] + vh(vxb).astype('int32x4') * (vl(va2)[0]).astype("int32x4")
                accs[3][0] = accs[3][0] + vl(vxb).astype('int32x4') * (vl(va3)[0]).astype("int32x4")
                accs[3][1] = accs[3][1] + vh(vxb).astype('int32x4') * (vl(va3)[0]).astype("int32x4")


            irb.emit(zz.vstore([0, 0], accs[0][0]))
            irb.emit(zz.vstore([0, 4], accs[0][1]))
            irb.emit(zz.vstore([1, 0], accs[1][0]))
            irb.emit(zz.vstore([1, 4], accs[1][1]))
            irb.emit(zz.vstore([2, 0], accs[2][0]))
            irb.emit(zz.vstore([2, 4], accs[2][1]))
            irb.emit(zz.vstore([3, 0], accs[3][0]))
            irb.emit(zz.vstore([3, 4], accs[3][1]))

            return irb.get()
        # body, reset, update
        return _instr(0), _instr(1), _instr(2)
    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(Z.op, _intrin_func, binds={X: Xb, W:Wb})


def _decl_spatial_pack_NCHWc(cfg, data, kernel, num_filter, kernel_size, stride, padding, layout, out_layout, out_dtype):
    wkl = None
    out_dtype = out_dtype or data.dtype
    N, CII, IH, IW, CIII = get_const_tuple(data.shape)
    COO, CII, KH, KW, CIII_, VC = get_const_tuple(kernel.shape)

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (KH, KW))
    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)

    OH = (IH + pad_top + pad_bottom - KH) // HSTR + 1
    OW = (IW + pad_left + pad_right - KW) // WSTR + 1
    data_pad = pad(data, [0, 0, pad_top, pad_left, 0], [0, 0, pad_bottom, pad_right, 0], name="data_pad")

    # ==================== define configuration space ====================
    n, coo, oh, ow, vc = cfg.axis(N), cfg.axis(COO), cfg.axis(OH), cfg.axis(OW), cfg.axis(VC)
    cii, ciii, kh, kw = cfg.reduce_axis(CII), cfg.reduce_axis(CIII), cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    oh, vh = cfg.define_split('tile_oh', oh, num_outputs=2, filter=lambda x: x.size[-1] <= 8)
    ow, vw = cfg.define_split('tile_ow', ow, num_outputs=2, filter=lambda x: x.size[-1] <= 8)

    cfg.define_reorder("reorder_0",
                       [n, coo, cii, oh, ow, kh, kw, vc, ciii, vh, vw],
                       policy='candidate', candidate=[
                           [n, coo, cii, oh, ow, kh, kw, vc, ciii, vh, vw],
                           [n, coo, cii, oh, ow, kh, kw, ciii, vh, vw, vc],
                           [n, coo, cii, oh, ow, kh, kw, vh, ciii, vw, vc],
                           [n, coo, oh, ow, cii, kh, kw, ciii, vh, vw, vc],
                           # [n, coo, cii, oh, ow, kh, kw, vc, vh, ciii, vw],
                           # [n, coo, cii, oh, ow, kh, kw, ciii, vh, vc, vw],
                           # [n, coo, oh, cii, ow, kh, kw, ciii, vh, vw, vc],
                       ])

    cfg.define_reorder("reorder_1",
                       [n, coo, oh, ow, vh, vw, vc],
                       policy='candidate', candidate=[
                           [n, coo, oh, ow, vh, vw, vc],
                           [n, coo, oh, ow, vc, vh, vw],
                           [n, coo, oh, ow, vh, vc, vw]
                       ])

    cfg.define_annotate("ann_reduce", [kh, kw, ciii], policy='try_unroll')
    cfg.define_annotate("ann_spatial", [vc], policy='try_vec')
    # cfg.define_annotate("ann_spatial", [vh, vw, vc], policy='try_unroll_vec')

    # fallback support
    # if cfg.is_fallback:
    #     if num_tile == 2:     # arm cpu
    #         ref_log = autotvm.tophub.load_reference_log('cpu', 'rk3399', 'conv2d', 'direct')
    #         cfg.fallback_with_reference_log(ref_log)
    #     elif num_tile == 3:  # mali gpu
    #         ref_log = autotvm.tophub.load_reference_log('mali', 'rk3399', 'conv2d', 'direct')
    #         cfg.fallback_with_reference_log(ref_log)
    # ====================================================================

    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    # input            = (N, CII, IH, IW, CIII)
    # -> transpose
    ############################################################
    # input_tile_shape = (N, CII, OH // VH, OH // VH, VH + KH, VW + KW, CIII)
    # oshape           = (N, COO, OH // VH, OW // VH, VH, VW, COOO)
    ############################################################
    # -> transpose
    # O_shape          = (N, COO, OH, OW, COOO)

    dvshape = (N, CII, OH // VH, OW // VW, VH*HSTR + KH-1, VW*WSTR + KW-1, CIII)
    ovshape = (N, COO, OH // VH, OW // VW, VH, VW, VC)
    oshape = (N, COO, OH, OW, VC)

    data_vec = tvm.compute(dvshape, lambda n, cii, h, w, vh, vw, ciii:
                           data_pad[n][cii][h*VH*HSTR+vh][w*VW*WSTR+vw][ciii],
                           name='data_vec')

    cii = tvm.reduce_axis((0, CII), name='cii')
    ciii = tvm.reduce_axis((0, CIII), name='ciii')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    # conv = tvm.compute(ovshape, lambda n, coo, h, w, vh, vw, vc: \
    #     tvm.sum((data_vec[n, cii, h, w, vh*HSTR+kh, vw*WSTR+kw, ciii].astype("int16") - tvm.const(12, dtype="int16")).astype("int32") *
    #             (kernel[coo, cii, kh, kw, ciii, vc].astype("int16") - tvm.const(71, dtype="int16")).astype("int32"),
    #             axis=[cii, ciii, kh, kw]), name='conv')

    conv = tvm.compute(ovshape, lambda n, coo, h, w, vh, vw, vc: \
        tvm.sum((data_vec[n, cii, h, w, vh*HSTR+kh, vw*WSTR+kw, ciii]).astype("int32") *
                (kernel[coo, cii, kh, kw, ciii, vc]).astype("int32"),
                axis=[cii, ciii, kh, kw]), name='conv')

    output = tvm.compute(oshape, lambda n, coo, h, w, vc: conv[n][coo][h//VH][w//VW][h%VH][w%VW][vc],
                         name='output_unpack', tag='spatial_conv2d_output')
    flops = 2 * N * COO * OH * OW * VC * KH * KW * CII * CIII
    cfg.add_flop(flops)
    return output

def _schedule_spatial_pack_NCHWc(cfg, s, output, last):
    """schedule implementation"""
    # import ipdb
    # ipdb.set_trace()
    conv = output.op.input_tensors[0]
    data_vec = conv.op.input_tensors[0]
    data_pad = data_vec.op.input_tensors[0]
    # s[data_pad].compute_inline()

    kernel_vec = conv.op.input_tensors[1]
    n, coo, oh, ow, vh, vw, vc = s[conv].op.axis
    _, dvcii, dvoh, dvow, dvvh, dvvw, dvciii = s[data_vec].op.axis
    cii, ciii, kh, kw = s[conv].op.reduce_axis
    data_pad = data_vec.op.input_tensors[0]
    if data_pad.op.name == "data_pad":
        assert type(data_pad.op) == tvm.tensor.ComputeOp
        has_padding = True
    else:
        pass
        assert type(data_pad.op) == tvm.tensor.PlaceholderOp
        has_padding = False
    cfg.define_knob('data_pad_inline', [0, 1, 2, 3, 4])

    if cfg['data_pad_inline'].val == 1 and has_padding:
        s[data_pad].compute_inline()
    if cfg['data_pad_inline'].val == 2 and has_padding:
        s[data_pad].vectorize(list(s[data_pad].op.axis)[-1])
    if cfg['data_pad_inline'].val == 3 and has_padding:
        s[data_pad].vectorize(list(s[data_pad].op.axis)[-1])
        s[data_pad].compute_at(s[data_vec], dvoh)
    if cfg['data_pad_inline'].val == 4 and has_padding:
        s[data_pad].vectorize(list(s[data_pad].op.axis)[-1])
        s[data_pad].compute_at(s[data_vec], dvow)

    cfg.define_knob('data_vec_inline', [0, 1, 2, 3])
    if cfg['data_vec_inline'].val == 1:
        s[data_vec].compute_at(s[conv], oh)
    if cfg['data_vec_inline'].val == 2:
        s[data_vec].compute_at(s[conv], ow)
    if cfg['data_vec_inline'].val == 3:
        s[data_vec].compute_at(s[conv], coo)

    # schedule conv
    cfg["reorder_0"].apply(s, conv, [n, coo, cii, oh, ow, kh, kw, vc, ciii, vh, vw])
    cfg["ann_reduce"].apply(s, conv, [kh, kw, ciii],
                            axis_lens=[get_const_int(kh.dom.extent),
                                       get_const_int(kw.dom.extent),
                                       get_const_int(ciii.dom.extent)],
                            max_unroll=16,
                            cfg=cfg)
    cfg["ann_spatial"].apply(s, conv, [vc],
                             axis_lens=[
                                        get_const_int(vc.dom.extent)],
                             max_unroll=16,
                             cfg=cfg)
    s[conv].vectorize(vc)

    # schedule fusion
    n, coo, h, w, vc = s[last].op.axis
    s[last].vectorize(vc)
    oh, vh = cfg['tile_oh'].apply(s, last, h)
    ow, vw = cfg['tile_ow'].apply(s, last, w)
    cfg["reorder_1"].apply(s, last, [n, coo, oh, ow, vh, vw, vc])
    if last != output:
        s[output].compute_inline()
        cfg["ann_spatial"].apply(s, last, [vc],
                                 axis_lens=[get_const_int(vc.dom.extent)],
                                 max_unroll=16,
                                 cfg=cfg)
    else:
        s[last].vectorize(vc)
        pass

    cfg.define_knob('conv_inline', [1, 2, 3])
    if cfg['conv_inline'].val == 1:
        s[conv].compute_at(s[last], ow)
    if cfg['conv_inline'].val == 2:
        s[conv].compute_at(s[last], oh)
    if cfg['conv_inline'].val == 3:
        s[conv].compute_at(s[last], coo)

    # s[conv].compute_at(s[last], ow)

    _, _, _, _, vh, vw, vc = s[data_vec].op.axis
    cfg["ann_spatial"].apply(s, data_vec, [vc],
                             axis_lens=[get_const_int(vc.dom.extent)],
                             max_unroll=16,
                             cfg=cfg)
    # s[data_vec].vectorize(vc)
    # s[data_vec].unroll(vw)

    if kernel_vec.op.name == 'kernel_vec':
        co, _, _, _, _ = s[kernel_vec].op.axis
        s[kernel_vec].pragma(co, 'debug_skip_region')
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel packing will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[kernel_vec].pragma(co, 'debug_skip_region')
        else:
            pass
    return s

@autotvm.template
def conv2d_NCHWc_direct_autotvm(s, ic, oc, kernel, pad, stride):
    # ic = ((ic + 16 - 1) // 16) * 16
    # oc = ((oc + 16 - 1) // 16) * 16
    cfg = autotvm.get_config()
    cfg.define_knob('BNInput', [4,]) # TODO, 8, 16
    cfg.define_knob('BNOutput', [8, 16]) # TODO 8, 16
    BNInput = cfg['BNInput'].val
    BNOutput = cfg['BNOutput'].val
    X = tvm.placeholder(shape=(1, ic // BNInput, s, s, BNInput), dtype="int8", name="X")
    W = tvm.placeholder(shape=(oc // BNOutput, ic // BNInput, kernel, kernel, BNInput, BNOutput), dtype="int8", name="W")

    Y = _decl_spatial_pack_NCHWc(cfg, X, W, num_filter=oc, kernel_size=kernel, stride=stride, padding=pad, layout="NCHW{}c".format(BNInput), out_layout="NCHW{}c".format(BNOutput), out_dtype="int32")
    s = tvm.create_schedule([Y.op])
    s = _schedule_spatial_pack_NCHWc(cfg, s, Y, Y)
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))
    return s, [X, W, Y]


# @autotvm.template
# def conv2d_NCHW_direct_autotvm(s, ic, oc, kernel, pad, stride):
#     ic = ((ic + 16 - 1) // 16) * 16
#     oc = ((oc + 16 - 1) // 16) * 16
#     cfg = autotvm.get_config()
#     X = tvm.placeholder(shape=(1, s, s, ic), dtype="float32", name="X")
#     W = tvm.placeholder(shape=(oc, ic, kernel, kernel), dtype="float32", name="W")
#     Y = unet_conv2d._decl_spatial_pack(cfg, X, W, stride, pad, layout="NCHW", out_dtype="float32", num_tile=2)

#     conv = Y.op.input_tensors[0]

#     data_vec = conv.op.input_tensors[0]
#     data_pad = data_vec.op.input_tensors[0]

#     s = tvm.create_schedule([Y.op])
#     s[data_pad].compute_inline()

#     kernel_vec = conv.op.input_tensors[1]
#     if kernel_vec.op.name == 'kernel_vec':
#         kernel = kernel_vec.op.input_tensors[0]
#     else:
#         kernel = kernel_vec
#     if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
#         s[kernel].compute_inline()
#     s = unet_conv2d._schedule_spatial_pack(cfg, s, data_vec, kernel_vec, conv, Y, Y)
#     print(tvm.lower(s, [X, W, Y], simple_mode=True))
#     return s, [X, W, Y]




Workload = collections.namedtuple("Workload", ["space", "input_channel", "output_channel", "kernel", "pad", "stride"])

def a(x, align=16):
    if x < align:
        return align
    return ((x + align - 1) // align) * align
WORKLOADS = [
        # Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
        # Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
        # Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
        # Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
        # Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
        # Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
        # Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
        # Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
        # # workloads of resnet34_v1 on imagenet, no extra workload required
        # # workloads of resnet50_v1 on imagenet
        # Workload('float32', 'float32', 56, 56, 64, 256, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 256, 64, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 28, 28, 128, 512, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 256, 512, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 28, 28, 512, 128, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1),

        # Workload(space=102, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=102, input_channel=32, output_channel=32, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=64, output_channel=64, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
        # Workload(space=128, input_channel=64, output_channel=64, kernel=3, pad=1, stride=1),
        # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),

        # # # Workload(space=12, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
    # Workload(space=64, input_channel=a(64), output_channel=a(64), kernel=3, pad=1, stride=1),
    # Workload(space=96, input_channel=a(32), output_channel=a(16), kernel=3, pad=1, stride=1),
        # Workload(space=96, input_channel=a(12), output_channel=a(24), kernel=3, pad=1, stride=1),
        # Workload(space=48, input_channel=a(24), output_channel=a(48), kernel=3, pad=1, stride=1),
        # Workload(space=24, input_channel=a(48), output_channel=a(96), kernel=3, pad=1, stride=1),
        # Workload(space=12, input_channel=a(96), output_channel=a(180), kernel=3, pad=1, stride=1),
        # Workload(space=6, input_channel=a(180), output_channel=a(220), kernel=3, pad=1, stride=1),
        # Workload(space=6, input_channel=a(220), output_channel=a(180), kernel=3, pad=1, stride=1),
        # Workload(space=12, input_channel=a(180), output_channel=a(96), kernel=3, pad=1, stride=1),
        # Workload(space=24, input_channel=a(96), output_channel=a(48), kernel=3, pad=1, stride=1),
        # Workload(space=48, input_channel=a(48), output_channel=a(24), kernel=3, pad=1, stride=1),
        # Workload(space=96, input_channel=a(24), output_channel=a(12), kernel=3, pad=1, stride=1),
        # Workload(space=192, input_channel=a(12), output_channel=1, kernel=3, pad=1, stride=1),
        # Workload(space=192, input_channel=1, output_channel=1, kernel=3, pad=1, stride=1),
        Workload(space=192, input_channel=4, output_channel=a(12), kernel=3, pad=1, stride=1),
]

target = tvm.target.arm_cpu("rasp3b")# 'llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu'
local_target = 'llvm -mcpu=core-avx2'

@click.command()
@click.option('--autotvm_number', default=10)
@click.option('--autotvm_repeat', default=2)
@click.option('--autotvm_n_trial', default=200)
@click.option('--autotvm_early_stopping', default=100)
@click.option('--autotvm_log', default="autotvm_direct_benchmark.log", type=str)
@click.option('--layout', type=click.Choice(["NCHW", "NCHWc"]), required=True)
@click.option('--tracker_port', default=9195)
@click.option('--local', is_flag=True, default=False)
def run(layout,
        autotvm_number,
        autotvm_repeat,
        autotvm_log,
        autotvm_n_trial,
        autotvm_early_stopping,
        tracker_port,
        local):
    logging.basicConfig(level=logging.DEBUG)
    for i, w in enumerate(WORKLOADS):
        try:
            # if w.in_filter % 16 != 0 or w.out_filter % 16 != 0:
            #     continue
            measure_option=autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=80),
                runner=autotvm.RPCRunner(
                    'rpi', '0.0.0.0', tracker_port,
                    number=autotvm_number,
                    repeat=autotvm_repeat,
                    timeout=80) if not local else
                autotvm.LocalRunner(
                    timeout=80,
                    number=autotvm_number,
                    repeat=autotvm_repeat)
            )

            task = autotvm.task.create(
                conv2d_NCHWc_direct_autotvm,
                args=(w.space, w.input_channel, w.output_channel, w.kernel, w.pad, w.stride),
                target=tvm.target.create(target if not local else local_target))
            print(task.config_space)
            tuner = autotvm.tuner.XGBTuner(task, feature_type="knob")
            tuner.tune(
                n_trial=autotvm_n_trial,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(
                        autotvm_n_trial,
                        prefix="{w.space}S, {w.input_channel} -> {w.output_channel}, {w.kernel}K, {w.pad}P, {w.stride}s, {layout}".format(w=w, layout=layout)),
                    autotvm.callback.log_to_file(str(autotvm_log))])
        except:
            logging.exception("Failed on workload: %s", w)

if __name__ == "__main__":
    run()
