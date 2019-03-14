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


def _decl_spatial_pack(cfg, data, kernel, strides, padding, dilation, layout, out_dtype, num_tile):
    assert layout == "NCHW", "Only support NCHW"
    # create workload according to raw arguments
    out_dtype = out_dtype or data.dtype
    N, CI, IH, IW = get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:
        pre_packed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        pre_packed = True
        CO, _, KH, KW, VC = get_const_tuple(kernel.shape)
        CO = CO * VC

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    OH = (IH + pad_top + pad_bottom - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1
    data_pad = pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_bottom, pad_right])
    flops = 2 * N * CO * OH * OW * KH * KW * CI
    print(flops)

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

    cfg.define_annotate("ann_reduce", [kh, kw], policy='try_unroll')
    cfg.define_annotate("ann_spatial", [vh, vw, vc], policy='try_unroll_vec')

    # fallback support
    if cfg.is_fallback:
        if num_tile == 2:     # arm cpu
            ref_log = autotvm.tophub.load_reference_log('arm_cpu', 'rk3399', 'conv2d', 'direct')
            cfg.fallback_with_reference_log(ref_log)
        elif num_tile == 3:  # mali gpu
            ref_log = autotvm.tophub.load_reference_log('mali', 'rk3399', 'conv2d', 'direct')
            cfg.fallback_with_reference_log(ref_log)
    # ====================================================================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    kvshape = (CO // VC, CI, KH, KW, VC)
    ovshape = (N, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (N, CO, OH, OW)

    if dilation_h != 1 or dilation_w != 1:
        # undilate input data
        dvshape = (N, OH // VH, OW // VW, CI, KH, KW, VH, VW)
        data_vec = tvm.compute(dvshape, lambda n, h, w, ci, kh, kw, vh, vw:
                               data_pad[n][ci][(h*VH+vh)*HSTR+kh*dilation_h]
                               [(w*VW+vw)*WSTR+kw*dilation_w],
                               name='data_vec_undilated')
    else:
        dvshape = (N, OH // VH, OW // VW, CI, VH*HSTR + KH-1, VW*WSTR + KW-1)
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

    if dilation_h != 1 or dilation_w != 1:
        conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
            tvm.sum(data_vec[n, h, w, ci, kh, kw, vh, vw].astype(out_dtype) *
                    kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                    axis=[ci, kh, kw]), name='conv')
    else:
        conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
            tvm.sum(data_vec[n, h, w, ci, vh*HSTR+kh, vw*WSTR+kw].astype(out_dtype) *
                    kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                    axis=[ci, kh, kw]), name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w:
                         conv[n][co//VC][h//VH][w//VW][h%VH][w%VW][co%VC],
                         name='output_unpack', tag='spatial_conv2d_output')
    cfg.add_flop(flops)
    return output

def _schedule_spatial_pack(cfg, s, output, last):
    """schedule implementation"""
    conv = output.op.input_tensors[0]
    data_vec = conv.op.input_tensors[0]
    data_pad = data_vec.op.input_tensors[0]
    # s[data_pad].compute_inline()

    kernel_vec = conv.op.input_tensors[1]

    n, co, oh, ow, vh, vw, vc = s[conv].op.axis
    ci, kh, kw = s[conv].op.reduce_axis

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
    s[last].reorder(n, co, oh, ow, vh, vw, vc)
    if last != output:
        s[output].compute_inline()
        cfg["ann_spatial"].apply(s, last, [vh, vw, vc],
                                 axis_lens=[cfg['tile_oh'].size[-1],
                                            cfg['tile_ow'].size[-1],
                                            cfg['tile_co'].size[-1]],
                                 max_unroll=16,
                                 cfg=cfg)
    s[conv].compute_at(s[last], ow)

    # mark parallel
    s[last].parallel(co)

    if data_vec.op.name == 'data_vec_undilated':
        _, h, _, _, _, _, _, _ = s[data_vec].op.axis
    else:
        _, h, _, _, _, _ = s[data_vec].op.axis
    s[data_vec].parallel(h)

    if kernel_vec.op.name == 'kernel_vec':
        co, _, _, _, _ = s[kernel_vec].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel packing will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[kernel_vec].pragma(co, 'debug_skip_region')
        else:
            s[kernel_vec].parallel(co)
    elif kernel_vec.op.name == 'kernel_vec_conv2d_transpose':  # for conv2d transpose
        co, _, _, _, _ = s[kernel_vec].op.axis
        s[kernel_vec].parallel(co)

    return s


@autotvm.template
def conv2d_NCHW_direct_autotvm(s, ic, oc, kernel, pad, stride):
    # ic = ((ic + 16 - 1) // 16) * 16
    # oc = ((oc + 16 - 1) // 16) * 16
    cfg = autotvm.get_config()
    X = tvm.placeholder(shape=(1, ic, s, s), dtype="int8", name="X")
    W = tvm.placeholder(shape=(oc, ic, kernel, kernel), dtype="int8", name="W")
    Y = _decl_spatial_pack(cfg, X, W, strides=stride, padding=pad, dilation=(1, 1), layout="NCHW", out_dtype="int32", num_tile=2)
    s = tvm.create_schedule([Y.op])
    s = _schedule_spatial_pack(cfg, s, Y, Y)
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
        Workload(space=192, input_channel=a(12), output_channel=1, kernel=3, pad=1, stride=1),
        Workload(space=192, input_channel=1, output_channel=1, kernel=3, pad=1, stride=1),
        Workload(space=192, input_channel=3, output_channel=a(12), kernel=3, pad=1, stride=1),
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
                conv2d_NCHW_direct_autotvm,
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
