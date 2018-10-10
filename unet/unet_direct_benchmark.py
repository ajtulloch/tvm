from __future__ import division

import click
from topi.util import get_const_int, const_matrix
from topi.nn.conv2d import Workload
import numpy as np
import tvm
import tvm.rpc
from tvm import autotvm
import unet_direct_NCHWc
import collections
import logging
import sys
import unet_conv2d

@autotvm.template
def conv2d_NCHWc_direct_autotvm(s, ic, oc, kernel, pad, stride):
    ic = ((ic + 16 - 1) // 16) * 16
    oc = ((oc + 16 - 1) // 16) * 16
    cfg = autotvm.get_config()
    cfg.define_knob('BNInput', [8, 16])
    cfg.define_knob('BNOutput', [8, 16])
    BNInput = cfg['BNInput'].val
    BNOutput = cfg['BNOutput'].val
    X = tvm.placeholder(shape=(1, ic // BNInput, s, s, BNInput), dtype="float32", name="X")
    W = tvm.placeholder(shape=(oc // BNOutput, ic // BNInput, kernel, kernel, BNInput, BNOutput), dtype="float32", name="W")

    Y = unet_direct_NCHWc.decl_direct_NCHWc(cfg, X, W, strides=stride, padding=pad, out_dtype="float32")
    s = unet_direct_NCHWc.schedule_direct_NCHWc(cfg, Y)
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))
    return s, [X, W, Y]


@autotvm.template
def conv2d_NCHW_direct_autotvm(s, ic, oc, kernel, pad, stride):
    ic = ((ic + 16 - 1) // 16) * 16
    oc = ((oc + 16 - 1) // 16) * 16
    cfg = autotvm.get_config()
    X = tvm.placeholder(shape=(1, s, s, ic), dtype="float32", name="X")
    W = tvm.placeholder(shape=(oc, ic, kernel, kernel), dtype="float32", name="W")
    Y = unet_conv2d._decl_spatial_pack(cfg, X, W, stride, pad, layout="NCHW", out_dtype="float32", num_tile=2)

    conv = Y.op.input_tensors[0]

    data_vec = conv.op.input_tensors[0]
    data_pad = data_vec.op.input_tensors[0]

    s = tvm.create_schedule([Y.op])
    s[data_pad].compute_inline()

    kernel_vec = conv.op.input_tensors[1]
    if kernel_vec.op.name == 'kernel_vec':
        kernel = kernel_vec.op.input_tensors[0]
    else:
        kernel = kernel_vec
    if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()
    s = unet_conv2d._schedule_spatial_pack(cfg, s, data_vec, kernel_vec, conv, Y, Y)
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))
    return s, [X, W, Y]




# Workload = collections.namedtuple("Workload", ["space", "input_channel", "output_channel", "kernel", "pad", "stride"])

WORKLOADS = [
        Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
        Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
        # workloads of resnet34_v1 on imagenet, no extra workload required
        # workloads of resnet50_v1 on imagenet
        Workload('float32', 'float32', 56, 56, 64, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 512, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1),

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
        # Workload(space=192, input_channel=3, output_channel=12, kernel=3, pad=1, stride=1),
        # Workload(space=96, input_channel=12, output_channel=24, kernel=3, pad=1, stride=1),
        # Workload(space=48, input_channel=24, output_channel=48, kernel=3, pad=1, stride=1),
        # Workload(space=24, input_channel=48, output_channel=96, kernel=3, pad=1, stride=1),
        # Workload(space=12, input_channel=96, output_channel=180, kernel=3, pad=1, stride=1),
        # Workload(space=6, input_channel=180, output_channel=220, kernel=3, pad=1, stride=1),
        # Workload(space=6, input_channel=220, output_channel=180, kernel=3, pad=1, stride=1),
        # Workload(space=12, input_channel=180, output_channel=96, kernel=3, pad=1, stride=1),
        # Workload(space=24, input_channel=96, output_channel=48, kernel=3, pad=1, stride=1),
        # Workload(space=48, input_channel=48, output_channel=24, kernel=3, pad=1, stride=1),
        # Workload(space=96, input_channel=24, output_channel=12, kernel=3, pad=1, stride=1),
        # Workload(space=192, input_channel=12, output_channel=1, kernel=3, pad=1, stride=1),
]

target = 'llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu'

@click.command()
@click.option('--autotvm_number', default=50)
@click.option('--autotvm_repeat', default=4)
@click.option('--autotvm_n_trial', default=200)
@click.option('--autotvm_early_stopping', default=100)
@click.option('--autotvm_log', default="autotvm_direct_benchmark.log", type=str)
@click.option('--layout', type=click.Choice(["NCHW", "NCHWc"]), required=True)
@click.option('--tracker_port', default=9195)
def run(layout,
        autotvm_number,
        autotvm_repeat,
        autotvm_log,
        autotvm_n_trial,
        autotvm_early_stopping,
        tracker_port):
    logging.basicConfig(level=logging.DEBUG)
    for i, w in enumerate(WORKLOADS):
        if w.in_filter % 16 != 0 or w.out_filter % 16 != 0:
            continue
        measure_option=autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.RPCRunner(
                'skl', 'localhost', tracker_port,
                number=autotvm_number,
                repeat=autotvm_repeat,
                timeout=10)
        )

        task = autotvm.task.create(
            conv2d_NCHWc_direct_autotvm if layout == "NCHWc" else conv2d_NCHW_direct_autotvm,
            args=(w.height, w.in_filter, w.out_filter, w.hkernel, w.hpad, w.hstride),
            target=tvm.target.create(target))
        print(task.config_space)
        tuner = autotvm.tuner.XGBTuner(task, feature_type="knob")
        tuner.tune(
            n_trial=autotvm_n_trial,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(
                    autotvm_n_trial,
                    prefix="{w.height}S, {w.in_filter} -> {w.out_filter}, {w.hkernel}K, {w.hpad}P, {w.hstride}s, {layout}".format(w=w, layout=layout)),
                autotvm.callback.log_to_file(str(autotvm_log))])

if __name__ == "__main__":
    run()
