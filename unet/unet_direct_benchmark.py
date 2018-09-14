from __future__ import division

from topi.util import get_const_int, const_matrix
import numpy as np
import tvm
import tvm.rpc
from tvm import autotvm
import unet_direct_NCHWc
import collections
import logging
import sys


@autotvm.template
def conv2d_NCHWc_direct_autotvm(s, ic, oc):
    ic = ((ic + 7) // 16) * 16
    oc = ((oc + 7) // 16) * 16
    cfg = autotvm.get_config()
    cfg.define_knob('BNInput', [8, 16])
    cfg.define_knob('BNOutput', [8, 16])
    BNInput = cfg['BNInput'].val
    BNOutput = cfg['BNOutput'].val
    X = tvm.placeholder(shape=(1, ic // BNInput, s, s, BNInput), dtype="float32", name="X")
    W = tvm.placeholder(shape=(oc // BNOutput, ic // BNInput, 3, 3, BNInput, BNOutput), dtype="float32", name="W")

    output = unet_direct_NCHWc.decl_direct_NCHWc(cfg, X, W, strides=1, padding=1, out_dtype="float32")
    s = unet_direct_NCHWc.schedule_direct_NCHWc(cfg, output)
    return s, [X, W, output]



Workload = collections.namedtuple("Workload", ["space", "input_channel", "output_channel", "kernel", "pad", "stride"])

WORKLOADS = [
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
        Workload(space=48, input_channel=48, output_channel=24, kernel=3, pad=1, stride=1),
        Workload(space=96, input_channel=24, output_channel=12, kernel=3, pad=1, stride=1),
        # Workload(space=192, input_channel=12, output_channel=1, kernel=3, pad=1, stride=1),
]

def run():
    logging.basicConfig(level=logging.DEBUG)
    for i, w in enumerate(WORKLOADS):
        measure_option=autotvm.measure_option(
            builder=autotvm.LocalBuilder(n_parallel=1, timeout=500),
            runner=autotvm.LocalRunner(
                number=50,
                repeat=4,
                timeout=500)
        )

        task = autotvm.task.create(
            conv2d_NCHWc_direct_autotvm,
            args=(w.space, w.input_channel, w.output_channel),
            target=tvm.target.create('llvm -mcpu=core-avx2'))
        print(task.config_space)
        tuner = autotvm.tuner.XGBTuner(task, feature_type="knob")
        job_name = 'conv2d_directbn_{w.space}_{w.input_channel}_{w.output_channel}'.format(w=w)
        # try:
        #     tuner.load_history(autotvm.record.load_from_file("{}.log".format(job_name)))
        # except Exception as e:
        #     logging.exception("Failed to load history file")
        n_trial = 200
        tuner.tune(
            n_trial=n_trial,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=job_name),
                autotvm.callback.log_to_file('{}.log'.format(job_name))])

if __name__ == "__main__":
    run()
