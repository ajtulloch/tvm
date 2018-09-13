from __future__ import division

from topi.util import get_const_int, const_matrix
import numpy as np
import tvm
import tvm.rpc
from tvm import autotvm
import unet_winograd_NCHWc
import collections
import logging
import sys


@autotvm.template
def conv2d_NCHWc_winograd_autotvm(s, ic, oc):
    ic = ((ic + 7) // 8) * 8
    oc = ((oc + 7) // 8) * 8
    cfg = autotvm.get_config()
    cfg.define_knob('unroll', [1])
    cfg.define_knob('compute_at', [0])
    cfg.define_knob('vectorize', [1])
    cfg.define_knob('tensorize', [1])
    cfg.define_knob('VK,VP', [(4, 24), (6, 16)])

    for intermediate in ["M", "A_T_dot_M", "input_tile", "B_T_dot_X", "V", "Y"]:
        cfg.define_knob("{}_COMPUTE_AT".format(intermediate), [0, 1])
    for intermediate in ["Y", "data_pad", "input_tile", "output"]:
        cfg.define_knob("{}_VECTORIZE".format(intermediate), [0, 1])
    for intermediate in ["input_tile", "V", "B_T_dot_X"]: # , "B_T_dot_X",
        cfg.define_knob("{}_REORDER_C".format(intermediate), [0, 1])

    cfg.define_knob('data_pad_inline', [0, 1])
    (VK, VP) = cfg['VK,VP'].val
    X = tvm.placeholder(shape=(1, ic // 8, s, s, 8), dtype="float32", name="X")
    W = tvm.placeholder(shape=(oc // 8, ic // 8, 3, 3, 8, 8), dtype="float32", name="W")

    output = unet_winograd_NCHWc.decl_winograd_NCHWc(cfg, X, W, strides=1, padding=1, out_dtype="float32", VK=VK, VP=VP)
    s = unet_winograd_NCHWc.schedule_winograd_NCHWc(cfg, output, VK=VK, VP=VP)
    if cfg.flop == 0:
        cfg.add_flop(2 * ic * oc * s * s * 3 * 3)
    print(tvm.lower(s, [X, W, output], simple_mode=True))
    return s, [X, W, output]


Workload = collections.namedtuple("Workload", ["space", "input_channel", "output_channel", "kernel", "pad", "stride"])

WORKLOADS = [
        Workload(space=102, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
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

def run():
    logging.basicConfig(level=logging.DEBUG)
    for i, w in enumerate(WORKLOADS):
        measure_option=autotvm.measure_option(
            builder=autotvm.LocalBuilder(n_parallel=1),
            runner=autotvm.LocalRunner(
                number=3,
                repeat=10,
                timeout=500)
        )

        task = autotvm.task.create(
            conv2d_NCHWc_winograd_autotvm,
            args=(w.space, w.input_channel, w.output_channel),
            target=tvm.target.create('llvm -mcpu=core-avx2'))
        print(task.config_space)
        tuner = autotvm.tuner.XGBTuner(task, feature_type="knob")
        job_name = 'conv2d_minimal_winograd_{w.space}_{w.input_channel}_{w.output_channel}'.format(w=w)
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
