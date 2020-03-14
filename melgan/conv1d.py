import logging
import sys

import numpy as np
import tvm
from tvm import te

# the module is called `autotvm`
from tvm import autotvm

# import logging
# import sys
# import topi
# from topi.nn.util import get_pad_tuple1d
# from topi.util import simplify
# from topi.nn import pad
# import numpy as np
# import tvm
# from tvm import te

# # the module is called `autotvm`
# from tvm import autotvm

@autotvm.template("tutorial/conv1d")
def my_conv1d(CI, CO, W, K):
    A = te.placeholder((1, CI, W), name='A')
    B = te.placeholder((CO, CI, K), name='B')
    def conv1d_ncw(data,
                   kernel,
                   strides=1,
                   padding=0,
                   dilation=1):
        out_dtype = "float32"
        batch, in_channels, data_width = topi.util.get_const_tuple(data.shape)
        out_channels, _, kernel_size = topi.util.get_const_tuple(kernel.shape)

        # Compute the output shape
        dilated_kernel_size = (kernel_size - 1) * dilation + 1
        pad_left, pad_right = get_pad_tuple1d(padding, (dilated_kernel_size, ))
        out_channels = simplify(out_channels)
        out_width = simplify(
            (data_width - dilated_kernel_size + pad_left + pad_right) // strides + 1)

        # Apply padding
        pad_before = [0, 0, pad_left]
        pad_after = [0, 0, pad_right]
        temp = pad(data, pad_before, pad_after, name='pad_temp')

        # Compute graph
        rc = te.reduce_axis((0, in_channels), name='rc')
        rw = te.reduce_axis((0, kernel_size), name='rw')
        return te.compute(
            (batch, out_channels, out_width),
            lambda b, c, w: te.sum(
                temp[b, rc, w * strides + rw * dilation].astype(out_dtype)
                * kernel[c, rc, rw].astype(out_dtype),
                axis=[rc, rw]),
            tag="conv1d_ncw")

    C = conv1d_ncw(A, B)

    s = te.create_schedule(C.op)

    # schedule
    n, c, w = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    cfg = autotvm.get_config()
    cfg.define_split("tile_c", c, num_outputs=2)
    cfg.define_split("tile_w", w, num_outputs=2)

    # schedule according to config
    co, ci = cfg["tile_c"].apply(s, C, c)
    wo, wi = cfg["tile_w"].apply(s, C, w)

    cfg.define_annotate('ann_reduce', [k], policy='try_unroll')
    cfg.define_annotate('ann_spatial', [ci, wi], policy='try_unroll_vec')
    cfg['ann_reduce'].apply(s, C, [k])
    cfg['ann_spatial'].apply(s, C, [ci, wi])
    s[C].reorder(n, co, wo, k, ci, wi)
    assert C.op.input_tensors[0].op.name == "pad_temp"
    s[C.op.input_tensors[0].op].compute_inline()
    return s, [A, B, C]


print("Got here")
CI, CO, W, K = 64, 64, 32, 3
task = autotvm.task.create("tutorial/conv1d", args=(CI, CO, W, K), target='llvm -mcpu=core-avx2')
print(task.config_space)


# logging config (for printing tuning log to the screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

# There are two steps for measuring a config: build and run.
# By default, we use all CPU cores to compile program. Then measure them sequentially.
# We measure 5 times and take average to reduce variance.
measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

# Begin tuning with RandomTuner, log records to file `matmul.log`
# You can use alternatives like XGBTuner.
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(n_trial=10,
           measure_option=measure_option,
           callbacks=[])
