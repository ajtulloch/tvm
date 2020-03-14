import tvm
from tvm import relay
import numpy as np
from tvm import autotvm
# import tvm.contrib.graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime as graph_runtime
import logging
import sys
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

def leaky_relu(x, alpha):
    return relay.nn.relu(x)

def relay_model(input_size, ngf, n_residual_layers):
    ratios = [8, 8, 2, 2]
    mult = int(2 ** len(ratios))

    x_var = relay.var('x', shape=[1, 32, input_size])
    x = x_var
    params = {}

    def conv1d_params(name, in_channel, out_channel, kernel_size):
        w, b = "{name}_w".format(name=name), "{name}_b".format(name=name)
        if w in params and b in params:
            return params[w], params[b]
        params[w] = relay.var(w, shape=[kernel_size, in_channel, out_channel])
        params[b] = relay.var(b, shape=[out_channel])
        return params[w], params[b]

    def conv1d_transpose_params(name, in_channel, out_channel, kernel_size):
        w, b = "{name}_w".format(name=name), "{name}_b".format(name=name)
        if w in params and b in params:
            return params[w], params[b]
        params[w] = relay.var(w, shape=[kernel_size, in_channel, out_channel])
        params[b] = relay.var(b, shape=[out_channel])
        return params[w], params[b]


    x = relay.nn.pad(x, pad_mode='reflect', pad_width=((0, 0), (3, 3), (0, 0)))
    (w, b) = conv1d_params("conv_first", input_size, mult * ngf, 7)
    x = relay.nn.bias_add(relay.nn.conv1d(x, w, data_layout="NWC", kernel_layout="WIO"), b, axis=2)

    # Upsample to raw audio scale
    for i, r in enumerate(ratios):
        (w, b) = conv1d_transpose_params("conv1_ratios_{i}".format(i=i), mult * ngf, mult * ngf // 2, r * 2)
        x = leaky_relu(x, 0.2)
        x = relay.nn.bias_add(relay.nn.conv1d_transpose(x, w, data_layout="NWC", kernel_layout="WOI", out_layout="NWC", strides=(r,), padding=(r // 2 + r % 2,), output_padding=(r % 2,)), b, axis=2)

        for j in range(n_residual_layers):
            # model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]
            def resnet_block(y, dim, dilation):
                (w_shortcut, b_shortcut) = conv1d_params("conv1d_shortcut_{i}_{j}".format(i=i, j=j), dim, dim, 1)
                y_shortcut = relay.nn.bias_add(relay.nn.conv1d(y, w_shortcut, data_layout="NWC", kernel_layout="WIO", out_layout="NWC", ), b_shortcut, axis=2)

                (w_block_0, b_block_0) = conv1d_params("conv1d_resblock_{i}_{j}_0".format(i=i, j=j), dim, dim, 3)
                (w_block_1, b_block_1) = conv1d_params("conv1d_resblock_{i}_{j}_1".format(i=i, j=j), dim, dim, 1)

                y = leaky_relu(y, 0.2)
                y = relay.nn.pad(x, pad_mode='reflect', pad_width=((0, 0), (dilation, dilation), (0, 0)))
                y = relay.nn.bias_add(relay.nn.conv1d(y, w_block_0, dilation=dilation, data_layout="NWC", kernel_layout="WIO", out_layout="NWC", ), b_block_0, axis=2)
                y = leaky_relu(y, 0.2)
                y = relay.nn.bias_add(relay.nn.conv1d(y, w_block_1, data_layout="NWC", kernel_layout="WIO", out_layout="NWC"), b_block_1, axis=2)
                return y + y_shortcut
            x = resnet_block(x, dim=mult * ngf // 2, dilation=3 ** j)
        mult //= 2

    x = leaky_relu(x, 0.2)
    x = relay.nn.pad(x, pad_mode='reflect', pad_width=((0, 0), (3, 3), (0, 0)))

    (w, b) = conv1d_params("conv_last", ngf, 1, 7)
    x = relay.nn.bias_add(relay.op.nn.conv1d(x, w, padding=0, data_layout="NWC", kernel_layout="WIO", out_layout="NWC"), b, axis=2)
    x = relay.tanh(x)
    outputs = relay.expr.Tuple([x])
    func = relay.Function(relay.analysis.free_vars(outputs), outputs)
    return func, x_var, params

func, x_var, params = relay_model(80, 32, 3)
tvm_params = {k: tvm.nd.array(np.random.randn(*v.type_annotation.concrete_shape).astype(np.float32)) for k, v in params.items()}
tvm_x_nd = np.random.randn(*x_var.type_annotation.concrete_shape).astype(np.float32)
module = tvm.ir.module.IRModule.from_expr(func)
print(module.astext(show_meta_data=False))

log_file = "melgan.log"
target = tvm.target.create("llvm -mcpu=core-avx2")


def tune():
    tuning_option = {
        'log_filename': log_file,
        'tuner': 'random',
        'early_stopping': None,

        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(number=2, repeat=2,
                                       min_repeat_ms=100),
        ),
    }


    # You can skip the implementation of this function for this tutorial.
    def tune_kernels(tasks,
                     measure_option,
                     tuner='gridsearch',
                     early_stopping=None,
                     log_filename='tuning.log'):

        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

            # create tuner
            if tuner == 'xgb' or tuner == 'xgb-rank':
                tuner_obj = autotvm.tuner.XGBTuner(task, loss_type='rank')
            elif tuner == 'random':
                tuner_obj = autotvm.tuner.RandomTuner(task)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            # do tuning
            print(task.config_space)
            n_trial=len(task.config_space)
            tuner_obj.tune(n_trial=20,
                           early_stopping=early_stopping,
                           measure_option=measure_option,
                           callbacks=[
                               autotvm.callback.progress_bar(n_trial, prefix=prefix),
                               autotvm.callback.log_to_file(log_filename)])


    # Use graph tuner to achieve graph level optimal schedules
    tasks = autotvm.task.extract_from_program(module["main"], target=target,
                                              params=tvm_params,
                                              ops=(
                                                  relay.op.get("nn.conv1d"),
                                                  relay.op.get("nn.conv1d_transpose"),
                                              ))

    # run tuning tasks
    tune_kernels(tasks, **tuning_option)


# tune()

def test():

    # compile kernels with graph-level best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                module, target=target, params=tvm_params)

        # upload parameters to device
        ctx = tvm.cpu()
        mod = graph_runtime.create(graph, lib, ctx)
        mod.set_input('x', tvm_x_nd)
        mod.set_input(**params)
        mod.run()
        mod.run()
        # evaluate
        # print("Evaluate inference time cost...")
        # ftimer = mod.module.time_evaluator("run", ctx, min_repeat_ms=100)
        # prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        # print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
        #       (np.mean(prof_res), np.std(prof_res)))

test()
# opt_level = 3
# target = tvm.target.create("llvm -mcpu=core-avx2")
# with tvm.relay.build_config(opt_level=opt_level):
#     graph, lib, params = tvm.relay.build_module.build(
#         module, target, params=tvm_params)
