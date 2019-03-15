from tvm import autotvm
import tvm.contrib.graph_runtime
import click
import logging
import mxnet as mx
import numpy as np
import os
import time
import tvm
from tvm import relay

import models

skl_target = tvm.target.create(
    'llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu')

local_target = tvm.target.create('llvm -mcpu=core-avx2')
rpi_target = tvm.target.arm_cpu('rasp3b')


# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=2000,
               early_stopping=100,
               log_filename='tuning.log',
               use_transfer_learning=False):
    for i, tsk in enumerate(tasks):
        print(tsk)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = autotvm.tuner.XGBTuner(
                tsk, loss_type='rank', feature_type="knob")
        elif tuner == 'ga':
            tuner_obj = autotvm.tuner.GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = autotvm.tuner.RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = autotvm.tuner.GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(log_filename):
                tuner_obj.load_history(
                    autotvm.record.load_from_file(log_filename))

        # do tuning
        print(tsk.config_space)
        tuner_obj.tune(
            n_trial=min(n_trial, len(tsk.config_space)),
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename)
            ])


@click.command()
@click.option('--align', default=8)
@click.option(
    '--model', type=click.Choice(['unet', 'resnet50']), required=True)
@click.option('--autotvm_number', default=50)
@click.option('--autotvm_repeat', default=4)
@click.option('--autotvm_n_trial', default=200)
@click.option('--autotvm_early_stopping', default=100)
@click.option('--autotvm_log', default="autotvm_unet_tuning.log", type=str)
@click.option('--tracker_port', default=9195)
@click.option('--opt_level', default=3)
@click.option(
    '--device', type=click.Choice(["skl", "rpi", "local"]), required=True)
def run(align, model, autotvm_number, autotvm_repeat, autotvm_log,
        autotvm_n_trial, autotvm_early_stopping, tracker_port, opt_level,
        device):
    logging.basicConfig(level=logging.INFO)
    target = dict(skl=skl_target, rpi=rpi_target, local=local_target)[device]
    print(target)
    sym, image_shape, output_shape = models.get_mxnet_symbol(model, align)
    sym, params = models.get_relay_sym(sym, image_shape)
    assert params

    data_shape = tuple([1] + list(image_shape))
    with relay.build_config(opt_level=opt_level):
        graph = relay.optimize(sym, target, params=params)

    from tvm.relay import quantize as qtz

    # with qtz.qconfig(skip_k_conv=0,
    #                  nbit_input=8,
    #                  nbit_weight=8,
    #                  global_scale=8,
    #                  dtype_input="int8",
    #                  dtype_weight="int8",
    #                  dtype_activation="int32",
    #                  store_lowbit_output=False,
    #                  debug_enabled_ops=None):
    with qtz.qconfig(skip_k_conv=0, global_scale=4.0, round_for_shift=False):
        qgraph = qtz.quantize(graph, params)
        qgraph = relay.ir_pass.infer_type(qgraph)
        # print(qgraph.astext(show_meta_data=False))

    with relay.build_config(opt_level=opt_level):
        qgraph = relay.optimize(qgraph, target, params=params)
        qgraph = relay.ir_pass.infer_type(qgraph)
        print("Optimized graph")
        print(qgraph.astext(show_meta_data=False))
    with relay.build_config(opt_level=opt_level):
        qgraph_json, lib, params = relay.build(qgraph, target, params=params)
    import netron
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix="tvm.json") as f:
        f.write(qgraph_json)
    netron.start(f.name, host="localhost")
    # with target:
    #     with relay.build_config(opt_level=opt_level):
    #         qgraph_json, lib, params = relay.build(qgraph, target, params=params)

    # with target:
    #     with relay.build_config(opt_level=2):
    #         tasks = autotvm.task.extract_from_program(
    #             qgraph, target=target, params=params, ops=(relay.op.nn.conv2d, ))
    # tasks = list(reversed(tasks))
    # logging.info("Got %s tasks", len(tasks))
    # for i, task in enumerate(tasks):
    #     logging.info("Task %s: %s", i, task)
    # # tasks = tasks[1:]
    # tune_tasks(
    #     tasks,
    #     measure_option=autotvm.measure_option(
    #         builder=autotvm.LocalBuilder(timeout=50),
    #         runner=autotvm.RPCRunner(
    #             device,
    #             '0.0.0.0',
    #             tracker_port,
    #             number=autotvm_number,
    #             repeat=autotvm_repeat,
    #             timeout=50)),
    #     n_trial=autotvm_n_trial,
    #     early_stopping=autotvm_early_stopping,
    #     log_filename=str(autotvm_log))


if __name__ == '__main__':
    run()
