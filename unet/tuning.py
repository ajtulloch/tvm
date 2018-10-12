from tvm import autotvm
import tvm.contrib.graph_runtime
import click
import logging
import mxnet as mx
import nnvm
import nnvm.compiler
import numpy as np
import os
import time
import tvm
import tvm_overrides
import unet_conv2d

import models

target = 'llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu'
local_target = 'llvm -mcpu=core-avx2'


def build_until_compile(graph, target=None, shape=None, dtype="float32",
                        params=None, target_host=None, layout=None):
    from nnvm.compiler.build_module import (
        BuildConfig, _update_shape_dtype, _graph, graph_attr, graph_util,
        _all_var_init, initialize_variables, optimize, _remove_noref_params, precompute_prune)
    target = target if target else tvm.target.current_target()
    if target is None:
        raise ValueError("Target is not set in env or passed as argument.")
    target = tvm.target.create(target)

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(target)
    else:
        tophub_context = autotvm.util.EmptyContext()

    with tophub_context:
        shape = shape if shape else {}
        if not isinstance(shape, dict):
            raise TypeError("require shape to be dict")
        for value in shape.values():
            if not all(isinstance(x, int) for x in value):
                raise TypeError("shape value must be int iterator")

        cfg = BuildConfig.current
        graph = graph if isinstance(graph, _graph.Graph) else _graph.create(graph)
        shape, dtype = _update_shape_dtype(shape, dtype, params)

        # correct layout if necessary
        layout = layout if layout else {}
        graph = graph_attr.set_layout_inputs(graph, layout)
        graph = graph.apply("CorrectLayout")
        index = graph.index
        layouts = graph.json_attr("layout")
        layout = {x: layouts[index.entry_id(x)] for x in index.input_names}

        # Initial pass do shape type inference
        ishape, _ = graph_util.infer_shape(graph, **shape)
        shape.update(zip(graph.index.input_names, ishape))
        if not isinstance(dtype, str):
            idtype, _ = graph_util.infer_dtype(graph, **dtype)
            dtype.update(zip(graph.index.input_names, idtype))
        # Initialize all variables specified in _all_var_init
        init_var = {}
        if _all_var_init:
            initialize_variables(shape, dtype)
        # Apply optimization
        with target:
            graph = optimize(graph, shape, dtype, layout)

        # Clear extra params without nodes.
        _remove_noref_params(params, graph)

        # Precompute prune
        graph, params = precompute_prune(graph, params)
        shape, dtype = _update_shape_dtype(shape, dtype, params)

        # Operator Fusion and generation
        graph = graph_attr.set_shape_inputs(graph, shape)
        graph = graph.apply("InferShape")
        graph = graph_attr.set_dtype_inputs(graph, dtype)
        graph._set_json_attr("target", str(target), "str")
        if target_host is not None:
            graph._set_json_attr("target_host", str(target_host), "str")

        return graph, params


# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=2000,
               early_stopping=100,
               log_filename='tuning.log',
               use_transfer_learning=False):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        print(tsk)
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type='rank', feature_type="knob")
        elif tuner == 'ga':
            tuner_obj = autotvm.tuner.GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = autotvm.tuner.RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = autotvm.tuner.GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


@click.command()
@click.option('--align', default=8)
@click.option('--model', type=click.Choice(['unet', 'resnet50']), required=True)
@click.option('--autotvm_number', default=50)
@click.option('--autotvm_repeat', default=4)
@click.option('--autotvm_n_trial', default=200)
@click.option('--autotvm_early_stopping', default=100)
@click.option('--autotvm_log', default="autotvm_unet_tuning.log", type=str)
@click.option('--tracker_port', default=9195)
@click.option('--opt_level', default=3)
def run(align,
        model,
        autotvm_number,
        autotvm_repeat,
        autotvm_log,
        autotvm_n_trial,
        autotvm_early_stopping,
        tracker_port,
        opt_level):
    logging.basicConfig(level=logging.DEBUG)
    sym, image_shape, output_shape = models.get_mxnet_symbol(model, align)
    sym, params = models.get_nnvm_sym(sym, image_shape)
    # import ipdb; ipdb.set_trace()
    # print(params)
    assert params

    data_shape = tuple([1] + list(image_shape))
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(sym, target, dict(data=data_shape), params=params)
    # with nnvm.compiler.build_config(opt_level=opt_level):
    #     sym, params = build_until_compile(sym, target, dict(data=data_shape), params=params)
    print("Succesfully built")
    # # print(sym.symbol().debug_str())
    # # import ipdb; ipdb.set_trace()
    with nnvm.compiler.build_config(opt_level=opt_level):
        tasks = autotvm.task.extract_from_graph(
            sym,
            target=target,
            shape=dict(data=data_shape),
            dtype="float32",
            symbols=[
                nnvm.sym.conv2d,
                nnvm.sym.contrib.conv2d_NCHWc,
            ]
        )
    print(tasks)
    # import ipdb; ipdb.set_trace()
    tune_tasks(tasks,
               measure_option=autotvm.measure_option(
                   builder=autotvm.LocalBuilder(timeout=50),
                   runner=autotvm.RPCRunner(
                       'skl', 'localhost', tracker_port,
                       number=autotvm_number,
                       repeat=autotvm_repeat,
                       timeout=50)
               ),
               n_trial=autotvm_n_trial,
               early_stopping=autotvm_early_stopping,
               log_filename=str(autotvm_log))

if __name__ == '__main__':
    run()
