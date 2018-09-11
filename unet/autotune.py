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
import unet
import topi

target = 'llvm -mcpu=core-avx2'
ctx = tvm.context(str(target), 0)

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
            init_var = initialize_variables(shape, dtype)
        # Apply optimization
        with target:
            graph = optimize(graph, shape, dtype, layout)

        # Clear extra params without nodes.
        _remove_noref_params(params, graph)

        # Precompute prune
        if params and cfg.pass_enabled("PrecomputePrune"):
            graph, params = precompute_prune(graph, params)
        return graph, params


# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=400,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type='rank')
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
@click.option('--align_8', default=0)
@click.option('--num_iter', default=10)
@click.option('--num_cycles', default=5)
@click.option('--opt_level', default=3)
def run(align_8, num_iter, num_cycles, opt_level):
    logging.basicConfig(level=logging.DEBUG)
    sym = unet.unet(align_8=align_8)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(data_shapes=[('data', (1, 3, 192, 192))])
    mod.init_params()
    sym, params = nnvm.frontend.from_mxnet(sym, arg_params=mod.get_params()[0], aux_params=mod.get_params()[1])
    assert params

    data_shape = (1, 3, 192, 192)
    out_shape = (1, 1, 192, 192)
    with nnvm.compiler.build_config(opt_level=opt_level):
        sym, params = build_until_compile(sym, target, dict(data=data_shape), params=params)
    print(sym.symbol().debug_str())

    tasks = autotvm.task.extract_from_graph(sym, target=target,
                                            shape=dict(data=data_shape), dtype="float32",
                                            symbols=(nnvm.sym.contrib.conv2d_NCHWc,
                                                     nnvm.sym.conv2d,
                                                     nnvm.sym.max_pool2d))
    print(tasks)
    tune_tasks(tasks,
               measure_option=autotvm.measure_option(
                   builder=autotvm.LocalBuilder(),
                   runner=autotvm.LocalRunner(
                       number=10,
                       timeout=5)
               ),
               log_filename="tuning.log")
    with autotvm.apply_history_best("tuning.log"):
        with nnvm.compiler.build_config(opt_level=opt_level):
            graph, lib, params = nnvm.compiler.build(sym, target, dict(data=data_shape), params=params)

    module = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    module.set_input('data', tvm.nd.array(np.random.uniform(size=(data_shape)).astype("float32")))
    rparams = {k: tvm.nd.array(v.shape, ctx) for k, v in params.items()}
    module.set_input(**params)
    module.run()
    out = module.get_output(0, tvm.nd.empty(out_shape, ctx=ctx))
    out.asnumpy()

    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    for i in range(num_cycles):
        prof_res = ftimer()
        print("TVM time: ", prof_res.mean)
        time.sleep(1)

if __name__ == '__main__':
    run()
