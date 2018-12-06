import mxnet as mx
import numpy as np
import nnvm
import nnvm.compiler

import tvm
from tvm import autotvm

import cPickle as pickle

from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime

import time
import click
import logging

TARGETS = dict(
    skl='llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu',
    local='llvm -mcpu=core-avx2'
)
@click.command()
@click.option('--symbol', type=click.Path())
@click.option('--params_pkl', type=click.Path())
@click.option('--inputs_pkl', type=click.Path())
@click.option('--num_iter', default=10000)
@click.option('--num_cycles', default=5)
@click.option('--opt_level', default=3)
@click.option('--autotvm_log', default="sparse_tuning.log", type=click.Path())
@click.option('--tracker_port', default=9195)
@click.option('--device', type=click.Choice(TARGETS.keys()))
@click.option('--layerwise', is_flag=True, default=False)
def run(
        symbol, params_pkl, inputs_pkl,
        num_iter, num_cycles, opt_level,
        autotvm_log, tracker_port, device, layerwise):
    logging.basicConfig(level=logging.DEBUG)
    target = TARGETS[device]

    with open(symbol, "rb") as f:
        sym = nnvm.graph.load_json(f.read()).symbol()

    with open(params_pkl, "rb") as f:
        params = pickle.load(f)
        params = {name: tvm.nd.array(np.atleast_1d(np.random.randn(*shape)).astype(dtype)) for name, (shape, dtype) in params.items()}

    with open(inputs_pkl, "rb") as f:
        inputs = pickle.load(f)
        inputs = {name: tvm.nd.array(v) for name, v in inputs.items()}

    with tvm.target.create(target):
        with autotvm.apply_history_best(str(autotvm_log)):
            with nnvm.compiler.build_config(opt_level=opt_level):
                graph, lib, new_params = nnvm.compiler.build(
                    sym,
                    target,
                    shape={name: v.asnumpy().shape for name, v in inputs.items()},
                    dtype={name: v.asnumpy().dtype for name, v in inputs.items()},
                    params=params
                )

    if device == "skl":
        tmp = tvm.contrib.util.tempdir()
        lib_fname = tmp.relpath('net.tar')
        with tvm.target.create(target):
            lib.export_library(lib_fname)
        tracker = tvm.rpc.connect_tracker('localhost', 9195)
        remote = tracker.request('skl')
        remote.upload(lib_fname)
        rlib = remote.load_module('net.tar')
        ctx = remote.cpu(0)
        module = debug_runtime.create(graph, rlib, ctx)
    else:
        ctx = tvm.context(str(target), 0)
        if layerwise:
            module = debug_runtime.create(graph, lib, ctx)
        else:
            module = graph_runtime.create(graph, lib, ctx)

    logging.debug(graph.symbol().debug_str())
    module.set_input(**inputs)
    module.zero_input(**new_params)
    for k, v in sorted(new_params.items()):
        print(k, v.shape)
    # if device == "skl":
    #     module.zero_input(**new_params)
    # else:
    #     module.set_input(**new_params)
    module.run()
    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    for i in range(num_cycles):
        prof_res = ftimer()
        print("TVM time: ", prof_res.mean)
        if layerwise:
            module.run()
        time.sleep(1)

if __name__ == '__main__':
    run()
