import unet

import mxnet as mx
import numpy as np
import nnvm
import nnvm.compiler

import tvm
import tvm_overrides

from tvm.contrib import graph_runtime as runtime
import time
import click
import logging

target = 'llvm -mcpu=core-avx2'
ctx = tvm.context(str(target), 0)

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
    # import ipdb
    # ipdb.set_trace()
    sym, params = nnvm.frontend.from_mxnet(sym, arg_params=mod.get_params()[0], aux_params=mod.get_params()[1])
    assert params

    data_shape = (1, 3, 192, 192)
    out_shape = (1, 1, 192, 192)
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(sym, target, dict(data=data_shape), params=params)
    module = runtime.create(graph, lib, ctx)
    logging.debug(graph.symbol().debug_str())
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
