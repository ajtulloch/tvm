
import mxnet as mx
import numpy as np
import nnvm
import nnvm.compiler

import tvm
import models

# import tvm_overrides
# import unet_conv2d

from tvm.contrib import graph_runtime as runtime
import time
import click
import logging

target = 'llvm -mcpu=core-avx2'
ctx = tvm.context(str(target), 0)

@click.command()
@click.option('--align', default=0)
@click.option('--num_iter', default=10)
@click.option('--num_cycles', default=5)
@click.option('--model', type=click.Choice(['unet', 'resnet50']), required=True)
@click.option('--opt_level', default=3)
def run(align, num_iter, num_cycles, model, opt_level):
    logging.basicConfig(level=logging.DEBUG)
    sym, image_shape, output_shape = models.get_mxnet_symbol(model, align)
    sym, params = models.get_nnvm_sym(sym, image_shape)
    assert params

    data_shape = tuple([1] + list(image_shape))
    out_shape = tuple([1] + list(output_shape))
    with tvm.target.create(target):
        with nnvm.compiler.build_config(opt_level=opt_level):
            graph, lib, params = nnvm.compiler.build(sym, target, dict(data=data_shape), params=params)
    module = runtime.create(graph, lib, ctx)
    logging.debug(graph.symbol().debug_str())
    with open("tvm_perf.log", "w") as f:
        f.write(graph.symbol().debug_str())
    module.set_input('data', tvm.nd.array(np.random.uniform(size=(data_shape)).astype("float32")))
    # rparams = {k: tvm.nd.array(v.shape, ctx) for k, v in params.items()}
    # module.set_input(**params)
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
