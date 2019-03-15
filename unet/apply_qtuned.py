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


@click.command()
@click.option('--align', default=8)
@click.option(
    '--model', type=click.Choice(['unet', 'resnet50']), required=True)
@click.option('--autotvm_log', default="autotvm_unet_tuning.log", type=str)
@click.option('--opt_level', default=3)
@click.option('--device', type=click.Choice(["skl", "rpi", "local"]), required=True)
@click.option('--num_iter', default=10)
@click.option('--num_cycles', default=5)
def run(align, model, autotvm_log, opt_level, device, num_iter, num_cycles):
    logging.basicConfig(level=logging.DEBUG)
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

    # with relay.build_config(opt_level=opt_level):
    #     qgraph_json, lib, params = relay.build(qgraph, target, params=params)
    with autotvm.apply_history_best(str(autotvm_log)):
        with target:
            with relay.build_config(opt_level=2):
                qgraph_json, lib, params = relay.build(qgraph, target, params=params)

    import netron
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix="tvm.json") as f:
        f.write(qgraph_json)
    netron.start(f.name, host="localhost")

    tmp = tvm.contrib.util.tempdir()
    lib_fname = tmp.relpath('net.tar')
    with tvm.target.create(target):
        lib.export_library(lib_fname)
    tracker = tvm.rpc.connect_tracker('0.0.0.0', 9195)
    remote = tracker.request(device)

    remote.upload(lib_fname)
    rlib = remote.load_module('net.tar')
    ctx = remote.cpu(0)

    module = tvm.contrib.graph_runtime.create(qgraph_json, rlib, ctx)
    module.set_input('data', tvm.nd.array(np.random.uniform(size=(data_shape)).astype("float32")))
    rparams = {k: tvm.nd.array(v.shape, ctx) for k, v in params.items()}
    # module.set_input(**rparams)
    module.run()
    out = module.get_output(0, tvm.nd.empty(output_shape, ctx=ctx))

    out.asnumpy()

    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    for i in range(1):
        prof_res = ftimer()
        # time.sleep(1)

    for i in range(num_cycles):
        prof_res = ftimer()
        print("TVM time: ", prof_res.mean)


if __name__ == '__main__':
    run()
