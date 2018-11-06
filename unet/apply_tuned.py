from tvm import autotvm
from tvm.contrib import graph_runtime
# from tvm.contrib.debugger import debug_runtime as graph_runtime
import tvm.contrib.util
import tvm.rpc
import click
import logging
import nnvm
import nnvm.compiler
import numpy as np
import os
import time
import tvm

import models


skl_target = tvm.target.create('llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu')
local_target = tvm.target.create('llvm -mcpu=core-avx2')
rpi_target = tvm.target.arm_cpu("rasp3b")

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def get_model_size(tar_file):
    import tempfile
    import tarfile
    import shutil
    import subprocess
    with tempfile.NamedTemporaryFile(delete=False) as f:
        shutil.copyfileobj(tarfile.open(tar_file).extractfile("lib.o"), f)
    with tempfile.NamedTemporaryFile(delete=False) as fzstd:
        subprocess.check_call(["zstd", "-19", "-f", "-o", fzstd.name, f.name])
    return sizeof_fmt(os.path.getsize(fzstd.name))

@click.command()
@click.option('--align', default=8)
@click.option('--num_iter', default=10)
@click.option('--num_cycles', default=5)
@click.option('--model',
              type=click.Choice(['unet', 'resnet50', 'unet_down_block', 'unet_up_block']),
              required=True)
@click.option('--autotvm_log', default="autotvm_unet_tuning.log", type=str)
@click.option('--tracker_port', default=9195)
@click.option('--opt_level', default=3)
@click.option('--device', type=click.Choice(["skl", "rpi", "local"]), required=True)
@click.option('--verbose', is_flag=True, default=False)
def run(align,
        num_iter,
        num_cycles,
        model,
        autotvm_log,
        tracker_port,
        opt_level,
        device,
        verbose):
    logging.basicConfig(level=logging.INFO if not verbose else logging.DEBUG)
    tracker = tvm.rpc.connect_tracker('localhost', 9195)
    remote = tracker.request(device)
    target = dict(skl=skl_target, rpi=rpi_target, local=local_target)[device]
    with autotvm.apply_history_best(str(autotvm_log)):
        sym, image_shape, output_shape = models.get_mxnet_symbol(model, align)
        data_shape = tuple([1] + list(image_shape))
        sym, params = models.get_nnvm_sym(sym, image_shape)
        with nnvm.compiler.build_config(opt_level=opt_level):
            with tvm.build_config(partition_const_loop=True):
                graph, lib, params = nnvm.compiler.build(sym, target, dict(data=data_shape), params=params)

    out_shape = tuple([1] + list(output_shape))


    tmp = tvm.contrib.util.tempdir()
    lib_fname = tmp.relpath('net.tar')
    with tvm.target.create(target):
        lib.export_library(lib_fname)
    logging.info("Model size: %s", get_model_size(lib_fname))
    remote.upload(lib_fname)
    rlib = remote.load_module('net.tar')
    ctx = remote.cpu(0)

    module = graph_runtime.create(graph, rlib, ctx)
    logging.debug(graph.symbol().debug_str())
    with open("apply_tuned.log", "w") as f:
        f.write(graph.symbol().debug_str())
    module.set_input('data', tvm.nd.array(np.random.uniform(size=(data_shape)).astype("float32")))
    rparams = {k: tvm.nd.array(v.shape, ctx) for k, v in params.items()}
    # module.set_input(**rparams)
    module.run()
    out = module.get_output(0, tvm.nd.empty(out_shape, ctx=ctx))

    out.asnumpy()

    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    for i in range(1):
        prof_res = ftimer()
        # time.sleep(1)

    for i in range(num_cycles):
        prof_res = ftimer()
        print("TVM time: ", prof_res.mean)
        # time.sleep(1)




if __name__ == "__main__":
    run()
