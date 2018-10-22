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

import tvm_overrides
import unet_conv2d

import models


target = 'llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu'

@click.command()
@click.option('--align', default=8)
@click.option('--model', type=click.Choice(['unet', 'resnet50']), required=True)
@click.option('--autotvm_log', default="autotvm_unet_tuning.log", type=str)
@click.option('--opt_level', default=3)
@click.option('--verbose', is_flag=True, default=False)
def run(align,
        model,
        autotvm_log,
        opt_level,
        verbose):
    logging.basicConfig(level=logging.INFO if not verbose else logging.DEBUG)

    with autotvm.apply_history_best(str(autotvm_log)):
        sym, image_shape, output_shape = models.get_mxnet_symbol(model, align)
        data_shape = tuple([1] + list(image_shape))
        sym, params = models.get_nnvm_sym(sym, image_shape)
        assert params
        with nnvm.compiler.build_config(opt_level=opt_level):
            graph, lib, params = nnvm.compiler.build(sym, target, dict(data=data_shape), params=params)
    with tvm.target.create(target):
        lib.export_library("model.tar")
    logging.info("Dumped model library to model.tar, extract with tar -xf model.tar")

if __name__ == "__main__":
    run()
