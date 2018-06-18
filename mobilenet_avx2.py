import sys
sys.path.append(".")
sys.path.append("./python")
import time
import argparse
import numpy as np
import tvm
import nnvm.compiler
import nnvm.testing
from tvm.contrib import util
from tvm.contrib import graph_runtime as runtime
import logging
import topi.x86
import topi.nn
target = 'llvm -mcpu=core-avx2'
ctx = tvm.context(target, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['resnet', 'mobilenet'],
        help="The model type.")
    parser.add_argument('--opt-level', type=int, default=1, help="Level of optimization.")
    parser.add_argument('--num-iter', type=int, default=50, help="Number of iteration during benchmark.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)
    opt_level = args.opt_level

    num_iter = args.num_iter
    batch_size = 1
    num_classes = 1000
    image_shape = (3, 224, 224)

    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)
    if args.model == 'resnet':
        net, params = nnvm.testing.resnet.get_workload(
            batch_size=1, image_shape=image_shape)
    elif args.model == 'mobilenet':
        net, params = nnvm.testing.mobilenet.get_workload(
            batch_size=1, image_shape=image_shape)
    else:
        raise ValueError('no benchmark prepared for {}.'.format(args.model))

    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(
            net, target, shape={"data": data_shape}, params=params)

    module = runtime.create(graph, lib, ctx)
    module.set_input('data', tvm.nd.array(np.random.uniform(size=(data_shape)).astype("float32")))
    module.set_input(**params)
    module.run()
    out = module.get_output(0, tvm.nd.empty(out_shape, ctx=ctx))
    out.asnumpy()

    print('benchmark args: {}'.format(args))
    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    for i in range(3):
        prof_res = ftimer()
        print(prof_res)
        # sleep for avoiding cpu overheat
        time.sleep(1)


if __name__ == '__main__':
    main()
