import unet

import mxnet as mx
import numpy as np
import nnvm

opt_level = 3
target = 'llvm -mcpu=core-avx2'

sym = unet.unet()
mod = mx.mod.Module(symbol=sym, context=mx.cpu())
mod.bind(for_training     = False,
         inputs_need_grad = False,
         data_shapes      = [('data', (1, 1, 192, 192))])

print(sym.debug_str())
sym, params = nnvm.frontend.from_mxnet(sym)
print(sym.debug_str())
print(params)
import nnvm.compiler
shape_dict = {'data': (1, 1, 192, 192)}
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
print(graph.symbol().debug_str())
print(graph.symbol().debug_str())
graph = graph.apply('InferShape').apply('InferType')
shapes = graph.json_attr('shape')
dtypes = graph.json_attr('dtype')
import json
jgraph = json.loads(graph.apply("SaveJSON").json_attr("json"))
name_shape = [(jgraph["nodes"][i]["name"], shape) for i, shape in enumerate(shapes)]
for el in name_shape:
    print(el)
