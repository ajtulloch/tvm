from __future__ import absolute_import as _abs
import numpy as np
import tvm
import nnvm.compiler
from nnvm.frontend.caffe2 import from_caffe2
from tvm.contrib import graph_runtime

from caffe2.proto import caffe2_pb2

from PIL import Image

def download(url, path, overwrite=False):
    import os
    if os.path.isfile(path) and not overwrite:
        print('File {} existed, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
    except:
        import urllib
        urllib.urlretrieve(url, path)


def load_caffe2_model(init, predict):
    init_net = caffe2_pb2.NetDef()
    with open(init, 'rb') as f:
        init_net.ParseFromString(f.read())

    predict_net = caffe2_pb2.NetDef()
    with open(predict, 'rb') as f:
        predict_net.ParseFromString(f.read())
    return init_net, predict_net


# test person segmentation
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
download(img_url, 'cat.png')
img = Image.open('cat.png').resize((192, 192))

image = np.asarray(img)
image = image.transpose((2, 0, 1))
x = image[np.newaxis, :]
print(x.shape)

init_net, predict_net = load_caffe2_model('seg_init_net_a.pb', 'seg_predict_net_a.pb')
sym, params = from_caffe2(init_net, predict_net)
print(sym.debug_str())

# assume first input name is data
input_name = sym.list_input_names()[0]
shape_dict = {input_name: x.shape}

opt_level = 3
target = 'llvm -mcpu=core-avx2'
graph = None
lib = None
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(
        sym, target, shape={input_name: x.shape}, params=params)
    print(graph.symbol().debug_str())

ctx = tvm.context(target, 0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
output_shape = (1, 1, 192, 192)
tvm_output = m.get_output(0, tvm.nd.empty(output_shape, dtype)).asnumpy()
print(tvm_output)
# top1 = np.argmax(tvm_output)

# print(top1)
# print(type(tvm_output))
# print(tvm_output)
