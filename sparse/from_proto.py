import click
from caffe2.proto import caffe2_pb2
import nnvm
import copy
import tvm
import logging
logging.basicConfig(level=logging.DEBUG)
import cPickle as pickle

from caffe2.python import workspace

def set_batch_size(input_dict, batch_size):
    (previous_batch_size, new_batch_size) = batch_size
    assert new_batch_size < previous_batch_size
    def f(k, v):
        if v.shape[0] == previous_batch_size:
            logging.info(
                "Changing batch size of %s (%s) from %s to %s",
                k, v.shape, previous_batch_size, new_batch_size)
            return v[:new_batch_size]
        logging.info(
            "Preserving batch size of %s (%s)",
            k, v.shape)
        return v
    return {k: f(k, v) for k, v in input_dict.items()}


@click.command()
@click.option('--init_net', type=click.Path())
@click.option('--input_init_net', type=click.Path())
@click.option('--pred_net', type=click.Path())
@click.option('--symbol', type=click.Path())
@click.option('--params_pkl', type=click.Path())
@click.option('--inputs_pkl', type=click.Path())
@click.option('--batch_size', type=(int, int), default=(20, 1))
def main(init_net, input_init_net, pred_net, symbol, params_pkl, inputs_pkl, batch_size):
    with open(init_net, "rb") as f:
        init_net = caffe2_pb2.NetDef()
        init_net.ParseFromString(f.read())
    with open(input_init_net, "rb") as f:
        input_init_net = caffe2_pb2.NetDef()
        input_init_net.ParseFromString(f.read())

    with open(pred_net, "rb") as f:
        pred_net = caffe2_pb2.NetDef()
        pred_net.ParseFromString(f.read())
    fake_init_net = copy.copy(init_net)

    fake_init_net.op.extend([op for op in input_init_net.op])
    sym, params = nnvm.frontend.from_caffe2(fake_init_net, pred_net)
    param_names = [o for op in init_net.op for o in op.output]
    input_names = [o for op in input_init_net.op for o in op.output]
    for param in params.keys():
        if param in input_names:
            logging.info("Removing parameter: %s", param)
            del params[param]
    ws = workspace.C.Workspace()
    ws.run(input_init_net)
    with open(symbol, "w") as f:
        f.write(nnvm.graph.create(sym).json())
    input_dict = {name: ws.fetch_blob(name) for name in ws.blobs.keys()}
    input_dict = set_batch_size(input_dict, batch_size)
    with open(inputs_pkl, "w") as f:
        pickle.dump(input_dict, f)
    params = {name: (v.asnumpy().shape, v.asnumpy().dtype) for name, v in params.items()}
    with open(params_pkl, "w") as f:
        pickle.dump(params, f)
    sym = nnvm.graph.load_json(open(symbol).read()).symbol()

if __name__ == "__main__":
    main()
