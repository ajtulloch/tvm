import click
import numpy as np
from caffe2.proto import caffe2_pb2
import copy
import logging
logging.basicConfig(level=logging.DEBUG)


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
@click.option('--num_iter', default=10000)
@click.option('--num_cycles', default=5)
@click.option('--batch_size', type=(int, int), default=(20, 1))
@click.option('--layerwise', is_flag=True, default=False)
def main(init_net, input_init_net, pred_net, num_iter, num_cycles, batch_size, layerwise):
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
    workspace.RunNetOnce(fake_init_net)
    for k in sorted(workspace.Blobs()):
        for op in init_net.op:
            if k in op.output:
                b = workspace.FetchBlob(k)
                print(k, b.shape)
        workspace.FeedBlob(k, np.zeros_like(b))
    ws = workspace.C.Workspace()
    ws.run(input_init_net)
    input_dict = {name: ws.fetch_blob(name) for name in ws.blobs.keys()}
    input_dict = set_batch_size(input_dict, batch_size)
    for k, v in input_dict.items():
        workspace.FeedBlob(k, v)
    pred_net.name = "benchmark"
    workspace.CreateNet(pred_net, True)
    for _ in range(num_cycles):
        workspace.BenchmarkNet("benchmark", num_iter, num_iter, layerwise)
    # for _ in range(num_cycles):
    #     workspace.BenchmarkNet("benchmark", num_iter, num_iter, layerwise)

if __name__ == "__main__":
    main()
