import click
import collections
import itertools
import json
import logging
import sys
import tempfile

import numpy as np
import scipy.sparse as sp

import topi
import torch
import tvm
import tvm.contrib.debugger.debug_runtime as debug_runtime
import tvm.contrib.graph_runtime as graph_runtime
import netron

from tvm import autotvm
from tvm import relay

cli = click.Group()

skl_target = tvm.target.create('llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu')

feat_dims = 24
aux_dims = 32
n_classes = 2 ** 8

BSR = collections.namedtuple(
    'BSR',
    ['data', 'indices', 'indptr', 'N', 'K', 'BS_R', 'BS_C', 'density'])


def random_bsr_matrix(M, N, BS_R, BS_C, density):
    Y = np.zeros((M, N), dtype="float32")
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r:r + BS_R, c:c + BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.indices.shape == (num_blocks, )
    assert s.indptr.shape == (M // BS_R + 1, )
    return s

def to_bf16(x):
    assert x.dtype == np.float32
    return ((x.view('<u4') + 2 ** 15) >> 16).astype("uint16")

def instantiate(param):
    if isinstance(param, BSR):
        param_np = random_bsr_matrix(M=param.N, N=param.K, BS_R=param.BS_R, BS_C=param.BS_C, density=param.density)
        return [
            (param.data.name_hint, tvm.ndarray.array(to_bf16(param_np.data))),
            (param.indices.name_hint, tvm.ndarray.array(param_np.indices.astype("int32"))),
            (param.indptr.name_hint, tvm.ndarray.array(param_np.indptr.astype("int32"))),
        ]
    elif param.type_annotation.dtype == "uint16":
        return [(
            param.name_hint,
            tvm.ndarray.array(
                to_bf16(
                    np.random.randn(*param.type_annotation.concrete_shape).astype("float32"))
            )
        )]
    else:
        assert param.type_annotation.dtype == "float32"
        return [(
            param.name_hint,
            tvm.ndarray.array(
                np.random.randn(*param.type_annotation.concrete_shape).astype(
                    param.type_annotation.dtype)
            )
        )]


def tune(func, params, log_name):
    with relay.build_config(opt_level=2):
        func = relay.optimize(func, target=skl_target, params=params)
        tasks = autotvm.task.extract_from_program(
            func, target=skl_target, params=params, ops=(relay.op.nn.dense,))
        for i, tsk in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

            tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type='rank', feature_type="knob")
            n_trial = 1000
            early_stopping = 200
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.RPCRunner(
                    'skl',
                    '0.0.0.0',
                    9195,
                    number=100,
                    repeat=5,
                    min_repeat_ms=1000,
                    timeout=100)
            )
            tuner_obj.tune(
                n_trial=min(n_trial, len(tsk.config_space)),
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    # autotvm.callback.progress_bar(n_trial, prefix=prefix),
                    autotvm.callback.log_to_file(log_name)
                ]
            )


@cli.command()
@click.option('--rnn-dims', type=int, required=True)
@click.option('--fc-dims', type=int, required=True)
@click.option('--density', type=float, required=True)
@click.option('--fc-density', type=float, required=True)
@click.option('--fc-bf16', type=int, required=True)
def run(rnn_dims, fc_dims, density, fc_density, fc_bf16):
    logging.basicConfig(level=logging.DEBUG)

    x = relay.var("x", shape=[1, rnn_dims], dtype="float32")
    h1 = relay.var("h1", shape=[1, rnn_dims], dtype="float32")

    def dense(X, W, B, **kwargs):
        return relay.nn.bias_add(relay.nn.dense(X, W), B)

    def sparse_dense(X, W, B, **kwargs):
        return relay.nn.bias_add(relay.nn.sparse_dense(X, W), B)

    def to_sparse(v, density, BS_R=16, BS_C=1):
        name = v.name_hint
        (N, K) = v.type_annotation.concrete_shape
        nnz = int(density * N * K)
        num_blocks = int(nnz / (BS_R * BS_C)) + 1
        v_data = relay.var(name + "_data", shape=(num_blocks, BS_R, BS_C), dtype="uint16")
        v_indices = relay.var(name + "_indices", shape=(num_blocks,), dtype="int32")
        v_indptr = relay.var(name + "_indptr", shape=(N // BS_R + 1,), dtype="int32")
        return BSR(data=v_data, indices=v_indices, indptr=v_indptr, N=N, K=K, BS_R=BS_R, BS_C=BS_C, density=density)

    def approx_sigmoid(v):
        x = relay.abs(v)
        x2 = v * v
        e = C(1.0) + x + x2 * C(0.5658) + C(0.143) * x2 * x2
        e_pos = e / (C(1) + e)
        e_neg = C(1) / (C(1) + e)
        return relay.where(relay.greater_equal(v, C(0.0)), e_pos, e_neg)

    def approx_tanh(v):
        x = relay.abs(v)
        x2 = v * v
        e = C(1.0) + x + x2 * C(0.5658) + C(0.143) * x2 * x2
        return relay.sign(v) * (e - C(1) / e) / (e + C(1) / e)

    def C(x):
        return relay.expr.const(x, "float32")

    def gru(X, H, W_X, W_H, B, **kwargs):
        XT = relay.nn.bias_add(relay.nn.sparse_dense(X, W_X), B)
        HT = relay.nn.sparse_dense(H, W_H)
        XT_gates = relay.split(XT, indices_or_sections=3, axis=1)
        HT_gates = relay.split(HT, indices_or_sections=3, axis=1)
        u_t = approx_sigmoid(XT_gates[0] + HT_gates[0])
        r_t = approx_sigmoid(XT_gates[1] + HT_gates[1])
        e_t = approx_tanh(r_t * HT_gates[2] + XT_gates[2])
        return u_t * HT_gates[0] + (C(1.0) - u_t) * e_t

    gru_0_W_X = to_sparse(
        relay.var("gru_0_W_X", shape=(3 * rnn_dims, rnn_dims), dtype="float32"), density=density)
    gru_0_W_H = to_sparse(
        relay.var("gru_0_W_H", shape=(3 * rnn_dims, rnn_dims), dtype="float32"), density=density)
    gru_0_B = relay.var("gru_0_B", shape=(3 * rnn_dims,), dtype="float32")

    h1_prime = gru(x, h1, gru_0_W_X, gru_0_W_H, gru_0_B)
    gru2_add1_o = x + h1_prime

    fc_1_W = to_sparse(relay.var("fc_1_W", shape=(fc_dims, rnn_dims, ), dtype="float32"),
                       density=density)
    fc_1_B = relay.var("fc_1_B", shape=(fc_dims,), dtype="float32")
    relu1_o = relay.nn.relu(sparse_dense(gru2_add1_o, fc_1_W, fc_1_B))

    if fc_density > 0:
        fc_3_W = to_sparse(relay.var("fc_3_W", shape=(n_classes, fc_dims), dtype="float32"), density=fc_density)
        fc_3_B = relay.var("fc_3_B", shape=(n_classes,), dtype="float32")
        fc_3_o = sparse_dense(relu1_o, fc_3_W, fc_3_B)
    else:
        # BFloat16 hard-coding.
        fc_3_W = relay.var(
            "fc_3_W", shape=(n_classes, fc_dims), dtype="uint16" if fc_bf16 else "float32")
        fc_3_B = relay.var("fc_3_B", shape=(n_classes,), dtype="float32")
        fc_3_o = dense(relu1_o, fc_3_W, fc_3_B)

    outputs = relay.expr.Tuple([fc_3_o])
    func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
    relay.ir_pass.infer_type(func)

    param_vars = [
        gru_0_W_X, gru_0_W_H, gru_0_B, fc_1_W, fc_1_B, fc_3_W, fc_3_B
    ]

    params = collections.OrderedDict([(k, v) for param in param_vars for (k, v) in instantiate(param)])

    param_size = sum(v.asnumpy().nbytes for v in params.values())
    logging.info("Total param size: %s KB", int(param_size / 1024.0))
    input_vars = [x, h1]
    inputs = collections.OrderedDict(
        [(
            param.name_hint,
            tvm.ndarray.array(torch.zeros(param.type_annotation.concrete_shape))
        ) for param in input_vars])

    log_name = tempfile.NamedTemporaryFile(delete=False, suffix=".tvm.log").name
    tune(func, params, log_name)

    with autotvm.apply_history_best(log_name):
        with relay.build_config(opt_level=3):
            func = relay.optimize(func, target=skl_target, params=params)
            func = relay.ir_pass.infer_type(func)
            graph, lib, new_params = relay.build_module.build(
                func, target=skl_target,  params=params)

    tmp = tvm.contrib.util.tempdir()
    lib_fname = tmp.relpath('net.tar')
    with skl_target:
        lib.export_library(lib_fname)
    tracker = tvm.rpc.connect_tracker('0.0.0.0', 9195)
    remote = tracker.request('skl')

    remote.upload(lib_fname)
    rlib = remote.load_module('net.tar')
    ctx = remote.cpu(0)
    r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
    r_inputs = {k: tvm.nd.array(v, ctx) for k, v in inputs.items()}
    module = graph_runtime.create(graph, rlib, ctx)
    module.set_input(**r_new_params)
    module.set_input(**r_inputs)
    ftimer = module.module.time_evaluator("run", ctx, 1000)
    ftimer()
    times = []
    for i in range(4):
        prof_res = ftimer()
        logging.info("TVM time: %.1fus", prof_res.mean * 10 ** 6)
        times.append(prof_res.mean)
    print(json.dumps(dict(rnn_dims=rnn_dims, fc_dims=fc_dims, density=density, fc_density=fc_density, fc_bf16=fc_bf16, avg_time=np.mean(times), param_size_bytes=param_size)))


@cli.command()
@click.pass_context
def search(ctx):
    import tqdm
    for (fc_density,
         rnn_dims,
         fc_dims,
         density,
         fc_bf16
    ) in tqdm.tqdm(itertools.product(
            [0, 0.4, 0.3, 0.2, 0.1],
            [512, 768, 1024],
            [512, 768, 1024],
            [0.02, 0.03, 0.04, 0.05],
            [1])):
        ctx.invoke(run, rnn_dims=rnn_dims, fc_dims=fc_dims, density=density, fc_density=fc_density, fc_bf16=fc_bf16)

if __name__ == "__main__":
    cli()
