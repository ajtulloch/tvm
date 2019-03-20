import tvm
import topi
from tvm import relay
from tvm import autotvm
import numpy as np
import itertools
import scipy.sparse as sp

import tvm.contrib.debugger.debug_runtime as debug_runtime
import tvm.contrib.graph_runtime as graph_runtime
import torch
import logging
import collections
import netron
import tempfile

dtype = "float32"

rnn_dims = 512
fc_dims = 512

feat_dims = 24
aux_dims = 32
n_classes = 2 ** 8

x = relay.var("x", shape=[1, 1], dtype=dtype)
h1 = relay.var("h1", shape=[1, rnn_dims], dtype=dtype)
h2 = relay.var("h2", shape=[1, rnn_dims], dtype=dtype)
m_t = relay.var("m_t", shape=[1, feat_dims], dtype=dtype)
a1_t = relay.var("a1_t", shape=[1, aux_dims], dtype=dtype)
a2_t = relay.var("a2_t", shape=[1, aux_dims], dtype=dtype)
a3_t = relay.var("a3_t", shape=[1, aux_dims], dtype=dtype)
a4_t = relay.var("a4_t", shape=[1, aux_dims], dtype=dtype)


concat0_o = relay.concatenate([x, m_t, a1_t], axis=1)

import collections
BSR = collections.namedtuple('CSR', ['data', 'indices', 'indptr', 'N', 'K', 'BS_R', 'BS_C', 'density'])

def sparse_dense(X, W, B, **kwargs):
    return relay.nn.bias_add(relay.nn.sparse_dense(X, W), B)

def to_sparse(v, density=0.04, BS_R=16, BS_C=1):
    name = v.name_hint
    (N, K) = v.type_annotation.concrete_shape
    nnz = int(density * N * K)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    v_data = relay.var(name + "_data", shape=(num_blocks, BS_R, BS_C), dtype=dtype)
    v_indices = relay.var(name + "_indices", shape=(num_blocks,), dtype="int32")
    v_indptr = relay.var(name + "_indptr", shape=(N // BS_R + 1,), dtype="int32")
    return BSR(data=v_data, indices=v_indices, indptr=v_indptr, N=N, K=K, BS_R=BS_R, BS_C=BS_C, density=density)

fc_0_W = to_sparse(relay.var("fc_0_W", shape=(rnn_dims, feat_dims + aux_dims + 1), dtype=dtype))
fc_0_B = relay.var("fc_0_B", shape=(rnn_dims,), dtype=dtype)
i_o = sparse_dense(concat0_o, fc_0_W, fc_0_B)

def approx_sigmoid(v):
    x = relay.abs(v)
    x2 = v * v
    e = C(1.0) + x + x2 * C(0.5658) + C(0.143) * x2 * x2
    e_pos = e / (C(1) + e)
    e_neg = C(1) / (C(1) + e)
    # TODO: ensure this returns good code.
    return relay.where(relay.greater_equal(v, C(0.0)), e_pos, e_neg)

def approx_tanh(v):
    x = relay.abs(v)
    x2 = v * v
    e = C(1.0) + x + x2 * C(0.5658) + C(0.143) * x2 * x2
    return relay.sign(v) * (e - C(1) / e) / (e + C(1) / e)

def C(x):
    return relay.expr.const(x, dtype)

def gru(X, H, W_X, W_H, B, **kwargs):
    XT = relay.nn.bias_add(relay.nn.sparse_dense(X, W_X), B)
    HT = relay.nn.sparse_dense(H, W_H)
    XT_gates = relay.split(XT, indices_or_sections=3, axis=1)
    HT_gates = relay.split(HT, indices_or_sections=3, axis=1)
    u_t = approx_sigmoid(XT_gates[0] + HT_gates[0])
    r_t = approx_sigmoid(XT_gates[1] + HT_gates[1])
    e_t = approx_tanh(r_t * HT_gates[2] + XT_gates[2])
    # u_t = relay.sigmoid(XT_gates[0] + HT_gates[0])
    # r_t = relay.sigmoid(XT_gates[1] + HT_gates[1])
    # e_t = relay.tanh(r_t * HT_gates[2] + XT_gates[2])
    return u_t * HT_gates[0] + (relay.expr.const(1.0, dtype=dtype) - u_t) * e_t

gru_0_W_X = to_sparse(relay.var("gru_0_W_X", shape=(3 * rnn_dims, rnn_dims), dtype=dtype))
gru_0_W_H = to_sparse(relay.var("gru_0_W_H", shape=(3 * rnn_dims, rnn_dims), dtype=dtype))
gru_0_B = relay.var("gru_0_B", shape=(3 * rnn_dims,), dtype=dtype)

h1_prime = gru(i_o, h1, gru_0_W_X, gru_0_W_H, gru_0_B)
gru2_add1_o = i_o + h1_prime

inp = relay.concatenate([gru2_add1_o, a2_t], axis=1)


gru_1_W_X = to_sparse(relay.var("gru_1_W_X", shape=(3 * rnn_dims, rnn_dims + aux_dims), dtype=dtype))
gru_1_W_H = to_sparse(relay.var("gru_1_W_H", shape=(3 * rnn_dims, rnn_dims), dtype=dtype))
gru_1_B = relay.var("gru_1_B",  shape=(3 * rnn_dims,), dtype=dtype)

h2_prime = gru(inp, h2, gru_1_W_X, gru_1_W_H, gru_1_B)
add1_o = gru2_add1_o + h2_prime

concat1_o = relay.concatenate([add1_o, a3_t], axis=1)


fc_1_W = to_sparse(relay.var("fc_1_W",
                   shape=(fc_dims, rnn_dims + aux_dims, ),
                   dtype=dtype))
fc_1_B = relay.var("fc_1_B",
                   shape=(fc_dims,),
                   dtype=dtype)
relu1_o = relay.nn.relu(sparse_dense(concat1_o, fc_1_W, fc_1_B))

concat2_o = relay.concatenate([relu1_o, a4_t], axis=1)

fc_2_W = to_sparse(relay.var("fc_2_W", shape=(fc_dims, fc_dims + aux_dims), dtype=dtype))
fc_2_B = relay.var("fc_2_B", shape=(fc_dims,), dtype=dtype)
relu2_o = relay.nn.relu(sparse_dense(concat2_o, fc_2_W, fc_2_B))

fc_3_W = to_sparse(relay.var("fc_3_W", shape=(n_classes, fc_dims), dtype=dtype))
fc_3_B = relay.var("fc_3_B", shape=(n_classes,), dtype=dtype)
fc_3_o = sparse_dense(relu2_o, fc_3_W, fc_3_B)

softmax_o = relay.nn.softmax(fc_3_o, axis=-1)

outputs = relay.expr.Tuple([fc_3_o])

func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
relay.ir_pass.infer_type(func)

param_vars = [
    fc_0_W, fc_0_B, gru_0_W_X, gru_0_W_H, gru_0_B, gru_1_W_X, gru_1_W_H, gru_1_B, fc_1_W, fc_1_B, fc_2_W, fc_2_B, fc_3_W, fc_3_B
]



def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype):
    Y = np.zeros((M, N), dtype=dtype)
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

def instantiate(param):
    if isinstance(param, BSR):
        import scipy.sparse as sp
        param_np = random_bsr_matrix(M=param.N, N=param.K, BS_R=param.BS_R, BS_C=param.BS_C, density=param.density, dtype='float32')
        print(param_np.data.shape)
        return [
            (param.data.name_hint, tvm.ndarray.array(param_np.data)),
            (param.indices.name_hint, tvm.ndarray.array(param_np.indices.astype("int32"))),
            (param.indptr.name_hint, tvm.ndarray.array(param_np.indptr.astype("int32"))),
        ]
    else:
        return [(
            param.name_hint,
            tvm.ndarray.array(
                np.random.randn(*param.type_annotation.concrete_shape).astype(
                    param.type_annotation.dtype)
            )
            )
        ]
params = collections.OrderedDict([(k, v) for param in param_vars for (k, v) in instantiate(param)])
print("Total param size: ", sum(v.asnumpy().nbytes for v in params.values()))
input_vars = [x, h1, h2, m_t, a1_t, a2_t, a3_t, a4_t]
inputs = collections.OrderedDict(
    [(
        param.name_hint,
        tvm.ndarray.array(torch.zeros(param.type_annotation.concrete_shape))
    ) for param in input_vars])


logging.basicConfig(level=logging.DEBUG)
skl_target = tvm.target.create('llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu')


def tune():
    global func
    with relay.build_config(opt_level=2):
        func = relay.optimize(func, target=skl_target, params=params)
        print(func.astext(show_meta_data=False))
        tasks = autotvm.task.extract_from_program(
            func, target=skl_target, params=params, ops=(relay.op.nn.dense,))
        for i, tsk in enumerate(tasks):
            print(tsk)
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
            log_filename = "synthesis_autotvm_skl.log"
            tuner_obj.tune(
                n_trial=min(n_trial, len(tsk.config_space)),
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(n_trial, prefix=prefix),
                    autotvm.callback.log_to_file(log_filename)
                ]
            )

if 0:
    tune()
    import sys
    sys.exit()


with autotvm.apply_history_best("synthesis_autotvm_skl.best.log"):
    with relay.build_config(opt_level=3):
        func = relay.optimize(func, target=skl_target, params=params)
        print(func.astext(show_meta_data=False))
        func = relay.ir_pass.infer_type(func)
        graph, lib, new_params = relay.build_module.build(
            func, target=skl_target,  params=params)

        for (k, v) in params.items():
            print(k, v.shape)
        for (k, v) in new_params.items():
            print(k, v.shape)
        # with tempfile.NamedTemporaryFile(delete=False, suffix="tvm.json") as f:
        #     f.write(graph.encode())
        # netron.start(f.name, host="localhost")


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
ftimer = module.module.time_evaluator("run", ctx, 10000)
for i in range(5):
    prof_res = ftimer()
    print("TVM time: ", prof_res.mean)

module = debug_runtime.create(graph, rlib, ctx)
module.set_input(**r_new_params)
module.set_input(**r_inputs)
module.run()
module.run()
module.run()
module.run()
module.run()
module.run()
