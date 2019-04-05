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
logging.basicConfig(level=logging.DEBUG)
import collections
import netron
import tempfile

import collections


BFLOAT16 = False
DEVICE = 'rpi'
# TARGET = tvm.target.create('llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu')
TARGET = tvm.target.rasp()
PORT = 9195
TUNE = False
log_filename = "lpcnet_no_bf16_autotvm_skl.log"


GRU_A_STATE_SIZE = 384
GRU_A_DENSITY = 0.1
GRU_B_STATE_SIZE = 16
FEATURE_DENSE2_OUT_SIZE = 128



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
        Y[r:r + BS_R, c:c + BS_C] = np.random.randn(BS_R, BS_C).astype("float32")
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
            (param.data.name_hint, tvm.ndarray.array(param_np.data.astype("uint16" if BFLOAT16 else "float32"))),
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

def to_sparse(v, density, BS_R=16, BS_C=1):
    name = v.name_hint
    (N, K) = v.type_annotation.concrete_shape
    nnz = int(density * N * K)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    v_data = relay.var(name + "_data", shape=(num_blocks, BS_R, BS_C), dtype="uint16" if BFLOAT16 else "float32")
    v_indices = relay.var(name + "_indices", shape=(num_blocks,), dtype="int32")
    v_indptr = relay.var(name + "_indptr", shape=(N // BS_R + 1,), dtype="int32")
    return BSR(data=v_data, indices=v_indices, indptr=v_indptr, N=N, K=K, BS_R=BS_R, BS_C=BS_C, density=density)


def approx_exp(x):
    x = relay.minimum(relay.maximum(x, C(-88.0)), C(88.0))
    x = C(127.0) + x * C(1.44268504)

    i = relay.cast(x, "int32")
    xf = relay.cast(i, "float32")
    x = x - xf
    Y = C(0.99992522) + x * (C(0.69583354) + x * (C(0.22606716) + x * C(0.078024523)))
    exponent = relay.left_shift(i, relay.expr.const(23, "int32"))
    exponent = relay.reinterpret(exponent, "float32")
    return exponent * Y

def approx_sigmoid(x):
    y = approx_exp(x)
    return y# / (y + C(1.0))

def approx_tanh(x):
    x = x * C(2.0)
    y = approx_exp(x)
    return y
    # return (y - C(1.0)) / (y + C(1.0))

def C(x):
    return relay.expr.const(x, "float32")

def gru(X0, X1, H, W_X0, W_X1, W_H, B_X, B_H, **kwargs):
    XT = relay.nn.bias_add(
        relay.nn.dense(X0, W_X0) + relay.nn.dense(X1, W_X1), B_X)
    HT = relay.nn.bias_add(relay.nn.dense(H, W_H), B_H)
    XT_gates = relay.split(XT, indices_or_sections=3, axis=1)
    HT_gates = relay.split(HT, indices_or_sections=3, axis=1)
    u_t = approx_sigmoid(XT_gates[0] + HT_gates[0])
    r_t = approx_sigmoid(XT_gates[1] + HT_gates[1])
    e_t = approx_tanh(r_t * HT_gates[2] + XT_gates[2])
    return e_t + u_t * (H - e_t)

def sparse_gru(XT, H, W_H, B_H, W_H_DIAG):
    HT = relay.nn.bias_add(relay.nn.sparse_dense(H, W_H), B_H)
    XT_gates = relay.split(XT, indices_or_sections=3, axis=1)
    HT_gates = relay.split(HT, indices_or_sections=3, axis=1)
    HT_gates_0 = HT_gates[0] + W_H_DIAG[0] * H
    HT_gates_1 = HT_gates[1] + W_H_DIAG[1] * H
    HT_gates_2 = HT_gates[2] + W_H_DIAG[2] * H
    u_t = approx_sigmoid(XT_gates[0] + HT_gates_0)
    r_t = approx_sigmoid(XT_gates[1] + HT_gates_1)
    e_t = approx_tanh(r_t * HT_gates_2 + XT_gates[2])
    return e_t + u_t * (H - e_t)


def dual_fc(x, W, B, A):
    # W = relay.cast(W, "uint16")
    dual_output = A * approx_tanh(relay.nn.bias_add(relay.nn.dense(x, W), B))
    dual_output_gates = relay.split(dual_output, indices_or_sections=2, axis=1)
    return dual_output_gates[0] + dual_output_gates[1]

gru_a_condition = relay.var("gru_a_condition", shape=(1, 3 * GRU_A_STATE_SIZE))
gru_b_condition = relay.var("gru_b_condition", shape=(1, FEATURE_DENSE2_OUT_SIZE))

last_sig = relay.var("last_sig", shape=[1], dtype="int32")
pred = relay.var("pred", shape=[1], dtype="int32")
last_exc = relay.var("last_exc", shape=[1], dtype="int32")

gru_a_hidden_state = relay.var("gru_a_hidden_state", shape=[1, GRU_A_STATE_SIZE])
gru_b_hidden_state = relay.var("gru_b_hidden_state", shape=[1, GRU_B_STATE_SIZE])

gru_a_embedding_sig = relay.var("gru_a_embedding_sig", shape=(256, 3 * GRU_A_STATE_SIZE,))
gru_a_embedding_pred = relay.var("gru_a_embedding_pred", shape=(256, 3 * GRU_A_STATE_SIZE,))
gru_a_embedding_last_exc = relay.var("gru_a_embedding_last_exc", shape=(256, 3 * GRU_A_STATE_SIZE,))

gru_a_input = gru_a_condition + relay.take(gru_a_embedding_sig, last_sig, axis=0) + relay.take(gru_a_embedding_pred, pred, axis=0) + relay.take(gru_a_embedding_last_exc, last_exc, axis=0)

gru_a_W_H = to_sparse(
    relay.var("gru_a_W_H",
              shape=(3 * GRU_A_STATE_SIZE, GRU_A_STATE_SIZE)),
    density=GRU_A_DENSITY)
gru_a_B_H = relay.var("gru_a_W_H", shape=(3 * GRU_A_STATE_SIZE, ))

gru_a_WD_0 = relay.var("gru_a_WD_0", shape=(1, GRU_A_STATE_SIZE,))
gru_a_WD_1 = relay.var("gru_a_WD_1", shape=(1, GRU_A_STATE_SIZE,))
gru_a_WD_2 = relay.var("gru_a_WD_2", shape=(1, GRU_A_STATE_SIZE,))


gru_a_next_hidden_state = sparse_gru(gru_a_input, gru_a_hidden_state, gru_a_W_H, gru_a_B_H, [gru_a_WD_0, gru_a_WD_1, gru_a_WD_2])


gru_b_W_H = relay.var("gru_b_W_H", shape=(3 * GRU_B_STATE_SIZE, GRU_B_STATE_SIZE))
gru_b_W_X0 = relay.var("gru_b_W_X0", shape=(3 * GRU_B_STATE_SIZE, GRU_A_STATE_SIZE))
gru_b_W_X1 = relay.var("gru_b_W_X1", shape=(3 * GRU_B_STATE_SIZE, FEATURE_DENSE2_OUT_SIZE))
gru_b_B_H = relay.var("gru_b_B_H", shape=(3 * GRU_B_STATE_SIZE, ))
gru_b_B_X = relay.var("gru_b_B_X", shape=(3 * GRU_B_STATE_SIZE, ))

gru_b_next_hidden_state = gru(gru_a_next_hidden_state, gru_b_condition, gru_b_hidden_state, gru_b_W_X0, gru_b_W_X1, gru_b_W_H, gru_b_B_X, gru_b_B_H)

dual_fc_W = relay.var("dual_fc_W", shape=(512, 16))
dual_fc_B = relay.var("dual_fc_B", shape=(512, ))
dual_fc_A = relay.var("dual_fc_A", shape=(512, ))

approx_softmax = approx_exp
output = approx_softmax(dual_fc(gru_b_next_hidden_state, dual_fc_W, dual_fc_B, dual_fc_A))


outputs = relay.expr.Tuple([output])

func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
relay.ir_pass.infer_type(func)

param_vars = [
    gru_a_W_H, gru_a_B_H,
    gru_a_WD_0, gru_a_WD_1, gru_a_WD_2,
    gru_a_embedding_sig, gru_a_embedding_pred, gru_a_embedding_last_exc,
    gru_b_W_H, gru_b_W_X0, gru_b_W_X1, gru_b_B_H, gru_b_B_X,
    dual_fc_W, dual_fc_B, dual_fc_A,
]

params = collections.OrderedDict([(k, v) for param in param_vars for (k, v) in instantiate(param)])

print("Total param size: ", sum(v.asnumpy().nbytes for v in params.values()))
print("Total param size, no emb: ",
      sum(v.asnumpy().nbytes for (k, v) in params.items() if "embedding" not in k))
for k, v in params.items():
    print(k, v.asnumpy().size)

input_vars = [
    last_sig,
    pred,
    last_exc,
    gru_a_condition,
    gru_b_condition,
    gru_a_hidden_state,
    gru_b_hidden_state,

]

inputs = collections.OrderedDict(
    [
        (
            param.name_hint,
            tvm.ndarray.array(
                np.zeros(param.type_annotation.concrete_shape).astype(
                    param.type_annotation.dtype))
        )
        for param in input_vars
    ]
)


def tune():
    global func
    with relay.build_config(opt_level=2):
        func = relay.optimize(func, target=TARGET, params=params)
        print(func.astext(show_meta_data=False))
        tasks = autotvm.task.extract_from_program(
            func, target=TARGET, params=params, ops=(relay.op.nn.dense,))
        for i, tsk in enumerate(tasks):
            print(tsk)
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

            tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type='rank', feature_type="knob")
            n_trial = 100
            early_stopping = 200
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.RPCRunner(
                    DEVICE,
                    '0.0.0.0',
                    PORT,
                    number=100,
                    repeat=3,
                    min_repeat_ms=1000,
                    timeout=100)
            )

            tuner_obj.tune(
                n_trial=min(n_trial, len(tsk.config_space)),
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(n_trial, prefix=prefix),
                    autotvm.callback.log_to_file(log_filename)
                ]
            )

if TUNE:
    tune()

with autotvm.apply_history_best(log_filename):
    with relay.build_config(opt_level=3):
        func = relay.optimize(func, target=TARGET, params=params)
        print(func.astext(show_meta_data=False))
        func = relay.ir_pass.infer_type(func)
        graph, lib, new_params = relay.build_module.build(
            func, target=TARGET,  params=params)

        for (k, v) in params.items():
            print(k, v.shape)
        for (k, v) in new_params.items():
            print(k, v.shape)


lib.save("lpcnet.o")
tmp = tvm.contrib.util.tempdir()
lib_fname = tmp.relpath('net.tar')
with TARGET:
    lib.export_library(lib_fname)
tracker = tvm.rpc.connect_tracker('0.0.0.0', PORT)
remote = tracker.request(DEVICE)

remote.upload(lib_fname)
rlib = remote.load_module('net.tar')
ctx = remote.cpu(0)
r_new_params = {k: tvm.nd.array(v, ctx) for k, v in new_params.items()}
r_inputs = {k: tvm.nd.array(v, ctx) for k, v in inputs.items()}
module = graph_runtime.create(graph, rlib, ctx)
module.set_input(**r_new_params)
module.set_input(**r_inputs)
ftimer = module.module.time_evaluator("run", ctx, number=100000)
for i in range(5):
    prof_res = ftimer()
    print("TVM time: {:.2f}us".format(prof_res.mean * 10 ** 6))
