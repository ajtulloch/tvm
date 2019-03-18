# coding: utf-8

import math
import os
import pickle
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
# from utils.display import *
# from utils.dsp import *
import argparse
import tvm
from tvm import relay
from tvm import autotvm

# from tvm.contrib.debugger import debug_runtime as runtime
import tvm.contrib.graph_runtime as runtime
import tvm.contrib.graph_runtime as graph_runtime

import topi.x86.dense

logging.basicConfig(level=logging.DEBUG)
skl_target = tvm.target.create(
    'llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu')

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
args = parser.parse_args()
print("args.device: ",args.device)
assert (args.device in ['cpu', 'cuda']), '--device must be either "cpu" or "cuda"'

def inv_delta(y, delta_scale=2):
    y_cumsum = np.array([y[0]])

    for i in np.arange(1, len(y)):
        y_cumsum = np.append(y_cumsum,np.array([y_cumsum[i-1]*0.99+y[i]]))

    y = y_cumsum/delta_scale

    return y


bits = 8
feature_dim = 19
feature_upsample_factors = (4,4,8)
sample_rate = 24000
DEVICE_NAME = args.device
DATA_PATH = "dataset_uxr_default_8bits_muLaw/"
GEN_PATH = "synthesis_audio/"

MODEL_FILE = "smoothedGTMFCC13_f0_periodicity_MuLaw_decayLR3_deltaScale2_randomTestSet_8bits_epoch550.pyt"
os.makedirs(GEN_PATH, exist_ok=True)

print(DATA_PATH)
print(GEN_PATH)
print('DEVICE_NAME: ', DEVICE_NAME)


# with open(f"{DATA_PATH}dataset_ids.pkl", "rb") as f:
#     test_ids = pickle.load(f)

# ## Define Model Classes
class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=5, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    def __init__(
        self, feat_dims, upsample_scales, compute_dims, res_blocks, res_out_dims, pad
    ):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1.0 / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent : -self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class RelayModel(object):
    def __init__(self, rnn_dims, fc_dims, bits, feat_dims, res_out_dims):
        self.rnn_dims = rnn_dims
        self.fc_dims = fc_dims
        self.bits = bits
        self.feat_dims = feat_dims
        self.res_out_dims = res_out_dims
        self.aux_dims = res_out_dims // 4
        self.n_classes = 2 ** bits
        self.dtype = "float32"
        self.tgt = "llvm -mcpu=core-avx2"
        self.tgt_host="llvm"
        self.ctx = tvm.context(self.tgt, 0)
        self.inputs = ["x", "h1", "h2", "m_t", "a1_t", "a2_t", "a3_t", "a4_t"]
        self.net = None
        self.params = None
        self.module = None
        self.tuning_option = {
            'log_filename': "speech_synthesis.log",
            'tuner': 'xgb',
            'early_stopping': None,

            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.RPCRunner(
                    'skl',
                    '0.0.0.0',
                    9195,
                    number=100,
                    repeat=5,
                    min_repeat_ms=1000,
                    timeout=100),

                # runner=autotvm.LocalRunner(number=10, repeat=1,
                #                            min_repeat_ms=1000),
            ),
        }

    def toTVMTensor(self, torch_tensor):
        return tvm.nd.array(torch_tensor.detach().numpy().astype(self.dtype),
                            self.ctx) if torch_tensor is not None else None

    def GRUCellSlow(self, name, input_size, hidden_size, x, h):
        w_ir = relay.var(name + ".w_ir", shape=[hidden_size, input_size], dtype=self.dtype)
        w_iz = relay.var(name + ".w_iz", shape=[hidden_size, input_size], dtype=self.dtype)
        w_in = relay.var(name + ".w_in", shape=[hidden_size, input_size], dtype=self.dtype)
        w_hr = relay.var(name + ".w_hr", shape=[hidden_size, input_size], dtype=self.dtype)
        w_hz = relay.var(name + ".w_hz", shape=[hidden_size, input_size], dtype=self.dtype)
        w_hn = relay.var(name + ".w_hn", shape=[hidden_size, input_size], dtype=self.dtype)
        b_ir = relay.var(name + ".b_ir", shape=[hidden_size], dtype=self.dtype)
        b_iz = relay.var(name + ".b_iz", shape=[hidden_size], dtype=self.dtype)
        b_in = relay.var(name + ".b_in", shape=[hidden_size], dtype=self.dtype)
        b_hr = relay.var(name + ".b_hr", shape=[hidden_size], dtype=self.dtype)
        b_hz = relay.var(name + ".b_hz", shape=[hidden_size], dtype=self.dtype)
        b_hn = relay.var(name + ".b_hn", shape=[hidden_size], dtype=self.dtype)

        ir_wo = relay.nn.dense(x, w_ir)
        ir_o = relay.nn.bias_add(ir_wo, b_ir)

        hr_wo = relay.nn.dense(h, w_hr)
        hr_o = relay.nn.bias_add(hr_wo, b_hr)

        r_ao = relay.add(ir_o, hr_o)
        r_o = relay.sigmoid(r_ao)

        iz_wo = relay.nn.dense(x, w_iz)
        iz_o = relay.nn.bias_add(iz_wo, b_iz)

        hz_wo = relay.nn.dense(h, w_hz)
        hz_o = relay.nn.bias_add(hz_wo, b_hz)

        z_ao = relay.add(iz_o, hz_o)
        z_o = relay.sigmoid(z_ao)

        in_wo = relay.nn.dense(x, w_in)
        in_o = relay.nn.bias_add(in_wo, b_in)

        hn_wo = relay.nn.dense(h, w_hn)
        hn_o = relay.nn.bias_add(hn_wo, b_hn)

        n1 = relay.multiply(r_o, hn_o)
        n_ao = relay.add(in_o, n1)
        n_o = relay.tanh(n_ao)

        ones = relay.ones((1, hidden_size), dtype=self.dtype)
        z_1 = relay.subtract(ones, z_o)
        n2 = relay.multiply(z_1, n_o)
        h2 = relay.multiply(z_o, h)
        h_prime = relay.add(n2, h2)
        return h_prime

    def GRUCell(self, name, input_size, hidden_size, x, h):
        w_ir = relay.var(name + ".w_ir", shape=[hidden_size * 3, input_size], dtype=self.dtype)
        w_hr = relay.var(name + ".w_hr", shape=[hidden_size * 3, input_size], dtype=self.dtype)
        b_ir = relay.var(name + ".b_ir", shape=[hidden_size * 3], dtype=self.dtype)
        b_hr = relay.var(name + ".b_hr", shape=[hidden_size * 3], dtype=self.dtype)


        ir_wo = relay.nn.dense(x, w_ir)
        ir_o = relay.nn.bias_add(ir_wo, b_ir)

        hr_wo = relay.nn.dense(h, w_hr)
        hr_o = relay.nn.bias_add(hr_wo, b_hr)

        z_ao = relay.add(ir_o, hr_o)
        z_o = relay.sigmoid(z_ao) # [1, H * 3]
        return relay.strided_slice(z_o, [0, 0], [1, hidden_size])

    def denseRelu(self, name, output_dim, input_dim, input,
                  add_bias=True, add_relu=True):
        w = relay.var(name+".w", shape=[output_dim, input_dim], dtype=self.dtype)
        o = relay.nn.dense(input, w)
        if add_bias:
            b = relay.var(name+".b", shape=[output_dim], dtype=self.dtype)
            o = relay.nn.bias_add(o, b)
        if add_relu:
            o = relay.nn.relu(o)
        return o

    def build(self):
        x = relay.var("x", shape=[1, 1], dtype=self.dtype)
        h1 = relay.var("h1", shape=[1, self.rnn_dims], dtype=self.dtype)
        h2 = relay.var("h2", shape=[1, self.rnn_dims], dtype=self.dtype)
        m_t = relay.var("m_t", shape=[1, self.feat_dims], dtype=self.dtype)
        a1_t = relay.var("a1_t", shape=[1, self.aux_dims], dtype=self.dtype)
        a2_t = relay.var("a2_t", shape=[1, self.aux_dims], dtype=self.dtype)
        a3_t = relay.var("a3_t", shape=[1, self.aux_dims], dtype=self.dtype)
        a4_t = relay.var("a4_t", shape=[1, self.aux_dims], dtype=self.dtype)

        concat0_o = relay.concatenate([x, m_t, a1_t], 1)

        i_o = self.denseRelu("fc_i", self.rnn_dims, self.feat_dims + self.aux_dims + 1,
                             concat0_o, add_bias=True, add_relu=False)

        h1_prime = self.GRUCell("gru1", self.rnn_dims, self.rnn_dims, i_o, h1)

        gru2_add1_o = relay.add(i_o, h1_prime)
        inp = relay.concatenate([gru2_add1_o, a2_t], 1)
        h2_prime = self.GRUCell("gru2", self.rnn_dims + self.aux_dims, self.rnn_dims, inp, h2)
        add1_o = relay.add(gru2_add1_o, h2_prime)

        concat1_o = relay.concatenate([add1_o, a3_t], 1)

        relu1_o = self.denseRelu("fc1", self.fc_dims,
                                 self.rnn_dims + self.aux_dims,
                                 concat1_o, add_bias=True, add_relu=True)

        concat2_o = relay.concatenate([relu1_o, a4_t], 1)

        relu2_o = self.denseRelu("fc2", self.fc_dims,
                                 self.fc_dims + self.aux_dims,
                                 concat2_o, add_bias=True, add_relu=True)

        fc3_o = self.denseRelu("fc3", self.n_classes,
                               self.fc_dims + self.aux_dims,
                               relu2_o, add_bias=True, add_relu=False)

        softmax_o = relay.nn.softmax(fc3_o, axis=-1)

        squeeze_o = relay.op.transform.squeeze(softmax_o, axis=None)

        output = squeeze_o
        return relay.Function(relay.ir_pass.free_vars(output), output)

    def getWorkload(self, pytorch_params):
        net = self.build()
        net = relay.ir_pass.infer_type(net)
        shape_dict = {
            v.name_hint : v.checked_type for v in net.params}
        net.astext()
        params = {}
        for k, v in shape_dict.items():
            if k in self.inputs:
                continue
            assert(k in pytorch_params)
            params[k] = self.toTVMTensor(pytorch_params[k])
        self.net = net
        self.params = params
        return net, params

    def initModel(self, pytorch_params):
        net, params = self.getWorkload(pytorch_params)
        # import pdb; pdb.set_trace()
        print(net.astext(show_meta_data=False))
        # with autotvm.apply_history_best(self.tuning_option["log_filename"]):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                net, target=skl_target,  params=params)
            # print(graph.astext(show_meta_data=False))
            import netron
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix="tvm.json") as f:
                f.write(graph.encode())
            netron.start(f.name, host="localhost")

        tmp = tvm.contrib.util.tempdir()
        lib_fname = tmp.relpath('net.tar')
        with skl_target:
            lib.export_library(lib_fname)
        tracker = tvm.rpc.connect_tracker('0.0.0.0', 9195)
        remote = tracker.request('skl')

        remote.upload(lib_fname)
        rlib = remote.load_module('net.tar')
        ctx = remote.cpu(0)

        module = graph_runtime.create(graph, rlib, ctx)
        module.set_input()
        inputs = {
            "x": torch.zeros(1, 1),
            "h1": torch.zeros(1, self.rnn_dims),
            "h2": torch.zeros(1, self.rnn_dims),
            "m_t": torch.zeros(1, self.feat_dims),
            "a1_t": torch.zeros(1, self.aux_dims),
            "a2_t": torch.zeros(1, self.aux_dims),
            "a3_t": torch.zeros(1, self.aux_dims),
            "a4_t": torch.zeros(1, self.aux_dims),
        }

        rparams = {k: tvm.nd.array(v.shape, ctx) for k, v in params.items()}
        # module.set_input(**rparams)
        module.run()
        out = module.get_output(0, tvm.nd.empty((256,), ctx=ctx))
        out.asnumpy()
        ftimer = module.module.time_evaluator("run", ctx, 100)
        for i in range(1):
            prof_res = ftimer()
            # time.sleep(1)

        for i in range(5):
            prof_res = ftimer()
            print("TVM time: ", prof_res.mean)

        return module
        # # upload parameters to device
        # self.ctx = tvm.cpu()
        # module = runtime.create(graph, lib, self.ctx)
        # module.set_input(**params)
        # self.module = module
        # return module

    def runOnce(self, inputs):
        tvm_inputs = {x : self.toTVMTensor(inputs[x]) for x in inputs}
        self.module.run(**tvm_inputs)
        ftimer = self.module.module.time_evaluator("run", self.ctx, number=100, repeat=3)
        return ftimer

    def tune(self):
        tasks = autotvm.task.extract_from_program(self.net, target=skl_target,
                                                  params=self.params, ops=(relay.op.nn.dense,))
        n_trial = 100
        measure_option = self.tuning_option["measure_option"]
        tuner = self.tuning_option["tuner"]
        early_stopping = self.tuning_option["early_stopping"]
        log_filename = self.tuning_option["log_filename"]

        for i, tsk in enumerate(tasks):
            print(tsk)
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

            # create tuner
            if tuner == 'xgb' or tuner == 'xgb-rank':
                tuner_obj = autotvm.tuner.XGBTuner(
                    tsk, loss_type='rank', feature_type="knob")
            elif tuner == 'ga':
                tuner_obj = autotvm.tuner.GATuner(tsk, pop_size=50)
            elif tuner == 'random':
                tuner_obj = autotvm.tuner.RandomTuner(tsk)
            elif tuner == 'gridsearch':
                tuner_obj = autotvm.tuner.GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            # if use_transfer_learning:
            #     if os.path.isfile(log_filename):
            #         tuner_obj.load_history(
            #             autotvm.record.load_from_file(log_filename))

            # do tuning
            print(tsk.config_space)
            tuner_obj.tune(
                n_trial=min(n_trial, len(tsk.config_space)),
                early_stopping=early_stopping,
                measure_option=measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(n_trial, prefix=prefix),
                    autotvm.callback.log_to_file(log_filename)
                ])

        # for i, tsk in enumerate(tasks):
        #     prefix = "[Task %2d/%2d] " % (i+1, len(tasks))
        #     # op_name = tsk.workload[0]
        #     # # if op_name == '':
        #     # #     func_create = 'topi_nn_dense'
        #     # # else:
        #     # #     continue
        #     # task = autotvm.task.create(func_create, args=tsk.args,
        #     #                            target=target, template_key='direct')
        #     # task.workload = tsk.workload

        #     # create tuner
        #     if tuner == 'xgb' or tuner == 'xgb-rank':
        #         tuner_obj = XGBTuner(task, loss_type='rank')
        #     elif tuner == 'ga':
        #         tuner_obj = GATuner(task, pop_size=50)
        #     elif tuner == 'random':
        #         tuner_obj = RandomTuner(task)
        #     elif tuner == 'gridsearch':
        #         tuner_obj = GridSearchTuner(task)
        #     else:
        #         raise ValueError("Invalid tuner: " + tuner)

        #     # do tuning
        #     n_trial=len(task.config_space)
        #     tuner_obj.tune(n_trial=n_trial,
        #                    early_stopping=early_stopping,
        #                    measure_option=measure_option,
        #                    callbacks=[
        #                        autotvm.callback.progress_bar(n_trial, prefix=prefix),
        #                        autotvm.callback.log_to_file(log_filename)])



class Model(nn.Module):
    def __init__(
        self,
        rnn_dims,
        fc_dims,
        bits,
        pad,
        upsample_factors,
        feat_dims,
        compute_dims,
        res_out_dims,
        res_blocks,
    ):
        super().__init__()
        self.n_classes = 2 ** bits
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.upsample = UpsampleNetwork(
            feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad
        )

        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)
        # num_params(self)
        self.relay = RelayModel(rnn_dims, fc_dims, bits, feat_dims, res_out_dims)
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        params = {
            "fc_i.w": self.I.weight,
            "fc_i.b": self.I.bias,
            "fc1.w" : self.fc1.weight,
            "fc1.b" : self.fc1.bias,
            "fc2.w" : self.fc2.weight,
            "fc2.b" : self.fc2.bias,
            "fc3.w" : self.fc3.weight,
            "fc3.b" : self.fc3.bias,
            "gru1.w_ir": rnn1.weight_ih.data, #[0:self.rnn_dims,],
            "gru1.w_hr": rnn1.weight_hh.data,#[0:self.rnn_dims,],
            "gru1.b_ir": rnn1.bias_ih.data,#[0:self.rnn_dims],
            "gru1.b_hr": rnn1.bias_hh.data, #[0:self.rnn_dims],
            "gru2.w_ir": rnn2.weight_ih.data,#[0:self.rnn_dims,],
            "gru2.w_hr": rnn2.weight_hh.data,# [0:self.rnn_dims,],
            "gru2.b_ir": rnn2.bias_ih.data, #[0:self.rnn_dims],
            "gru2.b_hr": rnn2.bias_hh.data,#[0:self.rnn_dims],
        }
        # self.relay.initModel(params)
        # self.relay.tune()
        with tvm.autotvm.apply_history_best("speech_synthesis.log"):
            self.relay = RelayModel(rnn_dims, fc_dims, bits, feat_dims, res_out_dims)
            self.relay.initModel(params)
    def maxDiff(self, a, b):
        x = a.asnumpy()
        y = b.asnumpy()
        z = np.abs(x - y).max()
        return z

    def generate(self, mels, save_path):
        self.eval()
        output = []
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        params = {
            "fc_i.w": self.I.weight,
            "fc_i.b": self.I.bias,
            "fc1.w" : self.fc1.weight,
            "fc1.b" : self.fc1.bias,
            "fc2.w" : self.fc2.weight,
            "fc2.b" : self.fc2.bias,
            "fc3.w" : self.fc3.weight,
            "fc3.b" : self.fc3.bias,
            "gru1.w_ir": rnn1.weight_ih.data[0:self.rnn_dims,],
            "gru1.w_iz": rnn1.weight_ih.data[self.rnn_dims:self.rnn_dims*2,],
            "gru1.w_in": rnn1.weight_ih.data[self.rnn_dims*2:self.rnn_dims*3,],
            "gru1.w_hr": rnn1.weight_hh.data, #[0:self.rnn_dims,],
            "gru1.w_hz": rnn1.weight_hh.data[self.rnn_dims:self.rnn_dims*2,],
            "gru1.w_hn": rnn1.weight_hh.data[self.rnn_dims*2:self.rnn_dims*3,],
            "gru1.b_ir": rnn1.bias_ih.data, #[0:self.rnn_dims],
            "gru1.b_iz": rnn1.bias_ih.data[self.rnn_dims:self.rnn_dims*2],
            "gru1.b_in": rnn1.bias_ih.data[self.rnn_dims*2:self.rnn_dims*3],
            "gru1.b_hr": rnn1.bias_hh.data, #[0:self.rnn_dims],
            "gru1.b_hz": rnn1.bias_hh.data[self.rnn_dims:self.rnn_dims*2],
            "gru1.b_hn": rnn1.bias_hh.data[self.rnn_dims*2:self.rnn_dims*3],
            "gru2.w_ir": rnn2.weight_ih.data[0:self.rnn_dims,],
            "gru2.w_iz": rnn2.weight_ih.data[self.rnn_dims:self.rnn_dims*2,],
            "gru2.w_in": rnn2.weight_ih.data[self.rnn_dims*2:self.rnn_dims*3,],
            "gru2.w_hr": rnn2.weight_hh.data, #[0:self.rnn_dims,],
            "gru2.w_hz": rnn2.weight_hh.data[self.rnn_dims:self.rnn_dims*2,],
            "gru2.w_hn": rnn2.weight_hh.data[self.rnn_dims*2:self.rnn_dims*3,],
            "gru2.b_ir": rnn2.bias_ih.data[0:self.rnn_dims],
            "gru2.b_iz": rnn2.bias_ih.data[self.rnn_dims:self.rnn_dims*2],
            "gru2.b_in": rnn2.bias_ih.data[self.rnn_dims*2:self.rnn_dims*3],
            "gru2.b_hr": rnn2.bias_hh.data, #[0:self.rnn_dims],
            "gru2.b_hz": rnn2.bias_hh.data[self.rnn_dims:self.rnn_dims*2],
            "gru2.b_hn": rnn2.bias_hh.data[self.rnn_dims*2:self.rnn_dims*3],
        }
        self.relay.initModel(params)
        # self.relay.tune()

        with torch.no_grad():
            start = time.time()
            x = getattr(torch.zeros(1, 1), DEVICE_NAME)()
            h1 = getattr(torch.zeros(1, self.rnn_dims), DEVICE_NAME)()
            h2 = getattr(torch.zeros(1, self.rnn_dims), DEVICE_NAME)()

            mels = getattr(torch.FloatTensor(mels), DEVICE_NAME)().unsqueeze(0)
            mels, aux = self.upsample(mels)

            aux_idx = [self.aux_dims * i for i in range(5)]
            a1 = aux[:, :, aux_idx[0] : aux_idx[1]]
            a2 = aux[:, :, aux_idx[1] : aux_idx[2]]
            a3 = aux[:, :, aux_idx[2] : aux_idx[3]]
            a4 = aux[:, :, aux_idx[3] : aux_idx[4]]

            seq_len = mels.size(1)

            prenet_compute_end_time = time.time()
            prenet_compute_time = prenet_compute_end_time-start

            print("Prenet compute time: {:.4f}ms; generates {} samples".format(prenet_compute_time*1000, seq_len))
            for i in range(seq_len):

                sample_start_time = time.time()
                m_t = mels[:, i, :]
                a1_t = a1[:, i, :]
                a2_t = a2[:, i, :]
                a3_t = a3[:, i, :]
                a4_t = a4[:, i, :]

                inputs = {
                    "x": x,
                    "h1": h1,
                    "h2": h2,
                    "m_t": m_t,
                    "a1_t": a1_t,
                    "a2_t": a2_t,
                    "a3_t": a3_t,
                    "a4_t": a4_t,
                }

                ro = tvm.nd.empty([256])
                ftimer = self.relay.runOnce(inputs)
                prof_res = np.array(ftimer().results)
                relay_compute_time = np.mean(prof_res)
                self.relay.module.get_output(0, ro)

                rnn1_start = time.time()
                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)
                # tvm.testing.assert_allclose(ro.asnumpy(), h1.detach().numpy(), atol=1e-4, rtol=1e-4)

                rnn1_compute_time = time.time()-rnn1_start

                rnn2_start = time.time()
                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)
                rnn2_compute_time = time.time()-rnn2_start
                # tvm.testing.assert_allclose(ro.asnumpy(), h2.detach().numpy(), atol=1e-4, rtol=1e-4)

                pytorch_start = time.time()
                fc1_start = time.time()
                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))
                fc1_compute_time = time.time()-fc1_start
                # import pdb; pdb.set_trace()
                # tvm.testing.assert_allclose(ro.asnumpy(), x.detach().numpy(), atol=1e-4, rtol=1e-4)

                fc2_start = time.time()
                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))
                fc2_compute_time = time.time()-fc2_start
                x = self.fc3(x)
                posterior = F.softmax(x, dim=1).view(-1)
                pytorch_compute_time = time.time() - pytorch_start
                # import pdb; pdb.set_trace()
                tvm.testing.assert_allclose(ro.asnumpy(), posterior.detach().numpy(), atol=1e-4, rtol=1e-4)

                distrib = torch.distributions.Categorical(posterior)
                sample = distrib.sample().float()/2**bits
                output.append(sample)
                x = getattr(torch.FloatTensor([[sample]]), DEVICE_NAME)()

                sample_end_time = time.time()
                sample_compute_time = sample_end_time-sample_start_time
                # print("i = {}, diff = {}".format(i, self.maxDiff(to, posterior)))
                if i%10 == 0:
                    speed = int((i + 1) / (time.time() - start))
                    print("{}/{} -- Avg.Speed: {:.2f} samples/sec".format(i + 1, seq_len, speed))
                    print("Single sample: {:.4f}ms; RNN1: {:.4f}ms; RNN2: {:.4f}ms; FC1: {:.4f}ms; FC2: {:.4f}ms; RELAY: {:.4f}ms; PYTORCH: {:.4f}ms;".format(sample_compute_time*1000, rnn1_compute_time*1000, rnn2_compute_time*1000, fc1_compute_time*1000, fc2_compute_time*1000, relay_compute_time*1000, pytorch_compute_time*1000))


        output = torch.stack(output).cpu().numpy()

        self.train()
        return output

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell


# ## Generate Samples
def generate():
    pass
    global output
    samples = len(test_ids[:])
    test_mels = [np.load(f"{DATA_PATH}mel/{id}.npy") for id in test_ids[:]]
    ground_truth = [np.load(f"{DATA_PATH}quant/{id}.npy") for id in test_ids[:]]

    for i, (gt, mel, test_id) in enumerate(zip(ground_truth, test_mels, test_ids)):
        print("\nGenerating: {}/{}    Sample rate:{}".format(i + 1, samples, sample_rate))
        output = model.generate(mel, f"{GEN_PATH}{test_id}")


model = getattr(Model(
    rnn_dims=512,
    fc_dims=512,
    bits=bits,
    pad=2,
    upsample_factors=feature_upsample_factors,
    feat_dims=feature_dim,
    compute_dims=128,
    res_out_dims=128,
    res_blocks=10,
), DEVICE_NAME)()

# model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE_NAME))
global step
step = 0



generate()
