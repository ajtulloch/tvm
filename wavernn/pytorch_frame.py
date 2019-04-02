import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections

torch.manual_seed(42)


rnn_dims = 512
fc_dims = 512

feat_dims = 19
aux_dims = 64
n_classes = 2 ** 8

T = 8

x_0 = torch.randn(1, 1)
h1_0 = torch.randn(1, rnn_dims)
m = torch.randn(1, T, feat_dims)
a1 = torch.randn(1, T, aux_dims)
a2 = torch.randn(1, T, aux_dims)

I = nn.Linear(feat_dims + aux_dims + 1, rnn_dims)
rnn1 = nn.GRUCell(rnn_dims, rnn_dims)
fc1 = nn.Linear(rnn_dims + aux_dims, fc_dims)
fc2 = nn.Linear(fc_dims, n_classes)

def sample(x_prob):
    gumbel = -torch.log(-torch.log(torch.tensor(np.random.uniform(size=x_prob.shape).astype("float32"))))
    result = np.zeros((1, 1), dtype="float32")
    result[:] = np.argmax(x_prob - gumbel)
    return torch.tensor(result / n_classes)

def reference_frame(a1, a2, m, x_0, h1_0):
    np.random.seed(10)
    with torch.no_grad():
        (x, h1) = (x_0, h1_0)
        T = a1.shape[0]
        outs = []
        for t in range(T):
            xconcat = torch.cat([x, m[0, t:t+1], a1[0, t:t+1]], dim=1)
            xconcat_trns = I(xconcat)
            h1 = rnn1(xconcat_trns, h1)
            xres = xconcat_trns + h1
            xres_concat = torch.cat([xres, a2[0, t:t+1]], dim=1)
            x_fc = F.relu(fc1(xres_concat))
            x_prob = fc2(x_fc)
            x = sample(x_prob)
            outs.append(x)
        return outs, h1


def reference():
    return reference_frame(a1, a2, m, x_0, h1_0)

def test_pytorch_reference():
    outs_ref, h1_ref = reference()
    outs_new, h1_new = reference_frame(a1, a2, m, x_0, h1_0)
    np.testing.assert_allclose(outs_ref, outs_new)
    np.testing.assert_allclose(h1_ref, h1_new)

def factored_premul_frame(a1, a2, m, x_0, h1_0):
    np.random.seed(10)
    with torch.no_grad():
        I_residual =  m[0] @ I.weight[:, 1:1 + feat_dims].transpose(1, 0) + a1[0] @ I.weight[:, 1 + feat_dims:].transpose(1, 0)
        fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)
        I_weight = I.weight[:]
        I_bias = I.bias[:]

        Ifactored = nn.Linear(1, rnn_dims)
        Ifactored.weight[:, :] = I.weight[:, :1]
        Ifactored.bias[:] = I.bias[:]

        (x, h1) = (x_0, h1_0)
        outs = []
        T = a1.shape[0]
        for t in range(T):
            xconcat_trns = Ifactored(x) + I_residual[t:t+1]

            def gru_cell(cell, x, h):
                xt = x @ cell.weight_ih.transpose(1, 0) + cell.bias_ih
                ht = h @ cell.weight_hh.transpose(1, 0) + cell.bias_hh
                assert xt.shape == (1, 3 * rnn_dims)
                assert ht.shape == (1, 3 * rnn_dims)

                reset_gate = torch.sigmoid(xt[:, 0:rnn_dims] + ht[:, 0:rnn_dims])
                input_gate = torch.sigmoid(xt[:, rnn_dims:2 * rnn_dims] + ht[:, rnn_dims:2 * rnn_dims])
                new_gate = torch.tanh(xt[:, 2 * rnn_dims:3 * rnn_dims] + reset_gate * ht[:, 2 * rnn_dims:3 * rnn_dims])
                return new_gate + input_gate * (h - new_gate)

            h1 = gru_cell(rnn1, xconcat_trns, h1)

            xres = xconcat_trns + h1

            fc1factored = nn.Linear(rnn_dims, fc_dims)
            fc1factored.weight[:, :] = fc1.weight[:, :rnn_dims]
            fc1factored.bias[:] = fc1.bias[:]

            x_fc = F.relu(fc1factored(xres) + fc1_residual[t:t+1])
            x_prob = fc2(x_fc)
            x = sample(x_prob)
            outs.append(x)
        return outs, h1

def build_wavernn_module():
    from tvm import relay
    import tvm

    with torch.no_grad():
        Ifactored = nn.Linear(1, rnn_dims)
        Ifactored.weight[:, :] = I.weight[:, :1]
        Ifactored.bias[:] = I.bias[:]

        fc1factored = nn.Linear(rnn_dims, fc_dims)
        fc1factored.weight[:, :] = fc1.weight[:, :rnn_dims]
        fc1factored.bias[:] = fc1.bias[:]

        params = {
            "I_W": tvm.ndarray.array(Ifactored.weight.detach().numpy()),
            "I_B": tvm.ndarray.array(Ifactored.bias.detach().numpy()),
            "rnn1_weight_ih": tvm.ndarray.array(rnn1.weight_ih.detach().numpy()),
            "rnn1_weight_hh": tvm.ndarray.array(rnn1.weight_hh.detach().numpy()),
            "rnn1_bias_ih": tvm.ndarray.array(rnn1.bias_ih.detach().numpy()),
            "rnn1_bias_hh": tvm.ndarray.array(rnn1.bias_hh.detach().numpy()),
            "fc1_W": tvm.ndarray.array(fc1factored.weight.detach().numpy()),
            "fc1_B": tvm.ndarray.array(fc1factored.bias.detach().numpy()),
            "fc2_W": tvm.ndarray.array(fc2.weight.detach().numpy()),
            "fc2_B": tvm.ndarray.array(fc2.bias.detach().numpy()),
        }

    Rx = relay.var("x", shape=[1, 1], dtype="float32")
    Rh1 = relay.var("h1", shape=[1, rnn_dims], dtype="float32")
    RI_residual = relay.var("I_residual", shape=[1, rnn_dims], dtype="float32")
    Rfc1_residual = relay.var("fc1_residual", shape=[1, fc_dims], dtype="float32")
    RI_W = relay.var("I_W", shape=(rnn_dims, 1), dtype="float32")
    RI_B = relay.var("I_B", shape=(rnn_dims,), dtype="float32")

    Cell = collections.namedtuple('Cell', ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'])

    def dense(X, W, B, **kwargs):
        return relay.nn.bias_add(relay.nn.dense(X, W), B)

    def gru_cell(cell, x, h):
        xt = dense(x, cell.weight_ih, cell.bias_ih)
        ht = dense(h, cell.weight_hh, cell.bias_hh)
        xt_gates = relay.split(xt, indices_or_sections=3, axis=1)
        ht_gates = relay.split(ht, indices_or_sections=3, axis=1)
        reset_gate = relay.sigmoid(xt_gates[0] + ht_gates[0])
        input_gate = relay.sigmoid(xt_gates[1] + ht_gates[1])
        new_gate = relay.tanh(xt_gates[2] + reset_gate * ht_gates[2])
        return new_gate + input_gate * (h - new_gate)

    xconcat_trns = dense(Rx, RI_W, RI_B) + RI_residual

    Rrnn1 = Cell(
        weight_ih=relay.var("rnn1_weight_ih", shape=(3 * rnn_dims, rnn_dims), dtype="float32"),
        weight_hh=relay.var("rnn1_weight_hh", shape=(3 * rnn_dims, rnn_dims), dtype="float32"),
        bias_ih=relay.var("rnn1_bias_ih", shape=(3 * rnn_dims, ), dtype="float32"),
        bias_hh=relay.var("rnn1_bias_hh", shape=(3 * rnn_dims, ), dtype="float32"),
    )
    h1 = gru_cell(Rrnn1, xconcat_trns, Rh1)
    xres = xconcat_trns + h1

    Rfc1_W = relay.var("fc1_W", shape=(fc_dims, rnn_dims), dtype="float32")
    Rfc1_B = relay.var("fc1_B", shape=(fc_dims,), dtype="float32")

    x_fc = relay.nn.relu(dense(xres, Rfc1_W, Rfc1_B) + Rfc1_residual)

    Rfc2_W = relay.var("fc2_W", shape=(n_classes, fc_dims), dtype="float32")
    Rfc2_B = relay.var("fc2_B", shape=(n_classes,), dtype="float32")

    x_prob = dense(x_fc, Rfc2_W, Rfc2_B)

    outputs = relay.expr.Tuple([x_prob, h1])
    func = relay.Function(relay.ir_pass.free_vars(outputs), outputs)
    func = relay.ir_pass.infer_type(func)
    graph, lib, params = relay.build_module.build(func, target="llvm", params=params)

    module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu(0))
    module.set_input(**params)
    return module

def factored_relay_frame(a1, a2, m, x_0, h1_0):
    import tvm
    np.random.seed(10)
    (x, h1) = (tvm.ndarray.array(x_0), tvm.ndarray.array(h1_0))
    outs = []
    T = a1.shape[0]
    module = build_wavernn_module()

    I_residual = m[0] @ I.weight[:, 1:1 + feat_dims].transpose(1, 0) + a1[0] @ I.weight[:, 1 + feat_dims:].transpose(1, 0)
    fc1_residual = a2[0] @ fc1.weight[:, rnn_dims:].transpose(1, 0)

    for t in range(T):
        inputs = {
            "x": x,
            "h1": h1,
            "I_residual": tvm.ndarray.array(I_residual[t:t+1].detach().numpy()),
            "fc1_residual": tvm.ndarray.array(fc1_residual[t:t+1].detach().numpy()),
        }

        module.set_input(**inputs)
        module.run()
        (x_prob, h1) = module.get_output(0), module.get_output(1)
        x = tvm.ndarray.array(sample(torch.tensor(x_prob.asnumpy())))
        outs.append(x.asnumpy()[0][0])
    return outs, h1.asnumpy()

def test_factored_premul_frame():
    outs_ref, h1_ref = reference()
    outs_new, h1_new = factored_premul_frame(a1, a2, m, x_0, h1_0)
    np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-4, atol=1e-4)

def test_relay_frame():
    outs_ref, h1_ref = reference()
    outs_new, h1_new = factored_relay_frame(a1, a2, m, x_0, h1_0)
    np.testing.assert_allclose(outs_ref, outs_new, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(h1_ref, h1_new, rtol=1e-4, atol=1e-4)
