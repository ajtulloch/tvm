import torch
from torch import nn

dtype = "float32"

rnn_dims = 128
fc_dims = 128

feat_dims = 24
aux_dims = 32
n_classes = 2 ** 8

x = torch.zeros(1, 1)
h1 = torch.zeros(1, rnn_dims)
h2 = torch.zeros(1, rnn_dims)
m_t = torch.zeros(1, feat_dims)
a1_t = torch.zeros(1, aux_dims)
a2_t = torch.zeros(1, aux_dims)
a3_t = torch.zeros(1, aux_dims)
a4_t = torch.zeros(1, aux_dims)


class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.I = nn.Linear(feat_dims + aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRUCell(rnn_dims, rnn_dims)
        self.rnn2 = nn.GRUCell(rnn_dims + aux_dims, rnn_dims)
        self.fc1 = nn.Linear(rnn_dims + aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, n_classes)

    def forward(self, x, h1, h2, m_t, a1_t, a2_t, a3_t, a4_t):
        concat0_o = torch.cat([x, m_t, a1_t], dim=1)
        i_o = self.I(concat0_o)
        # print(i_o.shape)
        h1_prime = self.rnn1(i_o, h1)
        gru2_add1_o = h1_prime + i_o
        inp = torch.cat([gru2_add1_o, a2_t], dim=1)
        # print(inp.shape)
        h2_prime = self.rnn2(inp, h2)
        add1_o = gru2_add1_o + h2_prime
        concat1_o = torch.cat([add1_o, a3_t], dim=1)
        # print(concat1_o.shape)
        relu1_o = nn.functional.relu(self.fc1(concat1_o))
        concat2_o = torch.cat([relu1_o, a4_t], dim=1)
        # print(concat2_o.shape)
        relu2_o = nn.functional.relu(self.fc2(concat2_o))
        fc_3_o = self.fc3(relu2_o)
        return (fc_3_o, h1_prime, h2_prime)

traced_f = torch.jit.trace(G(), (x, h1, h2, m_t, a1_t, a2_t, a3_t, a4_t))
print(traced_f.graph)
for i in range(5):
    import time
    t = time.perf_counter()
    for _ in range(1000):
        traced_f(x, h1, h2, m_t, a1_t, a2_t, a3_t, a4_t)
    t_end = time.perf_counter()
    print("PyTorch JIT time: {:.2f}us".format(((t_end - t) / 1000) * 10 ** 6))
