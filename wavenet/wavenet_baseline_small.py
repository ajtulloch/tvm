import torch
from torch import nn

dtype = "float32"

rnn_dims = 1024
fc_dims = 1024

feat_dims = 24
aux_dims = 32
n_classes = 2 ** 8

x = torch.zeros(1, rnn_dims)
h1 = torch.zeros(1, rnn_dims)

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.GRUCell(rnn_dims, rnn_dims)
        self.fc1 = nn.Linear(rnn_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, n_classes)

    def forward(self, x, h1):
        h1_prime = self.rnn1(x, h1)
        gru2_add1_o = h1_prime + x
        relu1_o = nn.functional.relu(self.fc1(gru2_add1_o))
        fc_3_o = self.fc3(relu1_o)
        return (fc_3_o, h1_prime)

traced_f = torch.jit.trace(G(), (x, h1,))
print(traced_f.graph)
for i in range(5):
    import time
    t = time.perf_counter()
    for _ in range(1000):
        traced_f(x, h1)
    t_end = time.perf_counter()
    print("PyTorch JIT time: {:.2f}us".format(((t_end - t) / 1000) * 10 ** 6))
