from __future__ import division
from topi.util import get_const_int, get_const_tuple, const_matrix
import numpy as np
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn import pad
import tvm


def decl_winograd(cfg, data, kernel, strides, padding, layout, out_dtype):
    N, CI, IH, IW = get_const_tuple(data.shape)
    CO, _, KH, KW = get_const_tuple(kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    assert layout == 'NCHW'
    assert KH == 3 and KW == 3 and HPAD == 1 and WPAD == 1 and HSTR == 1 and WSTR == 1
    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")


    A_data = np.array(
        [[1,  1,  1,   1,    1,    32,      32,    0],
         [0,  1,  -1,  2,   -2,  16,   -16,   0],
         [0,  1,  1,   4,    4,   8,    8,   0],
         [0,  1,  -1,  8,   -8,   4,   -4,   0],
         [0,  1,  1,   16,  16,   2,  2,   0],
         [0,  1,  -1,  32,  -32,  1,  -1,  1]],
        dtype=np.float32).T
    G_data = np.array(
        [[1,      0,     0],
         [-2/9,  -2/9,   -2/9],
         [-2/9,   2/9,   -2/9],
         [1/90,  1/45,   2/45],
         [1/90,  -1/45,  2/45],
         [1/45,    1/90, 1/180],
         [1/45,   -1/90, 1/190],
         [0,      0,     1]],
        dtype=np.float32)
    B_data = np.array(
        [[1,   0,    -21/4,    0,    21/4,     0,    -1,  0],
         [0,   1,      1,    -17/4,  -17/4,    1,    1,   0],
         [0,   -1,     1,    17/4,   -17/4,   -1,    1,   0],
         [0,  1/2,    1/4,   -5/2,   -5/4,     2,    1,   0],
         [0,  -1/2,   1/4,    5/2,   -5/4,    -2,    1,   0],
         [0,   2,      4,    -5/2,    -5,     1/2,   1,   0],
         [0,   -2,     4,     5/2,    -5,    -1/2,   1,   0],
         [0,   -1,     0,    21/4,     0,    -21/4,  0,   1]],
        dtype=np.float32).T

    m = A_data.shape[1]
    r = 3
    alpha = m + r - 1

    C = CI

    H = (IH + 2 * HPAD - 3) // HSTR + 1
    W = (IW + 2 * WPAD - 3) // WSTR + 1
    nH, nW = (H + m-1) // m, (W + m-1) // m

    VP = 4
    VK = 4

    def round_up(a, b): return ((a + b - 1) // b) * b
    K = round_up(CO, VK)
    P = round_up(N * nH * nW, VP)

    assert K % VK == 0
    assert P % VP == 0

    G = const_matrix(G_data, 'G')
    r_kh = tvm.reduce_axis((0, KH), 'r_kh')
    r_kw = tvm.reduce_axis((0, KW), 'r_kw')
    U = tvm.compute((K // VK, alpha, alpha, C, VK), lambda k, eps, nu, c, kk:
                    tvm.sum(kernel[k * VK + kk][c][r_kh][r_kw].astype(out_dtype) *
                            G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]), name='U')


    # pack input tile
    input_tile = tvm.compute((P // VP, C, alpha, alpha, VP),
                             lambda b, c, eps, nu, bb:
                             data_pad[(b*VP+bb) // (nH*nW)][c][(b*VP+bb) // nW % nH * m + eps]
                             [(b*VP+bb) % nW * m + nu],
                             name='d')

    # transform image
    B = const_matrix(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((P // VP, alpha, alpha, C, VP), lambda b, eps, nu, c, bb:
                    tvm.sum(input_tile[b][c][r_eps][r_nu][bb].astype(out_dtype) *
                            B[r_eps][eps] * B[r_nu][nu], axis=[r_eps, r_nu]), name='V')

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((K // VK, P // VP, alpha, alpha, VK, VP), lambda k, b, eps, nu, kk, bb:
                    tvm.sum(U[k][eps][nu][c][kk] *
                            V[b][eps][nu][c][bb], axis=c), name='M')

    # inverse transform
    A = const_matrix(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    Y = tvm.compute((K // VK, P // VP, m, m, VK, VP), lambda k, b, vh, vw, kk, bb:
                    tvm.sum(M[k][b][r_eps][r_nu][kk][bb] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]), name='Y')

    # unpack output
    def _output(n, k_, h, w):
        b_idx = n * nH * nW + (h//m) * nW + w//m
        b = b_idx // VP
        bb = b_idx % VP
        k = k_ // VK
        kk = k_ % VK
        return Y[k][b][h % m][w % m][kk][bb]

    output = tvm.compute((N, K, H, W), _output,
                         name='output', tag='winograd_conv_output')

    if cfg:
        cfg.add_flop(2 * N * K * H * W * KH * KW * C)
    return output

def schedule_winograd(cfg, output):
    s = tvm.create_schedule(output.op)
    return s
