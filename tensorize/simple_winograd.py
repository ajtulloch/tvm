from __future__ import division
from topi.util import get_const_int, get_const_tuple, const_matrix
import numpy as np
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn import pad
import tvm



def decl_winograd(cfg, data, kernel, strides, padding, layout, out_dtype):
    # return _baseline_winograd(cfg, data, kernel, strides, padding, layout, out_dtype)
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
         [1/45,   -1/90, 1/180],
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
    if K > CO:
        kernel_pad = pad(kernel, (0, 0, 0, 0), (K - CO, 0, 0, 0), name="kernel_pad")
    else:
        kernel_pad = kernel
    U = tvm.compute(
        (K // VK, alpha, alpha, C, VK), lambda k, eps, nu, c, kk:
        tvm.sum(kernel_pad[k * VK + kk][c][r_kh][r_kw].astype(out_dtype) *
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

    def compute_A_T_dot_M(k, b, eps, nu, kk, bb):
        temp_expr = {}

        for j in range(alpha):
            m1_add_m2 = M[k][b][1][j][kk][bb] + M[k][b][2][j][kk][bb]
            m1_sub_m2 = M[k][b][1][j][kk][bb] - M[k][b][2][j][kk][bb]
            m3_add_m4 = M[k][b][3][j][kk][bb] + M[k][b][4][j][kk][bb]
            m3_sub_m4 = M[k][b][3][j][kk][bb] - M[k][b][4][j][kk][bb]
            m5_add_m6 = M[k][b][5][j][kk][bb] + M[k][b][6][j][kk][bb]
            m5_sub_m6 = M[k][b][5][j][kk][bb] - M[k][b][6][j][kk][bb]
            s0 = M[k][b][0][j][kk][bb] + m1_add_m2
            s5 = M[k][b][7][j][kk][bb] + m1_sub_m2
            s1 = m1_sub_m2 + m5_sub_m6 * 16
            s4 = m1_add_m2 + m3_add_m4 * 16
            s2 = m1_add_m2 + 8 * m5_add_m6
            s3 = m1_sub_m2 + 8 * m3_sub_m4
            s0 = s0 + m5_add_m6 * 32
            s5 = s5 + m3_sub_m4 * 32
            s1 = s1 + m3_sub_m4 * 2
            s4 = s4 + m5_add_m6 * 2
            s0 = s0 + m3_add_m4
            s5 = s5 + m5_sub_m6
            s2 = s2 + m3_add_m4 * 4
            s3 = s3 + m5_sub_m6 * 4
            temp_expr[(0, j)] = s0
            temp_expr[(1, j)] = s1
            temp_expr[(2, j)] = s2
            temp_expr[(3, j)] = s3
            temp_expr[(4, j)] = s4
            temp_expr[(5, j)] = s5
        now = tvm.const(0.0, "float32")
        for ii in range(m):
            for jj in range(alpha):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    A_T_dot_M = tvm.compute((K // VK, P // VP, m, alpha, VK, VP), compute_A_T_dot_M, name="A_T_dot_M")

    def compute_X_dot_A(k, b, eps, nu, kk, bb):
        temp_expr = {}

        for i in range(m):
            m1_add_m2 = A_T_dot_M[k][b][i][1][kk][bb] + A_T_dot_M[k][b][i][2][kk][bb]
            m1_sub_m2 = A_T_dot_M[k][b][i][1][kk][bb] - A_T_dot_M[k][b][i][2][kk][bb]
            m3_add_m4 = A_T_dot_M[k][b][i][3][kk][bb] + A_T_dot_M[k][b][i][4][kk][bb]
            m3_sub_m4 = A_T_dot_M[k][b][i][3][kk][bb] - A_T_dot_M[k][b][i][4][kk][bb]
            m5_add_m6 = A_T_dot_M[k][b][i][5][kk][bb] + A_T_dot_M[k][b][i][6][kk][bb]
            m5_sub_m6 = A_T_dot_M[k][b][i][5][kk][bb] - A_T_dot_M[k][b][i][6][kk][bb]
            s0 = A_T_dot_M[k][b][i][0][kk][bb] + m1_add_m2
            s5 = A_T_dot_M[k][b][i][7][kk][bb] + m1_sub_m2
            s1 = m1_sub_m2 + m5_sub_m6 * 16
            s4 = m1_add_m2 + m3_add_m4 * 16
            s2 = m1_add_m2 + 8 * m5_add_m6
            s3 = m1_sub_m2 + 8 * m3_sub_m4
            s0 = s0 + m5_add_m6 * 32
            s5 = s5 + m3_sub_m4 * 32
            s1 = s1 + m3_sub_m4 * 2
            s4 = s4 + m5_add_m6 * 2
            s0 = s0 + m3_add_m4
            s5 = s5 + m5_sub_m6
            s2 = s2 + m3_add_m4 * 4
            s3 = s3 + m5_sub_m6 * 4
            temp_expr[(i, 0)] = s0
            temp_expr[(i, 1)] = s1
            temp_expr[(i, 2)] = s2
            temp_expr[(i, 3)] = s3
            temp_expr[(i, 4)] = s4
            temp_expr[(i, 5)] = s5
        now = tvm.const(0.0, "float32")
        for ii in range(m):
            for jj in range(m):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    Y = tvm.compute((K // VK, P // VP, m, m, VK, VP), compute_X_dot_A, name="Y")

    # unpack output
    def _output(n, k_, h, w):
        b_idx = n * nH * nW + (h//m) * nW + w//m
        b = b_idx // VP
        bb = b_idx % VP
        k = k_ // VK
        kk = k_ % VK
        return Y[k][b][h % m][w % m][kk][bb]

    output = tvm.compute((N, CO, H, W), _output,
                         name='output', tag='winograd_conv_output')

    if cfg:
        cfg.add_flop(2 * N * K * H * W * KH * KW * C)
    return output

def schedule_winograd(cfg, output):
    s = tvm.create_schedule(output.op)
    return s
