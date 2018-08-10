"""Example code to do convolution."""
from __future__ import division

from topi.util import get_const_int, const_matrix
import numpy as np
import tvm
import tvm.rpc
from tvm import autotvm
import simple_winograd

A_data = np.array([[1,  1,  1,   1,    1,    32,      32,    0],
                   [0,  1,  -1,  2,   -2,   16,   -16,   0],
                   [0,  1,  1,   4,    4,   8,    8,   0],
                   [0,  1,  -1,  8,   -8,   4,   -4,   0],
                   [0,  1,  1,   16,  16,   2,  2,   0],
                   [0,  1,  -1,  32,  -32,  1,  -1,  1]],
                  dtype=np.float32).T
m = A_data.shape[1]
r = 3
alpha = m + r - 1

HSTR = 1
WSTR = 1
HPAD = 1
WPAD = 1


# s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
# s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
# s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
# s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
# s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
# s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m

def A_T_dot_X(X):
    m1_add_m2 = X[1] + X[2]
    m1_sub_m2 = X[1] - X[2]
    m3_add_m4 = X[3] + X[4]
    m3_sub_m4 = X[3] - X[4]
    m5_add_m6 = X[5] + X[6]
    m5_sub_m6 = X[5] - X[6]
    s0 = X[0] + m1_add_m2
    s5 = X[7] + m1_sub_m2
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

    result = np.zeros((6, s0.size))
    result[0] = s0
    result[1] = s1
    result[2] = s2
    result[3] = s3
    result[4] = s4
    result[5] = s5
    return result

M = None

def decl_output_transform_minimal(cfg, X, M, VK, VP):

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

    N = get_const_int(X.shape[0])
    IH = get_const_int(X.shape[2])
    IW = get_const_int(X.shape[3])
    alpha = get_const_int(M.shape[0])

    K = get_const_int(M.shape[0]) * get_const_int(M.shape[4])
    P = get_const_int(M.shape[1]) * get_const_int(M.shape[5])

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
    OH = get_const_int((IH + 2 * HPAD - 3) // HSTR + 1)
    OW = get_const_int((IW + 2 * WPAD - 3) // WSTR + 1)
    nH, nW = get_const_int((OH + m-1) // m), get_const_int((OW + m-1) // m)

    # unpack output
    def _output(n, k, h, w):
        k_elem = k % VK
        k_tile = k // VK
        b = n * nH * nW + h // m * nW + w // m
        b_elem = b % VP
        b_tile = b // VP
        return Y[k_tile][b_tile][h % m][w % m][k_elem][b_elem]
    output = tvm.compute((N, K, OH, OW), _output,
                       name='output', tag='winograd_conv_output')
    return output


def decl_output_transform(cfg, X, M, VK, VP):
    N = get_const_int(X.shape[0])
    IH = get_const_int(X.shape[2])
    IW = get_const_int(X.shape[3])
    alpha = get_const_int(M.shape[0])

    K = get_const_int(M.shape[0]) * get_const_int(M.shape[4])
    P = get_const_int(M.shape[1]) * get_const_int(M.shape[5])

    # inverse transform
    A = const_matrix(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    Y = tvm.compute((K // VK, P // VP, m, m, VK, VP), lambda k, b, vh, vw, kk, bb:
                    tvm.sum(M[k][b][r_eps][r_nu][kk][bb] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]), name='Y')
    OH = get_const_int((IH + 2 * HPAD - 3) // HSTR + 1)
    OW = get_const_int((IW + 2 * WPAD - 3) // WSTR + 1)
    nH, nW = get_const_int((OH + m-1) // m), get_const_int((OW + m-1) // m)

    # unpack output
    def _output(n, k, h, w):
        k_elem = k % VK
        k_tile = k // VK
        b = n * nH * nW + h // m * nW + w // m
        b_elem = b % VP
        b_tile = b // VP
        return Y[k_tile][b_tile][h % m][w % m][k_elem][b_elem]
    output = tvm.compute((N, K, OH, OW), _output,
                       name='output', tag='winograd_conv_output')

    return output

def schedule_output_transform(cfg, output):
    s = tvm.create_schedule(output.op)
    Y = output.op.input_tensors[0]

    cfg.define_knob('reorder_kk', [0, 1])
    cfg.define_knob('vectorize_bb', [1])
    cfg.define_knob('unroll_vh', [1])
    cfg.define_knob('unroll_vw', [1])
    cfg.define_knob('unroll_r_eps', [1])
    cfg.define_knob('unroll_r_nu', [1])
    cfg.define_knob('compute_temp2_at_temp1', [1])
    cfg.define_knob('M_read_cache', [0, 1])
    if cfg['use_minimal'].val:
        temp_1 = output.op.input_tensors[0]
        temp_2 = temp_1.op.input_tensors[0]
        M = temp_2.op.input_tensors[0]
        for temp in [temp_1, temp_2]:
            k, b, eps, nu, kk, bb = s[temp].op.axis
            if cfg['reorder_kk'].val:
                s[temp].reorder(k, b, kk, eps, nu, bb)
            if cfg['vectorize_bb'].val:
                s[temp].vectorize(bb)
            if cfg['unroll_r_eps']:
                s[temp].unroll(eps)
            if cfg['unroll_r_nu']:
                s[temp].unroll(nu)
        (k, b, eps, nu, kk, bb) = s[temp_1].op.axis
        if cfg['compute_temp2_at_temp1'].val:
            s[temp_2].compute_at(s[temp_1], b)
        if cfg['M_read_cache'].val:
            MM = s.cache_read(M, 'global', [temp_2])
            (k, b, eps, nu, kk, bb) = s[temp_2].op.axis
            s[MM].compute_at(s[temp_2], b)
        pass
    else:
        M, A = Y.op.input_tensors
        s[A].compute_inline()

        k, b, vh, vw, kk, bb = s[Y].op.axis

        if cfg['reorder_kk'].val:
            s[Y].reorder(k, b, kk, vh, vw, bb)

        if cfg['vectorize_bb'].val:
            s[Y].vectorize(bb)

        r_eps, r_nu = s[Y].op.reduce_axis
        n, co, h, w = s[output].op.axis


        if cfg['unroll_vh'].val:
            s[Y].unroll(vh)

        if cfg['unroll_vw'].val:
            s[Y].unroll(vw)

        if cfg['unroll_r_eps'].val:
            s[Y].unroll(r_eps)
        if cfg['unroll_r_nu'].val:
            s[Y].unroll(r_nu)
    return s


@autotvm.template
def output_transform_autotvm(dtype):
    cfg = autotvm.get_config()
    cfg.define_knob('VK', [2, 4, 8, 16])
    cfg.define_knob('VP', [4, 8, 16])
    VK = cfg['VK'].val
    VP = cfg['VP'].val
    X = tvm.placeholder(shape=(1, 64, 56, 56), dtype="float32", name="X")
    W = tvm.placeholder(shape=(64, 64, 56, 56), dtype="float32", name="W")
    N = get_const_int(X.shape[0])
    IH = get_const_int(X.shape[2])
    IW = get_const_int(X.shape[3])
    OH = get_const_int((IH + 2 * HPAD - 3) // HSTR + 1)
    OW = get_const_int((IW + 2 * WPAD - 3) // WSTR + 1)
    nH, nW = get_const_int((OH + m-1) // m), get_const_int((OW + m-1) // m)

    def round_up(a, b):
        return ((a + b - 1) // b) * b
    P = round_up(N * nH * nW, VP)
    K = get_const_int(W.shape[0])
    assert K % VK == 0
    assert P % VP == 0

    cfg.define_knob('use_minimal', [1])
    M = tvm.placeholder(shape=(K // VK, P // VP, alpha, alpha, VK, VP), name="M")
    if cfg['use_minimal'].val:
        output = decl_output_transform_minimal(cfg, X, M, VK, VP)
    else:
        output = decl_output_transform(cfg, X, M, VK, VP)
    s = schedule_output_transform(cfg, output)
    # print(tvm.lower(s, [X, M, output], simple_mode=True))
    return s, [X, M, output]

@autotvm.template
def conv2d_winograd_autotvm(dtype):
    cfg = autotvm.get_config()
    cfg.define_knob('unroll', [0, 1])
    cfg.define_knob('compute_at', [0, 1])
    cfg.define_knob('vectorize', [0, 1])
    cfg.define_knob('VK', [2, 4, 8, 16])
    cfg.define_knob('VP', [4, 8, 16])
    VK = cfg['VK'].val
    VP = cfg['VP'].val
    N = 1
    S = 56
    CIn = 64
    COut = 64
    X = tvm.placeholder(shape=(N, CIn, S, S), dtype="float32", name="X")
    W = tvm.placeholder(shape=(64, 64, 3, 3), dtype="float32", name="W")

    output = simple_winograd.decl_winograd(cfg, X, W, strides=1, padding=1, layout="NCHW", out_dtype="float32", VK=VK, VP=VP)
    s = simple_winograd.schedule_winograd(cfg, output)
    cfg.add_flop(2 * N * CIn * COut * S * S * 3 * 3)
    return s, [X, W, output]



import logging
import sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

measure_option = autotvm.measure_option(
    measure_func='local',
    number=3)

task = autotvm.task.create(
    conv2d_winograd_autotvm,
    args=("float32",),
    target=tvm.target.create('llvm -mcpu=core-avx2'))
print(task.config_space)
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(
    n_trial=100,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file('conv2d_minimal_winograd.log')])
