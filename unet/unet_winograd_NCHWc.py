from __future__ import division

from topi.nn.util import get_pad_tuple
from tvm import autotvm

import numpy as np
import topi
import topi.nn.util
import tvm
import unet_intrinsics

def decl_winograd_NCHWc(cfg, data, kernel, strides, padding, VK, VP, out_dtype):
    # Assumption: NCHW8c input, NCHW8c output.

    # return _baseline_winograd(cfg, data, kernel, strides, padding, layout, out_dtype)
    N, CII, IH, IW, CIII = topi.util.get_const_tuple(data.shape)
    assert CIII == 8
    COO, CII_, KH, KW, COOO, CIII_ = topi.util.get_const_tuple(kernel.shape)
    assert CII_ == CII
    assert COOO == 8
    assert CIII_ == 8

    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    assert KH == 3 and KW == 3 and HPAD == 1 and WPAD == 1 and HSTR == 1 and WSTR == 1
    data_pad = topi.nn.pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")

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

    m = 6
    r = 3
    alpha = m + r - 1


    H = (IH + 2 * HPAD - 3) // HSTR + 1
    W = (IW + 2 * WPAD - 3) // WSTR + 1
    nH, nW = (H + m-1) // m, (W + m-1) // m

    def round_up(a, b): return ((a + b - 1) // b) * b

    # VK = COOO
    CO = COO * COOO
    C = CII * CIII
    K = round_up(CO, VK)
    P = round_up(N * nH * nW, VP)
    assert K % VK == 0
    assert P % VP == 0

    G = topi.util.const_matrix(G_data, 'G')
    r_kh = tvm.reduce_axis((0, KH), 'r_kh')
    r_kw = tvm.reduce_axis((0, KW), 'r_kw')
    assert K >= CO
    if K > CO:
        kernel_pad = topi.nn.pad(kernel, (0, 0, 0, 0, 0, 0), (K - CO, 0, 0, 0, 0, 0), name="kernel_pad")
    else:
        kernel_pad = kernel

    U = tvm.compute(
        (K // VK, alpha, alpha, C, VK), lambda k, eps, nu, c, kk:
        tvm.sum(kernel_pad[(k * VK + kk) // COOO][c // CIII][r_kh][r_kw][c % CIII][(k * VK + kk) % COOO].astype(out_dtype) *
                G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]), name='U')


    # pack input tile
    (d0, d1, d2, d3, d4) = topi.util.get_const_tuple(data_pad.shape)
    input_tile = tvm.compute((P // VP, C, alpha, alpha, VP),
                             lambda b, c, eps, nu, bb:
                             data_pad[(b*VP+bb) // (nH*nW)][c // CIII][(b*VP+bb) // nW % nH * m + eps]
                             [(b*VP+bb) % nW * m + nu][c % CIII] + tvm.const(0, data_pad.dtype) * data_pad[d0 - 1, d1 - 1, d2 - 1, d3 - 1, d4 - 1],
                             name='input_tile')

    def compute_B_T_dot_X(b, c, eps, nu, bb):
        temp_expr = {}
        for j in range(alpha):
            wd0 = input_tile[b][c][0][j][bb] - input_tile[b][c][6][j][bb]
            d4_sub_d2 = input_tile[b][c][4][j][bb] - input_tile[b][c][2][j][bb]
            wd7 = input_tile[b][c][7][j][bb] - input_tile[b][c][1][j][bb]
            d3_sub_d5 = input_tile[b][c][3][j][bb] - input_tile[b][c][5][j][bb]
            wd1 = input_tile[b][c][2][j][bb] + input_tile[b][c][6][j][bb]
            wd2 = input_tile[b][c][1][j][bb] + input_tile[b][c][5][j][bb]
            wd4 = input_tile[b][c][5][j][bb] + input_tile[b][c][1][j][bb] * 0.25
            wd5 = input_tile[b][c][6][j][bb] - input_tile[b][c][4][j][bb] * 5
            wd3 = input_tile[b][c][6][j][bb] + input_tile[b][c][2][j][bb] * 0.25
            wd6 = input_tile[b][c][1][j][bb] + input_tile[b][c][5][j][bb] * 0.25

            wd0 = wd0 + d4_sub_d2 * 5.25
            wd7 = wd7 + d3_sub_d5 * 5.25

            wd1 = wd1 - input_tile[b][c][4][j][bb] * 4.25
            wd2 = wd2 - input_tile[b][c][3][j][bb] * 4.25

            wd3 = wd3 - input_tile[b][c][4][j][bb] * 1.25
            wd5 = wd5 + input_tile[b][c][2][j][bb] * 4
            wd4 = wd4 - input_tile[b][c][3][j][bb] * 1.25
            wd6 = wd6 - input_tile[b][c][3][j][bb] * 1.25

            temp_expr[(0, j)] = wd0
            temp_expr[(1, j)] = wd1 + wd2
            temp_expr[(2, j)] = wd1 - wd2
            temp_expr[(3, j)] = wd3 + wd4 * 2
            temp_expr[(4, j)] = wd3 - wd4 * 2
            temp_expr[(5, j)] = wd5 + wd6 * 2
            temp_expr[(6, j)] = wd5 - wd6 * 2
            temp_expr[(7, j)] = wd7

        now = tvm.const(0.0, "float32")
        for ii in range(alpha):
            for jj in range(alpha):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    B_T_dot_X = tvm.compute((P // VP, C, alpha, alpha, VP), compute_B_T_dot_X, name="B_T_dot_X")

    def compute_X_dot_B(b, eps, nu, c, bb):
        temp_expr = {}

        for i in range(alpha):
            wd0 = B_T_dot_X[b][c][i][0][bb] - B_T_dot_X[b][c][i][6][bb]
            d4_sub_d2 = B_T_dot_X[b][c][i][4][bb] - B_T_dot_X[b][c][i][2][bb]
            wd7 = B_T_dot_X[b][c][i][7][bb] - B_T_dot_X[b][c][i][1][bb]
            d3_sub_d5 = B_T_dot_X[b][c][i][3][bb] - B_T_dot_X[b][c][i][5][bb]
            wd1 = B_T_dot_X[b][c][i][2][bb] + B_T_dot_X[b][c][i][6][bb]
            wd2 = B_T_dot_X[b][c][i][1][bb] + B_T_dot_X[b][c][i][5][bb]
            wd4 = B_T_dot_X[b][c][i][5][bb] + B_T_dot_X[b][c][i][1][bb] * 0.25
            wd5 = B_T_dot_X[b][c][i][6][bb] - B_T_dot_X[b][c][i][4][bb] * 5
            wd3 = B_T_dot_X[b][c][i][6][bb] + B_T_dot_X[b][c][i][2][bb] * 0.25
            wd6 = B_T_dot_X[b][c][i][1][bb] + B_T_dot_X[b][c][i][5][bb] * 0.25

            wd0 = wd0 + d4_sub_d2 * 5.25
            wd7 = wd7 + d3_sub_d5 * 5.25

            wd1 = wd1 - B_T_dot_X[b][c][i][4][bb] * 4.25
            wd2 = wd2 - B_T_dot_X[b][c][i][3][bb] * 4.25

            wd3 = wd3 - B_T_dot_X[b][c][i][4][bb] * 1.25
            wd5 = wd5 + B_T_dot_X[b][c][i][2][bb] * 4
            wd4 = wd4 - B_T_dot_X[b][c][i][3][bb] * 1.25
            wd6 = wd6 - B_T_dot_X[b][c][i][3][bb] * 1.25

            temp_expr[(i, 0)] = wd0
            temp_expr[(i, 1)] = wd1 + wd2
            temp_expr[(i, 2)] = wd1 - wd2
            temp_expr[(i, 3)] = wd3 + wd4 * 2
            temp_expr[(i, 4)] = wd3 - wd4 * 2
            temp_expr[(i, 5)] = wd5 + wd6 * 2
            temp_expr[(i, 6)] = wd5 - wd6 * 2
            temp_expr[(i, 7)] = wd7

        now = tvm.const(0.0, "float32")
        for ii in range(alpha):
            for jj in range(alpha):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now
    V = tvm.compute((P // VP, alpha, alpha, C, VP), compute_X_dot_B, name="V")

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute(
        (K // VK, P // VP, alpha, alpha, VK, VP),
        lambda k, b, eps, nu, kk, bb: tvm.sum(U[k][eps][nu][c][kk] * V[b][eps][nu][c][bb], axis=c),
        name='M')

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

        (d0, d1, d2, d3, d4, d5) = topi.util.get_const_tuple(M.shape)

        now = tvm.const(0.0, "float32") #* M[d0 - 1, d1 - 1, d2 - 1, d3 - 1, d4 - 1, d5 - 1]
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
    def _output(n, coo, h, w, cooo):
        b_idx = n * nH * nW + (h//m) * nW + w//m
        b = b_idx // VP
        bb = b_idx % VP
        k_ = coo * COOO + cooo
        k = k_ // VK
        kk = k_ % VK
        return Y[k][b][h % m][w % m][kk][bb]

    output = tvm.compute((N, COO, H, W, COOO), _output,
                         name='output', tag='winograd_conv_output')

    if cfg:
        cfg.add_flop(2 * N * K * H * W * KH * KW * C)

    return output

def schedule_winograd_NCHWc(cfg, output, VK, VP):
    s = tvm.create_schedule(output.op)
    if not cfg:
        return s
    Y = output.op.input_tensors[0]
    A_T_dot_M = Y.op.input_tensors[0]
    M = A_T_dot_M.op.input_tensors[0]
    U, V = M.op.input_tensors
    B_T_dot_X = V.op.input_tensors[0]
    input_tile = B_T_dot_X.op.input_tensors[0]
    data_pad = input_tile.op.input_tensors[0]
    # padding

    UNROLL = cfg['unroll'].val
    VECTORIZE = cfg['vectorize'].val
    TENSORIZE = cfg['tensorize'].val
    if cfg['data_pad_VECTORIZE'].val:
        s[data_pad].vectorize(list(s[data_pad].op.axis)[-1])

    if cfg['data_pad_inline'].val:
        s[data_pad].compute_inline()

    (b, c, eps, nu, bb) = s[input_tile].op.axis
    if cfg['input_tile_VECTORIZE'].val:
        s[input_tile].vectorize(bb)

    if cfg['input_tile_REORDER_C'].val:
        s[input_tile].reorder(b, eps, nu, c, bb)

    # transform kernel
    if isinstance(U.op, tvm.tensor.ComputeOp):
        kernel, G = U.op.input_tensors
        if isinstance(kernel.op, tvm.tensor.ComputeOp):
            s[kernel].compute_inline()

        s[G].compute_inline()
        k, eps, nu, c, kk, = s[U].op.axis
        # r_kh, r_kw = s[U].op.reduce_axis
        # s[U].reorder(k, c, eps, nu, r_kh, r_kw, kk)
        # s[U].unroll(eps)
        # s[U].unroll(nu)
        # s[U].unroll(r_kh)
        # s[U].unroll(r_kw)
        # s[U].vectorize(kk)
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(s[U].op.axis[0], 'debug_skip_region')
            s[G].pragma(s[G].op.axis[0], 'debug_skip_region')

    # if autotvm.GLOBAL_SCOPE.in_tuning:
    #     s[output].pragma(s[output].op.axis[0], 'debug_skip_region')

    (k, b, eps, nu, kk, bb) = A_T_dot_M.op.axis
    s[A_T_dot_M].reorder(b, k, eps, nu, kk, bb)
    if UNROLL:
        [s[A_T_dot_M].unroll(ax) for ax in [eps, nu, kk]]
    if VECTORIZE:
        s[A_T_dot_M].vectorize(bb)

    if cfg['M_COMPUTE_AT'].val:
        s[M].compute_at(s[A_T_dot_M], b)

    (k, b, eps, nu, kk, bb) = Y.op.axis
    s[Y].reorder(b, k, eps, nu, kk, bb)
    if UNROLL:
        [s[Y].unroll(ax) for ax in [eps, nu, kk]]

    if VECTORIZE:
        s[Y].vectorize(bb)

    if cfg['A_T_dot_M_COMPUTE_AT'].val:
        s[A_T_dot_M].compute_at(s[Y], b)

    # Schedule V
    (b, c, eps, nu, bb) = B_T_dot_X.op.axis
    if UNROLL:
        [s[B_T_dot_X].unroll(ax) for ax in [eps, nu]]
    if VECTORIZE:
        s[B_T_dot_X].vectorize(bb)

    if cfg['B_T_dot_X_REORDER_C'].val:
        s[B_T_dot_X].reorder(b, eps, nu, c, bb)

    if cfg['input_tile_COMPUTE_AT'].val:
        s[input_tile].compute_at(s[B_T_dot_X], b)

    (b, eps, nu, c, bb) = V.op.axis
    if UNROLL:
        [s[V].unroll(ax) for ax in [eps, nu]]
    if VECTORIZE:
        s[V].vectorize(bb)
    if cfg['V_REORDER_C'].val:
        s[V].reorder(b, eps, nu, c, bb)

    if cfg['B_T_dot_X_COMPUTE_AT'].val:
        s[B_T_dot_X].compute_at(s[V], b)

    (k, b, eps, nu, kk, bb) = M.op.axis
    if cfg['V_COMPUTE_AT'].val:
        s[V].compute_at(s[M], b)
    s[M].reorder(b, k, eps, nu, kk, bb)
    K = topi.util.get_const_int(M.op.reduce_axis[0].dom.extent)
    (tensorize_gemm, tensorize_code_path) = unet_intrinsics.gemm(M=VK, N=VP, K=K)

    s[M].tensorize(kk, tensorize_gemm)
    s[M].pragma(s[M].op.axis[0], "import_llvm", tensorize_code_path)

    (n, co, h, w, coo) = s[output].op.axis
    if cfg['output_VECTORIZE'].val:
        s[output].vectorize(coo)
    if cfg['Y_COMPUTE_AT'].val:
        s[Y].compute_at(s[output], co)
    return s
