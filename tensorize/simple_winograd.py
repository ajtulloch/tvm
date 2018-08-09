from __future__ import division
from topi.util import get_const_int, get_const_tuple, const_matrix
import numpy as np
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn import pad
import tvm


def _baseline_winograd(cfg, data, kernel, strides, padding, layout, out_dtype):
    tile_size = 4

    N, CI, IH, IW = get_const_tuple(data.shape)
    if len(kernel.shape) == 4:
        pre_computed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:
        pre_computed = True
        H_CAT, W_CAT, CO, CI, VC = get_const_tuple(kernel.shape)
        CO *= VC
        KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    assert layout == 'NCHW'
    assert KH == 3 and KW == 3 and HPAD == 1 and WPAD == 1 and HSTR == 1 and WSTR == 1
    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")

    if tile_size == 4:
        G_data = np.array([
            [1 / 4.0, 0, 0],
            [-1 / 6.0, -1 / 6.0, -1 / 6.0],
            [-1 / 6.0, 1 / 6.0, -1 / 6.0],
            [1 / 24.0, 1 / 12.0, 1 / 6.0],
            [1 / 24.0, -1 / 12.0, 1 / 6.0],
            [0, 0, 1]], dtype=np.float32)

        B_data = np.array([
            [4, 0, 0, 0, 0, 0],
            [0, -4, 4, -2, 2, 4],
            [-5, -4, -4, -1, -1, 0],
            [0, 1, -1, 2, -2, -5],
            [1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]], out_dtype)

        A_data = np.array([
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 2, 4, 8],
            [1, -2, 4, -8],
            [0, 0, 0, 1]], out_dtype)
    elif tile_size == 2:
        G_data = np.array([
            [1, 0, 0],
            [1.0/2, 1.0/2, 1.0/2],
            [1.0/2, -1.0/2, 1.0/2],
            [0, 0, 1]], np.float32)

        B_data = np.array([
            [1, 0, 0, 0],
            [0, 1, -1, 1],
            [-1, 1, 1, 0],
            [0, 0, 0, -1]], out_dtype)

        A_data = np.array([
            [1, 0],
            [1, 1],
            [1, -1],
            [0, -1]], out_dtype)
    else:
        raise ValueError("Unsupported tile size for winograd: " + str(tile_size))

    m = A_data.shape[1]
    r = 3
    alpha = m + r - 1
    K = CO
    C = CI

    H = (IH + 2 * HPAD - 3) // HSTR + 1
    W = (IW + 2 * WPAD - 3) // WSTR + 1
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    #cfg.define_split('tile_p', cfg.axis(P), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    #cfg.define_split('tile_k', cfg.axis(K), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    VP = 1#cfg['tile_p'].size[-1]
    VK = 1#cfg['tile_k'].size[-1]

    # pack input tile
    input_tile = tvm.compute((C, P // VP, alpha, alpha, VP),
                             lambda c, b, eps, nu, bb:
                             data_pad[(b*VP+bb) // (nH*nW)][c][(b*VP+bb) // nW % nH * m + eps]
                             [(b*VP+bb) % nW * m + nu],
                             name='d')

    # transform kernel
    if pre_computed:
        U = kernel
    else:
        G = const_matrix(G_data, 'G')
        r_kh = tvm.reduce_axis((0, KH), 'r_kh')
        r_kw = tvm.reduce_axis((0, KW), 'r_kw')
        U = tvm.compute((alpha, alpha, K // VK, C, VK), lambda eps, nu, k, c, kk:
                        tvm.sum(kernel[k * VK + kk][c][r_kh][r_kw].astype(out_dtype) *
                                G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]), name='U')

    # transform image
    B = const_matrix(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((alpha, alpha, P // VP, C, VP), lambda eps, nu, b, c, bb:
                    tvm.sum(input_tile[c][b][r_eps][r_nu][bb].astype(out_dtype) *
                            B[r_eps][eps] * B[r_nu][nu], axis=[r_eps, r_nu]), name='V')

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((alpha, alpha, K, P), lambda eps, nu, k, b:
                    tvm.sum(U[eps][nu][k // VK][c][k % VK] *
                            V[eps][nu][b // VP][c][b % VP], axis=c), name='M')

    # inverse transform
    A = const_matrix(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    Y = tvm.compute((K, P, m, m), lambda k, b, vh, vw:
                    tvm.sum(M[r_eps][r_nu][k][b] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]), name='Y')

    # unpack output
    output = tvm.compute((N, K, H, W), lambda n, k, h, w:
                         Y[k][n * nH * nW + (h//m) * nW + w//m][h % m][w % m],
                         name='output', tag='winograd_conv_output')
      

    # we have to manually assign effective GFLOP for winogard
    # cfg.add_flop(2 * N * K * H * W * KH * KW * C)
    return output

def schedule_winograd(cfg, output):
    s = tvm.create_schedule(output.op)
    return s

def decl_winograd(cfg, data, kernel, strides, padding, layout, out_dtype):
    return _baseline_winograd(cfg, data, kernel, strides, padding, layout, out_dtype)

    N, CI, IH, IW = get_const_tuple(data.shape)
    CO, _, KH, KW = get_const_tuple(kernel.shape)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    assert layout == 'NCHW'
    assert KH == 3 and KW == 3 and HPAD == 1 and WPAD == 1 and HSTR == 1 and WSTR == 1
    # data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    A_data = np.array([[1,  1,  1,   1,    1,    1,      1,    0],
                  [0,  1,  -1,  2,   -2,   1/2,   -1/2,   0],
                  [0,  1,  1,   4,    4,   1/4,    1/4,   0],
                  [0,  1,  -1,  8,   -8,   1/8,   -1/8,   0],
                  [0,  1,  1,   16,  16,   1/16,  1/16,   0],
                  [0,  1,  -1,  32,  -32,  1/32,  -1/32,  1]],
                 dtype=np.float32).T
    G_data = np.array([[1,      0,     0],
                  [-2/9,  -2/9,   -2/9],
                  [-2/9,   2/9,   -2/9],
                  [1/90,  1/45,   2/45],
                  [1/90,  -1/45,  2/45],
                  [32/45,    16/46, 8/45],
                  [32/45,   -16/45, 8/45],
                  [0,      0,     1]],
                 dtype=np.float32)
    B_data = np.array([[1,   0,    -21/4,    0,    21/4,     0,    -1,  0],
                  [0,   1,      1,    -17/4,  -17/4,    1,    1,   0],
                  [0,   -1,     1,    17/4,   -17/4,   -1,    1,   0],
                  [0,  1/2,    1/4,   -5/2,   -5/4,     2,    1,   0],
                  [0,  -1/2,   1/4,    5/2,   -5/4,    -2,    1,   0],
                  [0,   2,      4,    -5/2,    -5,     1/2,   1,   0],
                  [0,   -2,     4,     5/2,    -5,    -1/2,   1,   0],
                  [0,   -1,     0,    21/4,     0,    -21/4,  0,   1]],
                 dtype=np.float32).T

    m = 6
    r = 3
    alpha = m + r - 1
    assert alpha == 8
    assert B_data.shape == (alpha, alpha)
    assert G_data.shape == (alpha, 3)
    C = CI

    H = (IH + 2 * HPAD - 3) // HSTR + 1
    W = (IW + 2 * WPAD - 3) // WSTR + 1
    nH, nW = (H + m-1) // m, (W + m-1) // m

    def round_up(a, b): return ((a + b - 1) // b) * b

    VP = 1 #cfg['tile_p'].size[-1]
    VK = 1 #cfg['tile_k'].size[-1]

    P = round_up(N * nH * nW, VP)
    K = get_const_int(round_up(CO, VK))

    # pack input tile
    assert N == 1
    def _input_tile(b, c, eps, nu, bb):
        b_idx = b * VP + bb
        wp = b_idx % nW
        hp = (b_idx // nW) % nH
        n = 0 #(b_idx // nW // nH) % N
        w = wp * m + eps - padding
        h = hp * m + nu - padding
        return tvm.select(
            tvm.all(w >= 0, w < IW, h >= 0, h < IH),
            data[n][c][h][w],
            0.0
        )

    input_tile = tvm.compute(
        (P // VP, C, alpha, alpha, VP),
        _input_tile,
        name='input_tile')

    # transform kernel
    G = const_matrix(G_data, 'G')
    r_kh = tvm.reduce_axis((0, KH), 'r_kh')
    r_kw = tvm.reduce_axis((0, KW), 'r_kw')

    kernel_pad = pad(
        kernel, pad_before=(0, 0, 0, 0), pad_after=(K-CO, 0, 0, 0), name="kernel_pad")
    U = tvm.compute(
        (K // VK, alpha, alpha, C, VK),
        lambda k, eps, nu, c, kk: tvm.sum(
            kernel_pad[k * VK + kk][c][r_kh][r_kw].astype(out_dtype) * G[eps][r_kh] * G[nu][r_kw],
            axis=[r_kh, r_kw]),
        name='U')

    # transform image
    B = const_matrix(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute(
        (P // VP, alpha, alpha, C, VP),
        lambda b, eps, nu, c, bb: tvm.sum(
            input_tile[b][c][r_eps][r_nu][bb].astype(out_dtype) *
            B[r_eps][eps] * B[r_nu][nu], axis=[r_eps, r_nu]),
        name='V')

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute(
        (K // VK, P // VP, alpha, alpha, VK, VP),
        lambda k, b, eps, nu, kk, bb: tvm.sum(U[k][eps][nu][c][kk] * V[b][eps][nu][c][bb], axis=c),
        name='M')

    # inverse transform
    A = const_matrix(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    Y = tvm.compute((K // VK, P // VP, m, m, VK, VP), lambda k, b, vh, vw, kk, bb:
                    tvm.sum(
                        M[k][b][r_eps][r_nu][kk][bb] * A[r_eps][vh] * A[r_nu][vw],
                        axis=[r_eps, r_nu]), name='Y')

    # unpack output
    def _output(n, k, h, w):
        k_elem = k % VK
        k_tile = k // VK
        b = n * nH * nW + h // m * nW + w // m
        b_elem = b % VP
        b_tile = b // VP
        return Y[k_tile][b_tile][h % m][w % m][k_elem][b_elem]
    output = tvm.compute(
        (N, CO, H, W), _output,
        name='output', tag='winograd_conv_output')

    # we have to manually assign effective GFLOP for winogard
    if cfg:
        cfg.add_flop(2 * N * K * H * W * KH * KW * C)
    return output

