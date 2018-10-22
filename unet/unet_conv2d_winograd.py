# pylint: disable=invalid-name,unused-variable,no-else-return
from __future__ import absolute_import as _abs
from __future__ import division
import numpy as np

import tvm
from tvm import autotvm

from topi.generic import schedule_conv2d_nchw, schedule_conv2d_NCHWc_
from topi.util import traverse_inline, get_const_tuple, const_matrix
from topi.nn import pad, conv2d, conv2d_NCHWc, conv2d_alter_layout
from topi.nn.util import get_const_int, get_pad_tuple
import topi.nn

import os
UNROLL = int(os.environ.get('UNROLL', '1'))

def get_transform_matrices(m):
    assert m in (2, 4, 6)
    if m == 4:
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
            [0, 0, 0, 0, 0, 1]], dtype=np.float32)

        A_data = np.array([
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 2, 4, 8],
            [1, -2, 4, -8],
            [0, 0, 0, 1]], dtype=np.float32)

    elif m == 6:
        G_data = np.array([
            [1,      0,     0],
            [-2/9,  -2/9,   -2/9],
            [-2/9,   2/9,   -2/9],
            [1/90,  1/45,   2/45],
            [1/90,  -1/45,  2/45],
            [1/45,    1/90, 1/180],
            [1/45,   -1/90, 1/180],
            [0,      0,     1]
        ], dtype=np.float32)

        B_data = np.array([
            [1,   0,    -21/4,    0,    21/4,     0,    -1,  0],
            [0,   1,      1,    -17/4,  -17/4,    1,    1,   0],
            [0,   -1,     1,    17/4,   -17/4,   -1,    1,   0],
            [0,  1/2,    1/4,   -5/2,   -5/4,     2,    1,   0],
            [0,  -1/2,   1/4,    5/2,   -5/4,    -2,    1,   0],
            [0,   2,      4,    -5/2,    -5,     1/2,   1,   0],
            [0,   -2,     4,     5/2,    -5,    -1/2,   1,   0],
            [0,   -1,     0,    21/4,     0,    -21/4,  0,   1]
        ], dtype=np.float32).T

        A_data = np.array([
            [1,  1,  1,   1,    1,    32,      32,    0],
            [0,  1,  -1,  2,   -2,  16,   -16,   0],
            [0,  1,  1,   4,    4,   8,    8,   0],
            [0,  1,  -1,  8,   -8,   4,   -4,   0],
            [0,  1,  1,   16,  16,   2,  2,   0],
            [0,  1,  -1,  32,  -32,  1,  -1,  1]
        ], dtype=np.float32).T
    elif m == 2:
        G_data = np.array([
            [1, 0, 0],
            [1.0/2, 1.0/2, 1.0/2],
            [1.0/2, -1.0/2, 1.0/2],
            [0, 0, 1]], np.float32)

        B_data = np.array([
            [1, 0, 0, 0],
            [0, 1, -1, 1],
            [-1, 1, 1, 0],
            [0, 0, 0, -1]], np.float32)

        A_data = np.array([
            [1, 0],
            [1, 1],
            [1, -1],
            [0, -1]], np.float32)

    return const_matrix(A_data, "A"), const_matrix(B_data, "B"), const_matrix(G_data, "G")


def _decl_winograd_NCHWc(cfg, data, kernel, num_filter, kernel_size, stride, padding, layout, out_layout, out_dtype, m):
    # create workload according to raw arguments
    # wkl = _conv_NCHWc_arg_to_workload(
    #     data, kernel, num_filter, kernel_size,
    #     stride, padding, layout, out_layout, out_dtype)

    wkl = [] #None

    out_dtype = out_dtype or data.dtype
    N, CII, IH, IW, CIII = get_const_tuple(data.shape)
    COO, CII, KH, KW, CIII_, VC = get_const_tuple(kernel.shape)

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (KH, KW))
    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)

    OH = (IH + pad_top + pad_bottom - KH) // HSTR + 1
    OW = (IW + pad_left + pad_right - KW) // WSTR + 1
    data_pad = pad(data, [0, 0, pad_top, pad_left, 0], [0, 0, pad_bottom, pad_right, 0], name="data_pad")

    A, B, G = get_transform_matrices(m)
    r = 3
    alpha = m + r - 1

    def div_round_up(a, b):
        return (a + b - 1) // b

    assert all(k == 3 for k in (KH, KW))
    assert all(p == 1 for p in (pad_top, pad_left, pad_bottom, pad_right))
    assert all(s == 1 for s in (HSTR, WSTR))
    assert OH == IH
    assert OW == IW

    OH_M = div_round_up(OH, m)
    OW_M = div_round_up(OW, m)
    # Layouts:

    # input_tile_shape = (N, CII, OH // m, OH // m, alpha, alpha, CIII)
    # U_shape = (COO, CII, CIII, alpha, alpha, COOO)
    # V_shape = (N, CII, OH // m, OW // m, alpha, alpha, CIII)
    # M_shape = (N, COO, OH // m, OW // m, alpha, alpha, COOO)
    # Y_shape = (N, COO, OH // m, OW // m, m, m, COOO)
    # O_shape = (N, COO, OH, OW, COOO)

    n, coo, oh, ow, oh_m, ow_m, vc = \
        cfg.axis(N), cfg.axis(COO), cfg.axis(OH), cfg.axis(OW), \
        cfg.axis(OH_M), cfg.axis(OW_M), cfg.axis(VC)
    cii, ciii, kh, kw = cfg.reduce_axis(CII), cfg.reduce_axis(CIII), \
                        cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    eps, nu = cfg.axis(alpha), cfg.axis(alpha)
    vh, vw = cfg.axis(m), cfg.axis(m)
    r_eps, r_nu = cfg.axis(alpha), cfg.axis(alpha)
    cfg.define_reorder("reorder_M",
                       [n, coo, oh_m, ow_m, eps, nu, vc, cii, ciii],
                       policy='candidate', candidate=[
                           [n, coo, cii, oh_m, ow_m, eps, ciii, nu, vc],
                           # [n, coo, cii, oh_m, ow_m, ciii, nu, eps, vc],
                           # [n, coo, cii, oh_m, ow_m, nu, eps, ciii, vc],
                           # [n, coo, oh_m, ow_m, nu, eps, cii, ciii, vc],
                       ])

    cfg.define_reorder("reorder_V",
                       [n, cii, oh_m, ow_m, eps, nu, ciii, r_eps, r_nu],
                       policy='candidate', candidate=[
                           [n, cii, oh_m, ow_m, eps, r_eps, r_nu, nu, ciii],
                           # [n, cii, oh_m, ow_m, eps, nu, r_eps, r_nu, ciii],
                           # [n, cii, oh_m, ow_m, r_eps, r_nu, eps, nu, ciii],
                           # [n, cii, oh_m, ow_m, r_eps, r_nu, eps, nu, ciii],
                       ])

    cfg.define_reorder("reorder_Y",
                       [n, coo, oh_m, ow_m, vh, vw, vc, r_eps, r_nu],
                       policy='candidate', candidate=[
                           [n, coo, oh_m, ow_m, vh, r_eps, r_nu, vw, vc],
                           # [n, coo, oh_m, ow_m, vh, vw, r_eps, r_nu, vc],
                           # [n, coo, oh_m, ow_m, r_eps, r_nu, vh, vw, vc],
                           # [n, coo, oh_m, ow_m, r_eps, r_nu, vh, vw, vc],
                       ])


    input_tile = tvm.compute((N, CII, OH_M, OW_M, alpha, alpha, CIII),
                             lambda n, cii, oh_m, ow_m, eps, nu, ciii:
                             data_pad[n][cii][oh_m * m + eps][ow_m * m + nu][ciii],
                             name='input_tile')


    # transform kernel
    r_kh = tvm.reduce_axis((0, KH), 'r_kh')
    r_kw = tvm.reduce_axis((0, KW), 'r_kw')
    U = tvm.compute((COO, CII, CIII, alpha, alpha, VC),
                    lambda coo, cii, ciii, eps, nu, vc:
                    tvm.sum(kernel[coo][cii][r_kh][r_kw][ciii][vc].astype(out_dtype) *
                            G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]),
                    name='U')

    # transform image
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((N, CII, OH_M, OW_M, alpha, alpha, CIII),
                    lambda n, cii, oh_m, ow_m, eps, nu, ciii:
                    tvm.sum(input_tile[n][cii][oh_m][ow_m][r_eps][r_nu][ciii].astype(out_dtype) *
                            B[r_eps][eps] * B[r_nu][nu], axis=[r_eps, r_nu]), name='V')
    cii = tvm.reduce_axis((0, CII), name='cii')
    ciii = tvm.reduce_axis((0, CIII), name='ciii')

    # M_shape = (N, COO, OH // m, OW // m, alpha, alpha, COOO)
    M = tvm.compute((N, COO, OH_M, OW_M, alpha, alpha, VC),
                    lambda n, coo, oh_m, ow_m, eps, nu, vc:
                    tvm.sum(U[coo][cii][ciii][eps][nu][vc] * V[n][cii][oh_m][ow_m][eps][nu][ciii],
                            axis=[cii, ciii]),
                    name='M')

    # inverse transform
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    # Y_shape = (N, COO, OH // m, OW // m, m, m, COOO)
    Y = tvm.compute((N, COO, OH_M, OW_M, m, m, VC),
                    lambda n, coo, oh_m, ow_m, vh, vw, vc:
                    tvm.sum(M[n][coo][oh_m][ow_m][r_eps][r_nu][vc] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]),
                    name='Y')

    output = tvm.compute((N, COO, OH, OW, VC),
                         lambda n, coo, oh, ow, vc:
                         Y[n][coo][oh // m][ow // m][oh % m][ow % m][vc],
                         name='output', tag='winograd_conv2d_output',
                         attrs={'workload': wkl})
    cfg.add_flop(2 * N * COO * VC * OH * OW * KH * KW * CII * CIII)
    return output

def _schedule_winograd_NCHWc(cfg, s, output, last):
    Y = output.op.input_tensors[0]
    M, A = Y.op.input_tensors
    U, V = M.op.input_tensors
    input_tile, B = V.op.input_tensors
    data_pad = input_tile.op.input_tensors[0]

    # Inline the constants.
    s[A].compute_inline()
    s[B].compute_inline()

    # transform kernel
    if isinstance(U.op, tvm.tensor.ComputeOp):
        kernel, G = U.op.input_tensors
        s[G].compute_inline()
        coo, cii, eps, nu, ciii, vc = s[U].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(eps, 'debug_skip_region')
        else:
            pass
            # r_kh, r_kw = s[U].op.reduce_axis
            # s[U].reorder(k, c, eps, nu, r_kh, r_kw, kk)
            # for axis in [eps, nu, r_kh, r_kw]:
            #     s[U].unroll(axis)
            # s[U].vectorize(kk)
            # s[U].parallel(k)

        if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

    ############################################################
    # input tile
    n, cii, oh_m, ow_m, eps, nu, ciii = s[input_tile].op.axis
    # Vectorize the input tile
    s[input_tile].vectorize(ciii)

    # cfg.define_knob('data_pad_compute_location', [0, 1, 2, 3])
    cfg.define_knob('data_pad_compute_location', [3])
    if cfg['data_pad_compute_location'].val == 0:
        s[data_pad].compute_inline()
    if cfg['data_pad_compute_location'].val == 1:
        s[data_pad].compute_at(s[input_tile], cii)
        (_, _, _, _, dpcii) = s[data_pad].op.axis
        s[data_pad].vectorize(dpcii)
    if cfg['data_pad_compute_location'].val == 2:
        s[data_pad].compute_at(s[input_tile], oh_m)
        (_, _, _, _, dpcii) = s[data_pad].op.axis
        s[data_pad].vectorize(dpcii)
    if cfg['data_pad_compute_location'].val == 3:
        s[data_pad].compute_at(s[input_tile], ow_m)
        (_, _, _, _, dpcii) = s[data_pad].op.axis
        s[data_pad].vectorize(dpcii)

    ############################################################

    ############################################################
    # data_pad
    # s[data_pad].compute_inline()
    ############################################################

    ############################################################
    # transform image
    n, cii, oh_m, ow_m, eps, nu, ciii = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    cfg["reorder_V"].apply(s, V, [n, cii, oh_m, ow_m, eps, nu, ciii, r_eps, r_nu])

    cfg.define_annotate("reduce_V", [r_eps, r_nu, eps, nu],
                        policy='try_unroll')
    cfg['reduce_V'].apply(s, V, [r_eps, r_nu, eps, nu], cfg=cfg)
    s[V].vectorize(ciii)

    cfg.define_knob('input_tile_compute_location', [3])
    # cfg.define_knob('input_tile_compute_location', [0, 1, 2, 3])
    if cfg['input_tile_compute_location'].val == 1:
        s[input_tile].compute_at(s[V], cii)
    if cfg['input_tile_compute_location'].val == 2:
        s[input_tile].compute_at(s[V], oh_m)
    if cfg['input_tile_compute_location'].val == 3:
        s[input_tile].compute_at(s[V], ow_m)
    ############################################################

    ############################################################
    # batch gemm
    n, coo, oh_m, ow_m, eps, nu, vc = s[M].op.axis
    cii, ciii = s[M].op.reduce_axis
    s[M].vectorize(vc)
    cfg["reorder_M"].apply(s, M, [n, coo, oh_m, ow_m, eps, nu, vc, cii, ciii])

    cfg.define_annotate("reduce_M", [eps, nu],
                        policy='try_unroll')
    cfg['reduce_M'].apply(s, M, [eps, nu], cfg=cfg)

    # cfg.define_knob('V_compute_location', [0, 1, 2, 3])
    cfg.define_knob('V_compute_location', [0])
    if cfg['V_compute_location'].val == 1:
        s[V].compute_at(s[M], coo)
    if cfg['V_compute_location'].val == 2:
        s[V].compute_at(s[M], oh_m)
    if cfg['V_compute_location'].val == 3:
        s[V].compute_at(s[M], ow_m)

    ############################################################

    ############################################################
    # inverse transform
    s[A].compute_inline()
    n, coo, oh_m, ow_m, vh, vw, vc = s[Y].op.axis
    r_eps, r_nu = s[Y].op.reduce_axis

    cfg['reorder_Y'].apply(s, Y, [n, coo, oh_m, ow_m, vh, vw, vc, r_eps, r_nu])

    cfg.define_annotate("reduce_Y", [r_eps, r_nu, vh, vw],
                        policy='try_unroll')
    cfg['reduce_Y'].apply(s, Y, [r_eps, r_nu, vh, vw], cfg=cfg)

    # cfg.define_knob('M_compute_location', [0, 1, 2, 3])
    cfg.define_knob('M_compute_location', [2])
    if cfg['M_compute_location'].val == 1:
        s[M].compute_at(s[Y], coo)
    if cfg['M_compute_location'].val == 2:
        s[M].compute_at(s[Y], oh_m)
    if cfg['M_compute_location'].val == 3:
        s[M].compute_at(s[Y], ow_m)
    ############################################################

    ############################################################
    # output
    n, coo, oh, ow, vc = s[last].op.axis
    s[last].vectorize(vc)
    s[output].vectorize(vc)
    # cfg.define_knob('Y_compute_location', [0, 1, 2])
    cfg.define_knob('Y_compute_location', [1])
    if cfg['Y_compute_location'].val == 1:
        s[Y].compute_at(s[output], coo)
    if cfg['Y_compute_location'].val == 2:
        s[Y].compute_at(s[output], oh)
    if cfg['Y_compute_location'].val == 3:
        s[Y].compute_at(s[output], ow)
    ############################################################

    if output != last:
        s[output].compute_inline()
    return s
