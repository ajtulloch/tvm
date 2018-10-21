# pylint: disable=invalid-name,unused-variable,no-else-return
from __future__ import absolute_import as _abs

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


def _decl_winograd_NCHWc(cfg, data, kernel, num_filter, kernel_size, stride, padding, layout, out_layout, out_dtype):
    # import ipdb
    # ipdb.set_trace()
    # assert layout == "NCHW", "Only support NCHW"
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

    m = 6
    r = 3
    alpha = m + r - 1

    assert all(k == 3 for k in (KH, KW))
    assert all(p == 1 for p in (pad_top, pad_left, pad_bottom, pad_right))
    assert all(s == 1 for s in (HSTR, WSTR))
    assert OH == IH
    assert OW == IW

    # Layouts:

    # input_tile_shape = (N, CII, OH // m, OH // m, alpha, alpha, CIII)
    # U_shape = (COO, CII, alpha, alpha, CIII, COOO)
    # V_shape = (N, CII, OH // m, OW // m, alpha, alpha, CIII)
    # M_shape = (N, COO, OH // m, OW // m, alpha, alpha, COOO)
    # Y_shape = (N, COO, OH // m, OW // m, m, m, COOO)
    # O_shape = (N, COO, OH, OW, COOO)


    def div_round_up(a, b):
        return (a + b - 1) // b

    input_tile = tvm.compute((N, CII, div_round_up(OH, m), div_round_up(OW, m), alpha, alpha, CIII),
                             lambda n, cii, oh, ow, eps, nu, ciii:
                             data_pad[n][cii][oh * m + eps][ow * m + nu][ciii],
                             name='input_tile')


    # transform kernel
    G = const_matrix(G_data, 'G')
    r_kh = tvm.reduce_axis((0, KH), 'r_kh')
    r_kw = tvm.reduce_axis((0, KW), 'r_kw')
    U = tvm.compute((COO, CII, alpha, alpha, CIII, VC),
                    lambda coo, cii, eps, nu, ciii, vc:
                    tvm.sum(kernel[coo][cii][r_kh][r_kw][ciii][vc].astype(out_dtype) *
                            G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]),
                    name='U')

    # transform image
    B = const_matrix(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((N, CII, div_round_up(OH, m), div_round_up(OW, m), alpha, alpha, CIII),
                    lambda n, cii, oh, ow, eps, nu, ciii:
                    tvm.sum(input_tile[n][cii][oh][ow][r_eps][r_nu][ciii].astype(out_dtype) *
                            B[r_eps][eps] * B[r_nu][nu], axis=[r_eps, r_nu]), name='V')
    cii = tvm.reduce_axis((0, CII), name='cii')
    ciii = tvm.reduce_axis((0, CIII), name='ciii')

    # M_shape = (N, COO, OH // m, OW // m, alpha, alpha, COOO)
    M = tvm.compute((N, COO, div_round_up(OH, m), div_round_up(OW, m), alpha, alpha, VC),
                    lambda n, coo, oh, ow, eps, nu, vc:
                    tvm.sum(U[coo][cii][eps][nu][ciii][vc] * V[n][cii][oh][ow][eps][nu][ciii],
                            axis=[cii, ciii]),
                    name='M')

    # inverse transform
    A = const_matrix(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    # Y_shape = (N, COO, OH // m, OW // m, m, m, COOO)
    Y = tvm.compute((N, COO, div_round_up(OH, m), div_round_up(OW, m), m, m, VC),
                    lambda n, coo, oh, ow, vh, vw, vc:
                    tvm.sum(M[n][coo][oh][ow][r_eps][r_nu][vc] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]),
                    name='Y')

    output = tvm.compute((N, COO, OH, OW, VC),
                         lambda n, coo, oh, ow, vc:
                         Y[n][coo][oh // m][ow // m][oh % m][ow % m][vc],
                         name='output', tag='winograd_conv2d_output',
                         attrs={'workload': wkl})
    # cfg.add_flop(2 * N * COO * OH * OW * KH * KW * CII * CIII)
    return output

def _schedule_winograd_NCHWc(cfg, s, output, last):
    return s

    # Y = output.op.input_tensors[0]
    # M, A = Y.op.input_tensors
    # U, V = M.op.input_tensors
    # input_tile, B = V.op.input_tensors
    # data_pad = input_tile.op.input_tensors[0]

    # # padding
    # # s[data_pad].compute_inline()

    # # pack input tiles
    # # s[d].compute_inline()

    # # transform kernel
    # if isinstance(U.op, tvm.tensor.ComputeOp):
    #     kernel, G = U.op.input_tensors
    #     s[G].compute_inline()
    #     coo, cii, eps, nu, ciii, vc = s[U].op.axis
    #     if autotvm.GLOBAL_SCOPE.in_tuning:
    #         # kernel transformation will be pre-computed during compilation, so we skip
    #         # this part to make tuning records correct
    #         s[U].pragma(eps, 'debug_skip_region')
    #     else:
    #         pass
    #         # r_kh, r_kw = s[U].op.reduce_axis
    #         # s[U].reorder(k, c, eps, nu, r_kh, r_kw, kk)
    #         # for axis in [eps, nu, r_kh, r_kw]:
    #         #     s[U].unroll(axis)
    #         # s[U].vectorize(kk)
    #         # s[U].parallel(k)

    #     if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
    #         s[kernel].compute_inline()

    # # input tile
    # n, cii, oh, ow, eps, nu, ciii = s[input_tile].op.axis
    # s[input_tile].vectorize(ciii)


    # # data_pad
    # s[data_pad].compute_inline()

    # # transform image
    # n, cii, oh, ow, eps, nu, ciii = s[V].op.axis
    # r_eps, r_nu = s[V].op.reduce_axis
    # # s[V].reorder(b, c, eps, nu, r_eps, r_nu, bb)
    # for axis in [r_eps, r_nu, eps, nu]:
    #     if UNROLL:
    #         s[V].unroll(axis)
    # s[V].vectorize(ciii)


    # # batch gemm
    # n, coo, oh, ow, eps, nu, vc = s[M].op.axis
    # cii, ciii = s[M].op.reduce_axis
    # s[M].vectorize(vc)
    # # cfg.define_split('tile_c', c, num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    # # co, ci = cfg['tile_c'].apply(s, M, c)
    # # xo, xi = cfg['tile_p'].apply(s, M, b)
    # # s[M].reorder(eps, nu, xo, co, k, ci, xi)
    # # cfg.define_annotate('ann_reduce', [ci], policy='try_unroll')
    # # cfg.define_annotate('ann_spatial', [k, xi], policy='try_unroll_vec')
    # # cfg['ann_reduce'].apply(s, M, [ci],
    # #                         axis_lens=[cfg['tile_c'].size[-1]],
    # #                         max_unroll=16,
    # #                         cfg=cfg)
    # # cfg['ann_spatial'].apply(s, M, [k, xi])

    # # inverse transform
    # s[A].compute_inline()
    # n, coo, oh, ow, vh, vw, vc = s[Y].op.axis
    # r_eps, r_nu = s[Y].op.reduce_axis
    # for axis in [r_eps, r_nu, vh, vw]:
    #     if UNROLL:
    #         s[Y].unroll(axis)
    # s[Y].vectorize(vc)
    # # output

    # n, coo, h, w, vc = s[last].op.axis
    # s[last].vectorize(vc)
    # s[output].vectorize(vc)
    # # co, coi = cfg['tile_k'].apply(s, last, co)
    # # s[M].compute_at(s[last], co)
    # # s[last].parallel(co)

    # # MM = s.cache_read(M, 'global', [Y])
    # # m = get_const_int(V.shape[0]) + 1 - 3
    # # ho, wo, hi, wi = s[last].tile(h, w, m, m)
    # # s[Y].compute_at(s[last], wo)
    # # s[MM].compute_at(s[last], wo)

    # # if output != last:
    # #     s[output].compute_inline()
    # return s
