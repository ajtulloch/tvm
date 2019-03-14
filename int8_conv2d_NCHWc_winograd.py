# pylint: disable=invalid-name,unused-variable,no-else-return
from __future__ import absolute_import as _abs
from __future__ import division
import numpy as np

import tvm
from tvm import autotvm

from topi.util import traverse_inline, get_const_tuple, const_matrix
from topi.nn import pad, conv2d, conv2d_NCHWc, conv2d_alter_layout
from topi.nn.util import get_const_int, get_pad_tuple
import topi.nn


import click
from topi.util import get_const_int, const_matrix
from topi.nn.conv2d import Workload
import numpy as np
import tvm
import tvm.rpc
from tvm import autotvm
import collections
import logging
import sys


import os
UNROLL = int(os.environ.get('UNROLL', '1'))

def get_transform_matrices(m):
    assert m in (2, 4, 6)
    G_data = (np.array([
        [1, 0, 0],
        [1.0/2, 1.0/2, 1.0/2],
        [1.0/2, -1.0/2, 1.0/2],
        [0, 0, 1]], np.float32) * 2).astype(np.int8)

    B_data = np.array([
        [1, 0, 0, 0],
        [0, 1, -1, 1],
        [-1, 1, 1, 0],
        [0, 0, 0, -1]], np.int8)

    A_data = np.array([
        [1, 0],
        [1, 1],
        [1, -1],
        [0, -1]], np.int8)

    return const_matrix(A_data, "A"), const_matrix(B_data, "B"), const_matrix(G_data, "G")


def _decl_winograd_NCHWc(cfg, data, kernel, num_filter, kernel_size, stride, padding, layout, out_layout, out_dtype, m):
    # create workload according to raw arguments

    out_dtype = out_dtype or data.dtype
    N, CII, IH, IW, CIII = get_const_tuple(data.shape)
    COO, CII, KH, KW, CIII_, VC = get_const_tuple(kernel.shape)

    if (KH, KW) != (3, 3):
        COO, CII, CIII_, alpha_h, alpha_w, VC = get_const_tuple(kernel.shape)
        assert alpha_h == m + 3 - 1
        assert alpha_w == m + 3 - 1
        KH = 3
        KW = 3
        need_packing = False
        kernel_placeholder = tvm.placeholder((COO, CII, KH, KW, CIII_, VC), dtype="int8")
    else:
        kernel_placeholder = kernel
        need_packing = True

    wkl = None

    assert (KH, KW) == (3, 3)


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
    # import ipdb; ipdb.set_trace()
    # assert all(k == 3 for k in (KH, KW))
    assert all(p == 1 for p in (pad_top, pad_left, pad_bottom, pad_right))
    assert all(s == 1 for s in (HSTR, WSTR))
    assert OH == IH
    assert OW == IW

    OH_M = div_round_up(OH, m)
    OW_M = div_round_up(OW, m)
    # Layouts:

    # input            = (N, CII, IH, IW, CIII)
    # -> transpose
    ############################################################
    # input_tile_shape = (N, CII, OH // m, OH // m, alpha, alpha, CIII)
    # U_shape          = (COO, CII, CIII, alpha, alpha, COOO)
    # V_shape          = (N, CII, OH // m, OW // m, alpha, alpha, CIII)
    # M_shape          = (N, COO, OH // m, OW // m, alpha, alpha, COOO)
    # Y_shape          = (N, COO, OH // m, OW // m, m, m, COOO)
    ############################################################
    # -> transpose
    # O_shape          = (N, COO, OH, OW, COOO)

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
    if need_packing:
        r_kh = tvm.reduce_axis((0, KH), 'r_kh')
        r_kw = tvm.reduce_axis((0, KW), 'r_kw')
        U = tvm.compute((COO, CII, CIII, alpha, alpha, VC),
                        lambda coo, cii, ciii, eps, nu, vc:
                        tvm.sum(kernel[coo][cii][r_kh][r_kw][ciii][vc] * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]),
                        name='U')
    else:
        assert get_const_tuple(kernel.shape) == (COO, CII, CIII, alpha, alpha, VC)
        U = kernel

    # transform image
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((N, CII, OH_M, OW_M, alpha, alpha, CIII),
                    lambda n, cii, oh_m, ow_m, eps, nu, ciii:
                    tvm.sum(input_tile[n][cii][oh_m][ow_m][r_eps][r_nu][ciii].astype('int8') *
                            B[r_eps][eps] * B[r_nu][nu], axis=[r_eps, r_nu]), name='V')
    cii = tvm.reduce_axis((0, CII), name='cii')
    ciii = tvm.reduce_axis((0, CIII), name='ciii')

    # M_shape = (N, COO, OH // m, OW // m, alpha, alpha, COOO)
    M = tvm.compute((N, COO, OH_M, OW_M, alpha, alpha, VC),
                    lambda n, coo, oh_m, ow_m, eps, nu, vc:
                    tvm.sum(U[coo][cii][ciii][eps][nu][vc].astype(out_dtype) * V[n][cii][oh_m][ow_m][eps][nu][ciii].astype(out_dtype),
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
                         name='output', tag='winograd_conv2d_output')
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

    cfg.define_knob('data_pad_compute_location', [0, 1, 2, 3])
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

    s[V].vectorize(ciii)
    # import ipdb; ipdb.set_trace()
    cfg["reorder_V"].apply(s, V, [n, cii, oh_m, ow_m, eps, nu, ciii, r_eps, r_nu])

    cfg.define_annotate("reduce_V", [r_eps, r_nu, eps, nu],
                        policy='unroll')
    cfg['reduce_V'].apply(s, V, [r_eps, r_nu, eps, nu], cfg=cfg)


    cfg.define_knob('input_tile_compute_location', [0, 1, 2, 3])
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

    cfg.define_knob('V_compute_location', [0, 1, 2, 3])
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
    s[Y].vectorize(vc)

    cfg['reorder_Y'].apply(s, Y, [n, coo, oh_m, ow_m, vh, vw, vc, r_eps, r_nu])

    cfg.define_annotate("reduce_Y", [r_eps, r_nu, vh, vw],
                        policy='unroll')
    cfg['reduce_Y'].apply(s, Y, [r_eps, r_nu, vh, vw], cfg=cfg)

    cfg.define_knob('M_compute_location', [0, 1, 2, 3])
    if cfg['M_compute_location'].val == 1:
        s[M].compute_at(s[Y], coo)
    if cfg['M_compute_location'].val == 2:
        s[M].compute_at(s[Y], oh_m)
    if cfg['M_compute_location'].val == 3:
        s[M].compute_at(s[Y], ow_m)

    ############################################################

    ############################################################
    # output

    if output != last:
        s[output].compute_inline()

    n, coo, oh, ow, vc = s[last].op.axis
    s[last].vectorize(vc)

    OH = get_const_int(oh.dom.extent)
    OW = get_const_int(ow.dom.extent)
    mh = get_const_int(vh.dom.extent)
    mw = get_const_int(vw.dom.extent)
    cfg.define_knob('output_tile', [1])
    cfg.define_annotate('reduce_output', [cfg.axis(mh), cfg.axis(mw)], policy="try_unroll")
    output_tile = False
    if OH % mh == 0 and OW % mw == 0 and cfg['output_tile'].val == 1:
        # We can tile in OH
        output_tile = True
        oh, ow, ohi, owi = s[last].tile(oh, ow, mh, mw)
        cfg["reduce_output"].apply(s, last, [ohi, owi], cfg=cfg)

    cfg.define_knob('Y_compute_location', [0, 1, 2, 3])
    if cfg['Y_compute_location'].val == 1:
        s[Y].compute_at(s[last], coo)
    if cfg['Y_compute_location'].val == 2:
        s[Y].compute_at(s[last], oh)
    if cfg['Y_compute_location'].val == 3:
        s[Y].compute_at(s[last], ow)
    ############################################################

    return s

@autotvm.template
def conv2d_NCHWc_winograd_autotvm(s, ic, oc, kernel, pad, stride):
    ic = ((ic + 16 - 1) // 16) * 16
    oc = ((oc + 16 - 1) // 16) * 16
    kernel = 3
    pad = 1
    stride = 1
    cfg = autotvm.get_config()
    cfg.define_knob('BNInput', [16]) # TODO, 8, 16
    cfg.define_knob('BNOutput', [16]) # TODO 8, 16
    cfg.define_knob('m', [2]) # TODO 8, 16
    BNInput = cfg['BNInput'].val
    BNOutput = cfg['BNOutput'].val
    m = cfg['m'].val
    X = tvm.placeholder(shape=(1, ic // BNInput, s, s, BNInput), dtype="int8", name="X")
    W = tvm.placeholder(shape=(oc // BNOutput, ic // BNInput, kernel, kernel, BNInput, BNOutput), dtype="int8", name="W")

    Y = _decl_winograd_NCHWc(cfg, X, W, num_filter=oc, kernel_size=kernel, stride=stride, padding=pad, layout="NCHW{}c".format(BNInput), out_layout="NCHW{}c".format(BNOutput), out_dtype="int32", m=m)
    s = tvm.create_schedule([Y.op])
    s = _schedule_winograd_NCHWc(cfg, s, Y, Y)
    print(tvm.lower(s, [X, W, Y], simple_mode=True))
    return s, [X, W, Y]


# Workload = collections.namedtuple("Workload", ["space", "input_channel", "output_channel", "kernel", "pad", "stride"])

WORKLOADS = [
        # Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
        # Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
        # workloads of resnet34_v1 on imagenet, no extra workload required
        # workloads of resnet50_v1 on imagenet
        Workload('float32', 'float32', 56, 56, 64, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 512, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1),

        # Workload(space=102, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=102, input_channel=32, output_channel=32, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=64, output_channel=64, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
        # Workload(space=128, input_channel=64, output_channel=64, kernel=3, pad=1, stride=1),
        # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),

        # # # Workload(space=12, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
        # Workload(space=192, input_channel=3, output_channel=12, kernel=3, pad=1, stride=1),
        # Workload(space=96, input_channel=12, output_channel=24, kernel=3, pad=1, stride=1),
        # Workload(space=48, input_channel=24, output_channel=48, kernel=3, pad=1, stride=1),
        # Workload(space=24, input_channel=48, output_channel=96, kernel=3, pad=1, stride=1),
        # Workload(space=12, input_channel=96, output_channel=180, kernel=3, pad=1, stride=1),
        # Workload(space=6, input_channel=180, output_channel=220, kernel=3, pad=1, stride=1),
        # Workload(space=6, input_channel=220, output_channel=180, kernel=3, pad=1, stride=1),
        # Workload(space=12, input_channel=180, output_channel=96, kernel=3, pad=1, stride=1),
        # Workload(space=24, input_channel=96, output_channel=48, kernel=3, pad=1, stride=1),
        # Workload(space=48, input_channel=48, output_channel=24, kernel=3, pad=1, stride=1),
        # Workload(space=96, input_channel=24, output_channel=12, kernel=3, pad=1, stride=1),
        # Workload(space=192, input_channel=12, output_channel=1, kernel=3, pad=1, stride=1),
]

Workload = collections.namedtuple("Workload", ["space", "input_channel", "output_channel", "kernel", "pad", "stride"])

def a(x, align=16):
    if x < align:
        return align
    return ((x + align - 1) // align) * align
WORKLOADS = [
        # Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
        # Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
        # Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
        # Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
        # Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
        # Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
        # Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
        # Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
        # # workloads of resnet34_v1 on imagenet, no extra workload required
        # # workloads of resnet50_v1 on imagenet
        # Workload('float32', 'float32', 56, 56, 64, 256, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 256, 64, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 28, 28, 128, 512, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 256, 512, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 28, 28, 512, 128, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2),
        # Workload('float32', 'float32', 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1),

        # Workload(space=102, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=102, input_channel=32, output_channel=32, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=64, output_channel=64, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
        # Workload(space=128, input_channel=64, output_channel=64, kernel=3, pad=1, stride=1),
        # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),

        # # # Workload(space=12, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
    # Workload(space=64, input_channel=a(64), output_channel=a(64), kernel=3, pad=1, stride=1),
    # Workload(space=96, input_channel=a(32), output_channel=a(16), kernel=3, pad=1, stride=1),
        Workload(space=96, input_channel=a(12), output_channel=a(24), kernel=3, pad=1, stride=1),
        Workload(space=48, input_channel=a(24), output_channel=a(48), kernel=3, pad=1, stride=1),
        Workload(space=24, input_channel=a(48), output_channel=a(96), kernel=3, pad=1, stride=1),
        Workload(space=12, input_channel=a(96), output_channel=a(180), kernel=3, pad=1, stride=1),
        Workload(space=6, input_channel=a(180), output_channel=a(220), kernel=3, pad=1, stride=1),
        Workload(space=6, input_channel=a(220), output_channel=a(180), kernel=3, pad=1, stride=1),
        Workload(space=12, input_channel=a(180), output_channel=a(96), kernel=3, pad=1, stride=1),
        Workload(space=24, input_channel=a(96), output_channel=a(48), kernel=3, pad=1, stride=1),
        Workload(space=48, input_channel=a(48), output_channel=a(24), kernel=3, pad=1, stride=1),
        Workload(space=96, input_channel=a(24), output_channel=a(12), kernel=3, pad=1, stride=1),
        Workload(space=192, input_channel=a(12), output_channel=a(1), kernel=3, pad=1, stride=1),
        Workload(space=192, input_channel=a(1), output_channel=a(1), kernel=3, pad=1, stride=1),
]

target = tvm.target.arm_cpu("rasp3b")# 'llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu'
local_target = 'llvm -mcpu=core-avx2'

@click.command()
@click.option('--autotvm_number', default=10)
@click.option('--autotvm_repeat', default=5)
@click.option('--autotvm_n_trial', default=1000)
@click.option('--autotvm_early_stopping', default=1000)
@click.option('--autotvm_log', default="autotvm_direct_benchmark.log", type=str)
@click.option('--layout', type=click.Choice(["NCHW", "NCHWc"]), required=True)
@click.option('--tracker_port', default=9195)
@click.option('--local', is_flag=True, default=False)
def run(layout,
        autotvm_number,
        autotvm_repeat,
        autotvm_log,
        autotvm_n_trial,
        autotvm_early_stopping,
        tracker_port,
        local):
    logging.basicConfig(level=logging.DEBUG)
    for i, w in enumerate(WORKLOADS):
        # if w.in_filter % 16 != 0 or w.out_filter % 16 != 0:
        #     continue
        measure_option=autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=80),
            runner=autotvm.RPCRunner(
                'rpi', '0.0.0.0', tracker_port,
                number=autotvm_number,
                repeat=autotvm_repeat,
                min_repeat_ms=1000,
                timeout=80) if not local else
            autotvm.LocalRunner(
                timeout=80,
                number=autotvm_number,
                repeat=autotvm_repeat)
        )

        task = autotvm.task.create(
            conv2d_NCHWc_winograd_autotvm,
            args=(w.space, w.input_channel, w.output_channel, w.kernel, 1, w.stride),
            target=tvm.target.create(target if not local else local_target))
        print(task.config_space)
        tuner = autotvm.tuner.XGBTuner(task, feature_type="knob")
        tuner.tune(
            n_trial=autotvm_n_trial,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(
                    autotvm_n_trial,
                    prefix="{w.space}S, {w.input_channel} -> {w.output_channel}, {w.kernel}K, {w.pad}P, {w.stride}s, {layout}".format(w=w, layout=layout)),
                autotvm.callback.log_to_file(str(autotvm_log))])

if __name__ == "__main__":
    run()
