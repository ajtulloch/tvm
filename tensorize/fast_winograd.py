"""Example code to do convolution."""
from __future__ import division

from topi.util import get_const_tuple, get_const_int, const_matrix
from topi.nn.util import get_const_int, get_pad_tuple
import os
import numpy as np
import tvm
import tvm.rpc
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple, get_const_int
from topi.nn.pad import pad
from tvm.contrib import util
from topi import tag
import scipy.stats.mstats
import collections
from tvm import autotvm

A_data = np.array([[1,  1,  1,   1,    1,    1,      1,    0],
                   [0,  1,  -1,  2,   -2,   1/2,   -1/2,   0],
                   [0,  1,  1,   4,    4,   1/4,    1/4,   0],
                   [0,  1,  -1,  8,   -8,   1/8,   -1/8,   0],
                   [0,  1,  1,   16,  16,   1/16,  1/16,   0],
                   [0,  1,  -1,  32,  -32,  1/32,  -1/32,  1]],
                  dtype=np.float32).T
m = A_data.shape[1]
r = 3
alpha = m + r - 1

HSTR = 1
WSTR = 1
HPAD = 1
WPAD = 1

VK = 8
VP = 8

def decl_output_transform(cfg, X, M):
    N = get_const_int(X.shape[0])
    IH = get_const_int(X.shape[2])
    IW = get_const_int(X.shape[3])
    alpha = get_const_int(M.shape[0])

    K = get_const_int(M.shape[0]) * get_const_int(M.shape[4])
    P = get_const_int(M.shape[3]) * get_const_int(M.shape[5])

    # inverse transform
    A = const_matrix(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    Y = tvm.compute((K // VK, P // VP, m, m, K, P), lambda k, b, vh, vw, kk, bb:
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
    # output = tvm.compute((N, K, OH, OW), lambda n, k, h, w:
    #                    Y[k][n * nH * nW + (h//m) * nW + w//m][h % m][w % m],
    #                    name='output', tag='winograd_conv_output')
    output = tvm.compute((N, K, OH, OW), _output,
                       name='output', tag='winograd_conv_output')

    return output

def schedule_output_transform(cfg, output):
    s = tvm.create_schedule(output.op)
    Y = output.op.input_tensors[0]
    M, A = Y.op.input_tensors
    s[A].compute_inline()
    k, b, vh, vw, kk, bb = s[Y].op.axis
    cfg.define_knob('vectorize_bb', [0, 1])
    if cfg['vectorize_bb'].val:
        s[Y].vectorize(bb)
    r_eps, r_nu = s[Y].op.reduce_axis
    # s[Y].unroll(vh)
    # s[Y].unroll(vw)
    n, co, h, w = s[output].op.axis
    cfg.define_knob('tile_output', [0, 1])
    if cfg['tile_output'].val:
        pass
        # ho, wo, hi, wi = s[output].tile(h, w, m, m)
        # s[Y].compute_at(s[output], wo)

    cfg.define_knob('unroll_y', [0, 1])
    if cfg['unroll_y'].val:
        s[Y].unroll(r_eps)
        s[Y].unroll(r_nu)
    return s

@autotvm.template
def output_transform_autotvm(dtype):
    cfg = autotvm.get_config()
    X = tvm.placeholder(shape=(1, 64, 56, 56), dtype="float32", name="X")
    W = tvm.placeholder(shape=(64, 64, 56, 56), dtype="float32", name="W")
    N = get_const_int(X.shape[0])
    IH = get_const_int(X.shape[2])
    IW = get_const_int(X.shape[3])
    OH = get_const_int((IH + 2 * HPAD - 3) // HSTR + 1)
    OW = get_const_int((IW + 2 * WPAD - 3) // WSTR + 1)
    nH, nW = get_const_int((OH + m-1) // m), get_const_int((OW + m-1) // m)

    P = N * nH * nW
    K = get_const_int(W.shape[0])
    M = tvm.placeholder(shape=(K // VK, P // VP, alpha, alpha, VK, VP), name="M")
    output = decl_output_transform(cfg, X, M)
    s = schedule_output_transform(cfg, output)
    cfg.add_flop(2 * N * K * OH * OW * alpha * alpha)
    print(tvm.lower(s, [X, M, output], simple_mode=True))
    return s, [X, M, output]



import logging
import sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

measure_option = autotvm.measure_option(
    measure_func='local',
    number=10)


task = autotvm.task.create(
    output_transform_autotvm,
    args=("float32",),
    target=tvm.target.create('llvm -mcpu=core-avx2'))
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(
    n_trial=100,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file('output_transform_winograd.log')])
