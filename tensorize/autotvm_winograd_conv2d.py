import logging
import sys

import numpy as np
import tvm
from topi.util import get_const_tuple, get_const_int, const_matrix
from topi.nn.util import get_const_int, get_pad_tuple
from topi.nn import pad

# the module is called `autotvm`
from tvm import autotvm

MTile = 6
MMTile = 6
NTile = 8
KTile = 256
ARCH = "neon"
BITCODE_PATHS = [
    "tensorize/gemm__neon.bc"
]

@tvm.register_func("tvm_callback_llvm_bitcode_path")
def bitcode_paths():
    global BITCODE_PATHS
    return BITCODE_PATHS

# Tensorized
def intrin_gemm(M, N, K):
    assert M == MTile
    assert N == NTile
    dtype = 'float32'
    A = tvm.placeholder((K, M), dtype=dtype, name='A')
    B = tvm.placeholder((K, N), dtype=dtype, name='B')
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((M, N), lambda m, n:
                    tvm.sum(A[k, m] * B[k, n], axis=[k]), name='C')

    Ab = tvm.decl_buffer(A.shape, A.dtype,
                        name="A",
                        offset_factor=MTile,
                        strides=[M, 1])
    Bb = tvm.decl_buffer(B.shape, B.dtype,
                        name="B",
                        offset_factor=NTile,
                        strides=[N, 1])
    Cb = tvm.decl_buffer(C.shape, C.dtype,
                        name="C",
                        offset_factor=1,
                        strides=[tvm.var('ldc'), 1])

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def body():
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "sgemm_compute_{MTile}x{NTile}__{ARCH}".format(**globals()),
                K,
                irb.buffer_ptr(aa),
                aa.elem_offset,
                irb.buffer_ptr(bb),
                bb.elem_offset,
                irb.buffer_ptr(cc),
                cc.elem_offset,
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()

        def reset():
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "sgemm_reset_{MTile}x{NTile}__{ARCH}".format(**globals()),
                irb.buffer_ptr(cc),
                cc.elem_offset,
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()

        def update():
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "sgemm_update_{MTile}x{NTile}__{ARCH}".format(**globals()),
                K,
                irb.buffer_ptr(aa),
                aa.elem_offset,
                irb.buffer_ptr(bb),
                bb.elem_offset,
                irb.buffer_ptr(cc),
                cc.elem_offset,
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()
        return body(), reset(), update()

    with tvm.build_config():
        return tvm.decl_tensor_intrin(C.op,
                                      intrin_func,
                                      binds={A: Ab, B: Bb, C: Cb})

def _schedule_winograd(cfg, s, output, last):
    Y = output.op.input_tensors[0]
    M, A = Y.op.input_tensors
    U, V = M.op.input_tensors
    d, B = V.op.input_tensors
    data_pad = d.op.input_tensors[0]

    # padding
    s[data_pad].compute_inline()

    # pack input tiles
    s[d].compute_inline()

    # transform kernel
    if isinstance(U.op, tvm.tensor.ComputeOp):
        kernel, G = U.op.input_tensors
        s[G].compute_inline()
        eps, nu, k, c, kk, = s[U].op.axis
        r_kh, r_kw = s[U].op.reduce_axis
        s[U].reorder(k, c, eps, nu, r_kh, r_kw, kk)
        s[U].unroll(eps)
        s[U].unroll(nu)
        s[U].unroll(r_kh)
        s[U].unroll(r_kw)
        s[U].vectorize(kk)
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(k, 'debug_skip_region')
        else:
            s[U].parallel(k)

    # transform image
    DD = s.cache_read(d, 'global', [V])
    s[B].compute_inline()
    eps, nu, b, c, bb = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    s[V].reorder(b, c, eps, nu, r_eps, r_nu, bb)
    s[V].unroll(eps)
    s[V].unroll(nu)
    s[V].unroll(r_eps)
    s[V].unroll(r_nu)
    s[DD].compute_at(s[V], c)
    s[V].vectorize(bb)
    s[V].parallel(b)

    # batch gemm
    eps, nu, k, b = s[M].op.axis
    c = s[M].op.reduce_axis[0]
    cfg.define_split('tile_c', c, num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    co, ci = cfg['tile_c'].apply(s, M, c)
    xo, xi = cfg['tile_p'].apply(s, M, b)
    s[M].reorder(eps, nu, xo, co, k, ci, xi)
    cfg.define_annotate('ann_reduce', [ci], policy='try_unroll')
    cfg.define_annotate('ann_spatial', [k, xi], policy='try_unroll_vec')
    cfg['ann_reduce'].apply(s, M, [ci],
                            axis_lens=[cfg['tile_c'].size[-1]],
                            max_unroll=16,
                            cfg=cfg)
    cfg['ann_spatial'].apply(s, M, [k, xi])

    # inverse transform
    s[A].compute_inline()
    k, b, vh, vw = s[Y].op.axis
    r_eps, r_nu = s[Y].op.reduce_axis
    s[Y].unroll(vh)
    s[Y].unroll(vw)
    s[Y].unroll(r_eps)
    s[Y].unroll(r_nu)

    # output
    n, co, h, w = s[last].op.axis
    co, coi = cfg['tile_k'].apply(s, last, co)
    s[M].compute_at(s[last], co)
    s[last].parallel(co)

    MM = s.cache_read(M, 'global', [Y])
    m = get_const_int(V.shape[0]) + 1 - 3
    ho, wo, hi, wi = s[last].tile(h, w, m, m)
    s[Y].compute_at(s[last], wo)
    s[MM].compute_at(s[last], wo)

    if output != last:
        s[output].compute_inline()


def _decl_winograd(cfg, data, kernel, strides, padding, layout, out_dtype, tile_size):
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
    elif tile_size == 6:
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

        assert G_data.shape == (8, 3)
        assert A_data.shape == (8, 6)
        assert B_data.shape == (8, 8)
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

    cfg.define_split('tile_p', cfg.axis(P), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    fg.define_split('tile_k', cfg.axis(K), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    VP = cfg['tile_p'].size[-1]
    VK = cfg['tile_k'].size[-1]

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
        # U = tvm.placeholder((alpha, alpha, K // VK, C, VK), lambda eps, nu, k, c, kk:
        #                     tvm.sum(kernel[k * VK + kk][c][r_kh][r_kw].astype(out_dtype) *
        #                         G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]), name='U')
        U = tvm.placeholder((alpha, alpha, K // VK, C, VK), dtype="float32", name="U")


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
    cfg.add_flop(2 * N * K * H * W * KH * KW * C)
    return U, output


@autotvm.template
def conv2d(IH, IW, KH, KW, CIn, COut, dtype):
    cfg = autotvm.get_config()
    A = tvm.placeholder((1, IH, IW, CIn), dtype=dtype, name="A")
    W = tvm.placeholder((COut, CIn, KH, KW), dtype=dtype, name="W")
    U, Y = _decl_winograd(cfg, A, W, strides=1, padding=1, layout="NCHW", out_dtype="float32", tile_size=6)
    s = tvm.create_schedule(Y.op)
    _schedule_winograd(cfg, s, Y, Y)
    return s, [A, U, Y]

import collections

Workload = collections.namedtuple("Workload", ["space", "input_channel", "output_channel", "kernel", "pad", "stride"])
WORKLOADS = [
        Workload(space=192, input_channel=3, output_channel=12, kernel=3, pad=1, stride=1),
        Workload(space=96, input_channel=12, output_channel=24, kernel=3, pad=1, stride=1),
        Workload(space=48, input_channel=24, output_channel=48, kernel=3, pad=1, stride=1),
        Workload(space=24, input_channel=48, output_channel=96, kernel=3, pad=1, stride=1),
        Workload(space=12, input_channel=96, output_channel=180, kernel=3, pad=1, stride=1),
        Workload(space=6, input_channel=180, output_channel=220, kernel=3, pad=1, stride=1),
        Workload(space=6, input_channel=220, output_channel=180, kernel=3, pad=1, stride=1),
        Workload(space=12, input_channel=180, output_channel=96, kernel=3, pad=1, stride=1),
        Workload(space=24, input_channel=96, output_channel=48, kernel=3, pad=1, stride=1),
        Workload(space=48, input_channel=48, output_channel=24, kernel=3, pad=1, stride=1),
        Workload(space=96, input_channel=24, output_channel=12, kernel=3, pad=1, stride=1),
        Workload(space=192, input_channel=12, output_channel=1, kernel=3, pad=1, stride=1),
]

for i, w in enumerate(WORKLOADS):
    task = autotvm.task.create(
        conv2d,
        args=(w.space, w.space, w.kernel, w.kernel, w.input_channel, w.output_channel, 'float32'),
        target=tvm.target.rasp())
    print(task.config_space)

    # logging config (for printing tuning log to screen)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    measure_option = autotvm.measure_option(
        measure_func=autotvm.use_rpc("rpi", host="localhost", port=9190),
        number=3,
        parallel_num=5)

    # # begin tuning, log records to file `matmul.log`
    # tuner = autotvm.tuner.RandomTuner(task)
    # tuner.tune(n_trial=200,
    #            measure_option=measure_option,
    #            callbacks=[autotvm.callback.log_to_file('matmul.log')])
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=500,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('conv2d_xgb_segmentation__winograd_{i}_{w.space}_{w.kernel}_{w.input_channel}_{w.output_channel}.log'.format(i=i, w=w))])


# # apply history best from log file
# with autotvm.apply_history_best('matmul.log'):
#     with tvm.target.rasp():
#         s, arg_bufs = matmul(N, L, M, 'float32')
#         func = tvm.build(s, arg_bufs)

# # check correctness
# a_np = np.random.uniform(size=(N, L)).astype(np.float32)
# b_np = np.random.uniform(size=(L, M)).astype(np.float32)
# c_np = a_np.dot(b_np)

# c_tvm = tvm.nd.empty(c_np.shape)
# func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

# np.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
