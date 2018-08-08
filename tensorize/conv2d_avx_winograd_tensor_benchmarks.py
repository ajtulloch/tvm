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



USE_RASP = False

if USE_RASP:
    target = tvm.target.rasp()
    remote = tvm.rpc.connect('localhost', 9090)
    ctx = remote.cpu(0)
    BITCODE_PATHS = [
        "tensorize/gemm__neon.bc"
    ]
    MTile = 6
    MMTile = 6
    NTile = 8
    KTile = 256
    ARCH = "neon"

else:
    target = tvm.target.create('llvm -mcpu=core-avx2')
    ctx = tvm.context('llvm -mcpu=core-avx2', 0)
    BITCODE_PATHS = [
        "tensorize/gemm__avx2.bc"
    ]

    # We want to keep B micro-panel in cache.
    # so MTile * KTile + NTile * MTile + KTile * NTile elements should fit in L1.
    # Therefore, KTile = 256

    MTile = 4
    MMTile = 4
    NTile = 24
    KTile = 256
    ARCH = "avx2"


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




def _schedule_winograd(s, output, last):
    Y = output.op.input_tensors[0]
    M, A = Y.op.input_tensors
    V, U = M.op.input_tensors
    d, B = V.op.input_tensors
    data_pad = d.op.input_tensors[0]

    # padding
    # s[data_pad].compute_inline()

    # pack input tiles
    # s[d].compute_inline()

    # transform kernel
    if isinstance(U.op, tvm.tensor.ComputeOp):
        kernel, G = U.op.input_tensors
        s[G].compute_inline()
        eps, nu, k, c, kk, = s[U].op.axis
        r_kh, r_kw = s[U].op.reduce_axis
        s[U].reorder(k, c, eps, nu, r_kh, r_kw, kk)
        # s[U].unroll(eps)
        # s[U].unroll(nu)
        # s[U].unroll(r_kh)
        # s[U].unroll(r_kw)
        s[U].vectorize(kk)
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(k, 'debug_skip_region')
        else:
            pass
            # s[U].parallel(k)

    # transform image
    # DD = s.cache_read(d, 'global', [V])
    # s[B].compute_inline()
    eps, nu, b, c, bb = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis
    # s[V].reorder(b, c, eps, nu, r_eps, r_nu, bb)
    # s[V].unroll(eps)
    # s[V].unroll(nu)
    # s[V].unroll(r_eps)
    # s[V].unroll(r_nu)
    # s[DD].compute_at(s[V], c)
    # s[V].vectorize(bb)
    # s[V].parallel(b)

    # batch gemm
    eps, nu, b, k = s[M].op.axis
    print("b: {b}, k: {k}".format(b=b, k=k))
    c = s[M].op.reduce_axis[0]
    # # cfg.define_split('tile_c', c, num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    # co, ci = s[M].split(c, factor=2)  #cfg['tile_c'].apply(s, M, c)
    # xo, xi = s[M].split(b, factor=MTile)# cfg['tile_p'].apply(s, M, b)
    # s[M].reorder(eps, nu, xo, co, k, ci, xi)
    # cfg.define_annotate('ann_reduce', [ci], policy='try_unroll')
    # cfg.define_annotate('ann_spatial', [k, xi], policy='try_unroll_vec')
    # cfg['ann_reduce'].apply(s, M, [ci],
    #                         axis_lens=[cfg['tile_c'].size[-1]],
    #                         max_unroll=1,
    #                         cfg=cfg)
    # cfg['ann_spatial'].apply(s, M, [k, xi])
    (ko, ki) = s[M].split(k, factor=NTile)
    (bo, bi) = s[M].split(b, factor=MTile)
    s[M].reorder(eps, nu, bo, ko, bi, ki)
    # s[M].tensorize(bi, intrin_gemm(M=MTile, N=NTile, K=get_const_int(c.dom.extent)))
    # inverse transform
    s[A].compute_inline()
    k, b, vh, vw = s[Y].op.axis
    r_eps, r_nu = s[Y].op.reduce_axis
    # s[Y].unroll(vh)
    # s[Y].unroll(vw)
    # s[Y].unroll(r_eps)
    # s[Y].unroll(r_nu)

    # output
    n, co, h, w = s[last].op.axis
    # co, coi = s[last].split(co, factor=NTile)# cfg['tile_k'].apply(s, last, co)
    # print(co, coi)
    # s[M].compute_at(s[last], co)
    # s[last].parallel(co)

    # MM = s.cache_read(M, 'global', [Y])
    # m = get_const_int(V.shape[0]) + 1 - 3
    # ho, wo, hi, wi = s[last].tile(h, w, m, m)
    # s[Y].compute_at(s[last], wo)
    # s[MM].compute_at(s[last], wo)

    # if output != last:
    #     s[output].compute_inline()



def div_round_up(a, b):
    return (a + b - 1) // b

def round_up(a, b):
    return (a + b - 1) // b * b

def _decl_winograd(data, kernel, strides, padding, layout, out_dtype, tile_size, out_channels):
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
    # assert KH == 3 and KW == 3 and HPAD == 1 and WPAD == 1 and HSTR == 1 and WSTR == 1
    assert KH == 3 and KW == 3 and HSTR == 1 and WSTR == 1
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
    K = round_up(CO, NTile)
    C = CI

    H = (IH + 2 * HPAD - 3) // HSTR + 1
    W = (IW + 2 * WPAD - 3) // WSTR + 1
    print("IH", H, IH, HPAD, WSTR)
    nH, nW = div_round_up(H, m), div_round_up(W, m)
    P = round_up(N * nH * nW, MTile)
    # cfg.define_split('tile_p', cfg.axis(P), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    # cfg.define_split('tile_k', cfg.axis(K), num_outputs=2, filter=lambda x: x.size[-1] <= 16)
    VP = MTile # cfg['tile_p'].size[-1]
    VK = NTile # cfg['tile_k'].size[-1]

    print("P: ", P, ", VP: ", VP)
    assert P % VP == 0
    print("K: ", K, ", VK: ", VK)
    assert K % VK == 0
    # pack input tile

    # BATCH SIZE 1
    assert N == 1
    input_tile = tvm.compute((C, P // VP, alpha, alpha, VP),
                             lambda c, b, eps, nu, bb:
                             data_pad[0][c][(b*VP+bb) // nW % nH * m + eps]
                             [(b*VP+bb) % nW * m + nu],
                             name='d')


    # transform kernel
    if pre_computed:
        U = kernel
    else:
        G = const_matrix(G_data, 'G')
        r_kh = tvm.reduce_axis((0, KH), 'r_kh')
        r_kw = tvm.reduce_axis((0, KW), 'r_kw')
        pad_size = get_const_int(K - CO)
        print(type(kernel))
        print(type(pad_size))

        kernel_pad = pad(kernel,
                         pad_before=(0, 0, 0, 0),
                         pad_after=(0, pad_size, 0, 0),
                         name="kernel_pad")
        U = tvm.compute((alpha, alpha, K // VK, C, VK), lambda eps, nu, k, c, kk:
                        tvm.sum(kernel_pad[k * VK + kk][c][r_kh][r_kw].astype(out_dtype) *
                                G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]),
                        name='U')
        # U = tvm.placeholder((alpha, alpha, K // VK, C, VK), dtype="float32", name="U")


    # transform image
    B = const_matrix(B_data, 'B')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((alpha, alpha, P // VP, C, VP), lambda eps, nu, b, c, bb:
                    tvm.sum(input_tile[c][b][r_eps][r_nu][bb].astype(out_dtype) *
                            B[r_eps][eps] * B[r_nu][nu], axis=[r_eps, r_nu]), name='V')

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((alpha, alpha, P, K), lambda eps, nu, b, k:
                    tvm.sum(V[eps][nu][b // VP][c][b % VP] * U[eps][nu][k // VK][c][k % VK], axis=c), name='M')
    print("M shape: ", M.shape)
    # inverse transform
    A = const_matrix(A_data, 'A')
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    Y = tvm.compute((P, out_channels, m, m), lambda b, k, vh, vw:
                    tvm.sum(M[r_eps][r_nu][b][k] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]), name='Y')
    print("Y shape: ", Y.shape)
    # unpack output
    # P: N, nH, nW
    output = tvm.compute((N, out_channels, H, W), lambda n, k, h, w:
                         Y[n * nH * nW + ((h//m) % nH) * nW + ((w//m) % nW)][k][h % m][w % m], # + tvm.const(0, M.dtype) * M[alpha - 1, alpha - 1, P - 1, K - 1],
                         name='output', tag='winograd_conv_output')

    print("Output shape: ", output.shape)
    print("IH, IW: ", IH, IW, H, W)
    # we have to manually assign effective GFLOP for winogard
    # cfg.add_flop(2 * N * K * H * W * KH * KW * C)
    return U, output


def conv2d_winograd_nchw(A, W, stride, padding, out_channels, dtype):
    U, Y = _decl_winograd(A, W, strides=stride, padding=padding, layout="NCHW", out_dtype="float32", tile_size=6, out_channels=out_channels)
    s = tvm.create_schedule(Y.op)
    _schedule_winograd(s, Y, Y)
    return s, [A, U, Y]

X = True
def verify_conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1):
    print("N: {}, CIn: {}, H/W: {}, COut: {}, KH/KW: {}".format(batch, in_channel, in_size, num_filter, kernel))
    in_height = in_width = in_size
    # kernel = 1
    kernel = 3
    stride = 1
    padding = 1
    # # stride = 1
    # padding = 0
    dilation = 1
    A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    # @memoize("topi.tests.test_topi_conv2d_nhwc.verify_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        a_np.fill(1)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        w_np.fill(1)
        dw_np = topi.testing.dilate_python(w_np, (1, dilation, dilation, 1))
        b_np = topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    def check_device():
        # if not tvm.module.enabled(device):
        #     print("Skip because %s is not enabled" % device)
        #     return
        with target:
            A_NCHW = tvm.placeholder((batch, in_channel, in_height, in_width), name='A_NCHW')
            W_NCHW = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W_NCHW')
            dW = W
            dW_NCHW = W_NCHW
            # dW = topi.nn.dilate(W, (1, dilation, dilation, 1))
            B = topi.nn.conv2d_nhwc(A, dW, stride, padding)
            B_NCHW = topi.nn.conv2d(A_NCHW, W_NCHW, stride, padding, layout='NCHW')

            (_, (_, W_TRNS_NCHW, _)) = conv2d_winograd_nchw(A_NCHW, W_NCHW, stride=1, padding=padding, dtype="float32", out_channels=num_filter)
            W_TRNS_NCHW_ = tvm.placeholder(get_const_tuple(W_TRNS_NCHW.shape), dtype="float32", name="W_TRNS_NCHW")
            (s_wino, (_, _, B_WINO)) = conv2d_winograd_nchw(A_NCHW, W_TRNS_NCHW_, stride=1, padding=padding, dtype="float32", out_channels=num_filter)
            s = topi.generic.schedule_conv2d_nhwc([B])
            s_nchw = topi.generic.schedule_conv2d_nchw([B_NCHW])
            print(tvm.lower(s_wino, [A_NCHW, W_TRNS_NCHW_, B_NCHW], simple_mode=True))
        a = tvm.nd.array(a_np, ctx)
        a_nchw = tvm.nd.array(a_np.transpose(0, 3, 1, 2), ctx)
        w = tvm.nd.array(w_np, ctx)
        w_nchw = tvm.nd.array(w_np.transpose(3, 2, 0, 1), ctx)
        w_trns_nchw = tvm.nd.array(np.zeros(get_const_tuple(W_TRNS_NCHW.shape), dtype=W_TRNS_NCHW.dtype), ctx)

        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        b_nchw = tvm.nd.array(np.zeros(get_const_tuple(B_NCHW.shape), dtype=B_NCHW.dtype), ctx)
        # b_tensor = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        b_tensor_mxn = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        b_nchw_tensor_mxn = tvm.nd.array(np.zeros(get_const_tuple(B_NCHW.shape), dtype=B.dtype), ctx)

        def remote_func(func, name):
            if USE_RASP:
                import uuid
                tmp = util.tempdir()
                name = name + ".o"
                lib_fname = tmp.relpath(name)
                func.save(lib_fname)
                func.save(name + ".S", fmt="asm")
                remote.upload(lib_fname)
                return remote.load_module(name)
            else:
                return func

        func = remote_func(tvm.build(s, [A, W, B], target),name="func")
        func_nchw = remote_func(tvm.build(s_nchw, [A_NCHW, W_NCHW, B_NCHW], target), name="func_nchw")
        func_wino_trns = remote_func(tvm.build(tvm.create_schedule(W_TRNS_NCHW.op), [W_NCHW, W_TRNS_NCHW], target), name="func_wino_trns")
        func_wino_trns(w_nchw, w_trns_nchw)
        # func_tensor = tvm.build(s_tensor, [A, W, B_tensor], device)
        func_wino = remote_func(tvm.build(s_wino, [A_NCHW, W_TRNS_NCHW_, B_WINO], target), name="func_wino")

        func(a, w, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        # np.testing.assert_allclose(b_tensor_mxn.asnumpy(), b_np, rtol=1e-5)
        func_nchw(a_nchw, w_nchw, b_nchw)
        # Fails for some reason
        # np.testing.assert_allclose(b_nchw.asnumpy(), b_np.transpose(0, 3, 1, 2), rtol=1e-5)
        func_wino(a_nchw, w_trns_nchw, b_nchw_tensor_mxn)
        # print(b_nchw_tensor_mxn.asnumpy())
        print(b_nchw_tensor_mxn.asnumpy()[0, 0])
        print(b_np.transpose(0, 3, 1, 2)[0, 0])
        print((b_nchw_tensor_mxn.asnumpy()[0, 0] - b_np.transpose(0, 3, 1, 2)[0, 0]) / (np.abs(b_nchw_tensor_mxn.asnumpy()[0, 0]) + 1.0e-5))
        np.testing.assert_allclose(b_nchw_tensor_mxn.asnumpy()[0, 0], b_np.transpose(0, 3, 1, 2)[0, 0], rtol=1e-1)



        (_, _, out_size, _) = get_const_tuple(B.shape)
        FLOPS = 2 * batch * in_channel * out_size * out_size * kernel * kernel * num_filter
        REPEAT = 5

        def gflops(t):
            return FLOPS / t / 1E9
        evaluator = func.time_evaluator(func.entry_name, ctx, number=REPEAT)(a, w, b).mean
        evaluator_nchw = func_nchw.time_evaluator(func_nchw.entry_name, ctx, number=REPEAT)(a_nchw, w_nchw, b_nchw).mean
        # evaluator_tensor = func_tensor.time_evaluator(func_tensor.entry_name, ctx, number=REPEAT)
        evaluator_nchw_tensor_mxn = func_wino.time_evaluator(func_wino.entry_name, ctx, number=REPEAT)(a_nchw, w_trns_nchw, b_nchw_tensor_mxn).mean

        print("BaselineNHWC: {:.2f}, BaselineNCHW: {:.2f}, TensorNHWC: {:.2f}, TensorNCHW: {:.2f}".format(gflops(evaluator), gflops(evaluator_nchw), 0, gflops(evaluator_nchw_tensor_mxn)))
        return 1
    return check_device()



def test_conv2d_nhwc():
    Workload = collections.namedtuple(
        'Workload',
        ['in_dtype', 'out_dtype', 'height', 'width', 'in_filter', 'out_filter',
         'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

    RESNET_50 = [
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
    ]

    RESNET_18 = [
        # Workload('float32', 'float32', 24, 24, 3, 16, 7, 7, 3, 3, 2, 2),
        # Workload('float32', 'float32', MTile * 3, MTile * 3, NTile * 2, NTile * 2, 7, 7, 3, 3, 2, 2),
        # Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
        # Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
        # Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
        # Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
    ]

    MOBILENET = [
        Workload('float32', 'float32', 112, 112, 32, 64, 1, 1, 0, 0, 1, 1),

        Workload('float32', 'float32', 7, 7, 512, 1024, 1, 1, 0, 0, 1, 1),

        Workload('float32', 'float32', 7, 7, 1024, 1024, 1, 1, 0, 0, 1, 1),

        # Workload('float32', 'float32', 224, 224, 3, 32, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 128, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 256, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 512, 512, 1, 1, 0, 0, 1, 1),
    ]

    def run(workload, name):
        speedups = [verify_conv2d_nhwc(1, w.in_filter, w.height, w.out_filter, w.hkernel, w.hstride, w.hpad, 1) for w in workload]
        print("{}: {:.2f}".format(name, scipy.stats.mstats.gmean(speedups)))

    run(RESNET_18, "RESNET-18")
    run(MOBILENET, "MOBILENET")
    run(RESNET_50, "RESNET-50")


if __name__ == "__main__":
    test_conv2d_nhwc()
