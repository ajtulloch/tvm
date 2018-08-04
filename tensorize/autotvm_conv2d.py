import logging
import sys

import numpy as np
import tvm
from topi.util import get_const_tuple, get_const_int

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

@autotvm.template
def conv2d(IH, IW, KH, KW, CIn, COut, dtype):

    N = 1
    padding = 0
    stride = 1
    OH = (IH + 2*padding - KH) // stride + 1
    OW = (IW + 2*padding - KW) // stride + 1

    def div_round_up(a, b):
        return (a + b - 1) // b

    def round_up(a, b):
        return (a + b - 1) // b * b


    tile_in_k = CIn * KH * KW >= 2 * KTile
    K = CIn * KH * KW if not tile_in_k else round_up(CIn * KH * KW, KTile)

    A = tvm.placeholder((1, IH, IW, CIn), dtype=dtype, name="A")
    # [N * H * W // TILE, KH * KW * C, TILE]
    A_tile_shape = (div_round_up(N * OH * OW, MTile), K, MTile)

    def _A_tile(tile_idx, channel_idx, tile_elem):
        spatial_idx = tile_elem + tile_idx * MTile

        n = spatial_idx // OH // OW
        c_in = channel_idx % CIn
        c_kw = channel_idx // CIn % KW
        c_kh = channel_idx // CIn // KW
        h_in = spatial_idx // OW % OH * stride + c_kh - padding
        w_in = spatial_idx % OW * stride + c_kw - padding
        conds = []
        if padding != 0 or N * OH * OW % MTile != 0:
            conds += [
                n < N,
                0 <= h_in,
                h_in < IH,
                0 <= w_in,
                w_in < IW,
            ]

        # if padding != 0 or N * OH * OW % MTile != 0:
        #     conds += [n < N, h_in >= 0, h_in < IH, w_in >= 0, w_in < IW]
        if tile_in_k and CIn * KH * KW % KTile != 0:
            conds += [channel_idx < CIn * KH * KW]

        return tvm.select(tvm.all(*conds), A[n, h_in, w_in, c_in], 0.0) if conds else A[n, h_in, w_in, c_in]

    A_tile = tvm.compute(A_tile_shape, _A_tile, name="A_tile")

    W_tile_shape = (div_round_up(COut, NTile), K, NTile)

    # def _W_tile(tile_idx, channel_idx, tile_elem):
    #     c_out = tile_elem + tile_idx * NTile
    #     c_in = channel_idx % CIn
    #     c_kw = channel_idx // CIn % KW
    #     c_kh = channel_idx // CIn // KW
    #     conds = []
    #     if COut % NTile != 0:
    #         conds += [c_out < COut]
    #     if tile_in_k and CIn * KH * KW % KTile != 0:
    #         conds += [channel_idx < CIn * KH * KW]

    #     return tvm.select(tvm.all(*conds), W_[c_kh, c_kw, c_in, c_out], 0.0) if conds else W_[c_kh, c_kw, c_in, c_out]

    W_tile = tvm.placeholder(W_tile_shape, dtype=dtype, name="W_tile")

    k = tvm.reduce_axis((0, K), name='k')

    A_W_product = tvm.compute(
        (A_tile_shape[0] * MTile, W_tile_shape[0] * NTile),
        lambda m, n: tvm.sum(
            A_tile[m / MTile, k, m % MTile] * W_tile[n / NTile, k, n % NTile],
            axis=[k]),
        name='A_W_product')

    output_shape = (N, OH, OW, COut)

    def _unpack_output(n, h, w, c):
        m_idx = w + h * OW + n * OH * OW
        return A_W_product[m_idx, c] + tvm.const(0, A_W_product.dtype) * A_W_product[A_W_product.shape[0] - 1, A_W_product.shape[1] - 1]

    unpacked_nhwc = tvm.compute(
        output_shape,
        _unpack_output,
        name="A_W_product_NHWC",
        tag='conv2d_nhwc_tensor')

    s = tvm.create_schedule(unpacked_nhwc.op)
    x, y, z = A_tile.op.axis
    # zo, zi = s[A_tile].split(z, 8)
    # s[A_tile].reorder(x, zo, y, zi)
    # s[A_tile].vectorize(zi)
    s[A_tile].unroll(z)
    # s[A_tile].reorder(x, z, y)

    # xo, xi = s[A_tile].split(x, factor=4)
    # s[A_tile].reorder(xo, y, xi, z)

    M = get_const_int(A_W_product.op.axis[0].dom.extent)
    N = get_const_int(A_W_product.op.axis[1].dom.extent)
    assert M % MTile == 0
    MTileUnroll = 1
    for i in range(24, 0, -1):
        if M % (MTile * i) == 0:
            MTileUnroll = i
            break
    NTileUnroll = 1
    for i in range(6, 0, -1):
        if N % (NTile * i) == 0:
            NTileUnroll = i
            break

    K = get_const_int(A_W_product.op.reduce_axis[0].dom.extent)
    x_candidates = [(M / (i * MTile), i * MTile) for i in range(1, MTileUnroll + 1) if M % (i * MTile) == 0]
    y_candidates = [(N / (i * NTile), i * NTile) for i in range(1, NTileUnroll + 1) if N % (i * NTile) == 0]

    (x, y) = A_W_product.op.axis
    cfg = autotvm.get_config()
    cfg.define_split("tile_x", cfg.axis(x), num_outputs=2, policy='candidate', candidate=x_candidates)
    cfg.define_split("tile_y", cfg.axis(y), num_outputs=2, policy='candidate', candidate=y_candidates)
    cfg.define_knob("tile_in_k", [0, 1])
    cfg.define_knob("A_tile_compute_location", [0, 1, 2, 3])
    cfg.define_knob("A_tile_unroll", [0, 1])
    cfg.define_knob("output_fuse_or_tile", [0, 1])
    cfg.define_knob("output_vectorize", [0, 1])
    xo, xi = cfg["tile_x"].apply(s, A_W_product, x)
    yo, yi = cfg["tile_y"].apply(s, A_W_product, y)
    xii, xiii = s[A_W_product].split(xi, factor=MTile)
    yii, yiii = s[A_W_product].split(yi, factor=NTile)
    tile_in_k = K >= 2 * KTile and MTileUnroll > 1 and cfg['tile_in_k'].val

    def reorder_apply(self_, sch, op, axes, extra_axes):
        if len(axes) == len(self_.perm):
            new_order = [axes[i] for i in self_.perm]
        else:
            new_order = [axes[i] for i in self_.perm if i < len(axes)]
        new_order += extra_axes
        sch[op].reorder(*new_order)
        return new_order
    cfg.define_reorder("A_W_reorder_k", [xo, xii, yo, yii, ko], policy="all")
    cfg.define_reorder("A_W_reorder_no_k", [xo, xi, yo, yi], policy="all")
    if tile_in_k:

        k, = A_W_product.op.reduce_axis
        ko, ki = s[A_W_product].split(k, factor=KTile)
        reorder_apply(cfg["A_W_reorder_k"], s, A_W_product, [xo, xii, yo, yii, ko], extra_axes=[xiii, yiii, ki])
        # s[A_W_product].reorder(yo, xo, ko, yii, xii, xiii, yiii, ki)
        if cfg['A_tile_compute_location'].val == 1:
            s[A_tile].compute_at(s[A_W_product], xo)
        if cfg['A_tile_compute_location'].val == 2:
            s[A_tile].compute_at(s[A_W_product], xii)
        if cfg['A_tile_unroll'].val == 1:
            s[A_tile].unroll(A_tile.op.axis[2])
    else:
        reorder_apply(cfg["A_W_reorder_no_k"], s, A_W_product, [xo, xii, yo, yii], extra_axes=[xiii, yiii])
        # s[A_W_product].reorder(yo, xo, yii, xii, xiii, yiii)
        if cfg['A_tile_compute_location'].val == 1:
            s[A_tile].compute_at(s[A_W_product], xo)
        if cfg['A_tile_compute_location'].val == 2:
            s[A_tile].compute_at(s[A_W_product], xii)

    s[A_W_product].tensorize(xiii, intrin_gemm(M=MTile, N=NTile, K=KTile if tile_in_k else K))
    # s[A_W_product].unroll(xii)
    n, h, w, c = unpacked_nhwc.op.axis
    if cfg["output_fuse_or_tile"].val == 0:
        fused = s[unpacked_nhwc].fuse(n, h, w)
        if cfg["output_vectorize"].val:
            s[unpacked_nhwc].vectorize(c)
    if cfg["output_fuse_or_tile"].val == 1:
        nh = s[unpacked_nhwc].fuse(n, h)
        (nho, wo, nhi, wi) = s[unpacked_nhwc].tile(nh, w, 8, 8)
        if cfg["output_vectorize"].val:
            s[unpacked_nhwc].vectorize(c)
    # print(tvm.lower(s, [A, W_tile, unpacked_nhwc], simple_mode=True))
    cfg.add_flop(2 * 1 * CIn * get_const_int(unpacked_nhwc.shape[1]) * get_const_int(unpacked_nhwc.shape[2]) * KH * KW * COut)
    return s, [A, W_tile, unpacked_nhwc]

S = 26
K = 3
C = 256
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

    # use local cpu, measure 5 times for every config to reduce variance
    measure_option = autotvm.measure_option(
        measure_func=autotvm.use_rpc("rpi", host="localhost", port=9190),
        number=3)

    # # begin tuning, log records to file `matmul.log`
    # tuner = autotvm.tuner.RandomTuner(task)
    # tuner.tune(n_trial=200,
    #            measure_option=measure_option,
    #            callbacks=[autotvm.callback.log_to_file('matmul.log')])
    tuner = autotvm.tuner.XGBTuner(task, feature_type='knob')
    tuner.tune(n_trial=200,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file('conv2d_xgb_segmentation_tensor_reorder_{i}_{w.space}_{w.kernel}_{w.input_channel}_{w.output_channel}.log'.format(i=i, w=w))])


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
