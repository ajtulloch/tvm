import logging
import sys

import numpy as np
import tvm

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
def matmul(M, N, K, dtype):
    A = tvm.placeholder((M, K), name='A', dtype=dtype)
    # B = tvm.placeholder((K, N), name='B', dtype=dtype)

    APanel = tvm.compute(
        (M / MTile, K, MTile), lambda mtile, k, m: A[m + mtile * MTile, k], name='APanel')
    # BPanel = tvm.compute(
    #     (N / NTile, K, NTile), lambda ntile, k, n: B[k, n + ntile * NTile], name='BPanel')
    # APanel = tvm.placeholder(
    #     (M / MTile, K, MTile), dtype=dtype, name='APanel')
    BPanel = tvm.placeholder(
        (N / NTile, K, NTile), dtype=dtype, name='BPanel')

    print("APanel, ", APanel.shape)
    print("BPanel, ", BPanel.shape)
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute(
        (M, N),
        lambda m, n: tvm.sum(
            APanel[m / MTile, k, m % MTile] * BPanel[n / NTile, k, n % NTile],
            axis=[k]),
        name='C')
    s = tvm.create_schedule(C.op)
    # x, y, z = BPanel.op.axis
    # s[BPanel].vectorize(z)
    # x, y, z = APanel.op.axis
    # s[APanel].unroll(z)

    assert M % MTile == 0
    assert N % NTile == 0
    MTileUnroll = 1
    for i in range(M, 0, -1):
        if M % (MTile * i) == 0:
            MTileUnroll = i
            break
    NTileUnroll = 1
    for i in range(N, 0, -1):
        if N % (NTile * i) == 0:
                NTileUnroll = i
                break
    (x, y) = C.op.axis
    x_candidates = [(M / (i * MTile), i * MTile) for i in range(1, MTileUnroll + 1) if M % (i * MTile) == 0]
    y_candidates = [(N / (i * NTile), i * NTile) for i in range(1, NTileUnroll + 1) if N % (i * NTile) == 0]

    cfg = autotvm.get_config()
    cfg.define_split("tile_x", x, num_outputs=2, policy='candidate', candidate=x_candidates)
    cfg.define_split("tile_y", y, num_outputs=2, policy='candidate', candidate=y_candidates)
    cfg.define_knob("tile_in_k", [0, 1])
    cfg.define_knob("A_panel_tile", [0, 1])
    cfg.define_knob("A_panel_unroll", [0, 1])
    cfg.define_knob("A_panel_compute_inner", [0, 1])

    # cfg.define_split("tile_x", x, num_outputs=2, policy='candidate', candidate=[(i * MTile, M / (i * MTile)) for i in range(1, MTileUnroll + 1)])
    # cfg.define_split("tile_y", y, num_outputs=2, policy='candidate', candidate=[(i * NTile, N / (i * NTile)) for i in range(1, NTileUnroll + 1)])

    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)

    xii, xiii = s[C].split(xi, factor=MTile)
    yii, yiii = s[C].split(yi, factor=NTile)
    tile_in_k = K >= 2 * KTile and MTileUnroll > 1 and cfg['tile_in_k'].val
    if cfg['A_panel_tile']:
        if cfg['A_panel_compute_inner']:
            s[APanel].compute_at(s[C], xo)
        else:
            s[APanel].compute_at(s[C], xii)

    if cfg['A_panel_unroll']:
        s[APanel].unroll(s[APanel].op.axis[2])
    if tile_in_k:
        # k, = C.op.reduce_axis
        ko, ki = s[C].split(k, factor=KTile)
        s[C].reorder(yo, xo, ko, yii, xii, xiii, yiii, ki)
    else:
        s[C].reorder(yo, xo, yii, xii, xiii, yiii)

    s[C].tensorize(xiii, intrin_gemm(M=MTile, N=NTile, K=KTile if tile_in_k else K))
    # print(tvm.lower(s, [A, B, C], simple_mode=True))
    return s, [A, BPanel, C]

N, L, M = 66, 128, 256
task = autotvm.task.create(matmul, args=(N, L, M, 'float32'), target=tvm.target.rasp())
print(task.config_space)

# logging config (for printing tuning log to screen)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# use local cpu, measure 5 times for every config to reduce variance
measure_option = autotvm.measure_option(mode='rpc',
                                        rpc_tracker_addr=('localhost', 9190),
                                        rpc_device_key="rpi",
                                        number=3)

# # begin tuning, log records to file `matmul.log`
# tuner = autotvm.tuner.RandomTuner(task)
# tuner.tune(n_trial=200,
#            measure_option=measure_option,
#            callbacks=[autotvm.callback.log_to_file('matmul.log')])
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(n_trial=1000,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('matmul_xgb.log')])



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
