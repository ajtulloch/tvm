"""Example code to do convolution."""
import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple, get_const_int
from topi import tag

target = 'llvm -mcpu=core-avx2'

BITCODE_PATHS = [
    "gemmMxN__avx2.bc"
]

@tvm.register_func("tvm_callback_llvm_bitcode_path")
def bitcode_paths():
    global BITCODE_PATHS
    return BITCODE_PATHS


MTile = 4
MMTile = 4
NTile = 24

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
                "sgemm_only_4x24__avx2",
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
                "sgemm_reset_4x24__avx2",
                irb.buffer_ptr(cc),
                cc.elem_offset,
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()

        def update():
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "sgemm_update_4x24__avx2",
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

def conv2d_nhwc_tensor_mxn(A, W_, stride, padding):
    dtype = A.dtype
    (N, IH, IW, CIn) = get_const_tuple(A.shape)
    (KH, KW, CIn_, COut) = get_const_tuple(W_.shape)
    assert CIn == CIn_
    assert stride == 1, stride
    assert padding == 0


    OH = (IH + 2*padding - KH) // stride + 1
    OW = (IW + 2*padding - KW) // stride + 1

    def div_round_up(a, b):
        return (a + b - 1) // b

    def round_up(a, b):
        return (a + b - 1) // b * b

    # We need A_tile_shape to divide MMTile, not MTile.

    # [N * H * W // TILE, KH * KW * C, TILE]
    A_tile_shape = (div_round_up(N * OH * OW, MTile), CIn * KH * KW, MTile)

    def _A_tile(tile_idx, channel_idx, tile_elem):
        spatial_idx = tile_elem + tile_idx * MTile

        n = spatial_idx // OH // OW
        c_in = channel_idx % CIn
        c_kw = channel_idx // CIn % KW
        c_kh = channel_idx // CIn // KW
        h_in = spatial_idx // OW % OH*stride + c_kh
        w_in = spatial_idx % OW*stride + c_kw
        return tvm.select(
            spatial_idx < N * OH * OW, #tvm.all(n < N, h_in < IH, w_in < IW),
            A[n, h_in, w_in, c_in],
            0.0)

    A_tile = tvm.compute(A_tile_shape, _A_tile, name="A_tile")

    W_tile_shape = (div_round_up(COut, NTile), CIn * KH * KW, NTile)

    def _W_tile(tile_idx, channel_idx, tile_elem):
        c_out = tile_elem + tile_idx * NTile
        c_in = channel_idx % CIn
        c_kw = channel_idx // CIn % KW
        c_kh = channel_idx // CIn // KW
        return tvm.select(c_out < COut, W_[c_kh, c_kw, c_in, c_out], 0.0)

    W_tile = tvm.compute(W_tile_shape, _W_tile, name="W_tile")

    k = tvm.reduce_axis((0, CIn * KH * KW), name='k')

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

    return unpacked_nhwc

def schedule_conv2d_nhwc_tensor_mxn(outs):
    s = tvm.create_schedule([x.op for x in outs])
    output_op = outs[0].op
    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            else: # inject custom schedule
                if len(op.axis) == 4: # schedule bias + bn + relu
                    n, h, w, c = op.axis
                    fused = s[op].fuse(n, h, w)
                    s[op].parallel(fused)
                    s[op].vectorize(c)
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        if 'conv2d_nhwc' in op.tag:
            output = op.output(0)

            A_W_product = op.input_tensors[0]
            A_tile = A_W_product.op.input_tensors[0]
            x, y, z = A_tile.op.axis
            s[A_tile].unroll(z)
            # s[A_tile].reorder(x, z, y)
            # s[A_tile].vectorize(y)
            # xo, xi = s[A_tile].split(x, factor=4)
            # s[A_tile].reorder(xo, y, xi, z)
            W_tile = A_W_product.op.input_tensors[1]
            x, y, z = W_tile.op.axis
            # s[W_tile].reorder(x, z, y)
            s[W_tile].unroll(z)
            M = get_const_int(A_W_product.op.axis[0].dom.extent)
            assert M % MTile == 0
            MTileUnroll = 1
            for i in range(8, 0, -1):
                if M % (MTile * i) == 0:
                    MTileUnroll = i
                    break

            xo, yo, xi, yi = s[A_W_product].tile(A_W_product.op.axis[0], A_W_product.op.axis[1], MTile * MTileUnroll, NTile)
            s[A_W_product].reorder(xo, yo, xi, yi)
            s[A_tile].compute_at(s[A_W_product], xo)
            xii, xiii = s[A_W_product].split(xi, factor=MTile)
            k, = s[A_W_product].op.reduce_axis
            s[A_W_product].tensorize(xiii, intrin_gemm(M=MTile, N=NTile, K=get_const_int(k.dom.extent)))
            # s[A_W_product].unroll(xii)
            n, h, w, c = op.axis
            fused = s[op].fuse(n, h, w)
            # s[op].parallel(fused)
            s[op].vectorize(c)

    traverse(output_op)
    return s

X = True
def verify_conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1):
    print("N: {}, CIn: {}, H/W: {}, COut: {}, KH/KW: {}".format(batch, in_channel, in_size, num_filter, kernel))
    in_height = in_width = in_size
    # kernel = 1
    stride = 1
    padding = 0
    dilation = 1
    A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_conv2d_nhwc.verify_nhwc")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, dilation, dilation, 1))
        b_np = topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        with tvm.target.create(device):
            A_NCHW = tvm.placeholder((batch, in_channel, in_height, in_width), name='A_NCHW')
            W_NCHW = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W_NCHW')
            dW = W
            dW_NCHW = W_NCHW
            # dW = topi.nn.dilate(W, (1, dilation, dilation, 1))
            B = topi.nn.conv2d_nhwc(A, dW, stride, padding)
            B_NCHW = topi.nn.conv2d(A_NCHW, W_NCHW, stride, padding, layout='NCHW')

            # B_tensor = conv2d_nhwc_tensor_1x1(A, dW, stride, padding)
            B_tensor_mxn = conv2d_nhwc_tensor_mxn(A, dW, stride, padding)
            # s_tensor = schedule_conv2d_nhwc_tensor_1x1([B_tensor])
            s_tensor_mxn = schedule_conv2d_nhwc_tensor_mxn([B_tensor_mxn])

            global X
            if X:
                print(tvm.lower(s_tensor_mxn, [A, W, B_tensor_mxn], simple_mode=True))
            X = False


            s = topi.generic.schedule_conv2d_nhwc([B])
            s_nchw = topi.generic.schedule_conv2d_nchw([B_NCHW])
            # print(tvm.lower(s_nchw, [A_NCHW, W_NCHW, B_NCHW], simple_mode=True))
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        a_nchw = tvm.nd.array(a_np.transpose(0, 3, 1, 2), ctx)
        w = tvm.nd.array(w_np, ctx)
        w_nchw = tvm.nd.array(w_np.transpose(3, 2, 0, 1), ctx)

        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        b_nchw = tvm.nd.array(np.zeros(get_const_tuple(B_NCHW.shape), dtype=B_NCHW.dtype), ctx)
        # b_tensor = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        b_tensor_mxn = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A, W, B], device)
        func_nchw = tvm.build(s_nchw, [A_NCHW, W_NCHW, B_NCHW], device)
        # func_tensor = tvm.build(s_tensor, [A, W, B_tensor], device)
        func_tensor_mxn = tvm.build(s_tensor_mxn, [A, W, B_tensor_mxn], device)
        func(a, w, b)
        func_nchw(a_nchw, w_nchw, b_nchw)
        # func_tensor(a, w, b_tensor)
        func_tensor_mxn(a, w, b_tensor_mxn)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        np.testing.assert_allclose(b_nchw.asnumpy(), b_np.transpose(0, 3, 1, 2), rtol=1e-5)
        # np.testing.assert_allclose(b_tensor.asnumpy(), b_np, rtol=1e-5)
        np.testing.assert_allclose(b_tensor_mxn.asnumpy(), b_np, rtol=1e-5)

        FLOPS = 2 * batch * in_channel * in_size * in_size * kernel * kernel * num_filter
        REPEAT = 10

        def gflops(t):
            return FLOPS / t / 1E9
        evaluator = func.time_evaluator(func.entry_name, ctx, number=REPEAT)(a, w, b).mean
        evaluator_nchw = func_nchw.time_evaluator(func_nchw.entry_name, ctx, number=REPEAT)(a_nchw, w_nchw, b_nchw).mean
        # evaluator_tensor = func_tensor.time_evaluator(func_tensor.entry_name, ctx, number=REPEAT)
        evaluator_tensor_mxn = func_tensor_mxn.time_evaluator(func_tensor_mxn.entry_name, ctx, number=REPEAT)(a, w, b_tensor_mxn).mean

        print("BaselineNHWC: {:.2f}, BaselineNCHW: {:.2f}, TensorNHWC: {:.2f}".format(gflops(evaluator), gflops(evaluator_nchw), gflops(evaluator_tensor_mxn)))
        return evaluator_nchw / evaluator_tensor_mxn
    return check_device(target)



def test_conv2d_nhwc():
        # ResNet18 worklaods
    speedups = [
        # verify_conv2d_nhwc(1, 3, 224, 64, 7, 2, 3),
        # verify_conv2d_nhwc(1, 64, 56, 64, 3, 1, 1),
        # verify_conv2d_nhwc(1, 64, 56, 64, 1, 1, 0),
        # verify_conv2d_nhwc(1, 64, 56, 128, 3, 2, 1),
        # verify_conv2d_nhwc(1, 64, 56, 128, 1, 2, 0),
        # verify_conv2d_nhwc(1, 128, 28, 128, 3, 1, 1),
        # verify_conv2d_nhwc(1, 128, 28, 256, 3, 2, 1),
        # verify_conv2d_nhwc(1, 128, 28, 256, 1, 2, 0),
        # verify_conv2d_nhwc(1, 256, 14, 256, 3, 1, 1),
        # verify_conv2d_nhwc(1, 256, 14, 512, 3, 2, 1),
        # verify_conv2d_nhwc(1, 256, 14, 512, 1, 2, 0),
        verify_conv2d_nhwc(1, 512, 7, 512, 3, 1, 1),
    ]
    import scipy.stats.mstats
    print("GEOMEAN: {:.2f}".format(scipy.stats.mstats.gmean(speedups)))

    # # Vgg16 workloads
    # verify_conv2d_nhwc(1, 128, 122, 128, 3, 1, 1)
    # # Super resolution workloads
    # verify_conv2d_nhwc(1, 1, 224, 64, 5, 1, 2)
    # verify_conv2d_nhwc(1, 64, 224, 64, 3, 1, 1)
    # verify_conv2d_nhwc(1, 64, 224, 32, 3, 1, 1)
    # verify_conv2d_nhwc(1, 32, 224, 9, 3, 1, 1)
    # # dilation = 2
    # verify_conv2d_nhwc(1, 128, 122, 128, 3, 1, 1, dilation=2)

    # verify_conv2d_nhwc(1, 256, 32, 128, 3, 1, "SAME")
    # verify_conv2d_nhwc(1, 128, 16, 128, 5, 2, "SAME")
    # verify_conv2d_nhwc(1, 128, 16, 256, 5, 2, "SAME")
    # verify_conv2d_nhwc(1, 64, 16, 256, 5, 2, "SAME")
    # verify_conv2d_nhwc(1, 256, 32, 256, 3, 1, "VALID")
    # verify_conv2d_nhwc(1, 256, 32, 256, 3, 1, "VALID")
    # verify_conv2d_nhwc(1, 128, 16, 128, 5, 2, "VALID")
    # verify_conv2d_nhwc(1, 128, 16, 256, 5, 2, "VALID")
    # # dilation = 2
    # verify_conv2d_nhwc(1, 256, 32, 256, 3, 1, "SAME", dilation=2)


if __name__ == "__main__":
    test_conv2d_nhwc()
