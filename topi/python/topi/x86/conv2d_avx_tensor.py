import tvm
import numpy as np
import topi
import topi.testing
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

    with tvm.build_config():
        return tvm.decl_tensor_intrin(C.op,
                                      intrin_func,
                                      binds={A: Ab, B: Bb, C: Cb})

def conv2d_nchw_tensor(A, W_, stride, padding, layout):
    assert layout == "NCHW"
    dtype = A.dtype

    (N, CIn, H, W) = get_const_tuple(A.shape)
    (COut, CIn_, kh, kw) = get_const_tuple(W_.shape)
    assert kh == 1
    assert kw == 1
    assert CIn == CIn_
    assert stride == 1, stride
    assert padding == 0

    def div_round_up(a, b):
        return (a + b - 1) / b

    # We need A_tile_shape to divide MMTile, not MTile.
    A_tile_shape = (div_round_up(N * H * W, MTile), CIn, MTile)

    def _A_tile(tile_idx, c, tile_elem):
        linear_idx = tile_elem + tile_idx * MTile
        w = linear_idx % W
        linear_idx /= W
        h = linear_idx % H
        linear_idx /= H
        n = linear_idx
        return tvm.select(n < N, A[n, c, h, w], 0.0)

    A_tile = tvm.compute(A_tile_shape, _A_tile, name="A_tile")
    W_tile_shape = (div_round_up(COut, NTile), CIn, NTile)

    def _W_tile(tile_idx, c, tile_elem):
        linear_idx = tile_elem + tile_idx * NTile
        c_out = linear_idx
        return tvm.select(linear_idx < COut, W_[linear_idx, c, 0, 0], 0.0)

    W_tile = tvm.compute(W_tile_shape, _W_tile, name="W_tile")

    k = tvm.reduce_axis((0, CIn), name='k')


    A_W_product = tvm.compute(
        (A_tile_shape[0] * MTile, W_tile_shape[0] * NTile),
        lambda m, n: tvm.sum(
            A_tile[m / MTile, k, m % MTile] * W_tile[n / NTile, k, n % NTile],
            axis=[k]),
        name='A_W_product')
    output_shape = (N, COut, H, W)
    def _unpack_output(n, c, h, w):
        m_idx = w + h * H + n * H * W
        return A_W_product[m_idx, c]
    unpacked_nchw = tvm.compute(
        output_shape,
        _unpack_output,
        name="A_W_product_NCHW",
        tag='conv2d_nchw_tensor')

    return (unpacked_nchw, A_W_product)



def schedule_conv2d_nchw_tensor(outs):
    s = tvm.create_schedule([x.op for x in outs])
    output_op = outs[0].op
    gemm_op = outs[1].op
    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        print("Computing op: %s" % op)
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
        # print(op.tag)
        if 'conv2d_nchw' in op.tag:
            output = op.output(0)

            A_W_product = op.input_tensors[0]
            A_tile = A_W_product.op.input_tensors[0]
            x, y, z = A_tile.op.axis
            s[A_tile].unroll(z)
            xo, xi = s[A_tile].split(x, factor=4)
            s[A_tile].reorder(xo, y, xi, z)
            W_tile = A_W_product.op.input_tensors[1]
            # x, y, z = W_tile.op.axis
            # s[W_tile].unroll(z)
            print(A_tile, W_tile, A_W_product)
            M = get_const_int(A_W_product.op.axis[0].dom.extent)
            assert M % MTile == 0
            MTileUnroll = 1
            for i in range(8, 0, -1):
                if M % (MTile * i) == 0:
                    MTileUnroll = i
                    break

            xo, yo, xi, yi = s[A_W_product].tile(A_W_product.op.axis[0], A_W_product.op.axis[1], MTile * MTileUnroll, NTile)
            s[A_W_product].reorder(xo, yo, xi, yi)
            # s[A_W_product].compute_root()
            xii, xiii = s[A_W_product].split(xi, factor=MTile)
            k, = s[A_W_product].op.reduce_axis
            print("K", k, k.dom.extent)
            s[A_W_product].tensorize(xiii, intrin_gemm(M=MTile, N=NTile, K=get_const_int(k.dom.extent)))
            # s[A_W_product].unroll(xii)
            n, h, w, c = op.axis
            fused = s[op].fuse(n, h, w)
            s[op].parallel(fused)
            s[op].vectorize(c)

    traverse(output_op)
    return s


def verify_conv2d_nchw(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    # @memoize("topi.tests.test_topi_conv2d_nchw.verify_conv2d_nchw")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        # a_np.fill(1)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        # w_np.fill(2)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        b_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            # dW = topi.nn.dilate(W, (1, 1, dilation, dilation))
            B, G = conv2d_nchw_tensor(A, W, stride, padding, layout='NCHW')
            C = B # topi.nn.relu(B)

            s1 = schedule_conv2d_nchw_tensor([B, G])
            s2 = schedule_conv2d_nchw_tensor([C, G])
            print(tvm.lower(s2, [A, W, B], simple_mode=True))

            B_baseline = topi.nn.conv2d(A, W, stride, padding, layout='NCHW')
            s_baseline = topi.generic.schedule_conv2d_nchw([B_baseline])
            print("Baseline", tvm.lower(s_baseline, [A, W, B], simple_mode=True))

            print(tvm.lower(s1, [A, W, B], simple_mode=True))
            print(tvm.lower(s_baseline, [A, W, B_baseline], simple_mode=True))

        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)
        with tvm.build_config(auto_unroll_max_step=1400,
                              unroll_explicit=(device != "cuda")):
            func1 = tvm.build(s1, [A, W, B], device, name="conv2d_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func2 = tvm.build(s2, [A, W, C], device, name="relu_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
            func1(a, w, b)
            func2(a, w, c)
            REPEAT = 100
            FLOPS = 2 * batch * in_channel * in_size * in_size * kernel * kernel * num_filter
            evaluator1 = func1.time_evaluator(func1.entry_name, ctx, number=REPEAT)
            evaluator2 = func2.time_evaluator(func2.entry_name, ctx, number=REPEAT)
            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
            np.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)
            print('Tensor1x1: %f' % (FLOPS / evaluator1(a, w, b).mean / 1E9))

            func1_baseline = tvm.build(s_baseline, [A, W, B], device, name="conv2d_baseline_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))

            func1_baseline(a, w, b)
            np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
            evaluator_baseline = func1_baseline.time_evaluator(func1_baseline.entry_name, ctx, number=REPEAT)
            print('Tensor1x1 Baseline: %f' % (FLOPS / evaluator_baseline(a, w, b).mean / 1E9))
            print("M, N, K: ", batch * in_size * in_size, num_filter, kernel * kernel * in_channel)
    for device in [target]:
        check_device(device)

def test_conv2d_nchw():
    # ResNet18 worklaods
    # verify_conv2d_nchw(1, 3, 224, 64, 7, 2, 3)
    # verify_conv2d_nchw(1, 64, 56, 64, 3, 1, 1)
    # verify_conv2d_nchw(1, 2, 1, 1, 1, 1, 0)
    # verify_conv2d_nchw(1, 32, 9, 7, 1, 1, 0)
    verify_conv2d_nchw(1, 128, 28, 256, 1, 1, 0)

    # verify_conv2d_nchw(1, 64, 56, 128, 3, 2, 1)
    # verify_conv2d_nchw(1, 64, 56, 128, 1, 2, 0)
    # verify_conv2d_nchw(1, 128, 28, 128, 3, 1, 1)
    # verify_conv2d_nchw(1, 128, 28, 256, 3, 2, 1)
        # verify_conv2d_nchw(1, 256, 14, 256, 3, 1, 1)
    # verify_conv2d_nchw(1, 256, 14, 512, 3, 2, 1)
    # verify_conv2d_nchw(1, 256, 14, 512, 1, 2, 0)
    # verify_conv2d_nchw(1, 512, 7, 512, 3, 1, 1)
    # # Vgg16 workloads
    # verify_conv2d_nchw(1, 128, 122, 128, 3, 1, 1)
    # # Super resolution workloads
    # verify_conv2d_nchw(1, 1, 224, 64, 5, 1, 2)
    # verify_conv2d_nchw(1, 64, 224, 64, 3, 1, 1)
    # verify_conv2d_nchw(1, 64, 224, 32, 3, 1, 1)
    # verify_conv2d_nchw(1, 32, 224, 9, 3, 1, 1)
    # # dilation = 2
    # verify_conv2d_nchw(1, 128, 122, 128, 3, 1, 1, dilation=2)

if __name__ == "__main__":
    test_conv2d_nchw()
