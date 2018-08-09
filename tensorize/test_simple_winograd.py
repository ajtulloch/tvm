import simple_winograd
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_int, get_const_tuple
import collections

target = tvm.target.create('llvm -mcpu=core-avx2')
ctx = tvm.context('llvm -mcpu=core-avx2', 0)

def verify_conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1):
    print("N: {}, CIn: {}, H/W: {}, COut: {}, KH/KW: {}".format(batch, in_channel, in_size, num_filter, kernel))
    in_height = in_width = in_size
    kernel = 3
    stride = 1
    padding = 1
    dilation = 1
    A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        # a_np.fill(1)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        # w_np.fill(1)
        dw_np = topi.testing.dilate_python(w_np, (1, dilation, dilation, 1))
        b_np = topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    with target:
        A_NCHW = tvm.placeholder((batch, in_channel, in_height, in_width), name='A_NCHW')
        W_NCHW = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W_NCHW')
        dW = W
        dW_NCHW = W_NCHW
        B = topi.nn.conv2d_nhwc(A, dW, stride, padding)
        B_NCHW = topi.nn.conv2d(A_NCHW, W_NCHW, stride, padding, layout='NCHW')
        cfg = None
        B_NCHW_wino = simple_winograd.decl_winograd(
            cfg, A_NCHW, W_NCHW, stride, padding,
            layout="NCHW", out_dtype="float32")
        s = topi.generic.schedule_conv2d_nhwc([B])
        s_nchw = topi.generic.schedule_conv2d_nchw([B_NCHW])
        s_nchw_wino = simple_winograd.schedule_winograd(cfg, B_NCHW_wino)
        # print(tvm.lower(s_nchw_wino, [A_NCHW, W_NCHW, B_NCHW_wino], simple_mode=True))
    a = tvm.nd.array(a_np, ctx)
    a_nchw = tvm.nd.array(a_np.transpose(0, 3, 1, 2), ctx)
    w = tvm.nd.array(w_np, ctx)
    w_nchw = tvm.nd.array(w_np.transpose(3, 2, 0, 1), ctx)

    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    b_nchw = tvm.nd.array(np.zeros(get_const_tuple(B_NCHW.shape), dtype=B_NCHW.dtype), ctx)
    # b_tensor = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    b_nchw_wino = tvm.nd.array(np.zeros(get_const_tuple(B_NCHW.shape), dtype=B.dtype), ctx)

    def remote_func(func, name):
        return func

    func = remote_func(tvm.build(s, [A, W, B], target),name="func")
    func_nchw = remote_func(tvm.build(s_nchw, [A_NCHW, W_NCHW, B_NCHW], target), name="func_nchw")
    # func_tensor = tvm.build(s_tensor, [A, W, B_tensor], device)
    func_nchw_wino = remote_func(tvm.build(s_nchw_wino, [A_NCHW, W_NCHW, B_NCHW_wino], target), name="func_nchw_tensor_mxn")

    func(a, w, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
    func_nchw(a_nchw, w_nchw, b_nchw)
    np.testing.assert_allclose(b_nchw.asnumpy(), b_np.transpose(0, 3, 1, 2), rtol=1e-5)
    func_nchw_wino(a_nchw, w_nchw, b_nchw_wino)
    print(np.unravel_index(
        np.argmax(np.abs(b_nchw_wino.asnumpy() - b_np.transpose(0, 3, 1, 2)), axis=None),
        b_nchw_wino.shape))
    print(np.max(np.abs(b_nchw_wino.asnumpy() - b_np.transpose(0, 3, 1, 2)) / (np.abs(b_np.transpose(0, 3, 1, 2))) + 1e-5))

    np.testing.assert_allclose(b_nchw_wino.asnumpy(), b_np.transpose(0, 3, 1, 2), rtol=1e-3)

    return 1



def test_conv2d_nhwc():
    Workload = collections.namedtuple(
        'Workload',
        ['in_dtype', 'out_dtype', 'height', 'width', 'in_filter', 'out_filter',
         'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])
    WL = [
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
        Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 112, 112, 32, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 512, 1024, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 7, 7, 1024, 1024, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 128, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 256, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 512, 512, 1, 1, 0, 0, 1, 1),
    ]

    def run(workload, name):
        speedups = [verify_conv2d_nhwc(1, w.in_filter, w.height, w.out_filter, w.hkernel, w.hstride, w.hpad, 1) for w in workload]

    run(WL, "WL")


if __name__ == "__main__":
    test_conv2d_nhwc()
