import unet_conv2d_winograd
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_int, get_const_tuple
import collections
from tvm import autotvm
from topi.nn.conv2d import Workload

target = tvm.target.create('llvm -mcpu=core-avx2')
ctx = tvm.context('llvm -mcpu=core-avx2', 0)

def nchw_to_NCHWc(x, cc=8):
    (n, c, h, w) = x.shape
    assert c % cc == 0
    y = np.zeros(shape=(n, c // cc, h, w, cc), dtype=x.dtype)
    for cb in range(c // cc):
        y[:, cb, :, :, :] = x[:, cb * cc:(cb + 1) * cc, :, :].transpose(0, 2, 3, 1)
    return y

def weight_nchw_to_NCHWc(x, cc=8):
    (f, c, h, w) = x.shape
    assert c % cc == 0
    assert f % cc == 0
    y = np.zeros(shape=(f // cc, c // cc, h, w, cc, cc), dtype=x.dtype)
    for fcb in range(f // cc):
        for ccb in range(c // cc):
            y[fcb, ccb, :, :, :, :] = x[fcb * cc:(fcb + 1) * cc, ccb * cc:(ccb + 1) * cc, :, :].transpose(2, 3, 1, 0)
    return y

def NCHWc_to_nchw(y, cc=8):
    (n, cbs, h, w, cc_) = y.shape
    assert cc == cc_
    x = np.zeros(shape=(n, cbs * cc, h, w), dtype=y.dtype)
    for cb in range(cbs):
        x[:, cb * cc:(cb + 1) * cc, :, :] = y[:, cb, :, :, :].transpose(0, 3, 1, 2)
    return x

def verify_conv2d_NCHWc(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1):
    in_height = in_width = in_size
    assert kernel == 3
    assert padding == 1
    assert in_channel % 8 == 0
    assert num_filter % 8 == 0
    A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, num_filter), name='W')
    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype
    print("N: {}, CIn: {}, H/W: {}, COut: {}, KH/KW: {}".format(batch, in_channel, in_size, num_filter, kernel))
    def get_ref_data():
        np.random.seed(1)
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        # a_np.fill(1)
        w_np = np.random.uniform(size=w_shape).astype(dtype) * 1e-2
        # w_np.fill(1)
        dw_np = topi.testing.dilate_python(w_np, (1, dilation, dilation, 1))
        b_np = topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    with target:
        A_NCHW = tvm.placeholder((batch, in_channel, in_height, in_width), name='A_NCHW')
        A_NCHWc = tvm.placeholder((batch, in_channel // 8, in_height, in_width, 8), name='A_NCHWc')

        W_NCHW = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W_NCHW')
        W_NCHWc = tvm.placeholder((num_filter // 8, in_channel // 8, kernel, kernel, 8, 8), name='W_NCHW')


        dW = W
        dW_NCHW = W_NCHW
        B = topi.nn.conv2d_nhwc(A, dW, stride, padding)
        B_NCHW = topi.nn.conv2d(A_NCHW, W_NCHW, stride, padding, layout='NCHW')
        B_NCHWc = topi.nn.conv2d_NCHWc(A_NCHWc, W_NCHWc, num_filter, (kernel, kernel), stride, padding, layout='NCHW8c', out_layout="NCHW8c")
        cfg = autotvm.get_config()
        B_NCHWc_wino = unet_conv2d_winograd._decl_winograd_NCHWc(
            cfg, A_NCHWc, W_NCHWc, num_filter, kernel, stride, padding, layout="NCHW8c", out_layout="NCHW8c", out_dtype="float32", m=2)
        s = topi.generic.schedule_conv2d_nhwc([B])
        s_nchw = topi.generic.schedule_conv2d_nchw([B_NCHW])
        s_NCHWc_wino = tvm.create_schedule([B_NCHWc_wino.op])
        s_NCHWc = topi.generic.schedule_conv2d_NCHWc_([B_NCHWc])
        # print(tvm.lower(s_nchw_wino, [A_NCHW, W_NCHW, B_NCHW_wino], simple_mode=True))
        print(tvm.lower(s_NCHWc_wino, [A_NCHWc, W_NCHWc, B_NCHWc_wino], simple_mode=True))

    a = tvm.nd.array(a_np, ctx)
    a_nchw = tvm.nd.array(a_np.transpose(0, 3, 1, 2), ctx)

    a_NCHWc = tvm.nd.array(nchw_to_NCHWc(a_np.transpose(0, 3, 1, 2)), ctx)

    w = tvm.nd.array(w_np, ctx)
    w_nchw = tvm.nd.array(w_np.transpose(3, 2, 0, 1), ctx)
    w_NCHWc = tvm.nd.array(weight_nchw_to_NCHWc(w_np.transpose(3, 2, 0, 1)), ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    b_nchw = tvm.nd.array(np.zeros(get_const_tuple(B_NCHW.shape), dtype=B_NCHW.dtype), ctx)
    b_NCHWc = tvm.nd.array(nchw_to_NCHWc(np.zeros(get_const_tuple(B_NCHW.shape), dtype=B_NCHW.dtype)), ctx)

    b_NCHWc_wino = tvm.nd.array(nchw_to_NCHWc(np.zeros(get_const_tuple(B_NCHW.shape), dtype=B_NCHW.dtype)), ctx)
    # b_tensor = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    b_nchw_wino = tvm.nd.array(np.zeros(get_const_tuple(B_NCHW.shape), dtype=B.dtype), ctx)

    def remote_func(func, name):
        return func

    func = remote_func(tvm.build(s, [A, W, B], target),name="func")
    func_nchw = remote_func(tvm.build(s_nchw, [A_NCHW, W_NCHW, B_NCHW], target), name="func_nchw")
    # func_tensor = tvm.build(s_tensor, [A, W, B_tensor], device)
    func_NCHWc = remote_func(tvm.build(s_NCHWc, [A_NCHWc, W_NCHWc, B_NCHWc], target), name="func_NCHWc")
    func_NCHWc_wino = remote_func(tvm.build(s_NCHWc_wino, [A_NCHWc, W_NCHWc, B_NCHWc_wino], target), name="func_NCHWc_wino")

    func(a, w, b)
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
    func_nchw(a_nchw, w_nchw, b_nchw)
    np.testing.assert_allclose(b_nchw.asnumpy(), b_np.transpose(0, 3, 1, 2), rtol=1e-5)

    func_NCHWc(a_NCHWc, w_NCHWc, b_NCHWc)
    np.testing.assert_allclose(NCHWc_to_nchw(b_NCHWc.asnumpy()), b_np.transpose(0, 3, 1, 2), rtol=1e-5)
    func_NCHWc_wino(a_NCHWc, w_NCHWc, b_NCHWc_wino)
    print(b_np.transpose(0, 3, 1, 2).shape)
    print(np.unravel_index(
        np.argmax(np.abs(NCHWc_to_nchw(b_NCHWc_wino.asnumpy()) - b_np.transpose(0, 3, 1, 2)), axis=None),
        b_nchw_wino.shape))
    print(np.max(np.abs(NCHWc_to_nchw(b_NCHWc_wino.asnumpy()) - b_np.transpose(0, 3, 1, 2)) / (np.abs(b_np.transpose(0, 3, 1, 2))) + 1e-5))
    print(b_np.flatten()[:5])

    np.testing.assert_allclose(NCHWc_to_nchw(b_NCHWc_wino.asnumpy()), b_np.transpose(0, 3, 1, 2), rtol=1e-2)

    return 1



def test_conv2d_nhwc():
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
    ]

    for w in WORKLOADS:
        if w.in_filter % 8 != 0 or w.out_filter % 8 != 0 or w.hkernel != 3 or w.hstride != 1:
            continue
        verify_conv2d_NCHWc(1, w.in_filter, w.height, w.out_filter, w.hkernel, w.hstride, w.hpad)



if __name__ == "__main__":
    test_conv2d_nhwc()
