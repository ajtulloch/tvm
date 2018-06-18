# pylint: disable=invalid-name,unused-variable,unused-argument,invalid-name
"""1x1 Conv2D schedule on for Intel CPU"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm

from .. import tag
from ..util import get_const_tuple, get_const_int
from ..nn.conv2d import _get_schedule, _get_workload
from ..nn.util import infer_pad, infer_stride
from ..nn.pad import pad

AVXConv1x1Fwd = namedtuple('AVXConv1x1Fwd', ['ic_bn', 'oc_bn', 'oh_factor', 'ow_factor'])
USE_TENSOR = False

OPS_TO_GEMM = {}

def _get_default_schedule(wkl, simd_width):
    print("Getting default schedule")
    print(wkl, simd_width)
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    out_height = (wkl.height + 2 * HPAD - wkl.hkernel) // HSTR + 1
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    oc_bn = 1
    for bn in range(simd_width, 0, -1):
        if wkl.out_filter % bn == 0:
            oc_bn = bn
            break

    ic_bn = 1
    for bn in range(oc_bn, 0, -1):
        if wkl.in_filter % bn == 0:
            ic_bn = bn
            break
    for ow_factor in range(out_width, 0, -1):
        if out_width % ow_factor == 0:
            for oh_factor in range(out_height, 0, -1):
                if out_height % oh_factor == 0 and ow_factor * oh_factor < 32:
                    print("Returing AVXConv1x1Fwd")
                    return AVXConv1x1Fwd(ic_bn, oc_bn, oh_factor, ow_factor)

    raise ValueError("cannot decide default schedule for workload: {}".format(wkl))


def _declaration_conv(data, kernel, stride, padding, layout, out_dtype):
    print("Decl", data, kernel, stride, padding, layout, out_dtype)
    assert layout == 'NCHW', "only support NCHW convolution for AVX"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _get_schedule(wkl)
    print("Schedule: ", sch)

    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    batch_size, in_channel, in_height, in_width = get_const_tuple(data.shape)
    num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    pad_height = in_height + 2 * HPAD
    pad_width = in_width + 2 * WPAD

    out_height = (in_height + 2 * HPAD - kernel_height) // HSTR + 1
    out_width = (in_width + 2 * WPAD - kernel_width) // WSTR + 1

    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data
    shape = (batch_size, in_channel // sch.ic_bn, pad_height, pad_width, sch.ic_bn)
    data_vec = tvm.compute(shape, lambda n, C, h, w, c: data_pad[n, C * sch.ic_bn + c, h, w])

    shape = (num_filter // sch.oc_bn, in_channel // sch.ic_bn, sch.ic_bn, sch.oc_bn, 1, 1)
    kernel_vec = tvm.compute(shape, lambda CO, CI, ci, co, h, w:
                             kernel[CO * sch.oc_bn + co, CI * sch.ic_bn + ci, h, w],
                             name='kernel_vec')

    oshape = (batch_size, num_filter // sch.oc_bn, out_height, out_width, sch.oc_bn)
    ic = tvm.reduce_axis((0, in_channel), name='ic')
    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_vec[n, ic//sch.ic_bn, oh*HSTR, ow*WSTR, ic%sch.ic_bn] *
                               kernel_vec[oc_chunk, ic//sch.ic_bn, ic%sch.ic_bn, oc_block, 0, 0],
                               axis=[ic]), name='conv')

    oshape = (batch_size, num_filter, out_height, out_width)
    unpack = tvm.compute(oshape, lambda n, oc, oh, ow:
                         conv[n, oc // sch.oc_bn, oh, ow, oc % sch.oc_bn],
                         tag='conv2d_nchw')
    return unpack

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

def _declaration_conv_tensor(A, W_, stride, padding, layout, out_dtype):
    assert layout == "NCHW"
    assert out_dtype == A.dtype

    (N, CIn, H, W) = get_const_tuple(A.shape)
    (COut, CIn_, kh, kw) = get_const_tuple(W_.shape)
    assert kh == 1
    assert kw == 1
    assert CIn == CIn_
    assert stride == 1 or stride == (1, 1), stride
    assert padding == 0 or padding == (0, 0)

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

    OPS_TO_GEMM[unpacked_nchw.op] = A_W_product.op
    return unpacked_nchw

def _schedule_conv_tensor(s, op):
    output_op = op

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
            s[op].compute_inline()
            # n, h, w, c = op.axis
            # fused = s[op].fuse(n, h, w)
            # s[op].parallel(fused)
            # s[op].vectorize(c)

    traverse(output_op)
    return s



def _schedule_conv(s, data, data_pad, data_vec, kernel, kernel_vec, conv_out, output, last):
    print("Scheduling conv")
    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    if data_pad is None:
        stride = infer_stride(data, kernel, output)
    else:
        stride = infer_stride(data_pad, kernel, output)

    wkl = _get_workload(data, kernel, stride, padding, output.dtype)
    sch = _get_schedule(wkl)

    HPAD, WPAD = wkl.hpad, wkl.wpad
    DOPAD = (HPAD != 0 and WPAD != 0)

    A, W = data, kernel_vec
    A0, A1 = data_pad, data_vec
    # schedule data
    if DOPAD:
        s[A0].compute_inline()
    batch, ic_chunk, ih, ic_block, iw = s[A1].op.axis
    parallel_axis = s[A1].fuse(ic_chunk, ih)
    s[A1].parallel(parallel_axis)

    # schedule kernel pack
    oc_chunk, ic_chunk, oh, ow, ic_block, oc_block = s[W].op.axis
    s[W].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)
    if sch.oc_bn > 1:
        s[W].vectorize(oc_block)
    parallel_axis = s[W].fuse(oc_chunk, oh)
    s[W].parallel(parallel_axis)

    C, O0, O = conv_out, output, last
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=sch.oh_factor)
    s[C].vectorize(oc_block)

    s[CC].compute_at(s[C], oh_outer)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, = s[CC].op.reduce_axis

    ic_chunk, ic_block = s[CC].split(ic, factor=sch.ic_bn)

    oh_outer, oh_inner = s[CC].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=sch.ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, ic_chunk, ic_block, oh_inner, ow_inner, oc_block)
    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    if O0 != O:
        s[O0].compute_inline()
    batch, oc, oh, ow = s[O].op.axis

    oc_chunk, oc_block = s[O].split(oc, factor=sch.oc_bn)
    oh_outer, oh_inner = s[O].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[O].split(ow, factor=sch.ow_factor)
    s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

    parallel_axis = s[O].fuse(oc_chunk, oh_outer)
    s[C].compute_at(s[O], parallel_axis)
    s[O].vectorize(oc_block)

    s[O].parallel(parallel_axis)

    return s


def _declaration_conv_NCHWc(wkl, sch, data, kernel):
    print("Using NCHWc")
    out_dtype = wkl.out_dtype
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    batch_size = data.shape[0]
    out_height = (wkl.height + 2 * HPAD - wkl.hkernel) // HSTR + 1
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    oshape = (batch_size, wkl.out_filter//sch.oc_bn, out_height, out_width, sch.oc_bn)
    ic = tvm.reduce_axis((0, wkl.in_filter), name='ic')
    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_pad[n, ic//sch.ic_bn, oh*HSTR, ow*WSTR, ic%sch.ic_bn]
                               .astype(out_dtype) *
                               kernel[oc_chunk, ic // sch.ic_bn, ic % sch.ic_bn, oc_block, 0, 0],
                               axis=[ic]), name='conv2d_NCHWc', tag='conv2d_NCHWc')

    return conv


def _schedule_conv_NCHWc(s, wkl, sch, data, kernel, conv_out, last):
    print("Scheduling NCHWc")
    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, ic_block = s[A].op.axis
        parallel_axis = s[A].fuse(ic_chunk, ih)
        s[A].parallel(parallel_axis)

    C, O = conv_out, last
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[C].split(ow, factor=sch.ow_factor)
    s[C].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)
    s[C].vectorize(oc_block)

    parallel_axis = s[C].fuse(oc_chunk, oh_outer)
    s[CC].compute_at(s[C], parallel_axis)
    if C == O:
        s[C].parallel(parallel_axis)

    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, = s[CC].op.reduce_axis

    ic_chunk, ic_block = s[CC].split(ic, factor=sch.ic_bn)

    oh_outer, oh_inner = s[CC].split(oh, factor=sch.oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=sch.ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, ic_chunk, ic_block, oh_inner, ow_inner, oc_block)
    s[CC].fuse(oc_chunk, oh_outer)
    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    if C != O:
        batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
        oh_outer, oh_inner = s[O].split(oh, factor=sch.oh_factor)
        ow_outer, ow_inner = s[O].split(ow, factor=sch.ow_factor)
        s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

        parallel_axis = s[O].fuse(oc_chunk, oh_outer)
        s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s


if USE_TENSOR:
    _declaration_conv = _declaration_conv_tensor
    _schedule_conv = _schedule_conv_tensor
