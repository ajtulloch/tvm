from __future__ import division

from topi.nn.util import get_pad_tuple
from tvm import autotvm

import numpy as np
import topi
import topi.nn.util
import tvm
import unet_intrinsics

def decl_direct_NCHWc(cfg, data, kernel, strides, padding, out_dtype):
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    N, CII, IH, IW, CIII = topi.util.get_const_tuple(data.shape)
    COO, CII_, KH, KW, CIII_, COOO = topi.util.get_const_tuple(kernel.shape)
    OH = (IH + 2 * HPAD - KH) // HSTR + 1
    OW = (IW + 2 * WPAD - KW) // WSTR + 1


    data_pad = topi.nn.pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    cfg.define_split('tile_ow', cfg.axis(OW), num_outputs=2, filter=lambda x: x.size[-1] <= 6)

    # convolution
    oshape = (N, COO, OH, OW, COOO)

    cii = tvm.reduce_axis((0, CII), name='cii')
    ciii = tvm.reduce_axis((0, CIII), name='ciii')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_pad[n, cii, oh*HSTR+kh, ow*WSTR+kw, ciii]
                               .astype(out_dtype) *
                               kernel[oc_chunk, cii, kh, kw, ciii, oc_block],
                               axis=[cii, ciii, kh, kw]), name='conv2d_NCHWc', tag="conv2d_NCHWc")

    return conv

def schedule_direct_NCHWc(cfg, output):
    s = tvm.create_schedule(output.op)
    conv = output
    data_pad = output.op.input_tensors[0]
    assert data_pad.op.name == "data_pad"
    kernel = output.op.input_tensors[0]

    # schedule 5-D NCHW[x]c conv
    C, O = conv, conv
    CC = s.cache_write(C, 'global')

    cfg.define_knob('data_pad_inline', [0, 1])
    if cfg['data_pad_inline'].val:
        s[data_pad].compute_inline()

    _, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = cfg['tile_ow'].apply(s, C, ow)

    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    s[C].vectorize(oc_block)
    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    cii, ciii, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = cfg['tile_ow'].apply(s, CC, ow)

    (oc_chunk_ax, oh_ax, ow_chunk_ax, cii_ax, kh_ax, ciii_ax, kw_ax, ow_block_ax, oc_block_ax) = [
        cfg.axis(oc_chunk), cfg.axis(oh), cfg.axis(ow_chunk), cfg.axis(cii), cfg.axis(kh), cfg.axis(ciii), cfg.axis(kw), cfg.axis(ow_block), cfg.axis(oc_block)]

    cfg.define_reorder(
        "reorder_0",
        [oc_chunk_ax, oh_ax, ow_chunk_ax, cii_ax, kh_ax, ciii_ax, kw_ax, ow_block_ax, oc_block_ax],
        policy='candidate',
        candidate=[
            [oc_chunk_ax, oh_ax, ow_chunk_ax, cii_ax, kh_ax, kw_ax, ciii_ax, ow_block_ax, oc_block_ax],
            [oc_chunk_ax, oh_ax, ow_chunk_ax, cii_ax, kh_ax, ciii_ax, kw_ax, ow_block_ax, oc_block_ax],
            [oc_chunk_ax, oh_ax, ow_chunk_ax, cii_ax, ciii_ax, kh_ax, kw_ax, ow_block_ax, oc_block_ax],
            [oc_chunk_ax, oh_ax, ow_chunk_ax, cii_ax, kh_ax, kw_ax, ow_block_ax, ciii_ax, oc_block_ax],
        ]
    )

    cfg["reorder_0"].apply(
        s,
        CC,
        [oc_chunk, oh, ow_chunk, cii, kh, ciii, kw, ow_block, oc_block]
    )

    cfg.define_annotate('ann_reduce_k', [kw, kh], policy='try_unroll')
    cfg["ann_reduce_k"].apply(
        s,
        CC,
        [kh, kw],
        axis_lens=[
            topi.util.get_const_int(kh.dom.extent),
            topi.util.get_const_int(kw.dom.extent),
        ],
        cfg=cfg
    )
    cfg.define_annotate('ann_reduce_ow', [ow_block], policy='try_unroll')
    cfg["ann_reduce_ow"].apply(
        s,
        CC,
        [ow_block],
        axis_lens=[
            cfg['tile_ow'].size[-1]
        ],
        cfg=cfg
    )

    s[CC].vectorize(oc_block)
    return s
