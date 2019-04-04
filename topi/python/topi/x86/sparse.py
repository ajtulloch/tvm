from .. import generic, nn, tag
from ..util import traverse_inline, get_const_int
import tvm

@generic.schedule_sparse_dense.register(["cpu"])
def schedule_sparse_dense(outs):

    s = tvm.create_schedule([x.op for x in outs])
    def callback(op):
        if op.tag == "sparse_dense_csrmv" and op != outs[0].op:
            (_, vi) = s[op].op.axis
            s[op].vectorize(vi)
            (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 32)
            s[op].compute_at(s[outs[0]], yo)
            s[outs[0].op].vectorize(yi)
        if op.tag == "sparse_dense_bsrmv":
            Y_bsrmv = op.input_tensors[0]
            assert Y_bsrmv.op.tag == "sparse_dense_bsrmv_block"
            Y_reshape = op
            (m, nb, br) = s[Y_bsrmv].op.axis
            BS_R = get_const_int(br.dom.extent)
            (elem_idx, c) = s[Y_bsrmv].op.reduce_axis
            s[Y_bsrmv].reorder(nb, m, elem_idx, br, c)
            s[Y_bsrmv].vectorize(br)
            (mo, no) = s[Y_reshape].op.axis
            (noo, noi) = s[Y_reshape].split(no, BS_R)
            s[Y_reshape].unroll(noo)
            s[Y_bsrmv].compute_at(s[Y_reshape], noi)
            s[Y_reshape].vectorize(noi)
            s[Y_reshape].unroll(mo)
            if op != s[outs[0]].op:
                (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 32)
                s[Y_reshape].compute_at(s[outs[0]], yo)
                # s[outs[0].op].parallel(yo)
                s[outs[0].op].unroll(yo)
                s[outs[0].op].vectorize(yi)
            else:
                # pass
                # s[Y_reshape].parallel(noo)
                s[Y_reshape].unroll(noo)

    traverse_inline(s, outs[0].op, callback)
    return s
