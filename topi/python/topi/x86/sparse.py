from .. import generic, nn, tag
from ..util import traverse_inline
import tvm

@generic.schedule_sparse_dense.register(["cpu"])
def schedule_sparse_dense(outs):

    s = tvm.create_schedule([x.op for x in outs])
    if "sparse_dense" not in outs[0].op.tag:
        def callback(op):
            if "sparse_dense" in op.tag:
                (_, vi) = s[op].op.axis
                s[op].vectorize(vi)
                (yo, yi) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 32)
                s[op].compute_at(s[outs[0]], yo)
                s[outs[0].op].vectorize(yi)
        traverse_inline(s, outs[0].op, callback)
    return s
