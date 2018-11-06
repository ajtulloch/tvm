# pylint: disable=invalid-name, unused-variable
"""Schedule for pooling operators"""
import tvm
from .. import generic
from .. import tag

@generic.schedule_pool.register(["arm_cpu"])
def schedule_pool(outs, layout):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []
    output_op = outs[0].op

    def traverse(op, io):
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op, io)

        elif op.tag.startswith('pool'):
            # schedule the pooling op.
            kw, kh = op.reduce_axis
            s[op].unroll(kw)
            s[op].unroll(kh)
            s[op].vectorize(list(op.axis)[-1])
            s[op].compute_at(s[outs[0].op], io)

            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op, io)

        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

        scheduled_ops.append(op)

    io, ii = s[output_op].split(list(output_op.axis)[-1], 4)
    s[output_op].vectorize(ii)
    traverse(output_op, io)
    s[output_op].fuse(output_op.axis[0], output_op.axis[1])
    return s
