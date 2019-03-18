import tvm
import topi

dtype = "float32"

rnn_dims = 128
fc_dims = 128

feat_dims = 24
aux_dims = 32
n_classes = 2 ** 8

x = tvm.placeholder(name="x", shape=[1, 1], dtype=dtype)
h1 = tvm.placeholder(name="h1", shape=[1, rnn_dims], dtype=dtype)
h2 = tvm.placeholder(name="h2", shape=[1, rnn_dims], dtype=dtype)
m_t = tvm.placeholder(name="m_t", shape=[1, feat_dims], dtype=dtype)
a1_t = tvm.placeholder(name="a1_t", shape=[1, aux_dims], dtype=dtype)
a2_t = tvm.placeholder(name="a2_t", shape=[1, aux_dims], dtype=dtype)
a3_t = tvm.placeholder(name="a3_t", shape=[1, aux_dims], dtype=dtype)
a4_t = tvm.placeholder(name="a4_t", shape=[1, aux_dims], dtype=dtype)


def concatenate_2(A, B):
    (M, K0) = topi.util.get_const_tuple(A.shape)
    (M, K1) = topi.util.get_const_tuple(B.shape)
    def concat_default_ir(A, B, out):
        """Define IR for Dense"""
        irb = tvm.ir_builder.create()
        A_ptr = irb.buffer_ptr(A)
        B_ptr = irb.buffer_ptr(B)
        out_ptr = irb.buffer_ptr(out)
        with irb.for_range(0, M, name='m') as m:
            with irb.for_range(0, K0, for_type="serial", name='n') as n:
                out_ptr[m * (K0 + K1) + n] = A_ptr[m * K0 + n]
            with irb.for_range(0, K1, for_type="serial", name='n') as n:
                out_ptr[m * (K0 + K1 ) + n + K0] = B_ptr[m * K1 + n]
        return irb.get()

    return tvm.extern((M, K0 + K1), [A, B],
                        lambda ins, outs: concat_default_ir(ins[0], ins[1], outs[0]),
                        tag="concat", dtype=dtype, name='concat')

def concatenate_3(A, B, C):
    (M, K0) = topi.util.get_const_tuple(A.shape)
    (M, K1) = topi.util.get_const_tuple(B.shape)
    (M, K2) = topi.util.get_const_tuple(C.shape)
    def concat_default_ir(A, B, C, out):
        """Define IR for Dense"""
        irb = tvm.ir_builder.create()
        A_ptr = irb.buffer_ptr(A)
        B_ptr = irb.buffer_ptr(B)
        C_ptr = irb.buffer_ptr(C)
        out_ptr = irb.buffer_ptr(out)
        with irb.for_range(0, M, name='m') as m:
            with irb.for_range(0, K0, for_type="serial", name='n') as n:
                out_ptr[m * (K0 + K1 + K2) + n] = A_ptr[m * K0 + n]
            with irb.for_range(0, K1, for_type="serial", name='n') as n:
                out_ptr[m * (K0 + K1 + K2) + n + K0] = B_ptr[m * K1 + n]
            with irb.for_range(0, K2, for_type="serial", name='n') as n:
                out_ptr[m * (K0 + K1 + K2) + n + K0 + K1] = C_ptr[m * K2 + n]
        return irb.get()

    return tvm.extern((M, K0 + K1 + K2), [A, B, C],
                        lambda ins, outs: concat_default_ir(ins[0], ins[1], ins[2], outs[0]),
                        tag="concat", dtype=dtype, name='concat')

concat0_o = concatenate_3(x, m_t, a1_t)


def dense(X, W, B, **kwargs):
    (M, K_) = topi.util.get_const_tuple(X.shape)
    (K, N) = topi.util.get_const_tuple(W.shape)
    (N_, ) = topi.util.get_const_tuple(B.shape)
    assert K_ == K
    assert N_ == N
    k = tvm.reduce_axis((0, K), name='k')
    MM = tvm.compute(
        (M, N),
        lambda i, j: tvm.sum(X[i, k] * W[k, j], axis=k),
        name="dense_matmul",
        tag='dense',
        **kwargs
    )
    print(M, N)
    return tvm.compute(MM.shape, lambda i, j: MM[i, j] + B[j],
                       name="dense_biasadd",
                       tag=topi.tag.INJECTIVE)

fc_0_W = tvm.placeholder(name="fc_0_W",
                         shape=(feat_dims + aux_dims + 1, rnn_dims),
                         dtype=dtype)

fc_0_B = tvm.placeholder(name="fc_0_B",
                         shape=(rnn_dims,),
                         dtype=dtype)
i_o = dense(concat0_o, fc_0_W, fc_0_B)

def gru(X, H, W_X, W_H, B, **kwargs):
    (K_X, N) = topi.util.get_const_tuple(W_X.shape)
    (K_H, N_) = topi.util.get_const_tuple(W_H.shape)
    assert N % 3 == 0
    D = N // 3
    assert N_ == N
    assert D == rnn_dims
    (M, K_X_) = topi.util.get_const_tuple(X.shape)
    (M_, K_H_) = topi.util.get_const_tuple(H.shape)
    assert K_X_ == K_X
    assert K_H_ == K_H
    assert M_ == M
    k_x = tvm.reduce_axis((0, K_X), name='k_x')
    k_h = tvm.reduce_axis((0, K_H), name='k_h')
    XT = tvm.compute(
        (M, N),
        lambda i, j: tvm.sum(X[i, k_x] * W_X[k_x, j], axis=k_x),
        name="gru_XT",
        tag='dense',
    )
    HT = tvm.compute(
        (M, N),
        lambda i, j: tvm.sum(H[i, k_h] * W_H[k_h, j], axis=k_h),
        name="gru_HT",
        tag='dense',
    )
    def gru_cell(i, j):

        # u_t = tvm.sigmoid(XT[i, j] + HT[i, j] + B[j])
        # r_t = tvm.sigmoid(XT[i, j + D] + HT[i, j + D] + B[j + D])
        # e_t = tvm.tanh(r_t * HT[i, j + 2 * D] + XT[i, j + 2 * D] + B[j + 2 * D])
        u_t = XT[i, j] + HT[i, j] + B[j]
        r_t = XT[i, j + D] + HT[i, j + D] + B[j + D]
        e_t = r_t * HT[i, j + 2 * D] + XT[i, j + 2 * D] + B[j + 2 * D]

        return u_t * H[i, j] + (1 - u_t) * e_t

    return tvm.compute((M, D), lambda i, j: gru_cell(i, j),
                       name="gru_cell", tag=topi.tag.INJECTIVE)

gru_0_W_X = tvm.placeholder((rnn_dims, 3 * rnn_dims), name="gru_0_W_X", dtype=dtype)
gru_0_W_H = tvm.placeholder((rnn_dims, 3 * rnn_dims), name="gru_0_W_H", dtype=dtype)
gru_0_B = tvm.placeholder((3 * rnn_dims,), name="gru_0_B", dtype=dtype)

h1_prime = gru(i_o, h1, gru_0_W_X, gru_0_W_H, gru_0_B)
gru2_add1_o = tvm.compute(i_o.shape, lambda *i: i_o[i] + h1_prime[i], name="gru_residual_add", tag=topi.tag.INJECTIVE)

inp = concatenate_2(gru2_add1_o, a2_t)


gru_1_W_X = tvm.placeholder((rnn_dims + aux_dims, 3 * rnn_dims), name="gru_1_W_X", dtype=dtype)
gru_1_W_H = tvm.placeholder((rnn_dims, 3 * rnn_dims), name="gru_1_W_H", dtype=dtype)
gru_1_B = tvm.placeholder((3 * rnn_dims,), name="gru_1_B", dtype=dtype)

h2_prime = gru(inp, h2, gru_1_W_X, gru_1_W_H, gru_1_B)
add1_o = topi.add(gru2_add1_o, h2_prime)

concat1_o = concatenate_2(add1_o, a3_t)


fc_1_W = tvm.placeholder(name="fc_1_W",
                         shape=(rnn_dims + aux_dims, fc_dims),
                         dtype=dtype)
fc_1_B = tvm.placeholder(name="fc_1_B",
                         shape=(fc_dims,),
                         dtype=dtype)
relu1_o = topi.nn.relu(dense(concat1_o, fc_1_W, fc_1_B))

concat2_o = concatenate_2(relu1_o, a4_t)

fc_2_W = tvm.placeholder(name="fc_2_W",
                         shape=(fc_dims + aux_dims, fc_dims),
                         dtype=dtype)
fc_2_B = tvm.placeholder(name="fc_2_B",
                         shape=(fc_dims,),
                         dtype=dtype)
relu2_o = topi.nn.relu(dense(concat2_o, fc_2_W, fc_2_B))

# relu2_o = denseRelu("fc2", fc_dims,
#                          fc_dims + aux_dims,
#                          concat2_o, add_bias=True, add_relu=True)

fc_3_W = tvm.placeholder(name="fc_3_W",
                         shape=(fc_dims, n_classes),
                         dtype=dtype)
fc_3_B = tvm.placeholder(name="fc_3_B",
                         shape=(n_classes,),
                         dtype=dtype)
fc_3_o = dense(relu2_o, fc_3_W, fc_3_B)
softmax_o = topi.nn.softmax(fc_3_o, axis=-1)


params = [fc_0_W, fc_0_B, gru_0_W_X, gru_0_W_H, gru_0_B, gru_1_W_X, gru_1_W_H, gru_1_B, fc_1_W, fc_1_B, fc_2_W, fc_2_B, fc_3_W, fc_3_B]
inputs = [x, h1, h2, m_t, a1_t, a2_t, a3_t, a4_t]
outputs = [concat1_o]

s = tvm.create_schedule([output.op for output in outputs])

print(tvm.lower(s, params + inputs + outputs, simple_mode=True))

def block_1(op):
    if op.tag == "dense":
        C = op.output(0)
        x, y = s[C].op.axis
        k = s[C].op.reduce_axis[0]
        (M, N) = topi.util.get_const_tuple(C.shape)
        K = topi.util.get_const_int(k.dom.extent)
        yo, yi = s[C].split(y, 32)
        s[C].reorder(x, yo, k, yi)
        s[C].vectorize(yi)
    if op.name == "gru_residual_add":
        C = op.output(0)
        (co, ci) = s[C].split(s[C].op.axis[1], 32)
        s[C].vectorize(ci)
    print(op, op.name)

topi.util.traverse_inline(s, gru2_add1_o.op, block_1)
print(tvm.lower(s, params + inputs + outputs, simple_mode=True))
target = tvm.target.create("llvm -mcpu=core-avx2")

with target:
    func = tvm.build(s, params + inputs + outputs)

ctx = tvm.cpu()
ftimer = func.time_evaluator(func.entry_name, ctx, number=10000, repeat=10)

import numpy as np
np_args = [np.random.randn(*topi.util.get_const_tuple(x.shape)).astype(x.dtype) for x in params + inputs + outputs]
tvm_args = [tvm.ndarray.array(x) for x in np_args]
result = ftimer(*tvm_args)
print(result)
