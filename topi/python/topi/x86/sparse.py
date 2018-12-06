import tvm
from tvm import autotvm
from .. import generic, nn, tag
from ..util import traverse_inline, get_const_tuple
from ..sparse import sparse_lengths_sum_fused_8_bit_rowwise

from tvm.contrib import cblas

@generic.schedule_sparse_lengths_sum.register(["cpu"])
def schedule_sparse_lengths_sum(outs):
    """Schedule for sparse length sum op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    output_op = outs[0].op
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if 'sparse_lengths_sum' in op.tag:
            Y = op.output(0)
            (n, d) = s[Y].op.axis
            (gg,) = s[Y].op.reduce_axis
            s[Y].reorder(n, gg, d)
            s[Y].vectorize(d)
            if op != outs[0].op:
                s[Y].compute_at(s[output_op], s[output_op].op.axis[0])
    traverse_inline(s, outs[0].op, _callback)
    return s


def product(l):
    p = 1
    for i in l:
        p *= i
    return p

@autotvm.register_topi_compute(nn.batch_matmul, 'cpu', ['direct'])
def batch_matmul(cfg, A, B, trans_a=False, trans_b=False):
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    import topi
    assert len(A.shape) == len(B.shape), \
        "Shape mismatch between inputs"
    A_shape = get_const_tuple(A.shape)
    B_shape = get_const_tuple(B.shape)
    N = A_shape[-1] if trans_a else A_shape[-2]
    M = B_shape[-2] if trans_b else B_shape[-1]
    K = A_shape[-2] if trans_a else A_shape[-1]
    oshape = A_shape[:-2] + (N, M)
    if len(A.shape) > 3:
        batch_a = product(A.shape[:-2])
        batch_b = product(B.shape[:-2])
        A = topi.reshape(A, [batch_a] + A.shape[-2:])
        B = topi.reshape(B, [batch_b] + B.shape[-2:])

    cfg.define_knob('blas', [0, 1, 2])
    f = [batch_matmul_direct, batch_matmul_blas, batch_matmul_blas][cfg['blas'].val]
    C = f(cfg, A, B, trans_a, trans_b)
    cfg.add_flop(2 * product(oshape) * K)
    if len(A.shape) > 3:
        return topi.reshape(C, oshape)
    return C


def batch_matmul_direct(cfg, A, B, trans_a, trans_b):
    """The default implementation of batched_matmul in topi.

    Parameters
    ----------
    A: tvm.Tensor
        n-D with shape [b0, ... bn, N, K]

    B: tvm.Tensor
        n-D with shape [b0, ... bn, K, M]

    trans_a: bool
        Whether transpose A[-2:] or not

    trans_b: bool
        Whether transpose B[-2:] or not

    Returns
    -------
    output: tvm.Tensor
        n-D with shape [[b0, ... bn, N, M]
    """
    import topi
    A_shape = get_const_tuple(A.shape)
    B_shape = get_const_tuple(B.shape)
    batch_a = A_shape[0]
    batch_b = B_shape[0]
    N = A_shape[-1] if trans_a else A_shape[-2]
    M = B_shape[-2] if trans_b else B_shape[-1]
    K = A_shape[-2] if trans_a else A_shape[-1]
    if trans_a:
        A = topi.transpose(A, [0, 2, 1])
    if trans_b:
        B = topi.transpose(B, [0, 2, 1])
    k = tvm.reduce_axis((0, K), name='k')
    return tvm.compute((batch_a, N, M),
                       lambda b, y, x: tvm.sum(A[b, y, k] * B[b, k, x], axis=k),
                       tag='batch_matmul')

def batch_matmul_blas(cfg, A, B, trans_a, trans_b):
    iterative = cfg['blas'].val == 2
    return cblas.batch_matmul(
        A, B, transa=trans_a, transb=trans_b, iterative=iterative, tag="batch_matmul")

@autotvm.register_topi_schedule(generic.schedule_batch_matmul, 'cpu', ['direct'])
def schedule_dense(cfg, outs):
    """Schedule for dense operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config entity for this template
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        pass
    traverse_inline(s, outs[0].op, _callback)
    return s


def sls_fused_impl():
    src = """
    #include <immintrin.h>

    extern "C" int32_t embedding_lookup(
                                 const int32_t block_size,
                                 const int32_t output_size,
                                 const int32_t index_size,
                                 const uint8_t *input, const int64_t *indices,
                                 const int *lengths, float *out) {
      const int32_t prefdist_T0 = 16;
      const int32_t fused_block_size = block_size + 8;

      int64_t dataInd = 0;
      for (int64_t rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {
        float *op = &out[rangeIndex * block_size];
        int32_t j = 0;
        for (; j + 8 <= block_size; j += 8) {
          _mm256_storeu_ps(op + j, _mm256_setzero_ps());
        }
        for (; j < block_size; j++) {
          op[j] = 0.0f;
        }
        for (int64_t start = dataInd; dataInd < start + lengths[rangeIndex];
             ++dataInd) {
          const int64_t idx = indices[dataInd];
          float wgt = 1.f;
          float bio;
          const float *scale_bias = reinterpret_cast<const float *>(
              &input[idx * fused_block_size + block_size]);
          bio = wgt * scale_bias[1];
          wgt = wgt * scale_bias[0];
          __m256 vbio = _mm256_set1_ps(bio);
          __m256 vwgt = _mm256_set1_ps(wgt);
          const uint8_t *ip = &input[idx * fused_block_size];
          const int64_t next_T0 = (dataInd < index_size - prefdist_T0)
                                      ? (dataInd + prefdist_T0)
                                      : dataInd;
          const int64_t idx_pref_T0 = indices[next_T0];
          const uint8_t *ip_next_T0 = &input[idx_pref_T0 * fused_block_size];
          j = 0;
          for (; j + 8 <= block_size; j += 8) {
            _mm256_storeu_ps(
                &op[j], _mm256_fmadd_ps(
                            vwgt,
                            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                                reinterpret_cast<const __m128i *>(&ip[j])))),
                            _mm256_add_ps(_mm256_loadu_ps(&op[j]), vbio)));
            _mm_prefetch((&ip_next_T0[j]), _MM_HINT_T0);
          }
          for (; j < block_size; j++) {
            op[j] += wgt * ((float)ip[j]) + bio;
          }
        }
      }
      return 0;
    }
    """
    from tvm.contrib import clang
    return clang.create_llvm(src, options=["-O3", "-mavx2", "-mfma"])


@sparse_lengths_sum_fused_8_bit_rowwise.register(["cpu"])
def sparse_length_sum_fused_8_bit_rowwise_asm(data, indices, lengths):
    assert data.dtype == "uint8"
    assert indices.dtype == "int64"
    assert lengths.dtype == "int32"
    (I,) = get_const_tuple(indices.shape)
    (L,) = get_const_tuple(lengths.shape)
    (DD, B) = get_const_tuple(data.shape)
    fused_block_size = B
    block_size = B - 8
    output_size = L
    index_size = I
    def sparse_length_sum_fused_8_bit_rowwise_ir(data, indices, lengths, out):
        irb = tvm.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        indices_ptr = irb.buffer_ptr(indices)
        lengths_ptr = irb.buffer_ptr(lengths)
        assert not data.strides
        assert not out.strides
        out_ptr = irb.buffer_ptr(out)
        irb.scope_attr(tvm.const(1), "pragma_import_llvm", sls_fused_impl())
        irb.emit(tvm.call_extern(
                "int32", "embedding_lookup",
                B - 8, L, I, data_ptr, indices_ptr, lengths_ptr, out_ptr))
        return irb.get()
    sparse_length_sum = tvm.extern(
        (L, B - 8),
        [data, indices, lengths],
        lambda ins, outs: sparse_length_sum_fused_8_bit_rowwise_ir(
            ins[0], ins[1], ins[2], outs[0]),
        tag="sparse_length_sum",
        dtype="float32",
        name="sparse_length_sum")
    return sparse_length_sum
