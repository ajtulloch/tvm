"""
How to optimize GEMM on CPU
===========================
**Author**: `Jian Weng <https://github.com/were>`_, \
            `Ruofei Yu <https://github.com/yuruofeifei>`_

(TL;DR) TVM provides abstract interfaces which allows users to depict an algorithm and the
algorithm's implementing organization (the so-called schedule) separately. Typically, writing
algorithm in high-performance schedule breaks the algorithm's readability and modularity. Also,
trying various seemingly promising schedules is time-consuming. With the help of TVM, we can
try these schedules efficiently to enhance the performance.

In this tutorial, we will demonstrate how to use TVM to optimize square matrix multiplication
and achieve 200 times faster than baseline by simply adding 18 extra lines of code.

There are two important optimizations on intense computation applications executed on CPU:
    1. Increase the cache hit rate of memory access. Both complex numerical computation and hot-spot
       memory access can be accelerated from high cache hit rate. This requires us to transform the
       origin memory access pattern to the pattern fits the cache policy.
    2. SIMD (Single instruction multi-data), or we call it vector processing unit. Every time, a
       small batch of data, rather than a single grid, will be processed. This requires us to
       transform the data access pattern in the loop body in uniform pattern so that the LLVM
       backend can lower it to SIMD.

Actually, all the methodologies used in this tutorial is a subset of tricks mentioned in this
`repo <https://github.com/flame/how-to-optimize-gemm>`_. Some of them have been applied by TVM
abstraction automatically, but some of them cannot be simply applied due to TVM constraints.

All the experiment results mentioned below, are executed on 2015's 15' MacBook equipped with
Intel i7-4770HQ CPU. The cache line size should be 64 bytes for all the x86 CPUs.
"""

################################################################################################
# Preparation and Baseline
# ------------------------
# In this tutorial, we will demo how to use TVM to optimize matrix multiplication.
# Before actually demonstrating, we first define these variables.
# Then we write a baseline implementation, the simplest way to write a matrix multiplication in TVM.

import tvm
from topi.util import get_const_tuple
import numpy
import timeit

target = tvm.target.create("llvm -mcpu=core-avx2")
ARCH = "avx2"

BITCODE_PATHS = [
    "tensorize/gemm__{ARCH}.bc".format(ARCH=ARCH)
]

@tvm.register_func("tvm_callback_llvm_bitcode_path")
def bitcode_paths():
    global BITCODE_PATHS
    return BITCODE_PATHS

# Tensorized
def intrin_tile_gemm(KBlock, MTile, NTile, KTile):
    # assert (M, N) in MNTiles, (M, N)
    A = tvm.placeholder((KBlock, MTile, KTile), dtype='uint8', name='ATile')
    B = tvm.placeholder((KBlock, NTile, KTile), dtype='int8', name='BTile')
    kb = tvm.reduce_axis((0, KBlock), name='kblock')
    kt = tvm.reduce_axis((0, KTile), name='ktile')
    C = tvm.compute(
        (MTile, NTile),
        lambda m, n: tvm.sum(
            A[kb, m, kt].astype('int32') * B[kb, n, kt].astype('int32'),
            axis=[kb, kt]),
        name='C')

    Ab = tvm.decl_buffer(A.shape, A.dtype,
                        name="A",
                        offset_factor=KTile,
                        strides=[MTile * KTile, KTile, 1])
    Bb = tvm.decl_buffer(B.shape, B.dtype,
                        name="B",
                        offset_factor=KTile,
                        strides=[NTile * KTile, KTile, 1])
    Cb = tvm.decl_buffer(C.shape, C.dtype,
                        name="C",
                        offset_factor=1,
                        strides=[tvm.var("ldc"), 1])

    def intrin_func(ins, outs):
        aa, bb = ins
        print(outs[0])
        print(outs[0].strides[0])
        cc = outs[0]

        def body():
            irb = tvm.ir_builder.create()
            extern_call = tvm.call_extern(
                "int32",
                "qgemm_compute_{M}x{N}x{K}__{ARCH}".format(M=MTile, N=NTile, K=KTile, ARCH=ARCH),
                KBlock * KTile,
                irb.buffer_ptr(aa),
                aa.elem_offset,
                irb.buffer_ptr(bb),
                bb.elem_offset,
                irb.buffer_ptr(cc),
                cc.elem_offset,
                cc.strides[0])
            irb.emit(extern_call)
            return irb.get()
        return body()

    with tvm.build_config():
        return tvm.decl_tensor_intrin(C.op,
                                      intrin_func,
                                      binds={A: Ab, B: Bb, C: Cb})

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
M = 64
K = 64
N = 64
FLOPS = 2 * M * N * K
REPEAT = 10
# The default tensor type in tvm
dtype_input = "uint8"
dtype_output = "uint32"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
# target = 'llvm'
ctx = tvm.context(str(target), 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype_input), ctx)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype_input), ctx)


np_repeat = REPEAT
np_runing_time = timeit.timeit(setup='import numpy\n'
                                     'M = ' + str(M) + '\n'
                                     'K = ' + str(K) + '\n'
                                     'N = ' + str(N) + '\n'
                                     'dtype = "uint32"\n'
                                     'a = numpy.random.rand(M, K).astype(dtype)\n'
                                     'b = numpy.random.rand(K, N).astype(dtype)\n',
                               stmt='answer = numpy.dot(a, b)',
                               number=np_repeat)
print("Numpy running time: %f" % (FLOPS * np_repeat / np_runing_time / 1.0E9))

answer = numpy.dot(a.asnumpy(), b.asnumpy())

# Algorithm
k = tvm.reduce_axis((0, K), 'k')
A = tvm.placeholder((M, K), name='A', dtype=dtype_input)
B = tvm.placeholder((K, N), name='B', dtype=dtype_input)
C = tvm.compute(
           (M, N),
           lambda x, y: tvm.sum(A[x, k].astype(dtype_output) * B[k, y].astype(dtype_output), axis=k),
           name='C')

# Default schedule
s = tvm.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype_output), ctx)
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=REPEAT)
print('Baseline flops: %f' % (FLOPS / evaluator(a, b, c).mean / 1E9))


################################################################################################
# In TVM, we can always inspect lower level IR to debug or optimize our schedule.
# Here is the generated IR using our baseline schedule.

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################################
# Blocking
# --------
# A important trick to enhance the cache hit rate is blocking --- data chunk will be computed
# block by block. The memory access inside the block is a small neighbourhood which is with high
# memory locality. In this tutorial, I picked up 32 as the blocking factor. So the block will
# fill 32 * 32 * sizeof(float) which is 4KB in the cache whose total size is 32KB (L1 data cache)

bn = 32
s = tvm.create_schedule(C.op)

# Blocking by loop tiling
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
k, = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# Hoist reduction domain outside the blocking loop
s[C].reorder(xo, yo, ko, ki, xi, yi)

func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype = dtype_output), ctx)
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

# By simply tiling the loop 32x32, and hoisting ko, ki outside the blocking loops,
# we can see big speedup compared with the baseline.
evaluator = func.time_evaluator(func.entry_name, ctx, number=REPEAT)
print('Opt1: %f' % (FLOPS / evaluator(a, b, c).mean / 1E9))

################################################################################################
# Here is the generated IR after blocking.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################
# Vectorization
# -------------
# Another important trick is vectorization. When the memory access pattern is uniform,
# the compiler can detect this pattern and pass the continuous memory to vector processor. In TVM,
# we can use `vectorize` interface to hint the compiler this pattern, so that we can accelerate it vastly.
#
# In this tutorial, we chose to vectorize the inner loop row data since it is cache friendly.

s = tvm.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
k, = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

s[C].reorder(xo, yo, ko, ki, xi, yi)

# Vectorization
s[C].vectorize(yi)

func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype = dtype_output), ctx)
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=REPEAT)
print('Opt2: %f' % (FLOPS / evaluator(a, b, c).mean / 1E9))


################################################################################################
# Here is the generated IR after vectorization.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################
# Loop Permutation
# ----------------
# If we look at the above IR, we can see the inner loop row data is vectorized and
# B is transformed into PackedB. The traversal of PackedB is sequential now.
# So we will look at the access pattern of A. In current schedule, A is accessed column by column
# which is not cache friendly. If we change the nested loop order of ki and inner axes xi,
# the access pattern for A matrix is more cache friendly.

s = tvm.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
k, = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

# re-ordering
s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype = dtype_output), ctx)
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=REPEAT)
print('Opt3: %f' % (FLOPS / evaluator(a, b, c).mean / 1E9))


################################################################################################
# Here is the generated IR after loop permutation.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################
# Array Packing
# -------------
# Another important trick is array packing. This trick is to reorder the storage dimension of the
# array to convert the continuous access pattern on certain dimension to a sequential pattern after
# flattening.
#
# .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/array-packing.png
#      :align: center
#      :scale: 100%
#


###################################################################################################
# Just as it is shown in the figure above, after blocking the computations, we can observe the array
# access pattern of B (after flattening), which is regular but discontinuous. We expect that after
# some transformation we can get continuous access pattern. We can reorder a [16][16] array to
# a [16/4][16][4] array, so that the access pattern of B will be sequential when grabing
# the corresponding value from the packed array.
#

# We have to re-write the algorithm slightly.
packedB = tvm.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name='packedB')
C = tvm.compute((M, N),
                lambda x, y: tvm.sum(A[x, k].astype(dtype_output) * packedB[y / bn, k, y % bn].astype(dtype_output), axis=k),
                name = 'C')

s = tvm.create_schedule(C.op)

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
k, = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype = dtype_output), ctx)
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=REPEAT)
print('Opt4: %f' % (FLOPS / evaluator(a, b, c).mean / 1E9))


################################################################################################
# Here is the generated IR after array packing.

print(tvm.lower(s, [A, B, C], simple_mode=True))

################################################################################################
# Write cache for blocks
# ----------------------
# After blocking, the program will write result to C block by block, the access pattern
# is not sequential. So we can use a sequential cache array to hold the block results and
# write to C when all the block results are ready.
#

s = tvm.create_schedule(C.op)

# Allocate write cache
CC = s.cache_write(C, 'global')

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

# Write cache is computed at yo
s[CC].compute_at(s[C], yo)

# New inner axes
xc, yc = s[CC].op.axis

k, = s[CC].op.reduce_axis
ko, ki = s[CC].split(k, factor=4)
s[CC].reorder(ko, xc, ki, yc)
s[CC].unroll(ki)
s[CC].vectorize(yc)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype = dtype_output), ctx)
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=REPEAT)
print('Opt5: %f' % (FLOPS / evaluator(a, b, c).mean / 1E9))


################################################################################################
# Here is the generated IR after blocking.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################
# Parallel
# --------
# Futhermore, we can also utilize multi-core processors to do the thread-level parallelization.

s = tvm.create_schedule(C.op)

CC = s.cache_write(C, 'global')

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

s[CC].compute_at(s[C], yo)

xc, yc = s[CC].op.axis

k, = s[CC].op.reduce_axis
ko, ki = s[CC].split(k, factor=4)
s[CC].reorder(ko, xc, ki, yc)
s[CC].unroll(ki)
s[CC].vectorize(yc)

# parallel
s[C].parallel(xo)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

func = tvm.build(s, [A, B, C], target=target, name = 'mmult')
assert func

c = tvm.nd.array(numpy.zeros((M, N), dtype = dtype_output), ctx)
func(a, b, c)
numpy.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=REPEAT)
opt6_time = evaluator(a, b, c).mean
print('Opt6: %f' % (FLOPS / evaluator(a, b, c).mean / 1E9))

################################################################################################
# Here is the generated IR after parallelization.

print(tvm.lower(s, [A, B, C], simple_mode=True))

###################################################################################################

##################################################################################################
# Summary
# -------
# After applying the above simple optimizations with only 18 lines of code,
# our generated code can achieve 60% of the `numpy` performance with MKL.
# Note that the outputs on the web page reflect the running times on a non-exclusive
# Docker container, thereby they are *unreliable*. It is highly encouraged to run the
# tutorial by yourself to observe the performance gain acheived by TVM.

MTile = 8
NTile = 8
KTile = 4
# microkernel

# for m in M:
#     for n in N:
#         for kb in range(0, K, KB):
#             for ko in range(KB):
#                 c[m * N + n] += ((a[m * AStride + kb + ko] as int32)
#                              * (b[kb * packedN + n * KB + ko] as int32))

                # B[N * K] => BPacked[N / NB, K / KB, NB, KB]


ATile = tvm.placeholder(
    (M / MTile, K / KTile, MTile, KTile), dtype='uint8', name='ATile')
BTile = tvm.placeholder(
    (N / NTile, K / KTile, NTile, KTile), dtype='int8', name='BTile')
kb = tvm.reduce_axis((0, K / KTile), name='kblock')
kt = tvm.reduce_axis((0, KTile), name='ktile')
C = tvm.compute(
    (M / MTile, N / NTile, MTile, NTile),
    lambda mb, nb, mt, nt: tvm.sum(
        ATile[mb, kb, mt, kt].astype('int32') * BTile[nb, kb, nt, kt].astype('int32'),
        axis=[kb, kt]),
    name='C')

C = tvm.compute(
    (M, N),
    lambda m, n: tvm.sum(
        ATile[m // MTile, kb, m % MTile, kt].astype('int32') * BTile[n // NTile, kb, n % NTile, kt].astype('int32'),
        axis=[kb, kt]),
    name='C')

print("C", C.shape)
s = tvm.create_schedule(C.op)
(m, n) = s[C].op.axis
(mo, no, mi, ni) = s[C].tile(m, n, MTile, NTile)
s[C].tensorize(mi, intrin_tile_gemm(KBlock=K / KTile, MTile=MTile, NTile=NTile, KTile=KTile))
at = tvm.nd.array(numpy.random.randint(low=0, high=255, size=get_const_tuple(ATile.shape)).astype(ATile.dtype), ctx)
bt = tvm.nd.array(numpy.random.randint(low=-63, high=63, size=get_const_tuple(BTile.shape)).astype(BTile.dtype), ctx)
c = tvm.nd.array(numpy.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)


print("ATile.shape", ATile.shape)
print("BTile.shape", BTile.shape)
print("C.shape", C.shape)
print(tvm.lower(s, [ATile, BTile, C], simple_mode=True))
func = tvm.build(s, [ATile, BTile, C], target=target, name='tensorize')

evaluator = func.time_evaluator(func.entry_name, ctx, number=REPEAT)
opt6_time = evaluator(at, bt, c).mean
print('OptTensorize: %f' % (FLOPS / evaluator(at, bt, c).mean / 1E9))
print(c)

def ref(at, bt):
    result = numpy.zeros(c.shape, dtype=c.dtype)
    for m in range(M):
        for n in range(N):
            acc = 0
            for k in range(K):
                acc += at[m // MTile, k // KTile, m % MTile, k % KTile] * bt[n // NTile, k // KTile, n % NTile, k % KTile]
            result[m, n] = acc
    return result

numpy.testing.assert_allclose(c.asnumpy(), ref(at.asnumpy(), bt.asnumpy()))
