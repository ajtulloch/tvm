import tvm
import re
import os
import ctypes

def test_popcount():
    target = 'llvm -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon'

    def check_correct_assembly(type, elements, counts):
        n = tvm.convert(elements)
        A = tvm.placeholder(n, dtype=type, name='A')
        B = tvm.compute(A.shape, lambda i: tvm.popcount(A[i]), name='B')
        s = tvm.create_schedule(B.op)
        s[B].vectorize(s[B].op.axis[0])
        f = tvm.build(s, [A, B], target)

        # Verify we see the correct number of vpaddl and vcnt instructions in the assembly
        assembly = f.get_source('asm')
        matches = re.findall("vpaddl", assembly)
        assert (len(matches) == counts)
        matches = re.findall("vcnt", assembly)
        assert (len(matches) == 1)
    check_correct_assembly('uint16', 8, 1)
    check_correct_assembly('uint16', 4, 1)
    check_correct_assembly('uint32', 4, 2)
    check_correct_assembly('uint32', 2, 2)
    check_correct_assembly('uint64', 2, 3)

def test_vmlal_s16():
    target = 'llvm -target=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon'

    def check_correct_assembly(N):
        K = tvm.var("K")
        A = tvm.placeholder((K, N), dtype="int8", name='A')
        B = tvm.placeholder((K, N), dtype="int8", name='A')
        k = tvm.reduce_axis((0, K))
        C = tvm.compute((N, ), lambda n: tvm.sum(
            A[k, n].astype("int32") * B[k, n].astype("int32"), axis=[k]), name='C')
        s = tvm.create_schedule(C.op)
        s[C].vectorize(s[C].op.axis[0])
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        f = tvm.build(s, [A, B, C], target)

        # Verify we see the correct number of vmlal.s16 instructions
        assembly = f.get_source('asm')
        matches = re.findall("vmlal.s16", assembly)
        assert (len(matches) == N // 4)
    check_correct_assembly(4)
    check_correct_assembly(8)
    check_correct_assembly(16)

    def check_broadcast_correct_assembly(N):
        K = tvm.var("K")
        A = tvm.placeholder((K, N), dtype="int8", name='A')
        B = tvm.placeholder((K,), dtype="int8", name='A')
        k = tvm.reduce_axis((0, K))
        C = tvm.compute((N, ), lambda n: tvm.sum(
            A[k, n].astype("int32") * B[k].astype("int32"),
            axis=[k]), name='C')
        s = tvm.create_schedule(C.op)
        s[C].vectorize(s[C].op.axis[0])
        f = tvm.build(s, [A, B, C], target)

        # Verify we see the correct number of vmlal.s16 instructions
        assembly = f.get_source('asm')
        matches = re.findall("vmlal.s16", assembly)
        assert len(matches) == N // 4
    check_broadcast_correct_assembly(8)
    check_broadcast_correct_assembly(16)
    check_broadcast_correct_assembly(32)
    check_broadcast_correct_assembly(64)

if __name__ == "__main__":
    test_popcount()
    test_vmlal_s16()
