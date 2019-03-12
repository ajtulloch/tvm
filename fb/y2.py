import tvm

def _intrin_Kx4xint8_Kx8xint8_4x8_int32(K):
    X = tvm.placeholder((4, K), dtype="int16", name='X')
    W = tvm.placeholder((K, 8), dtype="int16", name='X')
    k = tvm.reduce_axis((0, K), name='k')
    Z = tvm.compute((4, 8), lambda i, j: tvm.sum(X[i, k].astype("int32") * W[k, j].astype("int32"), axis=[k]), name="Z")

    Xb = tvm.decl_buffer(X.shape, X.dtype,
                         name="Xb",
                         offset_factor=K,
                         strides=[tvm.var('ldX'), 1])
    Wb = tvm.decl_buffer(W.shape, W.dtype,
                         name="Wb",
                         offset_factor=8,
                         strides=[tvm.var('ldW'), 1])

    def _intrin_func(ins, outs):
        xx, ww = ins
        zz = outs[0]
        # vpadd = "llvm.arm.neon.vpadd.v8u8"
        # vpadalu = "llvm.arm.neon.vmlal.v4i32.v8i16"
        # args_1 = tvm.const(1, 'uint32')
        # args_2 = tvm.const(2, 'uint32')
        def vl(x):
            return tvm.call_pure_intrin('int16x4', 'vectorlow', x)

        def vh(x):
            return tvm.call_pure_intrin('int16x4', 'vectorhigh', x)

        def vi(x, i):
            return tvm.call_pure_intrin('int16x4', 'vectorindex', x, i)


        def _instr(index):
            irb = tvm.ir_builder.create()
            # if index == 1:
            #     irb.emit(zz.vstore([0, 0], tvm.const(0, 'int32x8')))
            #     irb.emit(zz.vstore([1, 0], tvm.const(0, 'int32x8')))
            #     irb.emit(zz.vstore([2, 0], tvm.const(0, 'int32x8')))
            #     irb.emit(zz.vstore([3, 0], tvm.const(0, 'int32x8')))
            #     return irb.get()

            accs = [[tvm.const(0, 'int32x4'), (tvm.const(0, 'int32x4'))] for _ in range(4)]
            assert K % 8 == 0
            for k in range(0, K, 8):
                va0l = xx.vload([0, k], 'int16x4')
                va1l = xx.vload([1, k], 'int16x4')
                va2l = xx.vload([2, k], 'int16x4')
                va3l = xx.vload([3, k], 'int16x4')
                # va0l = vl(va0)
                # va1l = vl(va1)
                # va2l = vl(va2)
                # va3l = vl(va3)

                vxbl = ww.vload([k, 0], 'int16x4')
                vxbh = ww.vload([k, 4], 'int16x4')

                accs[0][0] = accs[0][0] + vxbl.astype('int32x4') * (vi(va0l, 0)).astype("int32x4")
                accs[0][1] = accs[0][1] + vxbh.astype('int32x4') * (vi(va0l, 0)).astype("int32x4")
                accs[1][0] = accs[1][0] + vxbl.astype('int32x4') * (vi(va1l, 0)).astype("int32x4")
                accs[1][1] = accs[1][1] + vxbh.astype('int32x4') * (vi(va1l, 0)).astype("int32x4")
                accs[2][0] = accs[2][0] + vxbl.astype('int32x4') * (vi(va2l, 0)).astype("int32x4")
                accs[2][1] = accs[2][1] + vxbh.astype('int32x4') * (vi(va2l, 0)).astype("int32x4")
                accs[3][0] = accs[3][0] + vxbl.astype('int32x4') * (vi(va3l, 0)).astype("int32x4")
                accs[3][1] = accs[3][1] + vxbh.astype('int32x4') * (vi(va3l, 0)).astype("int32x4")

                vxbl = ww.vload([k + 1, 0], 'int16x4')
                vxbh = ww.vload([k + 1, 4], 'int16x4')


                accs[0][0] = accs[0][0] + vxbl.astype('int32x4') * (vi(va0l, 1)).astype("int32x4")
                accs[0][1] = accs[0][1] + vxbh.astype('int32x4') * (vi(va0l, 1)).astype("int32x4")
                accs[1][0] = accs[1][0] + vxbl.astype('int32x4') * (vi(va1l, 1)).astype("int32x4")
                accs[1][1] = accs[1][1] + vxbh.astype('int32x4') * (vi(va1l, 1)).astype("int32x4")
                accs[2][0] = accs[2][0] + vxbl.astype('int32x4') * (vi(va2l, 1)).astype("int32x4")
                accs[2][1] = accs[2][1] + vxbh.astype('int32x4') * (vi(va2l, 1)).astype("int32x4")
                accs[3][0] = accs[3][0] + vxbl.astype('int32x4') * (vi(va3l, 1)).astype("int32x4")
                accs[3][1] = accs[3][1] + vxbh.astype('int32x4') * (vi(va3l, 1)).astype("int32x4")

                vxbl = ww.vload([k + 2, 0], 'int16x4')
                vxbh = ww.vload([k + 2, 4], 'int16x4')

                accs[0][0] = accs[0][0] + vxbl.astype('int32x4') * (vi(va0l, 2)).astype("int32x4")
                accs[0][1] = accs[0][1] + vxbh.astype('int32x4') * (vi(va0l, 2)).astype("int32x4")
                accs[1][0] = accs[1][0] + vxbl.astype('int32x4') * (vi(va1l, 2)).astype("int32x4")
                accs[1][1] = accs[1][1] + vxbh.astype('int32x4') * (vi(va1l, 2)).astype("int32x4")
                accs[2][0] = accs[2][0] + vxbl.astype('int32x4') * (vi(va2l, 2)).astype("int32x4")
                accs[2][1] = accs[2][1] + vxbh.astype('int32x4') * (vi(va2l, 2)).astype("int32x4")
                accs[3][0] = accs[3][0] + vxbl.astype('int32x4') * (vi(va3l, 2)).astype("int32x4")
                accs[3][1] = accs[3][1] + vxbh.astype('int32x4') * (vi(va3l, 2)).astype("int32x4")

                vxbl = ww.vload([k + 3, 0], 'int16x4')
                vxbh = ww.vload([k + 3, 4], 'int16x4')

                accs[0][0] = accs[0][0] + vxbl.astype('int32x4') * (vi(va0l, 3)).astype("int32x4")
                accs[0][1] = accs[0][1] + vxbh.astype('int32x4') * (vi(va0l, 3)).astype("int32x4")
                accs[1][0] = accs[1][0] + vxbl.astype('int32x4') * (vi(va1l, 3)).astype("int32x4")
                accs[1][1] = accs[1][1] + vxbh.astype('int32x4') * (vi(va1l, 3)).astype("int32x4")
                accs[2][0] = accs[2][0] + vxbl.astype('int32x4') * (vi(va2l, 3)).astype("int32x4")
                accs[2][1] = accs[2][1] + vxbh.astype('int32x4') * (vi(va2l, 3)).astype("int32x4")
                accs[3][0] = accs[3][0] + vxbl.astype('int32x4') * (vi(va3l, 3)).astype("int32x4")
                accs[3][1] = accs[3][1] + vxbh.astype('int32x4') * (vi(va3l, 3)).astype("int32x4")

                # vxb = ww.vload([k + 4, 0], 'int8x8').astype('int16x8')

                # accs[0][0] = accs[0][0] + vxbl.astype('int32x4') * (vi(vh(va0), 0)).astype("int32x4")
                # accs[0][1] = accs[0][1] + vxbh.astype('int32x4') * (vi(vh(va0), 0)).astype("int32x4")
                # accs[1][0] = accs[1][0] + vxbl.astype('int32x4') * (vi(vh(va1), 0)).astype("int32x4")
                # accs[1][1] = accs[1][1] + vxbh.astype('int32x4') * (vi(vh(va1), 0)).astype("int32x4")
                # accs[2][0] = accs[2][0] + vxbl.astype('int32x4') * (vi(vh(va2), 0)).astype("int32x4")
                # accs[2][1] = accs[2][1] + vxbh.astype('int32x4') * (vi(vh(va2), 0)).astype("int32x4")
                # accs[3][0] = accs[3][0] + vxbl.astype('int32x4') * (vi(vh(va3), 0)).astype("int32x4")
                # accs[3][1] = accs[3][1] + vxbh.astype('int32x4') * (vi(vh(va3), 0)).astype("int32x4")

                # vxb = ww.vload([k + 5, 0], 'int8x8').astype('int16x8')

                # accs[0][0] = accs[0][0] + vxbl.astype('int32x4') * (vi(vh(va0), 1)).astype("int32x4")
                # accs[0][1] = accs[0][1] + vxbh.astype('int32x4') * (vi(vh(va0), 1)).astype("int32x4")
                # accs[1][0] = accs[1][0] + vxbl.astype('int32x4') * (vi(vh(va1), 1)).astype("int32x4")
                # accs[1][1] = accs[1][1] + vxbh.astype('int32x4') * (vi(vh(va1), 1)).astype("int32x4")
                # accs[2][0] = accs[2][0] + vxbl.astype('int32x4') * (vi(vh(va2), 1)).astype("int32x4")
                # accs[2][1] = accs[2][1] + vxbh.astype('int32x4') * (vi(vh(va2), 1)).astype("int32x4")
                # accs[3][0] = accs[3][0] + vxbl.astype('int32x4') * (vi(vh(va3), 1)).astype("int32x4")
                # accs[3][1] = accs[3][1] + vxbh.astype('int32x4') * (vi(vh(va3), 1)).astype("int32x4")

                # vxb = ww.vload([k + 6, 0], 'int8x8').astype('int16x8')

                # accs[0][0] = accs[0][0] + vxbl.astype('int32x4') * (vi(vh(va0), 2)).astype("int32x4")
                # accs[0][1] = accs[0][1] + vxbh.astype('int32x4') * (vi(vh(va0), 2)).astype("int32x4")
                # accs[1][0] = accs[1][0] + vxbl.astype('int32x4') * (vi(vh(va1), 2)).astype("int32x4")
                # accs[1][1] = accs[1][1] + vxbh.astype('int32x4') * (vi(vh(va1), 2)).astype("int32x4")
                # accs[2][0] = accs[2][0] + vxbl.astype('int32x4') * (vi(vh(va2), 2)).astype("int32x4")
                # accs[2][1] = accs[2][1] + vxbh.astype('int32x4') * (vi(vh(va2), 2)).astype("int32x4")
                # accs[3][0] = accs[3][0] + vxbl.astype('int32x4') * (vi(vh(va3), 2)).astype("int32x4")
                # accs[3][1] = accs[3][1] + vxbh.astype('int32x4') * (vi(vh(va3), 2)).astype("int32x4")


                # vxb = ww.vload([k + 7, 0], 'int8x8').astype('int16x8')

                # accs[0][0] = accs[0][0] + vxbl.astype('int32x4') * (vi(vh(va0), 3)).astype("int32x4")
                # accs[0][1] = accs[0][1] + vxbh.astype('int32x4') * (vi(vh(va0), 3)).astype("int32x4")
                # accs[1][0] = accs[1][0] + vxbl.astype('int32x4') * (vi(vh(va1), 3)).astype("int32x4")
                # accs[1][1] = accs[1][1] + vxbh.astype('int32x4') * (vi(vh(va1), 3)).astype("int32x4")
                # accs[2][0] = accs[2][0] + vxbl.astype('int32x4') * (vi(vh(va2), 3)).astype("int32x4")
                # accs[2][1] = accs[2][1] + vxbh.astype('int32x4') * (vi(vh(va2), 3)).astype("int32x4")
                # accs[3][0] = accs[3][0] + vxbl.astype('int32x4') * (vi(vh(va3), 3)).astype("int32x4")
                # accs[3][1] = accs[3][1] + vxbh.astype('int32x4') * (vi(vh(va3), 3)).astype("int32x4")

            print(accs[0][0])
            irb.emit(zz.vstore([0, 0], accs[0][0]))
            irb.emit(zz.vstore([0, 4], accs[0][1]))
            irb.emit(zz.vstore([1, 0], accs[1][0]))
            irb.emit(zz.vstore([1, 4], accs[1][1]))
            irb.emit(zz.vstore([2, 0], accs[2][0]))
            irb.emit(zz.vstore([2, 4], accs[2][1]))
            irb.emit(zz.vstore([3, 0], accs[3][0]))
            irb.emit(zz.vstore([3, 4], accs[3][1]))

            return irb.get()
        # body, reset, update
        return _instr(0)
    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(Z.op, _intrin_func, binds={X: Xb, W:Wb})

K = 32

X = tvm.placeholder((4, K), dtype="int16", name='X')
W = tvm.placeholder((K, 8), dtype="int16", name='X')
k = tvm.reduce_axis((0, K), name='k')
Z = tvm.compute((4, 8), lambda i, j: tvm.sum(X[i, k].astype("int32") * W[k, j].astype("int32"), axis=[k]), name="Z")
s = tvm.create_schedule(Z.op)
print(tvm.lower(s, [X, W, Z], simple_mode=True))
s[Z].tensorize(s[Z].op.axis[0], _intrin_Kx4xint8_Kx8xint8_4x8_int32(K))

target = tvm.target.arm_cpu("rasp3b")
with target:
    with tvm.build_config(dump_pass_ir=True):
        lib = tvm.build(s, [X, W, Z])
print(lib.get_source("asm"))
print(lib.get_source("ll"))

