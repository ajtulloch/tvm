import os
from tvm.contrib import rpc, util
import numpy as np
import tvm
import topi
from util import generate_quantized_np
from topi.util import get_const_tuple, simplify, get_const_int, equal_const_int

# Todo: option to accumulate in 32 bit or 16 bit
# Inputs of 8, 16, 32 bit
def intrin_popcount(m, k_i, w_b, x_b):
    type = 'uint8'
    w = tvm.placeholder((w_b, m, k_i), dtype=type, name='w')
    x = tvm.placeholder((x_b, k_i,), dtype=type, name='x')
    k = tvm.reduce_axis((0, k_i), name='k')
    bw = tvm.reduce_axis((0, w_b), name='bw')
    bx = tvm.reduce_axis((0, x_b), name='bx')
    z = tvm.compute((m,), lambda i:
                    tvm.sum(tvm.popcount(w[bw, i, k].astype('uint16') & x[bx, k].astype('uint16')) << (bw+bx).astype('uint16'),
                     axis=[bw, bx, k]), name='z')

    Wb = tvm.decl_buffer(w.shape, w.dtype,
                        name="W",
                        offset_factor=k_i,
                        strides=[tvm.var('ldw'), tvm.var('ldw'), 1]) 
    Xb = tvm.decl_buffer(x.shape, x.dtype,
                        name="X",
                        offset_factor=k_i,
                        strides=[tvm.var('ldw'), 1])

            
    def intrin_func(ins, outs):
        ww, xx = ins
        zz = outs[0]
        vpadd_id = tvm.const(647, 'uint32')
        vpadalu_id = tvm.const(646, 'uint32')
        args_1 = tvm.const(1, 'uint32')
        args_2 = tvm.const(2, 'uint32')
    
        def instr(index):
            irb = tvm.ir_builder.create()
            if index == 1:
                irb.emit(zz.vstore(0, tvm.const(0, 'uint16x8')))
            else:
                cnts8 = [None] * 8
                cnts4 = [None] * 4
                cnts2 = [None] * 2
                for bw in range(w_b):
                    for bx in range(x_b):
                        if k_i == 16:
                            for i in range(m):
                                ands = ww.vload([bw, i, 0], 'uint8x16') & xx.vload([bx, 0], 'uint8x16')
                                cnts = tvm.popcount(ands)
                                upper_half = tvm.call_pure_intrin('uint8x8', 'vectorhigh', cnts)
                                lower_half = tvm.call_pure_intrin('uint8x8', 'vectorlow', cnts)
                                cnts8[i] = upper_half + lower_half
                            for i in range(m/2):
                                cnts4[i] = tvm.call_pure_intrin('uint8x8', 'llvm_intrin', vpadd_id, args_1, cnts8[i*2], cnts8[i*2+1])
                            for i in range(m/4):
                                cnts2[i] = tvm.call_pure_intrin('uint8x8', 'llvm_intrin', vpadd_id, args_1, cnts4[i*2], cnts4[i*2+1])
                            cnts = tvm.call_pure_intrin('uint8x16', 'vectorcombine', cnts2[0], cnts2[1])
                            shifted_cnts = cnts << (bw+bx)
                            out = tvm.call_pure_intrin('uint16x8', 'llvm_intrin', vpadalu_id, args_2, zz.vload(0, 'uint16x8'), shifted_cnts)
                        else: # ki ==8
                            for i in range(m):
                                ands = ww.vload([bw, i, 0], 'uint8x8') & xx.vload([bx, 0], 'uint8x8')
                                cnts8[i] = tvm.popcount(ands)
                            for i in range(m/2):
                                cnts4[i] = tvm.call_pure_intrin('uint8x8', 'llvm_intrin', vpadd_id, args_1, cnts8[i*2], cnts8[i*2+1])
                            for i in range(m/4):
                                cnts2[i] = tvm.call_pure_intrin('uint8x8', 'llvm_intrin', vpadd_id, args_1, cnts4[i*2], cnts4[i*2+1])
                            cnts = tvm.call_pure_intrin('uint8x16', 'vectorcombine', cnts2[0], cnts2[1])
                            shifted_cnts = cnts << (bw+bx)
                            out = tvm.call_pure_intrin('uint16x8', 'llvm_intrin', vpadalu_id, args_2, zz.vload(0, 'uint16x8'), shifted_cnts)
                        irb.emit(zz.vstore(0, out))
            return irb.get()
        # body, reset, update
        return instr(0), instr(1), instr(2)
    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(z.op, intrin_func, binds={w: Wb, x:Xb})
