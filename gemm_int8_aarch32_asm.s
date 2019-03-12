	.text
	.syntax unified
	.eabi_attribute	67, "2.09"	@ Tag_conformance
	.eabi_attribute	6, 10	@ Tag_CPU_arch
	.eabi_attribute	7, 65	@ Tag_CPU_arch_profile
	.eabi_attribute	8, 1	@ Tag_ARM_ISA_use
	.eabi_attribute	9, 2	@ Tag_THUMB_ISA_use
	.fpu	neon
	.eabi_attribute	34, 1	@ Tag_CPU_unaligned_access
	.eabi_attribute	17, 1	@ Tag_ABI_PCS_GOT_use
	.eabi_attribute	20, 1	@ Tag_ABI_FP_denormal
	.eabi_attribute	21, 1	@ Tag_ABI_FP_exceptions
	.eabi_attribute	23, 3	@ Tag_ABI_FP_number_model
	.eabi_attribute	24, 1	@ Tag_ABI_align_needed
	.eabi_attribute	25, 1	@ Tag_ABI_align_preserved
	.eabi_attribute	28, 1	@ Tag_ABI_VFP_args
	.eabi_attribute	38, 1	@ Tag_ABI_FP_16bit_format
	.eabi_attribute	18, 4	@ Tag_ABI_PCS_wchar_t
	.eabi_attribute	26, 2	@ Tag_ABI_enum_size
	.eabi_attribute	14, 0	@ Tag_ABI_PCS_R9_use
	.file	"gemm_int8_aarch32_asm.cc"
	.globl	_Z26gemm_ukernel_4x8__neon_asmiPKaiiS0_iiPiii @ -- Begin function _Z26gemm_ukernel_4x8__neon_asmiPKaiiS0_iiPiii
	.p2align	2
	.type	_Z26gemm_ukernel_4x8__neon_asmiPKaiiS0_iiPiii,%function
	.code	32                      @ @_Z26gemm_ukernel_4x8__neon_asmiPKaiiS0_iiPiii
_Z26gemm_ukernel_4x8__neon_asmiPKaiiS0_iiPiii:
	.fnstart
@ %bb.0:
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	.setfp	r11, sp, #28
	add	r11, sp, #28
	.pad	#4
	sub	sp, sp, #4
	.vsave	{d8, d9, d10, d11, d12, d13, d14, d15}
	vpush	{d8, d9, d10, d11, d12, d13, d14, d15}
	.pad	#96
	sub	sp, sp, #96
	ldr	r12, [r11, #28]
	ldr	lr, [r11, #24]
	ldr	r4, [r11, #20]
	ldr	r5, [r11, #16]
	ldr	r6, [r11, #12]
	ldr	r7, [r11, #8]
	str	r0, [sp, #92]
	str	r1, [sp, #88]
	str	r2, [sp, #84]
	str	r3, [sp, #80]
	ldr	r0, [sp, #88]
	ldr	r1, [sp, #84]
	add	r0, r0, r1
	str	r0, [sp, #88]
	ldr	r0, [r11, #8]
	ldr	r1, [r11, #12]
	add	r0, r0, r1
	str	r0, [r11, #8]
	ldr	r0, [r11, #20]
	ldr	r1, [r11, #24]
	add	r0, r0, r1, lsl #2
	str	r0, [r11, #20]
	ldr	r0, [sp, #88]
	str	r0, [sp, #76]
	ldr	r0, [sp, #76]
	ldr	r1, [sp, #80]
	add	r0, r0, r1
	str	r0, [sp, #72]
	ldr	r0, [sp, #72]
	ldr	r1, [sp, #80]
	add	r0, r0, r1
	str	r0, [sp, #68]
	ldr	r0, [sp, #68]
	ldr	r1, [sp, #80]
	add	r0, r0, r1
	str	r0, [sp, #64]
	ldr	r0, [r11, #20]
	str	r0, [sp, #60]
	ldr	r0, [sp, #92]
	str	r0, [sp, #56]
	ldr	r0, [r11, #28]
	lsl	r0, r0, #2
	str	r0, [sp, #52]
	ldr	r0, [sp, #60]
	ldr	r1, [r11, #8]
	ldr	r2, [sp, #76]
	ldr	r3, [sp, #72]
	ldr	r8, [sp, #68]
	ldr	r9, [sp, #64]
	ldr	r10, [sp, #56]
	str	r0, [sp, #48]           @ 4-byte Spill
	ldr	r0, [sp, #52]
	str	r0, [sp, #44]           @ 4-byte Spill
	ldr	r0, [sp, #48]           @ 4-byte Reload
	str	r0, [sp, #40]           @ 4-byte Spill
	ldr	r0, [sp, #44]           @ 4-byte Reload
	str	r0, [sp, #36]           @ 4-byte Spill
	ldr	r0, [sp, #40]           @ 4-byte Reload
	str	r0, [sp, #32]           @ 4-byte Spill
	ldr	r0, [sp, #36]           @ 4-byte Reload
	str	r0, [sp, #28]           @ 4-byte Spill
	ldr	r0, [sp, #32]           @ 4-byte Reload
	str	r12, [sp, #24]          @ 4-byte Spill
	ldr	r12, [sp, #28]          @ 4-byte Reload
	@APP
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp0:
	vld1.8	{d1}, [r2]!
	vld1.8	{d3}, [r3]!
	vld1.8	{d5}, [r8]!
	vld1.8	{d7}, [r9]!
	vmovl.s8	q0, d1
	vmovl.s8	q1, d3
	vmovl.s8	q2, d5
	vmovl.s8	q3, d7
	vld1.8	{d9}, [r1]!
	vmovl.s8	q4, d9
	vmlal.s16	q8, d8, d0[0]
	vmlal.s16	q9, d9, d0[0]
	vmlal.s16	q10, d8, d2[0]
	vmlal.s16	q11, d9, d2[0]
	vmlal.s16	q12, d8, d4[0]
	vmlal.s16	q13, d9, d4[0]
	vmlal.s16	q14, d8, d6[0]
	vmlal.s16	q15, d9, d6[0]
	vld1.8	{d11}, [r1]!
	vmovl.s8	q5, d11
	vmlal.s16	q8, d10, d0[1]
	vmlal.s16	q9, d11, d0[1]
	vmlal.s16	q10, d10, d2[1]
	vmlal.s16	q11, d11, d2[1]
	vmlal.s16	q12, d10, d4[1]
	vmlal.s16	q13, d11, d4[1]
	vmlal.s16	q14, d10, d6[1]
	vmlal.s16	q15, d11, d6[1]
	vld1.8	{d9}, [r1]!
	vmovl.s8	q4, d9
	vmlal.s16	q8, d8, d0[2]
	vmlal.s16	q9, d9, d0[2]
	vmlal.s16	q10, d8, d2[2]
	vmlal.s16	q11, d9, d2[2]
	vmlal.s16	q12, d8, d4[2]
	vmlal.s16	q13, d9, d4[2]
	vmlal.s16	q14, d8, d6[2]
	vmlal.s16	q15, d9, d6[2]
	vld1.8	{d11}, [r1]!
	vmovl.s8	q5, d11
	vmlal.s16	q8, d10, d0[3]
	vmlal.s16	q9, d11, d0[3]
	vmlal.s16	q10, d10, d2[3]
	vmlal.s16	q11, d11, d2[3]
	vmlal.s16	q12, d10, d4[3]
	vmlal.s16	q13, d11, d4[3]
	vmlal.s16	q14, d10, d6[3]
	vmlal.s16	q15, d11, d6[3]
	vld1.8	{d9}, [r1]!
	vmovl.s8	q4, d9
	vmlal.s16	q8, d8, d1[0]
	vmlal.s16	q9, d9, d1[0]
	vmlal.s16	q10, d8, d3[0]
	vmlal.s16	q11, d9, d3[0]
	vmlal.s16	q12, d8, d5[0]
	vmlal.s16	q13, d9, d5[0]
	vmlal.s16	q14, d8, d7[0]
	vmlal.s16	q15, d9, d7[0]
	vld1.8	{d11}, [r1]!
	vmovl.s8	q5, d11
	vmlal.s16	q8, d10, d1[1]
	vmlal.s16	q9, d11, d1[1]
	vmlal.s16	q10, d10, d3[1]
	vmlal.s16	q11, d11, d3[1]
	vmlal.s16	q12, d10, d5[1]
	vmlal.s16	q13, d11, d5[1]
	vmlal.s16	q14, d10, d7[1]
	vmlal.s16	q15, d11, d7[1]
	vld1.8	{d9}, [r1]!
	vmovl.s8	q4, d9
	vmlal.s16	q8, d8, d1[2]
	vmlal.s16	q9, d9, d1[2]
	vmlal.s16	q10, d8, d3[2]
	vmlal.s16	q11, d9, d3[2]
	vmlal.s16	q12, d8, d5[2]
	vmlal.s16	q13, d9, d5[2]
	vmlal.s16	q14, d8, d7[2]
	vmlal.s16	q15, d9, d7[2]
	vld1.8	{d11}, [r1]!
	vmovl.s8	q5, d11
	vmlal.s16	q8, d10, d1[3]
	vmlal.s16	q9, d11, d1[3]
	vmlal.s16	q10, d10, d3[3]
	vmlal.s16	q11, d11, d3[3]
	vmlal.s16	q12, d10, d5[3]
	vmlal.s16	q13, d11, d5[3]
	vmlal.s16	q14, d10, d7[3]
	vmlal.s16	q15, d11, d7[3]
	subs	r10, r10, #8
	bne	.Ltmp0
	vst1.32	{d16, d17, d18, d19}, [r0], r12
	vst1.32	{d20, d21, d22, d23}, [r0], r12
	vst1.32	{d24, d25, d26, d27}, [r0], r12
	vst1.32	{d28, d29, d30, d31}, [r0], r12

	@NO_APP
	str	r0, [sp, #32]           @ 4-byte Spill
	mov	r0, r12
	str	r0, [sp, #20]           @ 4-byte Spill
	ldr	r0, [sp, #32]           @ 4-byte Reload
	str	r0, [sp, #60]
	str	r1, [r11, #8]
	str	r2, [sp, #76]
	str	r3, [sp, #72]
	str	r8, [sp, #68]
	str	r9, [sp, #64]
	str	r10, [sp, #56]
	ldr	r0, [sp, #20]           @ 4-byte Reload
	str	r0, [sp, #52]
	str	r7, [sp, #16]           @ 4-byte Spill
	str	r4, [sp, #12]           @ 4-byte Spill
	str	r5, [sp, #8]            @ 4-byte Spill
	str	r6, [sp, #4]            @ 4-byte Spill
	str	r12, [sp, #28]          @ 4-byte Spill
	str	lr, [sp]                @ 4-byte Spill
	sub	sp, r11, #96
	vpop	{d8, d9, d10, d11, d12, d13, d14, d15}
	add	sp, sp, #4
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.Lfunc_end0:
	.size	_Z26gemm_ukernel_4x8__neon_asmiPKaiiS0_iiPiii, .Lfunc_end0-_Z26gemm_ukernel_4x8__neon_asmiPKaiiS0_iiPiii
	.cantunwind
	.fnend
                                        @ -- End function

	.ident	"clang version 7.0.0 (tags/RELEASE_700/final)"
	.section	".note.GNU-stack","",%progbits
	.addrsig
	.addrsig_sym _Z26gemm_ukernel_4x8__neon_asmiPKaiiS0_iiPiii
	.eabi_attribute	30, 6	@ Tag_ABI_optimization_goals
