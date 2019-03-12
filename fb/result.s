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
	.eabi_attribute	38, 1	@ Tag_ABI_FP_16bit_format
	.eabi_attribute	14, 0	@ Tag_ABI_PCS_R9_use
	.file	"default_function"
	.globl	default_function        @ -- Begin function default_function
	.p2align	2
	.type	default_function,%function
	.code	32                      @ @default_function
default_function:
	.fnstart
@ %bb.0:                                @ %entry
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	sub	sp, sp, #12
	cmp	r2, #3
	bne	.LBB0_46
@ %bb.1:                                @ %assert_end
	ldr	r6, [r0]
	ldr	r3, [r0, #8]
	ldr	lr, [r0, #16]
	ldr	r0, [r6]
	str	r0, [sp, #4]            @ 4-byte Spill
	ldr	r0, [r6, #24]
	ldr	r7, [r1]
	ldr	r5, [r1, #4]
	cmp	r0, #0
	ldr	r4, [r1, #8]
	ldr	r10, [r6, #20]
	beq	.LBB0_4
@ %bb.2:                                @ %if_then
	ldr	r1, [r0, #8]
	cmp	r1, #1
	ldreq	r0, [r0]
	cmpeq	r0, #32
	beq	.LBB0_4
@ %bb.3:                                @ %assert_fail1
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.1
	movt	r0, :upper16:.L.str.1
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_4:                                @ %if_end
	ldr	r0, [r3, #24]
	ldr	r1, [r3]
	ldr	r9, [r3, #20]
	cmp	r0, #0
	ldr	r11, [r6, #4]
	ldr	r2, [r6, #8]
	str	r2, [sp, #8]            @ 4-byte Spill
	beq	.LBB0_7
@ %bb.5:                                @ %if_then3
	ldr	r2, [r0, #8]
	cmp	r2, #1
	ldreq	r0, [r0]
	cmpeq	r0, #8
	beq	.LBB0_7
@ %bb.6:                                @ %assert_fail5
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.2
	movt	r0, :upper16:.L.str.2
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_7:                                @ %if_end4
	ldr	r12, [lr, #24]
	ldr	r0, [lr]
	ldr	r8, [lr, #20]
	cmp	r12, #0
	beq	.LBB0_10
@ %bb.8:                                @ %if_then7
	ldr	r2, [r12, #8]
	cmp	r2, #1
	ldreq	r2, [r12]
	cmpeq	r2, #8
	beq	.LBB0_10
@ %bb.9:                                @ %assert_fail9
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.3
	movt	r0, :upper16:.L.str.3
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_10:                               @ %if_end8
	cmp	r7, #13
	bhi	.LBB0_27
@ %bb.11:                               @ %if_end8
	mov	r2, #1
	mov	r12, r1
	movw	r1, #8344
	tst	r1, r2, lsl r7
	beq	.LBB0_27
@ %bb.12:                               @ %assert_end12
	cmp	r5, #13
	bhi	.LBB0_28
@ %bb.13:                               @ %assert_end12
	mov	r1, #1
	movw	r2, #8344
	tst	r2, r1, lsl r5
	beq	.LBB0_28
@ %bb.14:                               @ %assert_end14
	cmp	r4, #13
	bhi	.LBB0_29
@ %bb.15:                               @ %assert_end14
	mov	r1, #1
	movw	r2, #8344
	tst	r2, r1, lsl r4
	beq	.LBB0_29
@ %bb.16:                               @ %assert_end16
	cmp	r11, #1
	bne	.LBB0_47
@ %bb.17:                               @ %assert_end18
	ldr	r1, [r6, #12]
	cmp	r1, #2
	bne	.LBB0_48
@ %bb.18:                               @ %assert_end20
	ldrb	r1, [r6, #16]
	cmp	r1, #0
	ldrbeq	r1, [r6, #17]
	cmpeq	r1, #8
	beq	.LBB0_20
.LBB0_19:                               @ %assert_fail21
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.9
	movt	r0, :upper16:.L.str.9
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_20:                               @ %assert_end20
	ldrh	r1, [r6, #18]
	cmp	r1, #1
	bne	.LBB0_19
@ %bb.21:                               @ %assert_end22
	ldr	r1, [r10]
	cmp	r1, #4
	bne	.LBB0_49
@ %bb.22:                               @ %assert_end24
	ldr	r1, [r10, #8]
	cmp	r1, #32
	bne	.LBB0_50
@ %bb.23:                               @ %assert_end26
	ldr	r2, [r6, #32]
	ldr	r1, [r6, #36]
	orrs	r1, r2, r1
	bne	.LBB0_51
@ %bb.24:                               @ %assert_end28
	ldr	r1, [r3, #12]
	cmp	r1, #2
	bne	.LBB0_52
@ %bb.25:                               @ %assert_end30
	ldrb	r1, [r3, #16]
	cmp	r1, #0
	ldrbeq	r1, [r3, #17]
	cmpeq	r1, #8
	beq	.LBB0_30
.LBB0_26:                               @ %assert_fail31
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.14
	movt	r0, :upper16:.L.str.14
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_27:                               @ %assert_fail11
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.4
	movt	r0, :upper16:.L.str.4
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_28:                               @ %assert_fail13
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.5
	movt	r0, :upper16:.L.str.5
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_29:                               @ %assert_fail15
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.6
	movt	r0, :upper16:.L.str.6
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_30:                               @ %assert_end30
	ldrh	r1, [r3, #18]
	cmp	r1, #1
	bne	.LBB0_26
@ %bb.31:                               @ %assert_end32
	ldr	r1, [r9]
	cmp	r1, #32
	bne	.LBB0_53
@ %bb.32:                               @ %assert_end34
	ldr	r1, [r9, #8]
	cmp	r1, #8
	bne	.LBB0_54
@ %bb.33:                               @ %assert_end36
	ldr	r2, [r3, #32]
	ldr	r1, [r3, #36]
	ldr	r4, [sp, #8]            @ 4-byte Reload
	orrs	r1, r2, r1
	bne	.LBB0_55
@ %bb.34:                               @ %assert_end38
	ldr	r1, [r3, #4]
	cmp	r1, #1
	bne	.LBB0_56
@ %bb.35:                               @ %assert_end40
	ldr	r1, [r3, #8]
	cmp	r4, r1
	bne	.LBB0_57
@ %bb.36:                               @ %assert_end42
	ldr	r1, [lr, #12]
	cmp	r1, #2
	bne	.LBB0_58
@ %bb.37:                               @ %assert_end44
	ldrb	r1, [lr, #16]
	cmp	r1, #0
	ldrbeq	r1, [lr, #17]
	cmpeq	r1, #32
	beq	.LBB0_39
.LBB0_38:                               @ %assert_fail45
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.21
	movt	r0, :upper16:.L.str.21
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_39:                               @ %assert_end44
	ldrh	r1, [lr, #18]
	cmp	r1, #1
	bne	.LBB0_38
@ %bb.40:                               @ %assert_end46
	ldr	r1, [r8]
	cmp	r1, #4
	bne	.LBB0_59
@ %bb.41:                               @ %assert_end48
	ldr	r1, [r8, #8]
	cmp	r1, #8
	bne	.LBB0_60
@ %bb.42:                               @ %assert_end50
	ldrd	r2, r3, [lr, #32]
	orrs	r1, r2, r3
	bne	.LBB0_61
@ %bb.43:                               @ %assert_end52
	ldr	r1, [lr, #4]
	cmp	r1, #1
	bne	.LBB0_62
@ %bb.44:                               @ %assert_end54
	ldr	r1, [lr, #8]
	cmp	r4, r1
	bne	.LBB0_63
@ %bb.45:                               @ %assert_end56
	ldr	r2, [sp, #4]            @ 4-byte Reload
	mov	r1, r12
	bl	.Ldefault_function_compute_
	mov	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_46:                               @ %assert_fail
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str
	movt	r0, :upper16:.L.str
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_47:                               @ %assert_fail17
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.7
	movt	r0, :upper16:.L.str.7
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_48:                               @ %assert_fail19
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.8
	movt	r0, :upper16:.L.str.8
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_49:                               @ %assert_fail23
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.10
	movt	r0, :upper16:.L.str.10
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_50:                               @ %assert_fail25
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.11
	movt	r0, :upper16:.L.str.11
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_51:                               @ %assert_fail27
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.12
	movt	r0, :upper16:.L.str.12
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_52:                               @ %assert_fail29
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.13
	movt	r0, :upper16:.L.str.13
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_53:                               @ %assert_fail33
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.15
	movt	r0, :upper16:.L.str.15
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_54:                               @ %assert_fail35
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.16
	movt	r0, :upper16:.L.str.16
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_55:                               @ %assert_fail37
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.17
	movt	r0, :upper16:.L.str.17
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_56:                               @ %assert_fail39
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.18
	movt	r0, :upper16:.L.str.18
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_57:                               @ %assert_fail41
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.19
	movt	r0, :upper16:.L.str.19
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_58:                               @ %assert_fail43
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.20
	movt	r0, :upper16:.L.str.20
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_59:                               @ %assert_fail47
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.22
	movt	r0, :upper16:.L.str.22
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_60:                               @ %assert_fail49
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.23
	movt	r0, :upper16:.L.str.23
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_61:                               @ %assert_fail51
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.24
	movt	r0, :upper16:.L.str.24
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_62:                               @ %assert_fail53
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.25
	movt	r0, :upper16:.L.str.25
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_63:                               @ %assert_fail55
	movw	r0, :lower16:__TVMAPISetLastError
	movt	r0, :upper16:__TVMAPISetLastError
	ldr	r1, [r0]
	movw	r0, :lower16:.L.str.26
	movt	r0, :upper16:.L.str.26
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.Lfunc_end0:
	.size	default_function, .Lfunc_end0-default_function
	.fnend
                                        @ -- End function
	.p2align	2               @ -- Begin function default_function_compute_
	.type	.Ldefault_function_compute_,%function
	.code	32                      @ @default_function_compute_
.Ldefault_function_compute_:
	.fnstart
@ %bb.0:                                @ %entry
	add	r3, r1, #64
	vld1.8	{d19}, [r2:64]
	vld1.8	{d16}, [r3:64]
	add	r3, r2, #8
	vld1.8	{d17}, [r3:64]
	add	r3, r2, #16
	vmovl.s8	q10, d17
	vld1.8	{d18}, [r1:64]
	vmovl.s8	q8, d16
	vld1.8	{d22}, [r3:64]
	add	r3, r1, #128
	add	r1, r1, #192
	vdup.16	d23, d20[0]
	vmovl.s8	q10, d19
	vmull.s16	q12, d23, d17
	vld1.8	{d26}, [r3:64]
	vmovl.s8	q9, d18
	add	r3, r2, #24
	vmovl.s8	q15, d22
	vld1.8	{d22}, [r1:64]
	add	r1, r0, #16
	vdup.16	d27, d20[0]
	vmovl.s8	q10, d26
	vmlal.s16	q12, d27, d19
	vld1.8	{d28}, [r3:64]
	mov	r3, #112
	vdup.16	d26, d30[0]
	vmull.s16	q15, d23, d16
	vmovl.s8	q14, d28
	vmlal.s16	q12, d26, d21
	vmovl.s8	q11, d22
	vmlal.s16	q15, d27, d18
	vdup.16	d28, d28[0]
	vmlal.s16	q12, d28, d23
	vmlal.s16	q15, d26, d20
	vst1.64	{d24, d25}, [r1:128]
	vmlal.s16	q15, d28, d22
	mov	r1, r0
	vst1.32	{d30, d31}, [r1:128], r3
	add	r3, r2, #40
	vld1.8	{d24}, [r3:64]
	add	r3, r2, #32
	vmovl.s8	q12, d24
	vld1.8	{d26}, [r3:64]
	add	r3, r2, #48
	vmovl.s8	q13, d26
	vdup.16	d24, d24[0]
	vmull.s16	q14, d24, d17
	vld1.8	{d25}, [r3:64]
	vmull.s16	q15, d24, d16
	add	r3, r2, #56
	vmovl.s8	q12, d25
	vdup.16	d26, d26[0]
	vmlal.s16	q14, d26, d19
	vmlal.s16	q15, d26, d18
	vld1.8	{d26}, [r3:64]
	add	r3, r0, #48
	vdup.16	d24, d24[0]
	vmovl.s8	q13, d26
	vmlal.s16	q14, d24, d21
	vmlal.s16	q15, d24, d20
	vdup.16	d24, d26[0]
	vmlal.s16	q14, d24, d23
	vmlal.s16	q15, d24, d22
	vst1.64	{d28, d29}, [r3:128]
	add	r3, r0, #32
	vst1.64	{d30, d31}, [r3:128]
	add	r3, r2, #72
	vld1.8	{d24}, [r3:64]
	add	r3, r2, #64
	vmovl.s8	q12, d24
	vld1.8	{d26}, [r3:64]
	add	r3, r2, #80
	vmovl.s8	q13, d26
	vdup.16	d24, d24[0]
	vmull.s16	q14, d24, d17
	vld1.8	{d25}, [r3:64]
	vmull.s16	q15, d24, d16
	add	r3, r2, #88
	vmovl.s8	q12, d25
	vdup.16	d26, d26[0]
	vmlal.s16	q14, d26, d19
	vmlal.s16	q15, d26, d18
	vld1.8	{d26}, [r3:64]
	add	r3, r0, #80
	vdup.16	d24, d24[0]
	vmovl.s8	q13, d26
	vmlal.s16	q14, d24, d21
	vmlal.s16	q15, d24, d20
	vdup.16	d24, d26[0]
	vmlal.s16	q14, d24, d23
	vmlal.s16	q15, d24, d22
	vst1.64	{d28, d29}, [r3:128]
	add	r3, r0, #64
	add	r0, r0, #96
	vst1.64	{d30, d31}, [r3:128]
	add	r3, r2, #104
	vld1.8	{d24}, [r3:64]
	add	r3, r2, #96
	vmovl.s8	q12, d24
	vld1.8	{d26}, [r3:64]
	add	r3, r2, #112
	add	r2, r2, #120
	vmovl.s8	q13, d26
	vdup.16	d24, d24[0]
	vmull.s16	q14, d24, d17
	vld1.8	{d25}, [r3:64]
	vmull.s16	q8, d24, d16
	vdup.16	d26, d26[0]
	vmovl.s8	q12, d25
	vmlal.s16	q14, d26, d19
	vmlal.s16	q8, d26, d18
	vld1.8	{d18}, [r2:64]
	vdup.16	d19, d24[0]
	vmovl.s8	q12, d18
	vmlal.s16	q14, d19, d21
	vmlal.s16	q8, d19, d20
	vdup.16	d18, d24[0]
	vmlal.s16	q14, d18, d23
	vmlal.s16	q8, d18, d22
	vst1.64	{d28, d29}, [r1:128]
	vst1.64	{d16, d17}, [r0:128]
	bx	lr
.Lfunc_end1:
	.size	.Ldefault_function_compute_, .Lfunc_end1-.Ldefault_function_compute_
	.cantunwind
	.fnend
                                        @ -- End function
	.type	__TVMAPISetLastError,%object @ @__TVMAPISetLastError
	.bss
	.weak	__TVMAPISetLastError
	.p2align	2
__TVMAPISetLastError:
	.long	0
	.size	__TVMAPISetLastError, 4

	.type	.L.str,%object          @ @.str
	.section	.rodata,"a",%progbits
.L.str:
	.asciz	"Assert fail: (num_args == 3), default_function: num_args should be 3"
	.size	.L.str, 69

	.type	.L.str.1,%object        @ @.str.1
.L.str.1:
	.asciz	"Assert fail: ((1 == int32(arg0.strides[1])) && (32 == int32(arg0.strides[0]))), arg0.strides: expected to be compact array"
	.size	.L.str.1, 123

	.type	.L.str.2,%object        @ @.str.2
.L.str.2:
	.asciz	"Assert fail: ((1 == int32(arg1.strides[1])) && (8 == int32(arg1.strides[0]))), arg1.strides: expected to be compact array"
	.size	.L.str.2, 122

	.type	.L.str.3,%object        @ @.str.3
.L.str.3:
	.asciz	"Assert fail: ((1 == int32(arg2.strides[1])) && (8 == int32(arg2.strides[0]))), arg2.strides: expected to be compact array"
	.size	.L.str.3, 122

	.type	.L.str.4,%object        @ @.str.4
.L.str.4:
	.asciz	"Assert fail: ((((arg0.code == 3) || (arg0.code == 13)) || (arg0.code == 7)) || (arg0.code == 4)), default_function: Expect arg[0] to be pointer"
	.size	.L.str.4, 144

	.type	.L.str.5,%object        @ @.str.5
.L.str.5:
	.asciz	"Assert fail: ((((arg1.code == 3) || (arg1.code == 13)) || (arg1.code == 7)) || (arg1.code == 4)), default_function: Expect arg[1] to be pointer"
	.size	.L.str.5, 144

	.type	.L.str.6,%object        @ @.str.6
.L.str.6:
	.asciz	"Assert fail: ((((arg2.code == 3) || (arg2.code == 13)) || (arg2.code == 7)) || (arg2.code == 4)), default_function: Expect arg[2] to be pointer"
	.size	.L.str.6, 144

	.type	.L.str.7,%object        @ @.str.7
.L.str.7:
	.asciz	"Assert fail: (dev_type == 1), device_type need to be 1"
	.size	.L.str.7, 55

	.type	.L.str.8,%object        @ @.str.8
.L.str.8:
	.asciz	"Assert fail: (2 == tvm_struct_get(arg0, 0, 4)), arg0.ndim is expected to equal 2"
	.size	.L.str.8, 81

	.type	.L.str.9,%object        @ @.str.9
.L.str.9:
	.asciz	"Assert fail: (((tvm_struct_get(arg0, 0, 5) == (uint8)0) && (tvm_struct_get(arg0, 0, 6) == (uint8)8)) && (tvm_struct_get(arg0, 0, 7) == (uint16)1)), arg0.dtype is expected to be int8"
	.size	.L.str.9, 182

	.type	.L.str.10,%object       @ @.str.10
.L.str.10:
	.asciz	"Assert fail: (int32(arg0.shape[0]) == 4), Argument arg0.shape[0] has an unsatisfied constraint"
	.size	.L.str.10, 95

	.type	.L.str.11,%object       @ @.str.11
.L.str.11:
	.asciz	"Assert fail: (int32(arg0.shape[1]) == 32), Argument arg0.shape[1] has an unsatisfied constraint"
	.size	.L.str.11, 96

	.type	.L.str.12,%object       @ @.str.12
.L.str.12:
	.asciz	"Assert fail: (tvm_struct_get(arg0, 0, 8) == (uint64)0), Argument arg0.byte_offset has an unsatisfied constraint"
	.size	.L.str.12, 112

	.type	.L.str.13,%object       @ @.str.13
.L.str.13:
	.asciz	"Assert fail: (2 == tvm_struct_get(arg1, 0, 4)), arg1.ndim is expected to equal 2"
	.size	.L.str.13, 81

	.type	.L.str.14,%object       @ @.str.14
.L.str.14:
	.asciz	"Assert fail: (((tvm_struct_get(arg1, 0, 5) == (uint8)0) && (tvm_struct_get(arg1, 0, 6) == (uint8)8)) && (tvm_struct_get(arg1, 0, 7) == (uint16)1)), arg1.dtype is expected to be int8"
	.size	.L.str.14, 182

	.type	.L.str.15,%object       @ @.str.15
.L.str.15:
	.asciz	"Assert fail: (int32(arg1.shape[0]) == 32), Argument arg1.shape[0] has an unsatisfied constraint"
	.size	.L.str.15, 96

	.type	.L.str.16,%object       @ @.str.16
.L.str.16:
	.asciz	"Assert fail: (int32(arg1.shape[1]) == 8), Argument arg1.shape[1] has an unsatisfied constraint"
	.size	.L.str.16, 95

	.type	.L.str.17,%object       @ @.str.17
.L.str.17:
	.asciz	"Assert fail: (tvm_struct_get(arg1, 0, 8) == (uint64)0), Argument arg1.byte_offset has an unsatisfied constraint"
	.size	.L.str.17, 112

	.type	.L.str.18,%object       @ @.str.18
.L.str.18:
	.asciz	"Assert fail: (1 == tvm_struct_get(arg1, 0, 10)), Argument arg1.device_type has an unsatisfied constraint"
	.size	.L.str.18, 105

	.type	.L.str.19,%object       @ @.str.19
.L.str.19:
	.asciz	"Assert fail: (dev_id == tvm_struct_get(arg1, 0, 9)), Argument arg1.device_id has an unsatisfied constraint"
	.size	.L.str.19, 107

	.type	.L.str.20,%object       @ @.str.20
.L.str.20:
	.asciz	"Assert fail: (2 == tvm_struct_get(arg2, 0, 4)), arg2.ndim is expected to equal 2"
	.size	.L.str.20, 81

	.type	.L.str.21,%object       @ @.str.21
.L.str.21:
	.asciz	"Assert fail: (((tvm_struct_get(arg2, 0, 5) == (uint8)0) && (tvm_struct_get(arg2, 0, 6) == (uint8)32)) && (tvm_struct_get(arg2, 0, 7) == (uint16)1)), arg2.dtype is expected to be int32"
	.size	.L.str.21, 184

	.type	.L.str.22,%object       @ @.str.22
.L.str.22:
	.asciz	"Assert fail: (int32(arg2.shape[0]) == 4), Argument arg2.shape[0] has an unsatisfied constraint"
	.size	.L.str.22, 95

	.type	.L.str.23,%object       @ @.str.23
.L.str.23:
	.asciz	"Assert fail: (int32(arg2.shape[1]) == 8), Argument arg2.shape[1] has an unsatisfied constraint"
	.size	.L.str.23, 95

	.type	.L.str.24,%object       @ @.str.24
.L.str.24:
	.asciz	"Assert fail: (tvm_struct_get(arg2, 0, 8) == (uint64)0), Argument arg2.byte_offset has an unsatisfied constraint"
	.size	.L.str.24, 112

	.type	.L.str.25,%object       @ @.str.25
.L.str.25:
	.asciz	"Assert fail: (1 == tvm_struct_get(arg2, 0, 10)), Argument arg2.device_type has an unsatisfied constraint"
	.size	.L.str.25, 105

	.type	.L.str.26,%object       @ @.str.26
.L.str.26:
	.asciz	"Assert fail: (dev_id == tvm_struct_get(arg2, 0, 9)), Argument arg2.device_id has an unsatisfied constraint"
	.size	.L.str.26, 107

	.type	__tvm_main__,%object    @ @__tvm_main__
	.weak	__tvm_main__
__tvm_main__:
	.asciz	"default_function"
	.size	__tvm_main__, 17


	.section	".note.GNU-stack","",%progbits
