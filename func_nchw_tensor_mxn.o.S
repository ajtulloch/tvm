	.text
	.syntax unified
	.eabi_attribute	67, "2.09"
	.cpu	cortex-a53
	.eabi_attribute	6, 14
	.eabi_attribute	7, 65
	.eabi_attribute	8, 1
	.eabi_attribute	9, 2
	.fpu	crypto-neon-fp-armv8
	.eabi_attribute	12, 3
	.eabi_attribute	36, 1
	.eabi_attribute	42, 1
	.eabi_attribute	34, 1
	.eabi_attribute	68, 3
	.eabi_attribute	15, 1
	.eabi_attribute	16, 1
	.eabi_attribute	17, 2
	.eabi_attribute	20, 1
	.eabi_attribute	21, 1
	.eabi_attribute	23, 3
	.eabi_attribute	24, 1
	.eabi_attribute	25, 1
	.eabi_attribute	28, 1
	.eabi_attribute	38, 1
	.eabi_attribute	18, 4
	.eabi_attribute	26, 2
	.eabi_attribute	14, 0
	.file	"default_function"
	.globl	default_function
	.p2align	3
	.type	default_function,%function
	.code	32
default_function:
	.fnstart
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	.pad	#12
	sub	sp, sp, #12
	cmp	r2, #3
	bne	.LBB0_59
	ldr	r5, [r1]
	ldmib	r1, {r4, r9}
	ldr	r6, [r0]
	ldr	lr, [r0, #8]
	ldr	r12, [r0, #16]
	ldr	r1, [r6, #24]
	ldr	r0, [r6]
	ldr	r7, [r6, #20]
	cmp	r1, #0
	str	r0, [sp, #4]
	beq	.LBB0_6
	add	r0, r1, #16
	vldr	d18, .LCPI0_67
	vld1.64	{d16, d17}, [r0]
	vmovn.i64	d16, q8
	vceq.i32	d16, d16, d18
	vmov.32	r0, d16[1]
	tst	r0, #1
	beq	.LBB0_5
	vmov.32	r0, d16[0]
	tst	r0, #1
	beq	.LBB0_5
	ldr	r0, [r1, #8]
	cmp	r0, #196
	ldreq	r0, [r1]
	cmpeq	r0, #100352
	beq	.LBB0_6
.LBB0_5:
	ldr	r0, .LCPI0_61
.LPC0_60:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_62
.LPC0_61:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_6:
	ldr	r2, [lr, #24]
	ldr	r0, [r6, #8]
	ldr	r1, [lr]
	ldr	r10, [lr, #20]
	ldr	r3, [r6, #4]
	cmp	r2, #0
	str	r0, [sp, #8]
	beq	.LBB0_11
	add	r0, r2, #16
	vmov.i32	d18, #0x1
	vld1.64	{d16, d17}, [r0]
	vmovn.i64	d16, q8
	vceq.i32	d16, d16, d18
	vmov.32	r0, d16[1]
	tst	r0, #1
	beq	.LBB0_10
	vmov.32	r0, d16[0]
	tst	r0, #1
	beq	.LBB0_10
	ldr	r0, [r2, #8]
	cmp	r0, #1
	ldreq	r0, [r2]
	cmpeq	r0, #512
	beq	.LBB0_11
.LBB0_10:
	ldr	r0, .LCPI0_63
.LPC0_62:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_64
.LPC0_63:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_11:
	ldr	r11, [r12, #24]
	ldr	r2, [r12]
	ldr	r8, [r12, #20]
	cmp	r11, #0
	beq	.LBB0_16
	add	r0, r11, #16
	vldr	d18, .LCPI0_67
	vld1.64	{d16, d17}, [r0]
	vmovn.i64	d16, q8
	vceq.i32	d16, d16, d18
	vmov.32	r0, d16[1]
	tst	r0, #1
	beq	.LBB0_15
	vmov.32	r0, d16[0]
	tst	r0, #1
	beq	.LBB0_15
	ldr	r0, [r11, #8]
	cmp	r0, #196
	ldreq	r0, [r11]
	cmpeq	r0, #100352
	beq	.LBB0_16
.LBB0_15:
	ldr	r0, .LCPI0_65
.LPC0_64:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_66
.LPC0_65:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_16:
	mov	r11, r1
	cmp	r5, #13
	bhi	.LBB0_35
	mov	r0, #1
	movw	r1, #8344
	tst	r1, r0, lsl r5
	beq	.LBB0_35
	cmp	r4, #13
	bhi	.LBB0_36
	mov	r0, #1
	movw	r1, #8344
	tst	r1, r0, lsl r4
	beq	.LBB0_36
	cmp	r9, #13
	bhi	.LBB0_37
	mov	r0, #1
	movw	r1, #8344
	tst	r1, r0, lsl r9
	beq	.LBB0_37
	cmp	r3, #1
	bne	.LBB0_60
	ldr	r0, [r6, #12]
	cmp	r0, #4
	bne	.LBB0_61
	ldrb	r0, [r6, #16]
	cmp	r0, #2
	ldrbeq	r0, [r6, #17]
	cmpeq	r0, #32
	beq	.LBB0_26
.LBB0_25:
	ldr	r0, .LCPI0_13
.LPC0_12:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_14
.LPC0_13:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_26:
	ldrh	r0, [r6, #18]
	cmp	r0, #1
	bne	.LBB0_25
	ldr	r0, [r7]
	cmp	r0, #1
	bne	.LBB0_63
	ldr	r0, [r7, #8]
	cmp	r0, #512
	bne	.LBB0_64
	ldr	r0, [r7, #16]
	cmp	r0, #14
	bne	.LBB0_65
	ldr	r0, [r7, #24]
	cmp	r0, #14
	bne	.LBB0_66
	ldrd	r0, r1, [r6, #32]
	orrs	r0, r0, r1
	bne	.LBB0_67
	ldr	r0, [lr, #12]
	cmp	r0, #4
	bne	.LBB0_68
	ldrb	r0, [lr, #16]
	cmp	r0, #2
	ldrbeq	r0, [lr, #17]
	cmpeq	r0, #32
	beq	.LBB0_38
.LBB0_34:
	ldr	r0, .LCPI0_27
.LPC0_26:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_28
.LPC0_27:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_35:
	ldr	r0, .LCPI0_3
.LPC0_2:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_4
.LPC0_3:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_36:
	ldr	r0, .LCPI0_5
.LPC0_4:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_6
.LPC0_5:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_37:
	ldr	r0, .LCPI0_7
.LPC0_6:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_8
.LPC0_7:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_38:
	ldrh	r0, [lr, #18]
	cmp	r0, #1
	bne	.LBB0_34
	ldr	r0, [r10]
	cmp	r0, #512
	bne	.LBB0_69
	ldr	r0, [r10, #8]
	cmp	r0, #512
	bne	.LBB0_70
	ldr	r0, [r10, #16]
	cmp	r0, #1
	bne	.LBB0_71
	ldr	r0, [r10, #24]
	cmp	r0, #1
	bne	.LBB0_72
	ldrd	r0, r1, [lr, #32]
	orrs	r0, r0, r1
	bne	.LBB0_73
	ldr	r0, [lr, #4]
	cmp	r0, #1
	bne	.LBB0_74
	ldr	r0, [lr, #8]
	ldr	r3, [sp, #8]
	cmp	r3, r0
	bne	.LBB0_75
	ldr	r0, [r12, #12]
	cmp	r0, #4
	bne	.LBB0_76
	ldrb	r0, [r12, #16]
	cmp	r0, #2
	ldrbeq	r0, [r12, #17]
	cmpeq	r0, #32
	beq	.LBB0_50
.LBB0_48:
	ldr	r0, .LCPI0_45
.LPC0_44:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_46
.LPC0_45:
	add	r0, pc, r0
.LBB0_49:
	blx	r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_50:
	ldrh	r0, [r12, #18]
	cmp	r0, #1
	bne	.LBB0_48
	ldr	r0, [r8]
	cmp	r0, #1
	bne	.LBB0_77
	ldr	r0, [r8, #8]
	cmp	r0, #512
	bne	.LBB0_78
	ldr	r0, [r8, #16]
	cmp	r0, #14
	bne	.LBB0_79
	ldr	r0, [r8, #24]
	cmp	r0, #14
	bne	.LBB0_80
	ldrd	r0, r1, [r12, #32]
	orrs	r0, r0, r1
	bne	.LBB0_81
	ldr	r0, [r12, #4]
	mov	r1, r11
	cmp	r0, #1
	bne	.LBB0_82
	ldr	r0, [r12, #8]
	cmp	r3, r0
	bne	.LBB0_83
	ldr	r0, [sp, #4]
	bl	.Ldefault_function_compute_
	mov	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.LBB0_59:
	ldr	r0, .LCPI0_1
.LPC0_0:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_2
.LPC0_1:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_60:
	ldr	r0, .LCPI0_9
.LPC0_8:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_10
.LPC0_9:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_61:
	ldr	r0, .LCPI0_11
.LPC0_10:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_12
.LPC0_11:
	add	r0, pc, r0
	b	.LBB0_49
	.p2align	3
.LCPI0_67:
	.long	14
	.long	1
.LBB0_63:
	ldr	r0, .LCPI0_15
.LPC0_14:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_16
.LPC0_15:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_64:
	ldr	r0, .LCPI0_17
.LPC0_16:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_18
.LPC0_17:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_65:
	ldr	r0, .LCPI0_19
.LPC0_18:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_20
.LPC0_19:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_66:
	ldr	r0, .LCPI0_21
.LPC0_20:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_22
.LPC0_21:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_67:
	ldr	r0, .LCPI0_23
.LPC0_22:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_24
.LPC0_23:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_68:
	ldr	r0, .LCPI0_25
.LPC0_24:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_26
.LPC0_25:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_69:
	ldr	r0, .LCPI0_29
.LPC0_28:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_30
.LPC0_29:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_70:
	ldr	r0, .LCPI0_31
.LPC0_30:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_32
.LPC0_31:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_71:
	ldr	r0, .LCPI0_33
.LPC0_32:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_34
.LPC0_33:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_72:
	ldr	r0, .LCPI0_35
.LPC0_34:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_36
.LPC0_35:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_73:
	ldr	r0, .LCPI0_37
.LPC0_36:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_38
.LPC0_37:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_74:
	ldr	r0, .LCPI0_39
.LPC0_38:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_40
.LPC0_39:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_75:
	ldr	r0, .LCPI0_41
.LPC0_40:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_42
.LPC0_41:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_76:
	ldr	r0, .LCPI0_43
.LPC0_42:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_44
.LPC0_43:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_77:
	ldr	r0, .LCPI0_47
.LPC0_46:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_48
.LPC0_47:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_78:
	ldr	r0, .LCPI0_49
.LPC0_48:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_50
.LPC0_49:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_79:
	ldr	r0, .LCPI0_51
.LPC0_50:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_52
.LPC0_51:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_80:
	ldr	r0, .LCPI0_53
.LPC0_52:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_54
.LPC0_53:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_81:
	ldr	r0, .LCPI0_55
.LPC0_54:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_56
.LPC0_55:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_82:
	ldr	r0, .LCPI0_57
.LPC0_56:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_58
.LPC0_57:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_83:
	ldr	r0, .LCPI0_59
.LPC0_58:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_60
.LPC0_59:
	add	r0, pc, r0
	b	.LBB0_49
	.p2align	2
.LCPI0_1:
.Ltmp0:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_0+8)-.Ltmp0)
.LCPI0_2:
	.long	.L.str-(.LPC0_1+8)
.LCPI0_3:
.Ltmp1:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_2+8)-.Ltmp1)
.LCPI0_4:
	.long	.L.str.4-(.LPC0_3+8)
.LCPI0_5:
.Ltmp2:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_4+8)-.Ltmp2)
.LCPI0_6:
	.long	.L.str.5-(.LPC0_5+8)
.LCPI0_7:
.Ltmp3:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_6+8)-.Ltmp3)
.LCPI0_8:
	.long	.L.str.6-(.LPC0_7+8)
.LCPI0_9:
.Ltmp4:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_8+8)-.Ltmp4)
.LCPI0_10:
	.long	.L.str.7-(.LPC0_9+8)
.LCPI0_11:
.Ltmp5:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_10+8)-.Ltmp5)
.LCPI0_12:
	.long	.L.str.8-(.LPC0_11+8)
.LCPI0_13:
.Ltmp6:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_12+8)-.Ltmp6)
.LCPI0_14:
	.long	.L.str.9-(.LPC0_13+8)
.LCPI0_15:
.Ltmp7:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_14+8)-.Ltmp7)
.LCPI0_16:
	.long	.L.str.10-(.LPC0_15+8)
.LCPI0_17:
.Ltmp8:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_16+8)-.Ltmp8)
.LCPI0_18:
	.long	.L.str.11-(.LPC0_17+8)
.LCPI0_19:
.Ltmp9:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_18+8)-.Ltmp9)
.LCPI0_20:
	.long	.L.str.12-(.LPC0_19+8)
.LCPI0_21:
.Ltmp10:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_20+8)-.Ltmp10)
.LCPI0_22:
	.long	.L.str.13-(.LPC0_21+8)
.LCPI0_23:
.Ltmp11:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_22+8)-.Ltmp11)
.LCPI0_24:
	.long	.L.str.14-(.LPC0_23+8)
.LCPI0_25:
.Ltmp12:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_24+8)-.Ltmp12)
.LCPI0_26:
	.long	.L.str.15-(.LPC0_25+8)
.LCPI0_27:
.Ltmp13:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_26+8)-.Ltmp13)
.LCPI0_28:
	.long	.L.str.16-(.LPC0_27+8)
.LCPI0_29:
.Ltmp14:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_28+8)-.Ltmp14)
.LCPI0_30:
	.long	.L.str.17-(.LPC0_29+8)
.LCPI0_31:
.Ltmp15:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_30+8)-.Ltmp15)
.LCPI0_32:
	.long	.L.str.18-(.LPC0_31+8)
.LCPI0_33:
.Ltmp16:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_32+8)-.Ltmp16)
.LCPI0_34:
	.long	.L.str.19-(.LPC0_33+8)
.LCPI0_35:
.Ltmp17:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_34+8)-.Ltmp17)
.LCPI0_36:
	.long	.L.str.20-(.LPC0_35+8)
.LCPI0_37:
.Ltmp18:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_36+8)-.Ltmp18)
.LCPI0_38:
	.long	.L.str.21-(.LPC0_37+8)
.LCPI0_39:
.Ltmp19:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_38+8)-.Ltmp19)
.LCPI0_40:
	.long	.L.str.22-(.LPC0_39+8)
.LCPI0_41:
.Ltmp20:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_40+8)-.Ltmp20)
.LCPI0_42:
	.long	.L.str.23-(.LPC0_41+8)
.LCPI0_43:
.Ltmp21:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_42+8)-.Ltmp21)
.LCPI0_44:
	.long	.L.str.24-(.LPC0_43+8)
.LCPI0_45:
.Ltmp22:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_44+8)-.Ltmp22)
.LCPI0_46:
	.long	.L.str.25-(.LPC0_45+8)
.LCPI0_47:
.Ltmp23:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_46+8)-.Ltmp23)
.LCPI0_48:
	.long	.L.str.26-(.LPC0_47+8)
.LCPI0_49:
.Ltmp24:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_48+8)-.Ltmp24)
.LCPI0_50:
	.long	.L.str.27-(.LPC0_49+8)
.LCPI0_51:
.Ltmp25:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_50+8)-.Ltmp25)
.LCPI0_52:
	.long	.L.str.28-(.LPC0_51+8)
.LCPI0_53:
.Ltmp26:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_52+8)-.Ltmp26)
.LCPI0_54:
	.long	.L.str.29-(.LPC0_53+8)
.LCPI0_55:
.Ltmp27:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_54+8)-.Ltmp27)
.LCPI0_56:
	.long	.L.str.30-(.LPC0_55+8)
.LCPI0_57:
.Ltmp28:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_56+8)-.Ltmp28)
.LCPI0_58:
	.long	.L.str.31-(.LPC0_57+8)
.LCPI0_59:
.Ltmp29:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_58+8)-.Ltmp29)
.LCPI0_60:
	.long	.L.str.32-(.LPC0_59+8)
.LCPI0_61:
.Ltmp30:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_60+8)-.Ltmp30)
.LCPI0_62:
	.long	.L.str.1-(.LPC0_61+8)
.LCPI0_63:
.Ltmp31:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_62+8)-.Ltmp31)
.LCPI0_64:
	.long	.L.str.2-(.LPC0_63+8)
.LCPI0_65:
.Ltmp32:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_64+8)-.Ltmp32)
.LCPI0_66:
	.long	.L.str.3-(.LPC0_65+8)
.Lfunc_end0:
	.size	default_function, .Lfunc_end0-default_function
	.fnend

	.p2align	3
	.type	.Ldefault_function_compute_,%function
	.code	32
.Ldefault_function_compute_:
	.fnstart
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	.pad	#4
	sub	sp, sp, #4
	.vsave	{d8, d9, d10, d11, d12, d13, d14, d15}
	vpush	{d8, d9, d10, d11, d12, d13, d14, d15}
	.pad	#168
	sub	sp, sp, #168
	str	r2, [sp, #12]
	mov	r7, r3
	mov	r4, r0
	mov	r0, #32
	mov	r9, r1
	mov	r10, #2
	mov	r2, #405504
	mov	r3, #0
	mov	r5, #0
	ldr	r8, .LCPI1_3
	mov	r1, r7
.LPC1_0:
	ldr	r8, [pc, r8]
	ldr	r6, [r8]
	str	r0, [sp, #4]
	mov	r0, #1
	str	r10, [sp]
	blx	r6
	ldr	r6, [r8]
	mov	r11, r0
	mov	r0, #32
	mov	r1, r7
	mov	r2, #1048576
	mov	r3, #0
	str	r10, [sp]
	str	r0, [sp, #4]
	mov	r0, #1
	blx	r6
	ldr	r6, [r8]
	str	r0, [sp, #20]
	mov	r0, #32
	mov	r1, r7
	mov	r2, #405504
	mov	r3, #0
	str	r10, [sp]
	str	r0, [sp, #4]
	mov	r0, #1
	str	r7, [sp, #16]
	blx	r6
	vldr	d16, .LCPI1_4
	add	r8, r4, #8
	vmov.i32	d17, #0xe
	vmov.i32	d18, #0xc4
	mov	r10, r11
	mov	lr, #0
	str	r0, [sp, #164]
	str	r11, [sp, #84]
.LBB1_1:
	cmp	lr, #32
	bhs	.LBB1_6
	add	r0, lr, lr, lsl #1
	lsl	r0, r0, #1
	vdup.32	d19, r0
	movw	r0, #9363
	vadd.i32	d19, d19, d16
	movt	r0, #37449
	vmov.32	r6, d19[1]
	mov	r7, r0
	lsr	r1, r6, #1
	umull	r1, r2, r1, r7
	lsr	r1, r2, #3
	lsr	r11, r2, #2
	vmov.32	r2, d19[0]
	umull	r1, r0, r1, r7
	lsr	r1, r0, #2
	lsl	r1, r1, #3
	sub	r0, r1, r0, lsr #2
	sub	r3, r11, r0, lsl #1
	lsr	r0, r2, #1
	umull	r0, r1, r0, r7
	lsr	r0, r1, #3
	lsr	r1, r1, #2
	umull	r0, r12, r0, r7
	movw	r7, #33437
	vmov.32	d20[0], r1
	lsr	r0, r12, #2
	movt	r7, #21399
	vmov.32	d20[1], r11
	lsl	r0, r0, #3
	vmls.i32	d19, d20, d17
	sub	r0, r0, r12, lsr #2
	sub	r12, r1, r0, lsl #1
	umull	r1, r0, r6, r7
	umull	r1, r2, r2, r7
	lsr	r1, r2, #6
	vmov.32	d22[0], r12
	lsr	r0, r0, #6
	mov	r2, #0
	vmov.32	d21[0], r1
	vmov.32	d22[1], r3
	mov	r1, #0
	vmov.32	d21[1], r0
	vmla.i32	d19, d22, d17
	mov	r0, r8
	vshl.i32	d20, d21, #9
.LBB1_3:
	ldr	r3, [r0, #-8]
	mov	r6, r10
	vdup.32	d21, r1
	vorr	d22, d19, d19
	add	r1, r1, #1
	vadd.i32	d21, d21, d20
	vmla.i32	d22, d21, d18
	str	r3, [r6, r2]!
	add	r2, r2, #24
	ldr	r3, [r0, #-4]
	cmp	r2, #12288
	str	r3, [r6, #4]
	ldr	r3, [r0]
	str	r3, [r6, #8]
	ldr	r3, [r0, #4]
	add	r0, r0, #784
	str	r3, [r6, #12]
	vmov.32	r3, d22[0]
	ldr	r3, [r4, r3, lsl #2]
	str	r3, [r6, #16]
	vmov.32	r3, d22[1]
	ldr	r3, [r4, r3, lsl #2]
	str	r3, [r6, #20]
	bne	.LBB1_3
	b	.LBB1_8
	.p2align	2
.LCPI1_3:
.Ltmp33:
	.long	__TVMBackendAllocWorkspace(GOT_PREL)-((.LPC1_0+8)-.Ltmp33)
	.p2align	3
.LCPI1_4:
	.long	4
	.long	5
.LBB1_6:
	mov	r0, #0
	mov	r1, r8
.LBB1_7:
	ldr	r2, [r1, #-8]
	mov	r3, r10
	str	r2, [r3, r0]!
	add	r0, r0, #24
	ldr	r2, [r1, #-4]
	cmp	r0, #12288
	str	r2, [r3, #4]
	ldr	r2, [r1]
	str	r2, [r3, #8]
	ldr	r2, [r1, #4]
	add	r1, r1, #784
	str	r5, [r3, #16]
	str	r5, [r3, #20]
	str	r2, [r3, #12]
	bne	.LBB1_7
.LBB1_8:
	add	lr, lr, #1
	add	r10, r10, #12288
	add	r8, r8, #24
	cmp	lr, #33
	bne	.LBB1_1
	ldr	lr, [sp, #20]
	mov	r12, #0
.LBB1_10:
	mov	r3, lr
	mov	r2, #0
.LBB1_11:
	add	r6, r9, r2
	add	r2, r2, #4
	add	r1, r6, #14336
	vldr	s0, [r6]
	add	r8, r6, #8192
	add	r4, r6, #10240
	add	r0, r6, #12288
	add	r7, r6, #2048
	add	r5, r6, #4096
	add	r6, r6, #6144
	cmp	r2, #2048
	vldr	s7, [r1]
	vldr	s3, [r6]
	vldr	s6, [r0]
	add	r0, r3, #32
	vldr	s2, [r5]
	vldr	s5, [r4]
	vldr	s1, [r7]
	vldr	s4, [r8]
	vst1.32	{d0, d1}, [r3:128]!
	vst1.64	{d2, d3}, [r3:128]
	mov	r3, r0
	bne	.LBB1_11
	add	r12, r12, #1
	add	lr, lr, #16384
	add	r9, r9, #16384
	cmp	r12, #64
	bne	.LBB1_10
	ldr	r10, [sp, #164]
	movw	r0, #8288
	movw	r11, #61440
	mov	r2, #0
	vmov.i32	q8, #0x0
	mov	r6, #10240
	movt	r11, #65535
	add	r12, r10, r0
	movw	r0, #4176
	mov	r5, r10
	add	r9, r10, r0
	movw	r0, #6224
	add	r7, r10, r0
	movw	r0, #8272
	add	lr, r10, r0
	movw	r0, #10320
	add	r0, r10, r0
	str	r0, [sp, #60]
	movw	r0, #8224
	add	r4, r10, r0
	movw	r0, #4112
	add	r8, r10, r0
	movw	r0, #6160
	add	r1, r10, r0
	movw	r0, #8208
	add	r3, r10, r0
	movw	r0, #10256
	add	r0, r10, r0
	str	r0, [sp, #52]
	mov	r0, r1
	ldr	r1, [sp, #20]
	str	r1, [sp, #160]
	mov	r1, r3
	mov	r3, #0
	str	r3, [sp, #48]
	str	r12, [sp, #64]
	str	r4, [sp, #56]
.LBB1_14:
	str	r9, [sp, #44]
	str	r7, [sp, #40]
	mov	r3, r7
	str	lr, [sp, #36]
	str	r8, [sp, #32]
	str	r8, [sp, #108]
	str	r0, [sp, #28]
	str	r0, [sp, #104]
	str	r1, [sp, #24]
	str	r1, [sp, #100]
	mov	r7, lr
	mov	r1, r2
	mov	r0, #0
	str	r5, [sp, #116]
	str	r0, [sp, #96]
	str	r2, [sp, #72]
	str	r5, [sp, #68]
.LBB1_15:
	str	r7, [sp, #88]
	str	r3, [sp, #92]
	add	r12, sp, #100
	ldr	r0, [sp, #52]
	str	r1, [sp, #112]
	ldm	r12, {r8, r10, r12}
	ldr	lr, [sp, #116]
	add	r0, r0, r1
	mov	r1, #0
.LBB1_16:
	add	r2, lr, r1
	mov	r3, r2
	add	r7, r2, #16
	vst1.32	{d16, d17}, [r3], r6
	vst1.32	{d16, d17}, [r7]
	add	r7, r2, #2048
	vst1.32	{d16, d17}, [r7]
	add	r7, r2, #2064
	vst1.32	{d16, d17}, [r7]
	add	r7, r2, #4096
	vst1.32	{d16, d17}, [r7]
	add	r7, r12, r1
	vst1.32	{d16, d17}, [r7]
	add	r7, r2, #6144
	add	r2, r2, #8192
	vst1.32	{d16, d17}, [r7]
	add	r7, r10, r1
	vst1.32	{d16, d17}, [r7]
	vst1.32	{d16, d17}, [r2]
	add	r2, r8, r1
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, r1
	add	r1, r1, #12288
	vst1.32	{d16, d17}, [r3]
	cmp	r1, #135168
	vst1.32	{d16, d17}, [r2]
	bne	.LBB1_16
	ldr	r8, [sp, #112]
	ldr	r6, [sp, #164]
	mov	r0, #11
	mov	r1, r8
.LBB1_18:
	add	r2, r6, r1
	subs	r0, r0, #1
	add	r3, r2, #32
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #48
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #2080
	add	r2, r2, #2096
	vst1.32	{d16, d17}, [r3]
	vst1.32	{d16, d17}, [r2]
	add	r2, r4, r1
	add	r1, r1, #12288
	add	r3, r2, r11
	add	r7, r3, #16
	vst1.32	{d16, d17}, [r3]
	vst1.32	{d16, d17}, [r7]
	add	r7, r3, #2048
	add	r3, r3, #2064
	vst1.32	{d16, d17}, [r7]
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #16
	vst1.32	{d16, d17}, [r2]
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #2048
	add	r2, r2, #2064
	vst1.32	{d16, d17}, [r3]
	vst1.32	{d16, d17}, [r2]
	bne	.LBB1_18
	ldr	r0, [sp, #60]
	ldr	r7, [sp, #116]
	ldr	lr, [sp, #64]
	ldr	r5, [sp, #92]
	ldr	r6, [sp, #88]
	mov	r1, #0
	add	r0, r0, r8
.LBB1_20:
	add	r2, r7, r1
	add	r3, r2, #64
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #80
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #2112
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #2128
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #4160
	vst1.32	{d16, d17}, [r3]
	add	r3, r9, r1
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #6208
	vst1.32	{d16, d17}, [r3]
	add	r3, r5, r1
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #8256
	add	r2, r2, #10304
	vst1.32	{d16, d17}, [r3]
	add	r3, r6, r1
	vst1.32	{d16, d17}, [r3]
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, r1
	add	r1, r1, #12288
	cmp	r1, #135168
	vst1.32	{d16, d17}, [r2]
	bne	.LBB1_20
	mov	r0, #11
	mov	r1, r8
	str	r9, [sp, #80]
.LBB1_22:
	ldr	r2, [sp, #164]
	subs	r0, r0, #1
	add	r2, r2, r1
	add	r3, r2, #96
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #112
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #2144
	add	r2, r2, #2160
	vst1.32	{d16, d17}, [r3]
	vst1.32	{d16, d17}, [r2]
	add	r2, lr, r1
	add	r1, r1, #12288
	add	r3, r2, r11
	add	r7, r3, #16
	vst1.32	{d16, d17}, [r3]
	vst1.32	{d16, d17}, [r7]
	add	r7, r3, #2048
	add	r3, r3, #2064
	vst1.32	{d16, d17}, [r7]
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #16
	vst1.32	{d16, d17}, [r2]
	vst1.32	{d16, d17}, [r3]
	add	r3, r2, #2048
	add	r2, r2, #2064
	vst1.32	{d16, d17}, [r3]
	vst1.32	{d16, d17}, [r2]
	bne	.LBB1_22
	ldr	r1, [sp, #96]
	mov	r0, #22
	mov	r4, #0
	mul	r0, r1, r0
	str	r0, [sp, #76]
	add	r0, r1, r1, lsl #5
	ldr	r1, [sp, #84]
	add	r0, r1, r0, lsl #12
	add	r1, r0, #122880
	add	r2, r0, #12288
	str	r1, [sp, #152]
	add	r1, r0, #110592
	str	r1, [sp, #148]
	add	r1, r0, #98304
	str	r1, [sp, #144]
	add	r1, r0, #86016
	str	r1, [sp, #140]
	add	r1, r0, #73728
	str	r1, [sp, #136]
	add	r1, r0, #61440
	str	r1, [sp, #132]
	add	r1, r0, #49152
	str	r1, [sp, #128]
	add	r1, r0, #36864
	str	r1, [sp, #124]
	add	r1, r0, #24576
	str	r1, [sp, #120]
	str	r0, [sp, #156]
.LBB1_24:
	ldr	r0, [sp, #164]
	ldr	r7, [sp, #156]
	mov	r9, #256
	mov	r6, #512
	mov	r3, r2
	mov	r5, #256
	add	lr, r0, r8
	ldr	r0, [sp, #160]
	add	r8, r8, #32
	mov	r11, lr
	add	r10, r0, r4
	mov	r0, #512
	add	r4, r4, #16384
	mov	r12, r10
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp34:
	vld1.32	{d4, d5, d6, d7}, [r12]!
	vld1.32	{d0, d1, d2}, [r7]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r9, r9, #1
	bne	.Ltmp34
	lsl	r6, r6, #2
	vld1.32	{d0, d1, d2, d3}, [r11]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r11], r6
	vld1.32	{d4, d5, d6, d7}, [r11]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r11], r6
	vld1.32	{d0, d1, d2, d3}, [r11]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r11], r6
	vld1.32	{d4, d5, d6, d7}, [r11]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r11], r6
	vld1.32	{d0, d1, d2, d3}, [r11]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r11], r6
	vld1.32	{d4, d5, d6, d7}, [r11]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r11]

	@NO_APP
	add	r7, lr, #12288
	mov	r6, r10
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp35:
	vld1.32	{d4, d5, d6, d7}, [r6]!
	vld1.32	{d0, d1, d2}, [r3]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r5, r5, #1
	bne	.Ltmp35
	lsl	r0, r0, #2
	vld1.32	{d0, d1, d2, d3}, [r7]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r7], r0
	vld1.32	{d4, d5, d6, d7}, [r7]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r7], r0
	vld1.32	{d0, d1, d2, d3}, [r7]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r7], r0
	vld1.32	{d4, d5, d6, d7}, [r7]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r7], r0
	vld1.32	{d0, d1, d2, d3}, [r7]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r7], r0
	vld1.32	{d4, d5, d6, d7}, [r7]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r7]

	@NO_APP
	ldr	r1, [sp, #120]
	add	r0, lr, #24576
	mov	r3, r10
	mov	r7, #256
	mov	r6, #512
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp36:
	vld1.32	{d4, d5, d6, d7}, [r3]!
	vld1.32	{d0, d1, d2}, [r1]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r7, r7, #1
	bne	.Ltmp36
	lsl	r6, r6, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r7, [sp, #124]
	add	r0, lr, #36864
	mov	r1, r10
	mov	r3, #256
	mov	r6, #512
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp37:
	vld1.32	{d4, d5, d6, d7}, [r1]!
	vld1.32	{d0, d1, d2}, [r7]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r3, r3, #1
	bne	.Ltmp37
	lsl	r6, r6, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r1, [sp, #128]
	add	r0, lr, #49152
	mov	r3, r10
	mov	r7, #256
	mov	r6, #512
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp38:
	vld1.32	{d4, d5, d6, d7}, [r3]!
	vld1.32	{d0, d1, d2}, [r1]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r7, r7, #1
	bne	.Ltmp38
	lsl	r6, r6, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r7, [sp, #132]
	add	r0, lr, #61440
	mov	r1, r10
	mov	r3, #256
	mov	r6, #512
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp39:
	vld1.32	{d4, d5, d6, d7}, [r1]!
	vld1.32	{d0, d1, d2}, [r7]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r3, r3, #1
	bne	.Ltmp39
	lsl	r6, r6, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r3, [sp, #136]
	add	r0, lr, #73728
	mov	r1, r10
	mov	r7, #256
	mov	r6, #512
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp40:
	vld1.32	{d4, d5, d6, d7}, [r1]!
	vld1.32	{d0, d1, d2}, [r3]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r7, r7, #1
	bne	.Ltmp40
	lsl	r6, r6, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r7, [sp, #140]
	add	r0, lr, #86016
	mov	r1, r10
	mov	r3, #256
	mov	r6, #512
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp41:
	vld1.32	{d4, d5, d6, d7}, [r1]!
	vld1.32	{d0, d1, d2}, [r7]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r3, r3, #1
	bne	.Ltmp41
	lsl	r6, r6, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r7, [sp, #144]
	add	r0, lr, #98304
	mov	r1, r10
	mov	r3, #256
	mov	r6, #512
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp42:
	vld1.32	{d4, d5, d6, d7}, [r1]!
	vld1.32	{d0, d1, d2}, [r7]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r3, r3, #1
	bne	.Ltmp42
	lsl	r6, r6, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r6
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r6
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r6, [sp, #148]
	add	r0, lr, #110592
	mov	r1, r10
	mov	r3, #256
	mov	r7, #512
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp43:
	vld1.32	{d4, d5, d6, d7}, [r1]!
	vld1.32	{d0, d1, d2}, [r6]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r3, r3, #1
	bne	.Ltmp43
	lsl	r7, r7, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r7
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r7
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r7
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r7
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r7
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r3, [sp, #152]
	add	r0, lr, #122880
	mov	r1, #256
	mov	r7, #512
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp44:
	vld1.32	{d4, d5, d6, d7}, [r10]!
	vld1.32	{d0, d1, d2}, [r3]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r1, r1, #1
	bne	.Ltmp44
	lsl	r7, r7, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r7
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r7
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r7
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r7
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r7
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	cmp	r4, #65536
	bne	.LBB1_24
	ldr	r0, [sp, #76]
	ldr	r1, [sp, #84]
	mov	r2, #8192
	orr	r0, r0, #1
	add	r0, r0, r0, lsl #1
	add	r0, r1, r0, lsl #11
	add	r1, r0, #122880
	mov	r9, r0
	add	r11, r0, #12288
	str	r1, [sp, #156]
	add	r1, r0, #110592
	str	r1, [sp, #152]
	add	r1, r0, #98304
	str	r1, [sp, #148]
	add	r1, r0, #86016
	str	r1, [sp, #144]
	add	r1, r0, #73728
	str	r1, [sp, #140]
	add	r1, r0, #61440
	str	r1, [sp, #136]
	add	r1, r0, #49152
	str	r1, [sp, #132]
	add	r1, r0, #36864
	str	r1, [sp, #128]
	add	r1, r0, #24576
	str	r1, [sp, #124]
	ldr	r12, [sp, #112]
.LBB1_26:
	ldr	r0, [sp, #164]
	mov	lr, r9
	mov	r10, #512
	mov	r3, #256
	mov	r6, r11
	mov	r5, #256
	add	r7, r0, r12
	ldr	r0, [sp, #160]
	add	r12, r12, #32
	mov	r8, r7
	add	r0, r0, r2
	add	r2, r2, #16384
	mov	r1, r0
	mov	r4, r0
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp45:
	vld1.32	{d4, d5, d6, d7}, [r1]!
	vld1.32	{d0, d1, d2}, [lr]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r3, r3, #1
	bne	.Ltmp45
	lsl	r10, r10, #2
	vld1.32	{d0, d1, d2, d3}, [r8]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r8], r10
	vld1.32	{d4, d5, d6, d7}, [r8]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r8], r10
	vld1.32	{d0, d1, d2, d3}, [r8]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r8], r10
	vld1.32	{d4, d5, d6, d7}, [r8]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r8], r10
	vld1.32	{d0, d1, d2, d3}, [r8]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r8], r10
	vld1.32	{d4, d5, d6, d7}, [r8]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r8]

	@NO_APP
	add	r1, r7, #12288
	mov	r3, #512
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp46:
	vld1.32	{d4, d5, d6, d7}, [r4]!
	vld1.32	{d0, d1, d2}, [r6]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r5, r5, #1
	bne	.Ltmp46
	lsl	r3, r3, #2
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r1]

	@NO_APP
	ldr	r4, [sp, #124]
	add	r1, r7, #24576
	mov	r3, #512
	mov	r6, r0
	mov	r5, #256
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp47:
	vld1.32	{d4, d5, d6, d7}, [r6]!
	vld1.32	{d0, d1, d2}, [r4]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r5, r5, #1
	bne	.Ltmp47
	lsl	r3, r3, #2
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r1]

	@NO_APP
	ldr	r5, [sp, #128]
	add	r1, r7, #36864
	mov	r3, #512
	mov	r6, r0
	mov	r4, #256
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp48:
	vld1.32	{d4, d5, d6, d7}, [r6]!
	vld1.32	{d0, d1, d2}, [r5]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r4, r4, #1
	bne	.Ltmp48
	lsl	r3, r3, #2
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r1]

	@NO_APP
	ldr	r3, [sp, #132]
	add	r1, r7, #49152
	mov	r6, #512
	mov	r5, r0
	mov	r4, #256
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp49:
	vld1.32	{d4, d5, d6, d7}, [r5]!
	vld1.32	{d0, d1, d2}, [r3]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r4, r4, #1
	bne	.Ltmp49
	lsl	r6, r6, #2
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r1], r6
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r1], r6
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r1], r6
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r1], r6
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r1], r6
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r1]

	@NO_APP
	ldr	r4, [sp, #136]
	add	r1, r7, #61440
	mov	r3, #512
	mov	r6, r0
	mov	r5, #256
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp50:
	vld1.32	{d4, d5, d6, d7}, [r6]!
	vld1.32	{d0, d1, d2}, [r4]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r5, r5, #1
	bne	.Ltmp50
	lsl	r3, r3, #2
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r1]

	@NO_APP
	ldr	r6, [sp, #140]
	add	r1, r7, #73728
	mov	r3, #512
	mov	r5, r0
	mov	r4, #256
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp51:
	vld1.32	{d4, d5, d6, d7}, [r5]!
	vld1.32	{d0, d1, d2}, [r6]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r4, r4, #1
	bne	.Ltmp51
	lsl	r3, r3, #2
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r1]

	@NO_APP
	ldr	r4, [sp, #144]
	add	r1, r7, #86016
	mov	r3, #512
	mov	r6, r0
	mov	r5, #256
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp52:
	vld1.32	{d4, d5, d6, d7}, [r6]!
	vld1.32	{d0, d1, d2}, [r4]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r5, r5, #1
	bne	.Ltmp52
	lsl	r3, r3, #2
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r1]

	@NO_APP
	ldr	r6, [sp, #148]
	add	r1, r7, #98304
	mov	r3, #512
	mov	r5, r0
	mov	r4, #256
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp53:
	vld1.32	{d4, d5, d6, d7}, [r5]!
	vld1.32	{d0, d1, d2}, [r6]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r4, r4, #1
	bne	.Ltmp53
	lsl	r3, r3, #2
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r1]

	@NO_APP
	ldr	r4, [sp, #152]
	add	r1, r7, #110592
	mov	r3, #512
	mov	r6, r0
	mov	r5, #256
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp54:
	vld1.32	{d4, d5, d6, d7}, [r6]!
	vld1.32	{d0, d1, d2}, [r4]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r5, r5, #1
	bne	.Ltmp54
	lsl	r3, r3, #2
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r1], r3
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r1], r3
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r1]

	@NO_APP
	add	r1, r7, #122880
	ldr	r7, [sp, #156]
	mov	r3, #256
	mov	r6, #512
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp55:
	vld1.32	{d4, d5, d6, d7}, [r0]!
	vld1.32	{d0, d1, d2}, [r7]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r3, r3, #1
	bne	.Ltmp55
	lsl	r6, r6, #2
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r1], r6
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r1], r6
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r1], r6
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r1], r6
	vld1.32	{d0, d1, d2, d3}, [r1]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r1], r6
	vld1.32	{d4, d5, d6, d7}, [r1]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r1]

	@NO_APP
	cmp	r2, #73728
	bne	.LBB1_26
	ldr	r0, [sp, #108]
	ldr	r9, [sp, #80]
	ldr	r3, [sp, #92]
	ldr	r7, [sp, #88]
	movw	r11, #61440
	vmov.i32	q8, #0x0
	mov	r6, #10240
	movt	r11, #65535
	add	r0, r0, #135168
	add	r9, r9, #135168
	add	r3, r3, #135168
	add	r7, r7, #135168
	str	r0, [sp, #108]
	ldr	r0, [sp, #104]
	add	r0, r0, #135168
	str	r0, [sp, #104]
	ldr	r0, [sp, #100]
	add	r0, r0, #135168
	str	r0, [sp, #100]
	ldr	r0, [sp, #116]
	add	r0, r0, #135168
	str	r0, [sp, #116]
	ldr	r0, [sp, #96]
	ldr	r1, [sp, #112]
	add	r0, r0, #1
	add	r1, r1, #135168
	str	r0, [sp, #96]
	cmp	r0, #3
	ldr	r4, [sp, #56]
	ldr	r2, [sp, #72]
	ldr	r5, [sp, #68]
	bne	.LBB1_15
	ldr	r0, [sp, #160]
	add	r5, r5, #128
	add	r2, r2, #128
	add	r0, r0, #65536
	str	r0, [sp, #160]
	ldr	r3, [sp, #48]
	ldr	r9, [sp, #44]
	ldr	r7, [sp, #40]
	ldr	lr, [sp, #36]
	ldr	r8, [sp, #32]
	ldr	r0, [sp, #28]
	ldr	r1, [sp, #24]
	add	r3, r3, #1
	add	r9, r9, #128
	add	r7, r7, #128
	add	lr, lr, #128
	add	r8, r8, #128
	add	r0, r0, #128
	add	r1, r1, #128
	cmp	r3, #16
	str	r3, [sp, #48]
	bne	.LBB1_14
	ldr	lr, [sp, #164]
	ldr	r11, [sp, #12]
	mov	r0, #0
	mov	r1, #4096
	mov	r3, #6144
	mov	r7, #8192
	mov	r10, #16384
.LBB1_30:
	mov	r12, #0
	str	r0, [sp, #160]
.LBB1_31:
	mov	r6, lr
	mov	r9, r11
	mov	r2, #10240
	ldr	r8, [r6, r12, lsl #9]!
	str	r8, [r9, r12]!
	add	r12, r12, #56
	ldr	r4, [r6, r1]
	ldr	r0, [r6, r3]
	mov	r3, #12288
	ldr	r1, [r6, r7]
	mov	r7, #14336
	ldr	r5, [r6, #2048]
	ldr	r2, [r6, r2]
	ldr	r8, [r6, r10]
	cmp	r12, #784
	ldr	r3, [r6, r3]
	ldr	r7, [r6, r7]
	str	r5, [r9, #4]
	str	r4, [r9, #8]
	add	r4, r9, #12
	mov	r5, #18432
	stm	r4, {r0, r1, r2, r3, r7}
	str	r8, [r9, #32]
	mov	r3, #24576
	mov	r2, #22528
	mov	r1, #20480
	mov	r4, #26624
	mov	r7, #8192
	ldr	r0, [r6, r5]
	ldr	r1, [r6, r1]
	ldr	r2, [r6, r2]
	ldr	r3, [r6, r3]
	ldr	r4, [r6, r4]
	str	r0, [r9, #36]
	str	r1, [r9, #40]
	str	r2, [r9, #44]
	str	r3, [r9, #48]
	mov	r1, #4096
	mov	r3, #6144
	str	r4, [r9, #52]
	bne	.LBB1_31
	ldr	r0, [sp, #160]
	add	r11, r11, #784
	add	lr, lr, #4
	add	r0, r0, #1
	cmp	r0, #512
	bne	.LBB1_30
	ldr	r4, .LCPI1_2
	mov	r0, #1
.LPC1_1:
	ldr	r4, [pc, r4]
	ldr	r5, [sp, #16]
	ldr	r2, [sp, #164]
	ldr	r3, [r4]
	mov	r1, r5
	blx	r3
	ldr	r3, [r4]
	ldr	r2, [sp, #20]
	mov	r0, #1
	mov	r1, r5
	blx	r3
	ldr	r3, [r4]
	ldr	r2, [sp, #84]
	mov	r0, #1
	mov	r1, r5
	add	sp, sp, #168
	vpop	{d8, d9, d10, d11, d12, d13, d14, d15}
	add	sp, sp, #4
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	bx	r3
	.p2align	2
.LCPI1_2:
.Ltmp56:
	.long	__TVMBackendFreeWorkspace(GOT_PREL)-((.LPC1_1+8)-.Ltmp56)
.Lfunc_end1:
	.size	.Ldefault_function_compute_, .Lfunc_end1-.Ldefault_function_compute_
	.fnend

	.globl	sgemm_compute_6x8__neon
	.p2align	2
	.type	sgemm_compute_6x8__neon,%function
	.code	32
sgemm_compute_6x8__neon:
	.fnstart
	.save	{r4, r5, r11, lr}
	push	{r4, r5, r11, lr}
	.setfp	r11, sp, #8
	add	r11, sp, #8
	.vsave	{d8, d9, d10, d11, d12, d13, d14, d15}
	vpush	{d8, d9, d10, d11, d12, d13, d14, d15}
	ldr	lr, [r11, #8]
	add	r1, r1, r2, lsl #2
	ldr	r12, [r11, #16]
	ldr	r4, [r11, #12]
	ldr	r5, [r11, #20]
	add	r3, r3, lr, lsl #2
	add	r2, r4, r12, lsl #2
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp57:
	vld1.32	{d4, d5, d6, d7}, [r3]!
	vld1.32	{d0, d1, d2}, [r1]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r0, r0, #1
	bne	.Ltmp57
	lsl	r5, r5, #2
	vst1.32	{d8, d9, d10, d11}, [r2], r5
	vst1.32	{d12, d13, d14, d15}, [r2], r5
	vst1.32	{d16, d17, d18, d19}, [r2], r5
	vst1.32	{d20, d21, d22, d23}, [r2], r5
	vst1.32	{d24, d25, d26, d27}, [r2], r5
	vst1.32	{d28, d29, d30, d31}, [r2]

	@NO_APP
	vpop	{d8, d9, d10, d11, d12, d13, d14, d15}
	pop	{r4, r5, r11, pc}
.Lfunc_end2:
	.size	sgemm_compute_6x8__neon, .Lfunc_end2-sgemm_compute_6x8__neon
	.cantunwind
	.fnend

	.globl	sgemm_reset_6x8__neon
	.p2align	2
	.type	sgemm_reset_6x8__neon,%function
	.code	32
sgemm_reset_6x8__neon:
	.fnstart
	add	r0, r0, r1, lsl #2
	vmov.i32	q8, #0x0
	lsl	r1, r2, #2
	add	r2, r0, #16
	vst1.32	{d16, d17}, [r0], r1
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #16
	vst1.32	{d16, d17}, [r0], r1
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #16
	vst1.32	{d16, d17}, [r0], r1
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #16
	vst1.32	{d16, d17}, [r0], r1
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #16
	vst1.32	{d16, d17}, [r0], r1
	vst1.32	{d16, d17}, [r2]
	vst1.32	{d16, d17}, [r0]!
	vst1.32	{d16, d17}, [r0]
	bx	lr
.Lfunc_end3:
	.size	sgemm_reset_6x8__neon, .Lfunc_end3-sgemm_reset_6x8__neon
	.cantunwind
	.fnend

	.globl	sgemm_update_6x8__neon
	.p2align	2
	.type	sgemm_update_6x8__neon,%function
	.code	32
sgemm_update_6x8__neon:
	.fnstart
	.save	{r4, r5, r11, lr}
	push	{r4, r5, r11, lr}
	.setfp	r11, sp, #8
	add	r11, sp, #8
	.vsave	{d8, d9, d10, d11, d12, d13, d14, d15}
	vpush	{d8, d9, d10, d11, d12, d13, d14, d15}
	ldr	lr, [r11, #8]
	add	r1, r1, r2, lsl #2
	ldr	r12, [r11, #16]
	ldr	r4, [r11, #12]
	ldr	r5, [r11, #20]
	add	r3, r3, lr, lsl #2
	add	r2, r4, r12, lsl #2
	@APP
	vmov.i32	q4, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q15, #0x0
.Ltmp58:
	vld1.32	{d4, d5, d6, d7}, [r3]!
	vld1.32	{d0, d1, d2}, [r1]!
	vmla.f32	q4, q2, d0[0]
	vmla.f32	q5, q3, d0[0]
	vmla.f32	q6, q2, d0[1]
	vmla.f32	q7, q3, d0[1]
	vmla.f32	q8, q2, d1[0]
	vmla.f32	q9, q3, d1[0]
	vmla.f32	q10, q2, d1[1]
	vmla.f32	q11, q3, d1[1]
	vmla.f32	q12, q2, d2[0]
	vmla.f32	q13, q3, d2[0]
	vmla.f32	q14, q2, d2[1]
	vmla.f32	q15, q3, d2[1]
	subs	r0, r0, #1
	bne	.Ltmp58
	lsl	r5, r5, #2
	vld1.32	{d0, d1, d2, d3}, [r2]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r2], r5
	vld1.32	{d4, d5, d6, d7}, [r2]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r2], r5
	vld1.32	{d0, d1, d2, d3}, [r2]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r2], r5
	vld1.32	{d4, d5, d6, d7}, [r2]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r2], r5
	vld1.32	{d0, d1, d2, d3}, [r2]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r2], r5
	vld1.32	{d4, d5, d6, d7}, [r2]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r2]

	@NO_APP
	vpop	{d8, d9, d10, d11, d12, d13, d14, d15}
	pop	{r4, r5, r11, pc}
.Lfunc_end4:
	.size	sgemm_update_6x8__neon, .Lfunc_end4-sgemm_update_6x8__neon
	.cantunwind
	.fnend

	.type	__TVMAPISetLastError,%object
	.bss
	.weak	__TVMAPISetLastError
	.p2align	2
__TVMAPISetLastError:
	.long	0
	.size	__TVMAPISetLastError, 4

	.type	.L.str,%object
	.section	.rodata,"a",%progbits
.L.str:
	.asciz	"Assert fail: (num_args == 3), default_function: num_args should be 3"
	.size	.L.str, 69

	.type	.L.str.1,%object
.L.str.1:
	.asciz	"Assert fail: ((((1 == int32(arg0.strides[3])) && ((1*14) == int32(arg0.strides[2]))) && (((1*14)*14) == int32(arg0.strides[1]))) && ((((1*14)*14)*512) == int32(arg0.strides[0]))), arg0.strides: expected to be compact array"
	.size	.L.str.1, 223

	.type	.L.str.2,%object
.L.str.2:
	.asciz	"Assert fail: ((((1 == int32(arg1.strides[3])) && ((1*1) == int32(arg1.strides[2]))) && (((1*1)*1) == int32(arg1.strides[1]))) && ((((1*1)*1)*512) == int32(arg1.strides[0]))), arg1.strides: expected to be compact array"
	.size	.L.str.2, 218

	.type	.L.str.3,%object
.L.str.3:
	.asciz	"Assert fail: ((((1 == int32(arg2.strides[3])) && ((1*14) == int32(arg2.strides[2]))) && (((1*14)*14) == int32(arg2.strides[1]))) && ((((1*14)*14)*512) == int32(arg2.strides[0]))), arg2.strides: expected to be compact array"
	.size	.L.str.3, 223

	.type	.L.str.4,%object
.L.str.4:
	.asciz	"Assert fail: ((((arg0.code == 3) || (arg0.code == 13)) || (arg0.code == 7)) || (arg0.code == 4)), default_function: Expect arg[0] to be pointer"
	.size	.L.str.4, 144

	.type	.L.str.5,%object
.L.str.5:
	.asciz	"Assert fail: ((((arg1.code == 3) || (arg1.code == 13)) || (arg1.code == 7)) || (arg1.code == 4)), default_function: Expect arg[1] to be pointer"
	.size	.L.str.5, 144

	.type	.L.str.6,%object
.L.str.6:
	.asciz	"Assert fail: ((((arg2.code == 3) || (arg2.code == 13)) || (arg2.code == 7)) || (arg2.code == 4)), default_function: Expect arg[2] to be pointer"
	.size	.L.str.6, 144

	.type	.L.str.7,%object
.L.str.7:
	.asciz	"Assert fail: (dev_type == 1), device_type need to be 1"
	.size	.L.str.7, 55

	.type	.L.str.8,%object
.L.str.8:
	.asciz	"Assert fail: (4 == tvm_struct_get(arg0, 0, 4)), arg0.ndim is expected to equal 4"
	.size	.L.str.8, 81

	.type	.L.str.9,%object
.L.str.9:
	.asciz	"Assert fail: (((tvm_struct_get(arg0, 0, 5) == (uint8)2) && (tvm_struct_get(arg0, 0, 6) == (uint8)32)) && (tvm_struct_get(arg0, 0, 7) == (uint16)1)), arg0.dtype is expected to be float32"
	.size	.L.str.9, 186

	.type	.L.str.10,%object
.L.str.10:
	.asciz	"Assert fail: (int32(arg0.shape[0]) == 1), Argument arg0.shape[0] has an unsatisfied constraint"
	.size	.L.str.10, 95

	.type	.L.str.11,%object
.L.str.11:
	.asciz	"Assert fail: (int32(arg0.shape[1]) == 512), Argument arg0.shape[1] has an unsatisfied constraint"
	.size	.L.str.11, 97

	.type	.L.str.12,%object
.L.str.12:
	.asciz	"Assert fail: (int32(arg0.shape[2]) == 14), Argument arg0.shape[2] has an unsatisfied constraint"
	.size	.L.str.12, 96

	.type	.L.str.13,%object
.L.str.13:
	.asciz	"Assert fail: (int32(arg0.shape[3]) == 14), Argument arg0.shape[3] has an unsatisfied constraint"
	.size	.L.str.13, 96

	.type	.L.str.14,%object
.L.str.14:
	.asciz	"Assert fail: (tvm_struct_get(arg0, 0, 8) == (uint64)0), Argument arg0.byte_offset has an unsatisfied constraint"
	.size	.L.str.14, 112

	.type	.L.str.15,%object
.L.str.15:
	.asciz	"Assert fail: (4 == tvm_struct_get(arg1, 0, 4)), arg1.ndim is expected to equal 4"
	.size	.L.str.15, 81

	.type	.L.str.16,%object
.L.str.16:
	.asciz	"Assert fail: (((tvm_struct_get(arg1, 0, 5) == (uint8)2) && (tvm_struct_get(arg1, 0, 6) == (uint8)32)) && (tvm_struct_get(arg1, 0, 7) == (uint16)1)), arg1.dtype is expected to be float32"
	.size	.L.str.16, 186

	.type	.L.str.17,%object
.L.str.17:
	.asciz	"Assert fail: (int32(arg1.shape[0]) == 512), Argument arg1.shape[0] has an unsatisfied constraint"
	.size	.L.str.17, 97

	.type	.L.str.18,%object
.L.str.18:
	.asciz	"Assert fail: (int32(arg1.shape[1]) == 512), Argument arg1.shape[1] has an unsatisfied constraint"
	.size	.L.str.18, 97

	.type	.L.str.19,%object
.L.str.19:
	.asciz	"Assert fail: (int32(arg1.shape[2]) == 1), Argument arg1.shape[2] has an unsatisfied constraint"
	.size	.L.str.19, 95

	.type	.L.str.20,%object
.L.str.20:
	.asciz	"Assert fail: (int32(arg1.shape[3]) == 1), Argument arg1.shape[3] has an unsatisfied constraint"
	.size	.L.str.20, 95

	.type	.L.str.21,%object
.L.str.21:
	.asciz	"Assert fail: (tvm_struct_get(arg1, 0, 8) == (uint64)0), Argument arg1.byte_offset has an unsatisfied constraint"
	.size	.L.str.21, 112

	.type	.L.str.22,%object
.L.str.22:
	.asciz	"Assert fail: (1 == tvm_struct_get(arg1, 0, 10)), Argument arg1.device_type has an unsatisfied constraint"
	.size	.L.str.22, 105

	.type	.L.str.23,%object
.L.str.23:
	.asciz	"Assert fail: (dev_id == tvm_struct_get(arg1, 0, 9)), Argument arg1.device_id has an unsatisfied constraint"
	.size	.L.str.23, 107

	.type	.L.str.24,%object
.L.str.24:
	.asciz	"Assert fail: (4 == tvm_struct_get(arg2, 0, 4)), arg2.ndim is expected to equal 4"
	.size	.L.str.24, 81

	.type	.L.str.25,%object
.L.str.25:
	.asciz	"Assert fail: (((tvm_struct_get(arg2, 0, 5) == (uint8)2) && (tvm_struct_get(arg2, 0, 6) == (uint8)32)) && (tvm_struct_get(arg2, 0, 7) == (uint16)1)), arg2.dtype is expected to be float32"
	.size	.L.str.25, 186

	.type	.L.str.26,%object
.L.str.26:
	.asciz	"Assert fail: (int32(arg2.shape[0]) == 1), Argument arg2.shape[0] has an unsatisfied constraint"
	.size	.L.str.26, 95

	.type	.L.str.27,%object
.L.str.27:
	.asciz	"Assert fail: (int32(arg2.shape[1]) == 512), Argument arg2.shape[1] has an unsatisfied constraint"
	.size	.L.str.27, 97

	.type	.L.str.28,%object
.L.str.28:
	.asciz	"Assert fail: (int32(arg2.shape[2]) == 14), Argument arg2.shape[2] has an unsatisfied constraint"
	.size	.L.str.28, 96

	.type	.L.str.29,%object
.L.str.29:
	.asciz	"Assert fail: (int32(arg2.shape[3]) == 14), Argument arg2.shape[3] has an unsatisfied constraint"
	.size	.L.str.29, 96

	.type	.L.str.30,%object
.L.str.30:
	.asciz	"Assert fail: (tvm_struct_get(arg2, 0, 8) == (uint64)0), Argument arg2.byte_offset has an unsatisfied constraint"
	.size	.L.str.30, 112

	.type	.L.str.31,%object
.L.str.31:
	.asciz	"Assert fail: (1 == tvm_struct_get(arg2, 0, 10)), Argument arg2.device_type has an unsatisfied constraint"
	.size	.L.str.31, 105

	.type	.L.str.32,%object
.L.str.32:
	.asciz	"Assert fail: (dev_id == tvm_struct_get(arg2, 0, 9)), Argument arg2.device_id has an unsatisfied constraint"
	.size	.L.str.32, 107

	.type	__TVMBackendAllocWorkspace,%object
	.bss
	.weak	__TVMBackendAllocWorkspace
	.p2align	2
__TVMBackendAllocWorkspace:
	.long	0
	.size	__TVMBackendAllocWorkspace, 4

	.type	__TVMBackendFreeWorkspace,%object
	.weak	__TVMBackendFreeWorkspace
	.p2align	2
__TVMBackendFreeWorkspace:
	.long	0
	.size	__TVMBackendFreeWorkspace, 4

	.type	__tvm_main__,%object
	.section	.rodata,"a",%progbits
	.weak	__tvm_main__
__tvm_main__:
	.asciz	"default_function"
	.size	__tvm_main__, 17


	.ident	"clang version 6.0.0 (tags/RELEASE_600/final)"
	.section	".note.GNU-stack","",%progbits
	.eabi_attribute	30, 1
