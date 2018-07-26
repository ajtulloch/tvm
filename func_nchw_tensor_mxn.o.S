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
	cmp	r0, #49
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
	cmpeq	r0, #2048
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
	cmp	r0, #49
	ldreq	r0, [r11]
	cmpeq	r0, #25088
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
	cmp	r0, #2048
	bne	.LBB0_64
	ldr	r0, [r7, #16]
	cmp	r0, #7
	bne	.LBB0_65
	ldr	r0, [r7, #24]
	cmp	r0, #7
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
	cmp	r0, #2048
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
	cmp	r0, #7
	bne	.LBB0_79
	ldr	r0, [r8, #24]
	cmp	r0, #7
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
	.long	7
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

	.p2align	2
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
	.pad	#72
	sub	sp, sp, #72
	str	r2, [sp, #8]
	mov	r7, r3
	mov	r4, r0
	mov	r0, #32
	mov	r9, r1
	mov	r10, #2
	mov	r2, #442368
	mov	r3, #0
	mov	r8, #0
	ldr	r11, .LCPI1_2
	mov	r1, r7
.LPC1_0:
	ldr	r11, [pc, r11]
	ldr	r6, [r11]
	str	r0, [sp, #4]
	mov	r0, #1
	str	r10, [sp]
	blx	r6
	ldr	r6, [r11]
	mov	r5, r0
	mov	r0, #32
	mov	r1, r7
	mov	r2, #4194304
	mov	r3, #0
	str	r10, [sp]
	str	r0, [sp, #4]
	mov	r0, #1
	blx	r6
	ldr	r6, [r11]
	str	r0, [sp, #16]
	mov	r0, #32
	mov	r1, r7
	mov	r2, #110592
	mov	r3, #0
	str	r10, [sp]
	str	r0, [sp, #4]
	mov	r0, #1
	str	r7, [sp, #12]
	blx	r6
	mov	r10, #5
	vmov.i32	q8, #0x7
	vmov.i32	q9, #0x31
	mov	r11, r5
	mov	r12, r4
	mov	lr, #0
	str	r0, [sp, #40]
	str	r5, [sp, #32]
.LBB1_1:
	cmp	lr, #8
	bhs	.LBB1_5
	movw	r7, #18725
	add	r3, lr, lr, lsl #1
	mov	r0, #2
	add	r0, r0, r3, lsl #1
	movt	r7, #9362
	umull	r1, r2, r0, r7
	str	r0, [sp, #60]
	sub	r1, r0, r2
	add	r1, r2, r1, lsr #1
	lsr	r0, r1, #2
	str	r0, [sp, #52]
	umull	r2, r0, r0, r7
	rsb	r2, r0, r1, lsr #2
	add	r0, r0, r2, lsr #1
	lsr	r2, r0, #2
	lsl	r2, r2, #3
	sub	r0, r2, r0, lsr #2
	rsb	r0, r0, r1, lsr #2
	str	r0, [sp, #68]
	mov	r0, #4
	add	r2, r0, r3, lsl #1
	umull	r0, r1, r2, r7
	str	r2, [sp, #56]
	sub	r0, r2, r1
	add	r0, r1, r0, lsr #1
	lsr	r1, r0, #2
	str	r1, [sp, #36]
	umull	r1, r2, r1, r7
	rsb	r1, r2, r0, lsr #2
	add	r1, r2, r1, lsr #1
	lsr	r2, r1, #2
	lsl	r2, r2, #3
	sub	r1, r2, r1, lsr #2
	rsb	r0, r1, r0, lsr #2
	str	r0, [sp, #64]
	mov	r0, #1
	orr	r2, r0, r3, lsl #1
	umull	r0, r1, r2, r7
	sub	r0, r2, r1
	add	r0, r1, r0, lsr #1
	lsr	r1, r0, #2
	str	r1, [sp, #28]
	umull	r5, r1, r1, r7
	rsb	r5, r1, r0, lsr #2
	add	r1, r1, r5, lsr #1
	lsr	r5, r1, #2
	lsl	r5, r5, #3
	sub	r1, r5, r1, lsr #2
	rsb	r0, r1, r0, lsr #2
	str	r0, [sp, #48]
	mov	r0, #3
	add	r3, r0, r3, lsl #1
	umull	r0, r1, r3, r7
	vmov.32	d21[0], r3
	sub	r0, r3, r1
	add	r0, r1, r0, lsr #1
	lsr	r1, r0, #2
	umull	r5, r6, r1, r7
	vmov.32	d23[0], r1
	rsb	r5, r6, r0, lsr #2
	add	r5, r6, r5, lsr #1
	lsr	r6, r5, #2
	lsl	r6, r6, #3
	sub	r5, r6, r5, lsr #2
	rsb	r0, r5, r0, lsr #2
	umull	r1, r5, r10, r7
	str	r0, [sp, #44]
	sub	r1, r10, r5
	ldr	r0, [sp, #36]
	add	r1, r5, r1, lsr #1
	lsr	r5, r1, #2
	umull	r5, r6, r5, r7
	rsb	r1, r6, r1, lsr #2
	movw	r5, #33437
	vmov.32	d23[1], r0
	ldr	r0, [sp, #28]
	movt	r5, #21399
	mov	r7, r5
	vmov.32	d22[0], r0
	ldr	r0, [sp, #52]
	vmov.32	d22[1], r0
	add	r0, r6, r1, lsr #1
	umull	r5, r6, r3, r7
	mov	r3, #0
	str	r0, [sp, #52]
	lsr	r5, r6, #4
	ldr	r1, [sp, #56]
	vmov.32	d25[0], r5
	ldr	r0, [sp, #60]
	umull	r5, r6, r1, r7
	vmov.32	d21[1], r1
	lsr	r5, r6, #4
	vmov.32	d20[0], r2
	vmov.32	d25[1], r5
	umull	r5, r6, r2, r7
	vmov.32	d20[1], r0
	umull	r1, r2, r10, r7
	lsr	r5, r6, #4
	lsr	r1, r2, #4
	mov	r2, #100352
	vmls.i32	q10, q11, q8
	vmov.32	d24[0], r5
	umull	r5, r6, r0, r7
	ldr	r0, [sp, #44]
	mla	r1, r1, r2, r10
	ldr	r2, [sp, #64]
	lsr	r5, r6, #4
	vmov.32	d24[1], r5
	mov	r5, #0
	vshl.i32	q11, q12, #11
	vmov.32	d27[0], r0
	ldr	r0, [sp, #48]
	vmov.32	d26[0], r0
	ldr	r0, [sp, #52]
	vmov.32	d27[1], r2
	ldr	r2, [sp, #68]
	lsr	r0, r0, #2
	vmov.32	d26[1], r2
	mov	r2, #49
	mls	r0, r0, r2, r1
	vmla.i32	q10, q13, q8
	mov	r2, #0
	add	r1, r4, r0, lsl #2
.LBB1_3:
	ldr	r0, [r12, r2]
	vdup.32	q12, r5
	vorr	q13, q10, q10
	mov	r6, r11
	add	r5, r5, #1
	vadd.i32	q12, q12, q11
	vmla.i32	q13, q12, q9
	str	r0, [r6, r3]!
	vmov.32	r0, d26[0]
	add	r3, r3, #24
	ldr	r0, [r4, r0, lsl #2]
	cmp	r3, #49152
	str	r0, [r6, #4]
	vmov.32	r0, d26[1]
	ldr	r0, [r4, r0, lsl #2]
	str	r0, [r6, #8]
	vmov.32	r0, d27[0]
	ldr	r0, [r4, r0, lsl #2]
	str	r0, [r6, #12]
	vmov.32	r0, d27[1]
	ldr	r0, [r4, r0, lsl #2]
	str	r0, [r6, #16]
	ldr	r0, [r1, r2]
	add	r2, r2, #196
	str	r0, [r6, #20]
	bne	.LBB1_3
	b	.LBB1_7
	.p2align	2
.LCPI1_2:
.Ltmp33:
	.long	__TVMBackendAllocWorkspace(GOT_PREL)-((.LPC1_0+8)-.Ltmp33)
.LBB1_5:
	mov	r1, #0
	mov	r2, r12
.LBB1_6:
	ldr	r0, [r2], #196
	mov	r3, r11
	str	r0, [r3, r1]!
	add	r1, r1, #24
	cmp	r1, #49152
	str	r8, [r3, #4]
	str	r8, [r3, #8]
	str	r8, [r3, #12]
	str	r8, [r3, #16]
	str	r8, [r3, #20]
	bne	.LBB1_6
.LBB1_7:
	add	lr, lr, #1
	add	r10, r10, #6
	add	r11, r11, #49152
	add	r12, r12, #24
	cmp	lr, #9
	bne	.LBB1_1
	ldr	lr, [sp, #16]
	mov	r12, #0
.LBB1_9:
	mov	r3, lr
	mov	r2, #0
.LBB1_10:
	add	r6, r9, r2
	add	r2, r2, #4
	add	r1, r6, #57344
	vldr	s0, [r6]
	add	r8, r6, #32768
	add	r4, r6, #40960
	add	r0, r6, #49152
	add	r7, r6, #8192
	add	r5, r6, #16384
	add	r6, r6, #24576
	cmp	r2, #8192
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
	bne	.LBB1_10
	add	r12, r12, #1
	add	lr, lr, #65536
	add	r9, r9, #65536
	cmp	r12, #64
	bne	.LBB1_9
	ldr	r1, [sp, #40]
	ldr	r10, [sp, #16]
	movw	r8, #12288
	movw	r0, #57360
	mov	r4, #0
	vmov.i32	q8, #0x0
	mov	r9, #108544
	movt	r8, #65535
	add	r12, r1, r0
	mov	r0, #0
	str	r12, [sp, #20]
.LBB1_13:
	str	r0, [sp, #24]
	mov	r6, #4
	mov	r1, r4
	ldr	lr, [sp, #40]
.LBB1_14:
	add	r5, lr, r1
	add	r7, r12, r1
	add	r1, r1, #32
	subs	r6, r6, #1
	mov	r3, r5
	add	r0, r5, #16
	add	r2, r5, #6144
	vst1.32	{d16, d17}, [r3], r9
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #2048
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #2064
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #4096
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, r8
	vst1.32	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #2048
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #8192
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #4096
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #10240
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #6144
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #12288
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #8192
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #14336
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #10240
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #16384
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #12288
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #18432
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #14336
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #20480
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #16384
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #22528
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #18432
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #24576
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #20480
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #26624
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #22528
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #28672
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #24576
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #30720
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #26624
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #32768
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #28672
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #34816
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #30720
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #36864
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #32768
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #38912
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #34816
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #40960
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #36864
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #43008
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #38912
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #45056
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #40960
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #47104
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #43008
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #49152
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #45056
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #51200
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #47104
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #53248
	vst1.32	{d16, d17}, [r2]
	add	r2, r0, #49152
	add	r0, r0, #51200
	vst1.32	{d16, d17}, [r2]
	add	r2, r5, #55296
	vst1.32	{d16, d17}, [r2]
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #57344
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #59392
	vst1.32	{d16, d17}, [r7]
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #2048
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #61440
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #4096
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #63488
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #6144
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #65536
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #8192
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #67584
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #10240
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #69632
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #12288
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #71680
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #14336
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #73728
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #16384
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #75776
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #18432
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #77824
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #20480
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #79872
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #22528
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #81920
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #24576
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #83968
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #26624
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #86016
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #28672
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #88064
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #30720
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #90112
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #32768
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #92160
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #34816
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #94208
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #36864
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #96256
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #38912
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #98304
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #40960
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #100352
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #43008
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #102400
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #45056
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #104448
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #47104
	vst1.32	{d16, d17}, [r0]
	add	r0, r5, #106496
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #49152
	vst1.32	{d16, d17}, [r0]
	add	r0, r7, #51200
	vst1.32	{d16, d17}, [r3]
	vst1.32	{d16, d17}, [r0]
	bne	.LBB1_14
	mov	r0, #0
	mov	lr, r10
	str	r10, [sp, #28]
	str	r4, [sp, #36]
.LBB1_16:
	str	r0, [sp, #44]
	add	r0, r0, r0, lsl #1
	mov	r10, #0
	ldr	r1, [sp, #32]
	add	r0, r1, r0, lsl #11
	add	r1, r0, #393216
	add	r8, r0, #98304
	mov	r9, r0
	add	r11, r0, #49152
	str	r1, [sp, #68]
	add	r1, r0, #344064
	str	r1, [sp, #64]
	add	r1, r0, #294912
	str	r1, [sp, #60]
	add	r1, r0, #245760
	str	r1, [sp, #56]
	add	r1, r0, #196608
	str	r1, [sp, #52]
	add	r1, r0, #147456
	str	r1, [sp, #48]
	ldr	r12, [sp, #40]
.LBB1_17:
	add	r2, r12, r4
	add	r7, lr, r10
	mov	r3, #256
	mov	r5, r9
	mov	r6, #512
	add	r10, r10, #65536
	add	r4, r4, #32
	mov	r0, r2
	mov	r1, r7
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
	vld1.32	{d4, d5, d6, d7}, [r1]!
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
	subs	r3, r3, #1
	bne	.Ltmp34
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
	add	r0, r2, #12288
	mov	r1, r11
	mov	r3, #256
	mov	r5, #512
	mov	r6, r7
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
	subs	r3, r3, #1
	bne	.Ltmp35
	lsl	r5, r5, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r5
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r5
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	add	r0, r2, #24576
	mov	r1, #256
	mov	r3, r8
	mov	r5, #512
	mov	r6, r7
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
	subs	r1, r1, #1
	bne	.Ltmp36
	lsl	r5, r5, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r5
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r5
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r3, [sp, #48]
	add	r0, r2, #36864
	mov	r1, #256
	mov	r5, #512
	mov	r6, r7
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
	subs	r1, r1, #1
	bne	.Ltmp37
	lsl	r5, r5, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r5
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r5
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r5, [sp, #52]
	add	r0, r2, #49152
	mov	r1, #256
	mov	r3, #512
	mov	r6, r7
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
	subs	r1, r1, #1
	bne	.Ltmp38
	lsl	r3, r3, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r3
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r3
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r3
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r3
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r3
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r3, [sp, #56]
	add	r0, r2, #61440
	mov	r1, #256
	mov	r5, #512
	mov	r6, r7
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
	subs	r1, r1, #1
	bne	.Ltmp39
	lsl	r5, r5, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r5
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r5
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r6, [sp, #60]
	add	r0, r2, #73728
	mov	r1, #256
	mov	r3, #512
	mov	r5, r7
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
	subs	r1, r1, #1
	bne	.Ltmp40
	lsl	r3, r3, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r3
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r3
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r3
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r3
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r3
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r3, [sp, #64]
	add	r0, r2, #86016
	mov	r1, #256
	mov	r5, #512
	mov	r6, r7
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
	subs	r1, r1, #1
	bne	.Ltmp41
	lsl	r5, r5, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r5
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r5
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r5
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	ldr	r1, [sp, #68]
	add	r0, r2, #98304
	mov	r2, #256
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
.Ltmp42:
	vld1.32	{d4, d5, d6, d7}, [r7]!
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
	subs	r2, r2, #1
	bne	.Ltmp42
	lsl	r3, r3, #2
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q4
	vadd.f32	q1, q1, q5
	vst1.32	{d0, d1, d2, d3}, [r0], r3
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q6
	vadd.f32	q3, q3, q7
	vst1.32	{d4, d5, d6, d7}, [r0], r3
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q8
	vadd.f32	q1, q1, q9
	vst1.32	{d0, d1, d2, d3}, [r0], r3
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q10
	vadd.f32	q3, q3, q11
	vst1.32	{d4, d5, d6, d7}, [r0], r3
	vld1.32	{d0, d1, d2, d3}, [r0]
	vadd.f32	q0, q0, q12
	vadd.f32	q1, q1, q13
	vst1.32	{d0, d1, d2, d3}, [r0], r3
	vld1.32	{d4, d5, d6, d7}, [r0]
	vadd.f32	q2, q2, q14
	vadd.f32	q3, q3, q15
	vst1.32	{d4, d5, d6, d7}, [r0]

	@NO_APP
	cmp	r10, #262144
	bne	.LBB1_17
	ldr	r0, [sp, #44]
	ldr	r4, [sp, #36]
	add	lr, lr, #8192
	add	r0, r0, #1
	cmp	r0, #8
	bne	.LBB1_16
	ldr	r0, [sp, #24]
	ldr	r10, [sp, #28]
	ldr	r12, [sp, #20]
	movw	r8, #12288
	add	r4, r4, #128
	vmov.i32	q8, #0x0
	mov	r9, #108544
	movt	r8, #65535
	add	r0, r0, #1
	add	r10, r10, #262144
	cmp	r0, #16
	bne	.LBB1_13
	ldr	r1, [sp, #8]
	ldr	r9, [sp, #40]
	mov	r0, #0
	mov	r3, #86016
	mov	r7, #88064
	mov	r6, #90112
	mov	r10, #92160
	mov	r12, #94208
	mov	lr, #96256
	mov	r8, #98304
	mov	r11, #0
.LBB1_21:
	mov	r5, r9
	mov	r4, r1
	ldr	r2, [r5, r0]!
	add	r0, r0, #4
	str	r2, [r4, r11]!
	add	r11, r11, #196
	ldr	r2, [r5, #2048]
	cmp	r11, #100352
	str	r2, [r4, #4]
	mov	r2, #4096
	ldr	r2, [r5, r2]
	str	r2, [r4, #8]
	mov	r2, #6144
	ldr	r2, [r5, r2]
	str	r2, [r4, #12]
	mov	r2, #8192
	ldr	r2, [r5, r2]
	str	r2, [r4, #16]
	mov	r2, #10240
	ldr	r2, [r5, r2]
	str	r2, [r4, #20]
	mov	r2, #12288
	ldr	r2, [r5, r2]
	str	r2, [r4, #24]
	mov	r2, #14336
	ldr	r2, [r5, r2]
	str	r2, [r4, #28]
	mov	r2, #16384
	ldr	r2, [r5, r2]
	str	r2, [r4, #32]
	mov	r2, #18432
	ldr	r2, [r5, r2]
	str	r2, [r4, #36]
	mov	r2, #20480
	ldr	r2, [r5, r2]
	str	r2, [r4, #40]
	mov	r2, #22528
	ldr	r2, [r5, r2]
	str	r2, [r4, #44]
	mov	r2, #24576
	ldr	r2, [r5, r2]
	str	r2, [r4, #48]
	mov	r2, #26624
	ldr	r2, [r5, r2]
	str	r2, [r4, #52]
	mov	r2, #28672
	ldr	r2, [r5, r2]
	str	r2, [r4, #56]
	mov	r2, #30720
	ldr	r2, [r5, r2]
	str	r2, [r4, #60]
	mov	r2, #32768
	ldr	r2, [r5, r2]
	str	r2, [r4, #64]
	mov	r2, #34816
	ldr	r2, [r5, r2]
	str	r2, [r4, #68]
	mov	r2, #36864
	ldr	r2, [r5, r2]
	str	r2, [r4, #72]
	mov	r2, #38912
	ldr	r2, [r5, r2]
	str	r2, [r4, #76]
	mov	r2, #40960
	ldr	r2, [r5, r2]
	str	r2, [r4, #80]
	mov	r2, #43008
	ldr	r2, [r5, r2]
	str	r2, [r4, #84]
	mov	r2, #45056
	ldr	r2, [r5, r2]
	str	r2, [r4, #88]
	mov	r2, #47104
	ldr	r2, [r5, r2]
	str	r2, [r4, #92]
	mov	r2, #49152
	ldr	r2, [r5, r2]
	str	r2, [r4, #96]
	mov	r2, #51200
	ldr	r2, [r5, r2]
	str	r2, [r4, #100]
	mov	r2, #53248
	ldr	r2, [r5, r2]
	str	r2, [r4, #104]
	mov	r2, #55296
	ldr	r2, [r5, r2]
	str	r2, [r4, #108]
	mov	r2, #57344
	ldr	r2, [r5, r2]
	str	r2, [r4, #112]
	mov	r2, #59392
	ldr	r2, [r5, r2]
	str	r2, [r4, #116]
	mov	r2, #61440
	ldr	r2, [r5, r2]
	str	r2, [r4, #120]
	mov	r2, #63488
	ldr	r2, [r5, r2]
	str	r2, [r4, #124]
	mov	r2, #65536
	ldr	r2, [r5, r2]
	str	r2, [r4, #128]
	mov	r2, #67584
	ldr	r2, [r5, r2]
	str	r2, [r4, #132]
	mov	r2, #69632
	ldr	r2, [r5, r2]
	str	r2, [r4, #136]
	mov	r2, #71680
	ldr	r2, [r5, r2]
	str	r2, [r4, #140]
	mov	r2, #73728
	ldr	r2, [r5, r2]
	str	r2, [r4, #144]
	mov	r2, #75776
	ldr	r2, [r5, r2]
	str	r2, [r4, #148]
	mov	r2, #77824
	ldr	r2, [r5, r2]
	str	r2, [r4, #152]
	mov	r2, #79872
	ldr	r2, [r5, r2]
	str	r2, [r4, #156]
	mov	r2, #81920
	ldr	r2, [r5, r2]
	str	r2, [r4, #160]
	mov	r2, #83968
	ldr	r2, [r5, r2]
	str	r2, [r4, #164]
	ldr	r2, [r5, r3]
	str	r2, [r4, #168]
	ldr	r2, [r5, r7]
	str	r2, [r4, #172]
	ldr	r2, [r5, r6]
	str	r2, [r4, #176]
	ldr	r2, [r5, r10]
	str	r2, [r4, #180]
	ldr	r2, [r5, r12]
	str	r2, [r4, #184]
	ldr	r2, [r5, lr]
	str	r2, [r4, #188]
	ldr	r2, [r5, r8]
	str	r2, [r4, #192]
	bne	.LBB1_21
	ldr	r4, .LCPI1_1
	mov	r0, #1
	mov	r2, r9
.LPC1_1:
	ldr	r4, [pc, r4]
	ldr	r5, [sp, #12]
	ldr	r3, [r4]
	mov	r1, r5
	blx	r3
	ldr	r3, [r4]
	ldr	r2, [sp, #16]
	mov	r0, #1
	mov	r1, r5
	blx	r3
	ldr	r3, [r4]
	ldr	r2, [sp, #32]
	mov	r0, #1
	mov	r1, r5
	add	sp, sp, #72
	vpop	{d8, d9, d10, d11, d12, d13, d14, d15}
	add	sp, sp, #4
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	bx	r3
	.p2align	2
.LCPI1_1:
.Ltmp43:
	.long	__TVMBackendFreeWorkspace(GOT_PREL)-((.LPC1_1+8)-.Ltmp43)
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
.Ltmp44:
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
	bne	.Ltmp44
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
.Ltmp45:
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
	bne	.Ltmp45
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
	.asciz	"Assert fail: ((((1 == int32(arg0.strides[3])) && ((1*7) == int32(arg0.strides[2]))) && (((1*7)*7) == int32(arg0.strides[1]))) && ((((1*7)*7)*2048) == int32(arg0.strides[0]))), arg0.strides: expected to be compact array"
	.size	.L.str.1, 219

	.type	.L.str.2,%object
.L.str.2:
	.asciz	"Assert fail: ((((1 == int32(arg1.strides[3])) && ((1*1) == int32(arg1.strides[2]))) && (((1*1)*1) == int32(arg1.strides[1]))) && ((((1*1)*1)*2048) == int32(arg1.strides[0]))), arg1.strides: expected to be compact array"
	.size	.L.str.2, 219

	.type	.L.str.3,%object
.L.str.3:
	.asciz	"Assert fail: ((((1 == int32(arg2.strides[3])) && ((1*7) == int32(arg2.strides[2]))) && (((1*7)*7) == int32(arg2.strides[1]))) && ((((1*7)*7)*512) == int32(arg2.strides[0]))), arg2.strides: expected to be compact array"
	.size	.L.str.3, 218

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
	.asciz	"Assert fail: (int32(arg0.shape[1]) == 2048), Argument arg0.shape[1] has an unsatisfied constraint"
	.size	.L.str.11, 98

	.type	.L.str.12,%object
.L.str.12:
	.asciz	"Assert fail: (int32(arg0.shape[2]) == 7), Argument arg0.shape[2] has an unsatisfied constraint"
	.size	.L.str.12, 95

	.type	.L.str.13,%object
.L.str.13:
	.asciz	"Assert fail: (int32(arg0.shape[3]) == 7), Argument arg0.shape[3] has an unsatisfied constraint"
	.size	.L.str.13, 95

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
	.asciz	"Assert fail: (int32(arg1.shape[1]) == 2048), Argument arg1.shape[1] has an unsatisfied constraint"
	.size	.L.str.18, 98

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
	.asciz	"Assert fail: (int32(arg2.shape[2]) == 7), Argument arg2.shape[2] has an unsatisfied constraint"
	.size	.L.str.28, 95

	.type	.L.str.29,%object
.L.str.29:
	.asciz	"Assert fail: (int32(arg2.shape[3]) == 7), Argument arg2.shape[3] has an unsatisfied constraint"
	.size	.L.str.29, 95

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
