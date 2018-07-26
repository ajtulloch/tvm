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
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	b	.Ldefault_function_compute_
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

	.p2align	2
	.type	.Ldefault_function_compute_,%function
	.code	32
.Ldefault_function_compute_:
	.fnstart
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	.setfp	r11, sp, #28
	add	r11, sp, #28
	.pad	#20
	sub	sp, sp, #20
	str	r2, [r11, #-44]
	mov	r4, r3
	mov	r9, r1
	mov	r6, r0
	ldr	r7, .LCPI1_0
.LPC1_0:
	ldr	r7, [pc, r7]
	ldr	r5, [r7]
	sub	sp, sp, #8
	mov	r10, #32
	mov	r8, #2
	mov	r0, #1
	mov	r1, r4
	mov	r2, #401408
	mov	r3, #0
	stm	sp, {r8, r10}
	blx	r5
	add	sp, sp, #8
	ldr	r7, [r7]
	mov	r5, r0
	push	{r8, r10}
	mov	r0, #1
	mov	r1, r4
	mov	r2, #1048576
	mov	r3, #0
	blx	r7
	add	sp, sp, #8
	str	r5, [r11, #-40]
	str	r6, [r11, #-36]
	mov	r7, r0
	sub	r1, r11, #40
	mov	r2, #0
	ldr	r6, .LCPI1_1
.LPC1_1:
	ldr	r6, [pc, r6]
	ldr	r3, [r6]
	ldr	r0, .LCPI1_2
.LPC1_2:
	add	r0, pc, r0
	blx	r3
	cmp	r0, #0
	bne	.LBB1_4
	mov	r0, sp
	sub	r1, r0, #8
	mov	sp, r1
	stmdb	r0, {r7, r9}
	mov	r2, #0
	ldr	r3, [r6]
	ldr	r0, .LCPI1_3
.LPC1_3:
	add	r0, pc, r0
	blx	r3
	cmp	r0, #0
	bne	.LBB1_4
	mov	r0, sp
	sub	r1, r0, #16
	mov	sp, r1
	ldr	r2, [r11, #-44]
	str	r5, [r0, #-16]
	str	r7, [r0, #-12]
	ldr	r3, [r6]
	str	r2, [r0, #-8]
	mov	r2, #0
	ldr	r0, .LCPI1_4
.LPC1_4:
	add	r0, pc, r0
	blx	r3
	cmp	r0, #0
	bne	.LBB1_4
	ldr	r6, .LCPI1_5
	mov	r0, #1
	mov	r1, r4
	mov	r2, r7
.LPC1_5:
	ldr	r6, [pc, r6]
	ldr	r3, [r6]
	blx	r3
	ldr	r3, [r6]
	mov	r0, #1
	mov	r1, r4
	mov	r2, r5
	blx	r3
	mov	r0, #0
.LBB1_4:
	sub	sp, r11, #28
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
	.p2align	2
.LCPI1_0:
.Ltmp33:
	.long	__TVMBackendAllocWorkspace(GOT_PREL)-((.LPC1_0+8)-.Ltmp33)
.LCPI1_1:
.Ltmp34:
	.long	__TVMBackendParallelLaunch(GOT_PREL)-((.LPC1_1+8)-.Ltmp34)
.LCPI1_2:
	.long	.L__tvm_parallel_lambda-(.LPC1_2+8)
.LCPI1_3:
	.long	.L__tvm_parallel_lambda.34-(.LPC1_3+8)
.LCPI1_4:
	.long	.L__tvm_parallel_lambda.35-(.LPC1_4+8)
.LCPI1_5:
.Ltmp35:
	.long	__TVMBackendFreeWorkspace(GOT_PREL)-((.LPC1_5+8)-.Ltmp35)
.Lfunc_end1:
	.size	.Ldefault_function_compute_, .Lfunc_end1-.Ldefault_function_compute_
	.fnend

	.p2align	2
	.type	.L__tvm_parallel_lambda,%function
	.code	32
.L__tvm_parallel_lambda:
	.fnstart
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	.pad	#28
	sub	sp, sp, #28
	cmp	r0, #6
	stmib	sp, {r0, r1}
	bgt	.LBB2_17
	ldr	r0, [sp, #8]
	ldr	r6, [r0, #4]
	ldm	r2, {r0, r1}
	ldr	r12, [sp, #4]
	rsb	r3, r12, r12, lsl #3
	str	r6, [sp, #20]
	add	lr, r0, r3, lsl #13
	add	r5, r1, r3, lsl #4
	rsb	r1, r6, r6, lsl #3
	lsl	r3, r1, #4
	lsl	r1, r1, #13
	add	r4, lr, #49152
	add	r7, lr, #40960
	add	r0, lr, #32768
	add	r6, r5, #60
	add	r9, lr, #24576
	add	r10, lr, #16384
	add	r11, lr, #8192
	str	r3, [sp, #16]
	str	r1, [sp, #12]
.LBB2_2:
	mov	r1, #0
	str	r6, [sp, #24]
.LBB2_3:
	ldr	r8, [r6, #-60]
	mov	r2, lr
	str	r8, [r2, r1]!
	add	r1, r1, #16
	ldr	r3, [r6, #-56]
	cmp	r1, #8192
	str	r3, [r2, #4]
	ldr	r3, [r6, #-4]
	str	r3, [r2, #8]
	ldr	r3, [r6]
	add	r6, r6, #784
	str	r3, [r2, #12]
	bne	.LBB2_3
	mov	r1, #0
	mov	r6, r11
.LBB2_5:
	add	r2, r5, r1
	add	r1, r1, #784
	ldr	r3, [r2, #8]
	cmp	r1, #401408
	str	r3, [r6]
	ldr	r3, [r2, #12]
	str	r3, [r6, #4]
	ldr	r3, [r2, #64]
	str	r3, [r6, #8]
	ldr	r2, [r2, #68]
	str	r2, [r6, #12]
	add	r6, r6, #16
	bne	.LBB2_5
	mov	r1, #0
	mov	r6, r10
.LBB2_7:
	add	r2, r5, r1
	add	r1, r1, #784
	ldr	r3, [r2, #16]
	cmp	r1, #401408
	str	r3, [r6]
	ldr	r3, [r2, #20]
	str	r3, [r6, #4]
	ldr	r3, [r2, #72]
	str	r3, [r6, #8]
	ldr	r2, [r2, #76]
	str	r2, [r6, #12]
	add	r6, r6, #16
	bne	.LBB2_7
	mov	r1, #0
	mov	r6, r9
.LBB2_9:
	add	r2, r5, r1
	add	r1, r1, #784
	ldr	r3, [r2, #24]
	cmp	r1, #401408
	str	r3, [r6]
	ldr	r3, [r2, #28]
	str	r3, [r6, #4]
	ldr	r3, [r2, #80]
	str	r3, [r6, #8]
	ldr	r2, [r2, #84]
	str	r2, [r6, #12]
	add	r6, r6, #16
	bne	.LBB2_9
	mov	r1, #0
	mov	r6, r0
.LBB2_11:
	add	r2, r5, r1
	add	r1, r1, #784
	ldr	r3, [r2, #32]
	cmp	r1, #401408
	str	r3, [r6]
	ldr	r3, [r2, #36]
	str	r3, [r6, #4]
	ldr	r3, [r2, #88]
	str	r3, [r6, #8]
	ldr	r2, [r2, #92]
	str	r2, [r6, #12]
	add	r6, r6, #16
	bne	.LBB2_11
	mov	r1, #0
	mov	r6, r7
.LBB2_13:
	add	r2, r5, r1
	add	r1, r1, #784
	ldr	r3, [r2, #40]
	cmp	r1, #401408
	str	r3, [r6]
	ldr	r3, [r2, #44]
	str	r3, [r6, #4]
	ldr	r3, [r2, #96]
	str	r3, [r6, #8]
	ldr	r2, [r2, #100]
	str	r2, [r6, #12]
	add	r6, r6, #16
	bne	.LBB2_13
	mov	r1, #0
	mov	r6, r4
.LBB2_15:
	add	r2, r5, r1
	add	r1, r1, #784
	ldr	r3, [r2, #48]
	cmp	r1, #401408
	str	r3, [r6]
	ldr	r3, [r2, #52]
	str	r3, [r6, #4]
	ldr	r3, [r2, #104]
	str	r3, [r6, #8]
	ldr	r2, [r2, #108]
	str	r2, [r6, #12]
	add	r6, r6, #16
	bne	.LBB2_15
	ldr	r1, [sp, #16]
	ldr	r6, [sp, #24]
	ldr	r2, [sp, #12]
	add	r5, r5, r1
	add	r6, r6, r1
	ldr	r1, [sp, #20]
	add	r4, r4, r2
	add	r7, r7, r2
	add	r0, r0, r2
	add	r9, r9, r2
	add	r10, r10, r2
	add	r11, r11, r2
	add	lr, lr, r2
	add	r12, r12, r1
	cmp	r12, #7
	blt	.LBB2_2
.LBB2_17:
	ldr	r0, .LCPI2_0
.LPC2_0:
	ldr	r0, [pc, r0]
	ldr	r2, [r0]
	ldmib	sp, {r0, r1}
	blx	r2
	mov	r0, #0
	add	sp, sp, #28
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
	.p2align	2
.LCPI2_0:
.Ltmp36:
	.long	__TVMBackendParallelBarrier(GOT_PREL)-((.LPC2_0+8)-.Ltmp36)
.Lfunc_end2:
	.size	.L__tvm_parallel_lambda, .Lfunc_end2-.L__tvm_parallel_lambda
	.fnend

	.p2align	2
	.type	.L__tvm_parallel_lambda.34,%function
	.code	32
.L__tvm_parallel_lambda.34:
	.fnstart
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	.pad	#28
	sub	sp, sp, #28
	cmp	r0, #7
	str	r1, [sp]
	str	r0, [sp, #4]
	bgt	.LBB3_7
	ldr	r0, [sp]
	mov	r5, #4096
	mov	r6, #6144
	mov	r7, #8192
	mov	r4, #10240
	ldr	r3, [r0, #4]
	ldm	r2, {r0, r1}
	ldr	r2, [sp, #4]
	add	r8, r1, r2, lsl #14
	add	r0, r0, r2, lsl #14
	lsl	r2, r3, #14
	str	r3, [sp, #24]
	mov	r1, #0
	mov	r3, #12288
	str	r2, [sp, #20]
	add	lr, r0, #16
	mov	r2, #14336
.LBB3_2:
	ldr	r10, [sp, #4]
	str	r1, [sp, #8]
	str	lr, [sp, #12]
	str	r8, [sp, #16]
.LBB3_3:
	mov	r1, #0
	mov	r11, lr
.LBB3_4:
	mov	r12, r8
	ldr	r9, [r12, r1]!
	add	r1, r1, #4
	cmp	r1, #2048
	str	r9, [r11, #-16]
	ldr	r0, [r12, #2048]
	str	r0, [r11, #-12]
	ldr	r0, [r12, r5]
	str	r0, [r11, #-8]
	ldr	r0, [r12, r6]
	str	r0, [r11, #-4]
	ldr	r0, [r12, r7]
	str	r0, [r11]
	ldr	r0, [r12, r4]
	str	r0, [r11, #4]
	ldr	r0, [r12, r3]
	str	r0, [r11, #8]
	ldr	r0, [r12, r2]
	str	r0, [r11, #12]
	add	r11, r11, #32
	bne	.LBB3_4
	ldr	r0, [sp, #20]
	add	lr, lr, r0
	add	r8, r8, r0
	ldr	r0, [sp, #24]
	add	r10, r10, r0
	cmp	r10, #8
	blt	.LBB3_3
	ldr	r1, [sp, #8]
	ldr	lr, [sp, #12]
	ldr	r8, [sp, #16]
	add	r1, r1, #1
	add	lr, lr, #131072
	add	r8, r8, #131072
	cmp	r1, #8
	bne	.LBB3_2
.LBB3_7:
	ldr	r0, .LCPI3_0
.LPC3_0:
	ldr	r0, [pc, r0]
	ldr	r1, [sp]
	ldr	r2, [r0]
	ldr	r0, [sp, #4]
	blx	r2
	mov	r0, #0
	add	sp, sp, #28
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
	.p2align	2
.LCPI3_0:
.Ltmp37:
	.long	__TVMBackendParallelBarrier(GOT_PREL)-((.LPC3_0+8)-.Ltmp37)
.Lfunc_end3:
	.size	.L__tvm_parallel_lambda.34, .Lfunc_end3-.L__tvm_parallel_lambda.34
	.fnend

	.p2align	2
	.type	.L__tvm_parallel_lambda.35,%function
	.code	32
.L__tvm_parallel_lambda.35:
	.fnstart
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	.setfp	r11, sp, #28
	add	r11, sp, #28
	.pad	#4
	sub	sp, sp, #4
	.vsave	{d8, d9, d10, d11}
	vpush	{d8, d9, d10, d11}
	.pad	#104
	sub	sp, sp, #104
	bfc	sp, #0, #4
	mov	r6, sp
	ldr	r7, [r2]
	ldr	r3, [r2, #4]
	ldr	r2, [r2, #8]
	mov	r5, #16
	str	r1, [r6, #12]
	ldr	r1, [r1, #4]
	str	r2, [r6, #68]
	add	r2, r3, r0, lsl #14
	str	r1, [r6, #36]
	lsl	r1, r1, #14
	str	r2, [r6, #24]
	add	r2, r7, #8
	str	r1, [r6, #32]
	mov	r1, #0
	str	r2, [r6, #40]
	str	r1, [r6, #20]
.LBB4_1:
	cmp	r0, #7
	str	r0, [r6, #16]
	bgt	.LBB4_10
	ldr	r0, [r6, #20]
	ldr	r10, [r6, #24]
	ldr	r1, [r6, #16]
	lsl	r0, r0, #3
	str	r0, [r6, #28]
.LBB4_3:
	ldr	r0, [r6, #28]
	str	r1, [r6, #44]
	add	r0, r1, r0
	rsb	r0, r0, r0, lsl #3
	lsl	r0, r0, #3
	str	r0, [r6, #48]
	sub	r0, sp, #128
	bic	lr, r0, #15
	mov	sp, lr
	ldr	r8, [r6, #40]
	add	r1, lr, #64
	add	r9, lr, #32
	add	r0, lr, #96
	mov	r2, #0
	str	r0, [r6, #88]
	str	r1, [r6, #64]
	str	r9, [r6, #60]
.LBB4_4:
	ldr	r0, [r6, #48]
	str	r2, [r6, #56]
	str	r8, [r6, #52]
	add	r0, r2, r0
	rsb	r0, r0, r0, lsl #3
	lsl	r0, r0, #1
	str	r0, [r6, #72]
	mov	r0, #0
	str	r0, [r6, #92]
.LBB4_5:
	vmov.i32	q11, #0x0
	mov	r4, lr
	mov	r0, #112
	vmov.i32	q14, #0x0
	vmov.i32	q8, #0x0
	vmov.i32	q15, #0x0
	vmov.i32	q10, #0x0
	vmov.i32	q9, #0x0
	vmov.i32	q13, #0x0
	vmov.i32	q12, #0x0
	str	r8, [r6, #84]
	vst1.64	{d22, d23}, [r4:128], r0
	ldr	r0, [r6, #88]
	vst1.64	{d22, d23}, [r1:128]
	vst1.64	{d22, d23}, [r9:128]
	vst1.64	{d22, d23}, [r4:128]
	add	r4, lr, #80
	vst1.64	{d22, d23}, [r4:128]
	add	r4, lr, #48
	vst1.64	{d22, d23}, [r4:128]
	add	r4, lr, #16
	vst1.64	{d22, d23}, [r4:128]
	mov	r4, #0
	vst1.64	{d22, d23}, [r0:128]
.LBB4_6:
	add	r0, r10, r4
	sub	r2, r8, #8
	sub	r7, r8, #4
	add	r12, r8, #4
	vld1.32	{d4[], d5[]}, [r8:32], r5
	add	r4, r4, #32
	vld1.32	{d0, d1}, [r0:128]!
	vld1.32	{d2[], d3[]}, [r12:32]
	vld1.32	{d6[], d7[]}, [r7:32]
	vld1.32	{d8[], d9[]}, [r2:32]
	cmp	r4, #16384
	vld1.64	{d10, d11}, [r0:128]
	vfma.f32	q11, q1, q0
	vfma.f32	q8, q2, q0
	vfma.f32	q10, q3, q0
	vfma.f32	q13, q4, q0
	vfma.f32	q14, q1, q5
	vfma.f32	q15, q2, q5
	vfma.f32	q9, q3, q5
	vfma.f32	q12, q4, q5
	bne	.LBB4_6
	mov	r0, lr
	ldr	r3, [r6, #92]
	vmov.32	r2, d26[0]
	movw	r8, #4704
	movw	r12, #5488
	vst1.32	{d26, d27}, [r0:128]!
	vst1.64	{d24, d25}, [r0:128]
	mov	r0, r9
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d18, d19}, [r0:128]
	mov	r0, r1
	ldr	r1, [r6, #68]
	vst1.32	{d16, d17}, [r0:128]!
	vst1.64	{d30, d31}, [r0:128]
	ldr	r0, [r6, #88]
	mov	r4, r1
	mov	r7, r1
	vst1.32	{d22, d23}, [r0:128]!
	vst1.64	{d28, d29}, [r0:128]
	ldr	r0, [r6, #72]
	add	r0, r3, r0
	add	r3, r3, #1
	str	r2, [r4, r0, lsl #3]!
	cmp	r3, #7
	add	r2, r4, #784
	vst1.32	{d26[1]}, [r2:32]
	add	r2, r4, #1568
	vst1.32	{d27[0]}, [r2:32]
	add	r2, r4, #2352
	vst1.32	{d27[1]}, [r2:32]
	add	r2, r4, #3136
	vst1.32	{d24[0]}, [r2:32]
	add	r2, r4, #3920
	vst1.32	{d24[1]}, [r2:32]
	add	r2, r4, r8
	vst1.32	{d25[0]}, [r2:32]
	add	r2, r4, r12
	vst1.32	{d25[1]}, [r2:32]
	mov	r2, #4
	orr	r0, r2, r0, lsl #3
	vmov.32	r2, d20[0]
	str	r2, [r7, r0]!
	add	r0, r7, #784
	vst1.32	{d20[1]}, [r0:32]
	add	r0, r7, #1568
	vst1.32	{d21[0]}, [r0:32]
	add	r0, r7, #2352
	vst1.32	{d21[1]}, [r0:32]
	add	r0, r7, #3136
	vst1.32	{d18[0]}, [r0:32]
	add	r0, r7, #3920
	vst1.32	{d18[1]}, [r0:32]
	add	r0, r7, r8
	vst1.32	{d19[0]}, [r0:32]
	add	r0, r7, r12
	vst1.32	{d19[1]}, [r0:32]
	add	r0, r4, #56
	vst1.32	{d16[0]}, [r0:32]
	ldr	r0, [lr, #68]
	ldr	r2, [lr, #72]
	ldr	r7, [lr, #76]
	ldr	r1, [lr, #80]
	ldr	r12, [lr, #88]
	ldr	r8, [lr, #92]
	ldr	r9, [lr, #96]
	str	r0, [r6, #76]
	ldr	r0, [lr, #84]
	str	r0, [r6, #80]
	ldr	r0, [r6, #76]
	str	r0, [r4, #840]
	ldr	r0, [r6, #80]
	str	r2, [r4, #1624]
	str	r7, [r4, #2408]
	str	r1, [r4, #3192]
	str	r0, [r4, #3976]
	movw	r0, #4760
	str	r12, [r4, r0]
	movw	r0, #5544
	str	r8, [r4, r0]
	str	r9, [r4, #60]
	add	r9, lr, #100
	ldm	r9, {r0, r1, r2, r9}
	ldr	r7, [lr, #116]
	ldr	r12, [lr, #120]
	ldr	r8, [lr, #124]
	str	r3, [r6, #92]
	str	r0, [r4, #844]
	movw	r0, #4764
	str	r1, [r4, #1628]
	str	r2, [r4, #2412]
	str	r9, [r4, #3196]
	str	r7, [r4, #3980]
	ldr	r1, [r6, #64]
	ldr	r9, [r6, #60]
	str	r12, [r4, r0]
	movw	r0, #5548
	str	r8, [r4, r0]
	ldr	r8, [r6, #84]
	add	r8, r8, #8192
	bne	.LBB4_5
	ldr	r2, [r6, #56]
	ldr	r8, [r6, #52]
	add	r2, r2, #1
	add	r8, r8, #57344
	cmp	r2, #7
	bne	.LBB4_4
	ldr	r0, [r6, #32]
	ldr	r1, [r6, #44]
	add	r10, r10, r0
	ldr	r0, [r6, #36]
	add	r1, r1, r0
	cmp	r1, #8
	blt	.LBB4_3
.LBB4_10:
	ldr	r0, [r6, #24]
	add	r0, r0, #131072
	str	r0, [r6, #24]
	ldr	r0, [r6, #20]
	add	r0, r0, #1
	str	r0, [r6, #20]
	cmp	r0, #8
	ldr	r0, [r6, #16]
	bne	.LBB4_1
	ldr	r1, .LCPI4_0
.LPC4_0:
	ldr	r1, [pc, r1]
	ldr	r2, [r1]
	ldr	r1, [r6, #12]
	blx	r2
	mov	r0, #0
	sub	sp, r11, #64
	vpop	{d8, d9, d10, d11}
	add	sp, sp, #4
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
	.p2align	2
.LCPI4_0:
.Ltmp38:
	.long	__TVMBackendParallelBarrier(GOT_PREL)-((.LPC4_0+8)-.Ltmp38)
.Lfunc_end4:
	.size	.L__tvm_parallel_lambda.35, .Lfunc_end4-.L__tvm_parallel_lambda.35
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
.Ltmp39:
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
	bne	.Ltmp39
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
.Lfunc_end5:
	.size	sgemm_compute_6x8__neon, .Lfunc_end5-sgemm_compute_6x8__neon
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
.Lfunc_end6:
	.size	sgemm_reset_6x8__neon, .Lfunc_end6-sgemm_reset_6x8__neon
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
.Ltmp40:
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
	bne	.Ltmp40
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
.Lfunc_end7:
	.size	sgemm_update_6x8__neon, .Lfunc_end7-sgemm_update_6x8__neon
	.cantunwind
	.fnend

	.type	__TVMAPISetLastError,%object
	.bss
	.weak	__TVMAPISetLastError
	.p2align	2
__TVMAPISetLastError:
	.long	0
	.size	__TVMAPISetLastError, 4

	.type	__TVMBackendParallelLaunch,%object
	.weak	__TVMBackendParallelLaunch
	.p2align	2
__TVMBackendParallelLaunch:
	.long	0
	.size	__TVMBackendParallelLaunch, 4

	.type	__TVMBackendParallelBarrier,%object
	.weak	__TVMBackendParallelBarrier
	.p2align	2
__TVMBackendParallelBarrier:
	.long	0
	.size	__TVMBackendParallelBarrier, 4

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
