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
	vldr	d18, .LCPI0_68
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
	cmp	r0, #14336
	ldreq	r0, [r1]
	cmpeq	r0, #401408
	beq	.LBB0_6
.LBB0_5:
	ldr	r0, .LCPI0_62
.LPC0_60:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_63
.LPC0_61:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_6:
	ldr	r0, [lr]
	ldr	r1, [lr, #24]
	ldr	r10, [lr, #20]
	ldr	r3, [r6, #4]
	str	r0, [sp]
	ldr	r0, [r6, #8]
	cmp	r1, #0
	str	r0, [sp, #8]
	beq	.LBB0_11
	add	r0, r1, #16
	vldr	d18, .LCPI0_69
	vld1.64	{d16, d17}, [r0]
	vmovn.i64	d16, q8
	vceq.i32	d16, d16, d18
	vmov.32	r0, d16[1]
	tst	r0, #1
	beq	.LBB0_10
	vmov.32	r0, d16[0]
	tst	r0, #1
	beq	.LBB0_10
	ldr	r0, [r1, #8]
	cmp	r0, #131072
	ldreq	r0, [r1]
	cmpeq	r0, #131072
	beq	.LBB0_11
.LBB0_10:
	ldr	r0, .LCPI0_64
.LPC0_62:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_65
.LPC0_63:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_11:
	ldr	r11, [r12, #24]
	ldr	r1, [r12]
	ldr	r8, [r12, #20]
	cmp	r11, #0
	beq	.LBB0_16
	add	r0, r11, #16
	vldr	d18, .LCPI0_69
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
	cmp	r0, #3584
	ldreq	r0, [r11]
	cmpeq	r0, #50176
	beq	.LBB0_16
.LBB0_15:
	ldr	r0, .LCPI0_66
.LPC0_64:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_67
.LPC0_65:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_16:
	cmp	r5, #13
	bhi	.LBB0_35
	mov	r0, #1
	movw	r2, #8344
	tst	r2, r0, lsl r5
	beq	.LBB0_35
	cmp	r4, #13
	bhi	.LBB0_36
	mov	r0, #1
	movw	r2, #8344
	tst	r2, r0, lsl r4
	beq	.LBB0_36
	cmp	r9, #13
	bhi	.LBB0_37
	mov	r0, #1
	movw	r2, #8344
	tst	r2, r0, lsl r9
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
	ldr	r0, .LCPI0_14
.LPC0_12:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_15
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
	cmp	r0, #28
	bne	.LBB0_64
	ldr	r0, [r7, #16]
	cmp	r0, #28
	bne	.LBB0_65
	ldr	r0, [r7, #24]
	cmp	r0, #512
	bne	.LBB0_66
	ldr	r0, [r6, #32]
	ldr	r3, [r6, #36]
	orrs	r0, r0, r3
	bne	.LBB0_68
	ldr	r0, [lr, #12]
	cmp	r0, #4
	bne	.LBB0_69
	ldrb	r0, [lr, #16]
	cmp	r0, #2
	ldrbeq	r0, [lr, #17]
	cmpeq	r0, #32
	beq	.LBB0_38
.LBB0_34:
	ldr	r0, .LCPI0_28
.LPC0_26:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_29
.LPC0_27:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_35:
	ldr	r0, .LCPI0_4
.LPC0_2:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_5
.LPC0_3:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_36:
	ldr	r0, .LCPI0_6
.LPC0_4:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_7
.LPC0_5:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_37:
	ldr	r0, .LCPI0_8
.LPC0_6:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_9
.LPC0_7:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_38:
	ldrh	r0, [lr, #18]
	cmp	r0, #1
	bne	.LBB0_34
	ldr	r0, [r10]
	cmp	r0, #1
	bne	.LBB0_70
	ldr	r0, [r10, #8]
	cmp	r0, #1
	bne	.LBB0_71
	ldr	r0, [r10, #16]
	cmp	r0, #512
	bne	.LBB0_72
	ldr	r0, [r10, #24]
	cmp	r0, #256
	bne	.LBB0_73
	ldr	r0, [lr, #32]
	ldr	r3, [lr, #36]
	orrs	r0, r0, r3
	bne	.LBB0_74
	ldr	r0, [lr, #4]
	cmp	r0, #1
	bne	.LBB0_75
	ldr	r0, [lr, #8]
	ldr	r3, [sp, #8]
	cmp	r3, r0
	bne	.LBB0_76
	ldr	r0, [r12, #12]
	cmp	r0, #4
	bne	.LBB0_77
	ldrb	r0, [r12, #16]
	cmp	r0, #2
	ldrbeq	r0, [r12, #17]
	cmpeq	r0, #32
	beq	.LBB0_50
.LBB0_48:
	ldr	r0, .LCPI0_46
.LPC0_44:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_47
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
	bne	.LBB0_78
	ldr	r0, [r8, #8]
	cmp	r0, #14
	bne	.LBB0_79
	ldr	r0, [r8, #16]
	cmp	r0, #14
	bne	.LBB0_80
	ldr	r0, [r8, #24]
	cmp	r0, #256
	bne	.LBB0_81
	ldr	r0, [r12, #32]
	ldr	r5, [r12, #36]
	orrs	r0, r0, r5
	bne	.LBB0_82
	ldr	r0, [r12, #4]
	cmp	r0, #1
	bne	.LBB0_83
	ldr	r0, [r12, #8]
	cmp	r3, r0
	bne	.LBB0_84
	ldr	r0, [sp, #4]
	ldr	r2, [sp]
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	b	.Ldefault_function_compute_
.LBB0_59:
	ldr	r0, .LCPI0_2
.LPC0_0:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_3
.LPC0_1:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_60:
	ldr	r0, .LCPI0_10
.LPC0_8:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_11
.LPC0_9:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_61:
	ldr	r0, .LCPI0_12
.LPC0_10:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_13
.LPC0_11:
	add	r0, pc, r0
	b	.LBB0_49
	.p2align	3
.LCPI0_68:
	.long	512
	.long	1
.LBB0_63:
	ldr	r0, .LCPI0_16
.LPC0_14:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_17
.LPC0_15:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_64:
	ldr	r0, .LCPI0_18
.LPC0_16:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_19
.LPC0_17:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_65:
	ldr	r0, .LCPI0_20
.LPC0_18:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_21
.LPC0_19:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_66:
	ldr	r0, .LCPI0_22
.LPC0_20:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_23
.LPC0_21:
	add	r0, pc, r0
	b	.LBB0_49
	.p2align	3
.LCPI0_69:
	.long	256
	.long	1
.LBB0_68:
	ldr	r0, .LCPI0_24
.LPC0_22:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_25
.LPC0_23:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_69:
	ldr	r0, .LCPI0_26
.LPC0_24:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_27
.LPC0_25:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_70:
	ldr	r0, .LCPI0_30
.LPC0_28:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_31
.LPC0_29:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_71:
	ldr	r0, .LCPI0_32
.LPC0_30:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_33
.LPC0_31:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_72:
	ldr	r0, .LCPI0_34
.LPC0_32:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_35
.LPC0_33:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_73:
	ldr	r0, .LCPI0_36
.LPC0_34:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_37
.LPC0_35:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_74:
	ldr	r0, .LCPI0_38
.LPC0_36:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_39
.LPC0_37:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_75:
	ldr	r0, .LCPI0_40
.LPC0_38:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_41
.LPC0_39:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_76:
	ldr	r0, .LCPI0_42
.LPC0_40:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_43
.LPC0_41:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_77:
	ldr	r0, .LCPI0_44
.LPC0_42:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_45
.LPC0_43:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_78:
	ldr	r0, .LCPI0_48
.LPC0_46:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_49
.LPC0_47:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_79:
	ldr	r0, .LCPI0_50
.LPC0_48:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_51
.LPC0_49:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_80:
	ldr	r0, .LCPI0_52
.LPC0_50:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_53
.LPC0_51:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_81:
	ldr	r0, .LCPI0_54
.LPC0_52:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_55
.LPC0_53:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_82:
	ldr	r0, .LCPI0_56
.LPC0_54:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_57
.LPC0_55:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_83:
	ldr	r0, .LCPI0_58
.LPC0_56:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_59
.LPC0_57:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_84:
	ldr	r0, .LCPI0_60
.LPC0_58:
	ldr	r0, [pc, r0]
	ldr	r1, [r0]
	ldr	r0, .LCPI0_61
.LPC0_59:
	add	r0, pc, r0
	b	.LBB0_49
	.p2align	2
.LCPI0_2:
.Ltmp0:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_0+8)-.Ltmp0)
.LCPI0_3:
	.long	.L.str-(.LPC0_1+8)
.LCPI0_4:
.Ltmp1:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_2+8)-.Ltmp1)
.LCPI0_5:
	.long	.L.str.4-(.LPC0_3+8)
.LCPI0_6:
.Ltmp2:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_4+8)-.Ltmp2)
.LCPI0_7:
	.long	.L.str.5-(.LPC0_5+8)
.LCPI0_8:
.Ltmp3:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_6+8)-.Ltmp3)
.LCPI0_9:
	.long	.L.str.6-(.LPC0_7+8)
.LCPI0_10:
.Ltmp4:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_8+8)-.Ltmp4)
.LCPI0_11:
	.long	.L.str.7-(.LPC0_9+8)
.LCPI0_12:
.Ltmp5:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_10+8)-.Ltmp5)
.LCPI0_13:
	.long	.L.str.8-(.LPC0_11+8)
.LCPI0_14:
.Ltmp6:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_12+8)-.Ltmp6)
.LCPI0_15:
	.long	.L.str.9-(.LPC0_13+8)
.LCPI0_16:
.Ltmp7:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_14+8)-.Ltmp7)
.LCPI0_17:
	.long	.L.str.10-(.LPC0_15+8)
.LCPI0_18:
.Ltmp8:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_16+8)-.Ltmp8)
.LCPI0_19:
	.long	.L.str.11-(.LPC0_17+8)
.LCPI0_20:
.Ltmp9:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_18+8)-.Ltmp9)
.LCPI0_21:
	.long	.L.str.12-(.LPC0_19+8)
.LCPI0_22:
.Ltmp10:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_20+8)-.Ltmp10)
.LCPI0_23:
	.long	.L.str.13-(.LPC0_21+8)
.LCPI0_24:
.Ltmp11:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_22+8)-.Ltmp11)
.LCPI0_25:
	.long	.L.str.14-(.LPC0_23+8)
.LCPI0_26:
.Ltmp12:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_24+8)-.Ltmp12)
.LCPI0_27:
	.long	.L.str.15-(.LPC0_25+8)
.LCPI0_28:
.Ltmp13:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_26+8)-.Ltmp13)
.LCPI0_29:
	.long	.L.str.16-(.LPC0_27+8)
.LCPI0_30:
.Ltmp14:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_28+8)-.Ltmp14)
.LCPI0_31:
	.long	.L.str.17-(.LPC0_29+8)
.LCPI0_32:
.Ltmp15:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_30+8)-.Ltmp15)
.LCPI0_33:
	.long	.L.str.18-(.LPC0_31+8)
.LCPI0_34:
.Ltmp16:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_32+8)-.Ltmp16)
.LCPI0_35:
	.long	.L.str.19-(.LPC0_33+8)
.LCPI0_36:
.Ltmp17:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_34+8)-.Ltmp17)
.LCPI0_37:
	.long	.L.str.20-(.LPC0_35+8)
.LCPI0_38:
.Ltmp18:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_36+8)-.Ltmp18)
.LCPI0_39:
	.long	.L.str.21-(.LPC0_37+8)
.LCPI0_40:
.Ltmp19:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_38+8)-.Ltmp19)
.LCPI0_41:
	.long	.L.str.22-(.LPC0_39+8)
.LCPI0_42:
.Ltmp20:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_40+8)-.Ltmp20)
.LCPI0_43:
	.long	.L.str.23-(.LPC0_41+8)
.LCPI0_44:
.Ltmp21:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_42+8)-.Ltmp21)
.LCPI0_45:
	.long	.L.str.24-(.LPC0_43+8)
.LCPI0_46:
.Ltmp22:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_44+8)-.Ltmp22)
.LCPI0_47:
	.long	.L.str.25-(.LPC0_45+8)
.LCPI0_48:
.Ltmp23:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_46+8)-.Ltmp23)
.LCPI0_49:
	.long	.L.str.26-(.LPC0_47+8)
.LCPI0_50:
.Ltmp24:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_48+8)-.Ltmp24)
.LCPI0_51:
	.long	.L.str.27-(.LPC0_49+8)
.LCPI0_52:
.Ltmp25:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_50+8)-.Ltmp25)
.LCPI0_53:
	.long	.L.str.28-(.LPC0_51+8)
.LCPI0_54:
.Ltmp26:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_52+8)-.Ltmp26)
.LCPI0_55:
	.long	.L.str.29-(.LPC0_53+8)
.LCPI0_56:
.Ltmp27:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_54+8)-.Ltmp27)
.LCPI0_57:
	.long	.L.str.30-(.LPC0_55+8)
.LCPI0_58:
.Ltmp28:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_56+8)-.Ltmp28)
.LCPI0_59:
	.long	.L.str.31-(.LPC0_57+8)
.LCPI0_60:
.Ltmp29:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_58+8)-.Ltmp29)
.LCPI0_61:
	.long	.L.str.32-(.LPC0_59+8)
.LCPI0_62:
.Ltmp30:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_60+8)-.Ltmp30)
.LCPI0_63:
	.long	.L.str.1-(.LPC0_61+8)
.LCPI0_64:
.Ltmp31:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_62+8)-.Ltmp31)
.LCPI0_65:
	.long	.L.str.2-(.LPC0_63+8)
.LCPI0_66:
.Ltmp32:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_64+8)-.Ltmp32)
.LCPI0_67:
	.long	.L.str.3-(.LPC0_65+8)
.Lfunc_end0:
	.size	default_function, .Lfunc_end0-default_function
	.fnend

	.p2align	2
	.type	.Ldefault_function_compute_,%function
	.code	32
.Ldefault_function_compute_:
	.fnstart
	.save	{r4, r5, r6, r7, r8, r10, r11, lr}
	push	{r4, r5, r6, r7, r8, r10, r11, lr}
	.setfp	r11, sp, #24
	add	r11, sp, #24
	.pad	#8
	sub	sp, sp, #8
	mov	r6, r0
	ldr	r0, .LCPI1_0
	mov	r4, r3
	mov	r8, r2
	mov	r7, r1
.LPC1_0:
	ldr	r0, [pc, r0]
	ldr	r5, [r0]
	sub	sp, sp, #8
	movw	r2, #51200
	mov	r0, #32
	mov	r1, #2
	mov	r3, #0
	str	r1, [sp]
	str	r0, [sp, #4]
	movt	r2, #22
	mov	r0, #1
	mov	r1, r4
	blx	r5
	add	sp, sp, #8
	mov	r5, r0
	sub	r1, r11, #32
	mov	r2, #0
	str	r5, [r11, #-32]
	str	r6, [r11, #-28]
	ldr	r6, .LCPI1_1
.LPC1_1:
	ldr	r6, [pc, r6]
	ldr	r3, [r6]
	ldr	r0, .LCPI1_2
.LPC1_2:
	add	r0, pc, r0
	blx	r3
	cmp	r0, #0
	bne	.LBB1_3
	mov	r0, sp
	sub	r1, r0, #16
	mov	sp, r1
	str	r7, [r0, #-16]
	str	r5, [r0, #-12]
	str	r8, [r0, #-8]
	ldr	r3, [r6]
	mov	r2, #0
	ldr	r0, .LCPI1_3
.LPC1_3:
	add	r0, pc, r0
	blx	r3
	cmp	r0, #0
	bne	.LBB1_3
	ldr	r0, .LCPI1_4
	mov	r1, r4
	mov	r2, r5
.LPC1_4:
	ldr	r0, [pc, r0]
	ldr	r3, [r0]
	mov	r0, #1
	blx	r3
	mov	r0, #0
.LBB1_3:
	sub	sp, r11, #24
	pop	{r4, r5, r6, r7, r8, r10, r11, pc}
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
.Ltmp35:
	.long	__TVMBackendFreeWorkspace(GOT_PREL)-((.LPC1_4+8)-.Ltmp35)
.Lfunc_end1:
	.size	.Ldefault_function_compute_, .Lfunc_end1-.Ldefault_function_compute_
	.fnend

	.p2align	2
	.type	.L__tvm_parallel_lambda,%function
	.code	32
.L__tvm_parallel_lambda:
	.fnstart
	.save	{r4, r5, r6, lr}
	push	{r4, r5, r6, lr}
	ldr	r1, [r1, #4]
	mov	lr, #27
	add	r3, r1, #26
	sdiv	r1, r3, r1
	add	r3, r0, #1
	mul	r12, r1, r3
	mul	r0, r1, r0
	cmp	r12, #27
	movge	r12, lr
	cmp	r0, #27
	movlt	lr, r0
	cmp	lr, r12
	bge	.LBB2_7
	movw	r4, #51200
	mvn	r0, r0
	ldm	r2, {r1, r2}
	mov	r5, #13824
	cmn	r0, #28
	movt	r4, #65535
	mvnle	r0, #27
	add	r3, r4, #512
	mls	r3, r0, r5, r3
	rsb	r0, r0, r0, lsl #3
	sub	r0, r4, r0, lsl #11
	add	r3, r1, r3, lsl #2
	add	r2, r2, r0, lsl #2
.LBB2_2:
	mov	r1, #0
	mov	r0, r3
	mov	r4, r2
.LBB2_3:
	mov	r5, #0
.LBB2_4:
	ldr	r6, [r4, r5, lsl #2]
	str	r6, [r0, r5, lsl #2]
	add	r5, r5, #1
	cmp	r5, #512
	bne	.LBB2_4
	add	r1, r1, #1
	add	r0, r0, #2048
	add	r4, r4, #2048
	cmp	r1, #27
	bne	.LBB2_3
	add	lr, lr, #1
	add	r3, r3, #55296
	add	r2, r2, #57344
	cmp	lr, r12
	blt	.LBB2_2
.LBB2_7:
	mov	r0, #0
	pop	{r4, r5, r6, pc}
.Lfunc_end2:
	.size	.L__tvm_parallel_lambda, .Lfunc_end2-.L__tvm_parallel_lambda
	.cantunwind
	.fnend

	.p2align	2
	.type	.L__tvm_parallel_lambda.34,%function
	.code	32
.L__tvm_parallel_lambda.34:
	.fnstart
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	.pad	#4
	sub	sp, sp, #4
	.vsave	{d8, d9, d10, d11, d12, d13, d14, d15}
	vpush	{d8, d9, d10, d11, d12, d13, d14, d15}
	.pad	#984
	sub	sp, sp, #984
	ldr	r1, [r1, #4]
	mov	r5, #196
	add	r3, r1, #195
	sdiv	r1, r3, r1
	add	r3, r0, #1
	mul	r3, r1, r3
	mul	r0, r1, r0
	cmp	r3, #196
	movge	r3, r5
	cmp	r0, #196
	movlt	r5, r0
	str	r3, [sp, #12]
	cmp	r5, r3
	bge	.LBB3_5
	ldr	r1, [r2]
	mvn	r0, r0
	cmn	r0, #197
	mvnle	r0, #196
	str	r1, [sp, #8]
	ldmib	r2, {r1, r2}
	str	r2, [sp, #4]
	movw	r2, #64512
	movt	r2, #65535
	sub	r0, r2, r0, lsl #10
	add	r3, r1, r0, lsl #2
	mov	r0, #32
.LBB3_2:
	ldr	r1, [sp, #8]
	vmov.i32	q10, #0x0
	str	r5, [sp, #124]
	str	r3, [sp, #120]
	vmov.i32	q8, #0x0
	mov	lr, #0
	vmov.i32	q6, #0x0
	vmov.i32	q7, #0x0
	vmov.i32	q1, #0x0
	vmov.i32	q2, #0x0
	vmov.i32	q4, #0x0
	vmov.i32	q14, #0x0
	vmov.i32	q5, #0x0
	vmov.i32	q15, #0x0
	vmov.i32	q0, #0x0
	vmov.i32	q11, #0x0
	vmov.i32	q12, #0x0
	vmov.i32	q13, #0x0
	add	r8, r1, r5, lsl #10
	add	r1, r8, #976
	add	r12, r8, #384
	add	r6, r8, #256
	add	r9, r8, #192
	add	r10, r8, #128
	add	r7, r8, #64
	str	r1, [sp, #116]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #912
	str	r1, [sp, #112]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #848
	str	r1, [sp, #108]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #784
	str	r1, [sp, #104]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #720
	str	r1, [sp, #100]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #656
	str	r1, [sp, #92]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #592
	str	r1, [sp, #84]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #528
	str	r1, [sp, #80]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #464
	str	r1, [sp, #72]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #400
	str	r1, [sp, #64]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #336
	str	r1, [sp, #60]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #272
	str	r1, [sp, #52]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #208
	str	r1, [sp, #44]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #144
	str	r1, [sp, #40]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #80
	str	r1, [sp, #32]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #48
	str	r1, [sp, #28]
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #32
	str	r1, [sp, #20]
	vst1.64	{d20, d21}, [r1:128]
	mov	r1, r8
	vst1.32	{d20, d21}, [r1:128]!
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #960
	str	r1, [sp, #96]
	vst1.32	{d20, d21}, [r1:128], r0
	vst1.32	{d20, d21}, [r1:128]!
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #896
	str	r1, [sp, #88]
	vst1.32	{d20, d21}, [r1:128], r0
	vst1.32	{d20, d21}, [r1:128]!
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #832
	str	r1, [sp, #76]
	vst1.32	{d20, d21}, [r1:128], r0
	vst1.32	{d20, d21}, [r1:128]!
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #768
	str	r1, [sp, #68]
	vst1.32	{d20, d21}, [r1:128], r0
	vst1.32	{d20, d21}, [r1:128]!
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #704
	str	r1, [sp, #56]
	vst1.32	{d20, d21}, [r1:128], r0
	vst1.32	{d20, d21}, [r1:128]!
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #640
	str	r1, [sp, #48]
	vst1.32	{d20, d21}, [r1:128], r0
	vst1.32	{d20, d21}, [r1:128]!
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #576
	str	r1, [sp, #36]
	vst1.32	{d20, d21}, [r1:128], r0
	vst1.32	{d20, d21}, [r1:128]!
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #512
	str	r1, [sp, #24]
	vst1.32	{d20, d21}, [r1:128], r0
	vst1.32	{d20, d21}, [r1:128]!
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #448
	str	r1, [sp, #16]
	vst1.32	{d20, d21}, [r1:128], r0
	vst1.32	{d20, d21}, [r1:128]!
	vst1.64	{d20, d21}, [r1:128]
	mov	r1, r12
	vst1.32	{d20, d21}, [r1:128], r0
	vst1.32	{d20, d21}, [r1:128]!
	vst1.64	{d20, d21}, [r1:128]
	add	r1, r8, #320
	mov	r2, r1
	vst1.32	{d20, d21}, [r2:128], r0
	vst1.32	{d20, d21}, [r2:128]!
	vst1.64	{d20, d21}, [r2:128]
	mov	r2, r6
	vst1.32	{d20, d21}, [r2:128], r0
	vst1.32	{d20, d21}, [r2:128]!
	vst1.64	{d20, d21}, [r2:128]
	mov	r2, r9
	vst1.32	{d20, d21}, [r2:128], r0
	vst1.32	{d20, d21}, [r2:128]!
	vst1.64	{d20, d21}, [r2:128]
	mov	r2, r10
	vst1.32	{d20, d21}, [r2:128], r0
	vst1.32	{d20, d21}, [r2:128]!
	vst1.64	{d20, d21}, [r2:128]
	mov	r2, r7
	vst1.32	{d20, d21}, [r2:128], r0
	vst1.32	{d20, d21}, [r2:128]!
	vst1.64	{d20, d21}, [r2:128]
	movw	r2, #9363
	movt	r2, #37449
	smmla	r2, r5, r2, r5
	asr	r5, r2, #3
	add	r2, r5, r2, lsr #31
	mov	r5, #53248
	mla	r2, r2, r5, r3
	add	r3, sp, #208
	ldr	r5, [sp, #4]
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #192
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #176
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #272
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #160
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #256
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #240
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #336
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #224
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #320
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #304
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #400
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #288
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #384
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #368
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #464
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #352
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #448
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #432
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #528
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #416
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #512
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #496
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #592
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #480
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #576
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #560
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #656
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #544
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #640
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #624
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #720
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #608
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #704
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #688
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #784
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #672
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #768
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #752
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #848
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #736
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #832
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #816
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #912
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #800
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #896
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #880
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #864
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #944
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
	add	r3, sp, #928
	vstmia	r3, {d16, d17}
	vmov.i32	q8, #0x0
.LBB3_3:
	add	r3, sp, #128
	add	r11, r5, #64
	add	r4, r2, lr
	add	lr, lr, #4
	vstmia	r3, {d12, d13}
	add	r3, sp, #144
	vld1.32	{d18, d19}, [r11:128], r0
	vorr	q6, q4, q4
	vorr	q4, q1, q1
	vorr	q1, q15, q15
	vorr	q15, q13, q13
	vorr	q13, q11, q11
	cmp	lr, #2048
	vstmia	r3, {d14, d15}
	add	r3, sp, #960
	vorr	q7, q5, q5
	vorr	q5, q2, q2
	vorr	q2, q0, q0
	vorr	q0, q14, q14
	vorr	q14, q12, q12
	vorr	q12, q10, q10
	vstmia	r3, {d16, d17}
	add	r3, sp, #272
	vld1.32	{d16[], d17[]}, [r4:32]
	add	r4, r5, #128
	vldmia	r3, {d6, d7}
	add	r3, sp, #272
	vld1.32	{d22, d23}, [r4:128], r0
	vld1.32	{d20, d21}, [r11:128]!
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, sp, #256
	vldmia	r3, {d18, d19}
	add	r3, sp, #256
	vfma.f32	q9, q8, q10
	vorr	q10, q12, q12
	vorr	q12, q14, q14
	vorr	q14, q0, q0
	vorr	q0, q2, q2
	vorr	q2, q5, q5
	vorr	q5, q7, q7
	vstmia	r3, {d18, d19}
	add	r3, sp, #144
	vldmia	r3, {d14, d15}
	add	r3, sp, #336
	vldmia	r3, {d18, d19}
	add	r3, sp, #336
	vfma.f32	q9, q8, q11
	vorr	q11, q13, q13
	vorr	q13, q15, q15
	vorr	q15, q1, q1
	vorr	q1, q4, q4
	vorr	q4, q6, q6
	vstmia	r3, {d18, d19}
	add	r3, sp, #128
	vld1.32	{d18, d19}, [r4:128]!
	vldmia	r3, {d12, d13}
	add	r3, sp, #320
	vldmia	r3, {d6, d7}
	add	r3, sp, #320
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r11:128]
	vstmia	r3, {d6, d7}
	add	r3, sp, #240
	vldmia	r3, {d6, d7}
	add	r3, sp, #240
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #192
	vstmia	r3, {d6, d7}
	add	r3, sp, #304
	vldmia	r3, {d6, d7}
	add	r3, sp, #304
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vstmia	r3, {d6, d7}
	add	r3, sp, #400
	vldmia	r3, {d6, d7}
	add	r3, sp, #400
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vstmia	r3, {d6, d7}
	add	r3, sp, #384
	vldmia	r3, {d6, d7}
	add	r3, sp, #384
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #256
	vstmia	r3, {d6, d7}
	add	r3, sp, #368
	vldmia	r3, {d6, d7}
	add	r3, sp, #368
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vstmia	r3, {d6, d7}
	add	r3, sp, #464
	vldmia	r3, {d6, d7}
	add	r3, sp, #464
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vstmia	r3, {d6, d7}
	add	r3, sp, #448
	vldmia	r3, {d6, d7}
	add	r3, sp, #448
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #320
	vstmia	r3, {d6, d7}
	add	r3, sp, #432
	vldmia	r3, {d6, d7}
	add	r3, sp, #432
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vstmia	r3, {d6, d7}
	add	r3, sp, #528
	vldmia	r3, {d6, d7}
	add	r3, sp, #528
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vstmia	r3, {d6, d7}
	add	r3, sp, #512
	vldmia	r3, {d6, d7}
	add	r3, sp, #512
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #384
	vstmia	r3, {d6, d7}
	add	r3, sp, #496
	vldmia	r3, {d6, d7}
	add	r3, sp, #496
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vstmia	r3, {d6, d7}
	add	r3, sp, #592
	vldmia	r3, {d6, d7}
	add	r3, sp, #592
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vstmia	r3, {d6, d7}
	add	r3, sp, #576
	vldmia	r3, {d6, d7}
	add	r3, sp, #576
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #448
	vstmia	r3, {d6, d7}
	add	r3, sp, #560
	vldmia	r3, {d6, d7}
	add	r3, sp, #560
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vstmia	r3, {d6, d7}
	add	r3, sp, #656
	vldmia	r3, {d6, d7}
	add	r3, sp, #656
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vstmia	r3, {d6, d7}
	add	r3, sp, #640
	vldmia	r3, {d6, d7}
	add	r3, sp, #640
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #512
	vstmia	r3, {d6, d7}
	add	r3, sp, #624
	vldmia	r3, {d6, d7}
	add	r3, sp, #624
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vstmia	r3, {d6, d7}
	add	r3, sp, #720
	vldmia	r3, {d6, d7}
	add	r3, sp, #720
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vstmia	r3, {d6, d7}
	add	r3, sp, #704
	vldmia	r3, {d6, d7}
	add	r3, sp, #704
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #576
	vstmia	r3, {d6, d7}
	add	r3, sp, #688
	vldmia	r3, {d6, d7}
	add	r3, sp, #688
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vstmia	r3, {d6, d7}
	add	r3, sp, #784
	vldmia	r3, {d6, d7}
	add	r3, sp, #784
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vstmia	r3, {d6, d7}
	add	r3, sp, #768
	vldmia	r3, {d6, d7}
	add	r3, sp, #768
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #640
	vstmia	r3, {d6, d7}
	add	r3, sp, #752
	vldmia	r3, {d6, d7}
	add	r3, sp, #752
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vstmia	r3, {d6, d7}
	add	r3, sp, #848
	vldmia	r3, {d6, d7}
	add	r3, sp, #848
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vstmia	r3, {d6, d7}
	add	r3, sp, #832
	vldmia	r3, {d6, d7}
	add	r3, sp, #832
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #704
	vstmia	r3, {d6, d7}
	add	r3, sp, #816
	vldmia	r3, {d6, d7}
	add	r3, sp, #816
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vstmia	r3, {d6, d7}
	add	r3, sp, #912
	vldmia	r3, {d6, d7}
	add	r3, sp, #912
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vstmia	r3, {d6, d7}
	add	r3, sp, #896
	vldmia	r3, {d6, d7}
	add	r3, sp, #896
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #768
	vstmia	r3, {d6, d7}
	add	r3, sp, #880
	vldmia	r3, {d6, d7}
	add	r3, sp, #880
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vstmia	r3, {d6, d7}
	add	r3, sp, #944
	vldmia	r3, {d6, d7}
	add	r3, sp, #944
	vfma.f32	q6, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vfma.f32	q7, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #832
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vstmia	r3, {d6, d7}
	add	r3, sp, #176
	vldmia	r3, {d6, d7}
	add	r3, sp, #176
	vfma.f32	q1, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vfma.f32	q2, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #896
	vfma.f32	q4, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vfma.f32	q14, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vfma.f32	q15, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #960
	vfma.f32	q0, q8, q9
	vld1.32	{d18, d19}, [r4:128], r0
	vfma.f32	q11, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vfma.f32	q12, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #48
	vfma.f32	q13, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	add	r4, r5, #32
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	mov	r4, r5
	vstmia	r3, {d6, d7}
	add	r3, sp, #192
	vldmia	r3, {d6, d7}
	add	r3, sp, #192
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r4:128]!
	vstmia	r3, {d6, d7}
	add	r3, r5, #80
	vfma.f32	q10, q8, q9
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #160
	vldmia	r3, {d6, d7}
	add	r3, sp, #160
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r4:128]
	vstmia	r3, {d6, d7}
	add	r3, sp, #208
	vldmia	r3, {d6, d7}
	add	r3, sp, #208
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #144
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #224
	vldmia	r3, {d6, d7}
	add	r3, sp, #224
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #208
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #288
	vldmia	r3, {d6, d7}
	add	r3, sp, #288
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #272
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #352
	vldmia	r3, {d6, d7}
	add	r3, sp, #352
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #336
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #416
	vldmia	r3, {d6, d7}
	add	r3, sp, #416
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #400
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #480
	vldmia	r3, {d6, d7}
	add	r3, sp, #480
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #464
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #544
	vldmia	r3, {d6, d7}
	add	r3, sp, #544
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #528
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #608
	vldmia	r3, {d6, d7}
	add	r3, sp, #608
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #592
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #672
	vldmia	r3, {d6, d7}
	add	r3, sp, #672
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #656
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #736
	vldmia	r3, {d6, d7}
	add	r3, sp, #736
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #720
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #800
	vldmia	r3, {d6, d7}
	add	r3, sp, #800
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #784
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #864
	vldmia	r3, {d6, d7}
	add	r3, sp, #864
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #848
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #928
	vldmia	r3, {d6, d7}
	add	r3, sp, #928
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, r5, #912
	vld1.64	{d18, d19}, [r3:128]
	add	r3, r5, #976
	add	r5, r5, #1024
	vfma.f32	q5, q8, q9
	vld1.64	{d18, d19}, [r3:128]
	add	r3, sp, #960
	vldmia	r3, {d6, d7}
	add	r3, sp, #960
	vfma.f32	q3, q8, q9
	vstmia	r3, {d6, d7}
	add	r3, sp, #960
	vldmia	r3, {d16, d17}
	bne	.LBB3_3
	add	lr, sp, #208
	vst1.32	{d20, d21}, [r8:128]!
	ldr	r2, [sp, #20]
	ldr	r5, [sp, #124]
	ldr	r3, [sp, #120]
	vldmia	lr, {d18, d19}
	add	lr, sp, #192
	add	r5, r5, #1
	add	r3, r3, #4096
	vst1.64	{d18, d19}, [r8:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #176
	vst1.64	{d18, d19}, [r2:128]
	vldmia	lr, {d18, d19}
	ldr	r2, [sp, #28]
	add	lr, sp, #160
	vst1.64	{d18, d19}, [r2:128]
	vldmia	lr, {d18, d19}
	ldr	r2, [sp, #32]
	add	lr, sp, #272
	vst1.64	{d18, d19}, [r2:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #256
	ldr	r2, [sp, #40]
	vst1.32	{d18, d19}, [r7:128], r0
	vldmia	lr, {d18, d19}
	add	lr, sp, #240
	vst1.32	{d18, d19}, [r7:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #224
	vst1.64	{d18, d19}, [r7:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #336
	vst1.64	{d18, d19}, [r2:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #320
	ldr	r2, [sp, #44]
	vst1.32	{d18, d19}, [r10:128], r0
	vldmia	lr, {d18, d19}
	add	lr, sp, #304
	vst1.32	{d18, d19}, [r10:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #288
	vst1.64	{d18, d19}, [r10:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #400
	vst1.64	{d18, d19}, [r2:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #384
	ldr	r2, [sp, #52]
	vst1.32	{d18, d19}, [r9:128], r0
	vldmia	lr, {d18, d19}
	add	lr, sp, #368
	vst1.32	{d18, d19}, [r9:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #352
	vst1.64	{d18, d19}, [r9:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #464
	vst1.64	{d18, d19}, [r2:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #448
	ldr	r2, [sp, #60]
	vst1.32	{d18, d19}, [r6:128], r0
	vldmia	lr, {d18, d19}
	add	lr, sp, #432
	vst1.32	{d18, d19}, [r6:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #416
	vst1.64	{d18, d19}, [r6:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #528
	vst1.64	{d18, d19}, [r2:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #512
	vst1.32	{d18, d19}, [r1:128], r0
	vldmia	lr, {d18, d19}
	add	lr, sp, #496
	vst1.32	{d18, d19}, [r1:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #480
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	ldr	r1, [sp, #64]
	add	lr, sp, #592
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #576
	ldr	r1, [sp, #72]
	vst1.32	{d18, d19}, [r12:128], r0
	vldmia	lr, {d18, d19}
	add	lr, sp, #560
	vst1.32	{d18, d19}, [r12:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #544
	vst1.64	{d18, d19}, [r12:128]
	vldmia	lr, {d18, d19}
	add	lr, sp, #656
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	ldr	r1, [sp, #16]
	add	lr, sp, #640
	vst1.32	{d18, d19}, [r1:128], r0
	vldmia	lr, {d18, d19}
	add	lr, sp, #624
	vst1.32	{d18, d19}, [r1:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #608
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	ldr	r1, [sp, #80]
	add	lr, sp, #720
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	ldr	r1, [sp, #24]
	add	lr, sp, #704
	vst1.32	{d18, d19}, [r1:128], r0
	vldmia	lr, {d18, d19}
	add	lr, sp, #688
	vst1.32	{d18, d19}, [r1:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #672
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	ldr	r1, [sp, #84]
	add	lr, sp, #784
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	ldr	r1, [sp, #36]
	add	lr, sp, #768
	vst1.32	{d18, d19}, [r1:128], r0
	vldmia	lr, {d18, d19}
	add	lr, sp, #752
	vst1.32	{d18, d19}, [r1:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #736
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	ldr	r1, [sp, #92]
	add	lr, sp, #848
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	ldr	r1, [sp, #48]
	add	lr, sp, #832
	vst1.32	{d18, d19}, [r1:128], r0
	vldmia	lr, {d18, d19}
	add	lr, sp, #816
	vst1.32	{d18, d19}, [r1:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #800
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	ldr	r1, [sp, #100]
	add	lr, sp, #912
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	ldr	r1, [sp, #56]
	add	lr, sp, #896
	vst1.32	{d18, d19}, [r1:128], r0
	vldmia	lr, {d18, d19}
	add	lr, sp, #880
	vst1.32	{d18, d19}, [r1:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #864
	vst1.64	{d18, d19}, [r1:128]
	vldmia	lr, {d18, d19}
	ldr	r1, [sp, #104]
	add	lr, sp, #944
	vst1.64	{d18, d19}, [r1:128]
	ldr	r1, [sp, #68]
	vldmia	lr, {d18, d19}
	add	lr, sp, #928
	vst1.32	{d12, d13}, [r1:128], r0
	vst1.32	{d14, d15}, [r1:128]!
	vst1.64	{d18, d19}, [r1:128]
	ldr	r1, [sp, #108]
	vldmia	lr, {d18, d19}
	vst1.64	{d18, d19}, [r1:128]
	ldr	r1, [sp, #76]
	vst1.32	{d2, d3}, [r1:128], r0
	vst1.32	{d4, d5}, [r1:128]!
	vst1.64	{d8, d9}, [r1:128]
	ldr	r1, [sp, #112]
	vst1.64	{d10, d11}, [r1:128]
	ldr	r1, [sp, #88]
	vst1.32	{d28, d29}, [r1:128], r0
	vst1.32	{d30, d31}, [r1:128]!
	vst1.64	{d0, d1}, [r1:128]
	ldr	r1, [sp, #116]
	vst1.64	{d16, d17}, [r1:128]
	ldr	r1, [sp, #96]
	vst1.32	{d22, d23}, [r1:128], r0
	vst1.32	{d24, d25}, [r1:128]!
	vst1.64	{d26, d27}, [r1:128]
	ldr	r1, [sp, #12]
	cmp	r5, r1
	blt	.LBB3_2
.LBB3_5:
	mov	r0, #0
	add	sp, sp, #984
	vpop	{d8, d9, d10, d11, d12, d13, d14, d15}
	add	sp, sp, #4
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
.Lfunc_end3:
	.size	.L__tvm_parallel_lambda.34, .Lfunc_end3-.L__tvm_parallel_lambda.34
	.cantunwind
	.fnend

	.globl	sgemm_compute_6x8__neon
	.p2align	2
	.type	sgemm_compute_6x8__neon,%function
	.code	32
sgemm_compute_6x8__neon:
	.fnstart
	.save	{r4, r5, r6, r7, r11, lr}
	push	{r4, r5, r6, r7, r11, lr}
	.setfp	r11, sp, #16
	add	r11, sp, #16
	.pad	#824
	sub	sp, sp, #824
	.pad	#2048
	sub	sp, sp, #2048
	bfc	sp, #0, #4
	add	lr, sp, #2048
	add	r12, lr, #40
	add	lr, sp, #1024
	add	lr, lr, #24
	ldr	r4, [r11, #20]
	ldr	r4, [r11, #16]
	ldr	r4, [r11, #12]
	ldr	r4, [r11, #8]
	str	r0, [sp, #508]
	str	r1, [sp, #504]
	str	r2, [sp, #500]
	str	r3, [sp, #496]
	ldr	r0, [sp, #504]
	ldr	r1, [sp, #500]
	add	r0, r0, r1, lsl #2
	str	r0, [sp, #504]
	ldr	r0, [sp, #496]
	ldr	r1, [r11, #8]
	add	r0, r0, r1, lsl #2
	str	r0, [sp, #496]
	ldr	r0, [r11, #12]
	ldr	r1, [r11, #16]
	add	r0, r0, r1, lsl #2
	str	r0, [r11, #12]
	mov	r0, #0
	str	r0, [sp, #556]
	add	r1, sp, #556
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #512
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #528
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #480
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #604]
	add	r1, sp, #604
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #560
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #576
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #464
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #652]
	add	r1, sp, #652
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #608
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #624
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #448
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #692]
	add	r1, sp, #692
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #656
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #672
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #432
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #756]
	add	r1, sp, #756
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #720
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #736
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #416
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #820]
	add	r1, sp, #820
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #784
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #800
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #400
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #884]
	add	r1, sp, #884
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #848
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #864
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #384
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #956]
	add	r1, sp, #956
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #912
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #928
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #368
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1004]
	add	r1, sp, #1004
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #960
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #976
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #352
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1044]
	add	r1, sp, #1024
	add	r1, r1, #20
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #1008
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1024
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #336
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1108]
	add	r1, sp, #1024
	add	r1, r1, #84
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #1072
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1088
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #320
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1172]
	add	r0, sp, #1024
	add	r0, r0, #148
	vld1.32	{d16[], d17[]}, [r0:32]
	add	r0, sp, #1136
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #1152
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #304
	vst1.64	{d16, d17}, [r0]
.LBB4_1:
	ldr	r0, [sp, #504]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #272
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #256
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #288
	vst1.64	{d16, d17}, [r1]
	ldr	r0, [sp, #504]
	add	r0, r0, #16
	vld1.32	{d16}, [r0]
	vstr	d16, [sp, #240]
	vldr	d16, [sp, #240]
	vstr	d16, [sp, #232]
	vldr	d16, [sp, #232]
	vstr	d16, [sp, #248]
	ldr	r0, [sp, #504]
	add	r0, r0, #24
	str	r0, [sp, #504]
	ldr	r0, [sp, #496]
	str	r0, [sp, #1276]
	ldr	r0, [sp, #1276]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #1248
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #1232
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r2, sp, #208
	vst1.64	{d16, d17}, [r2]
	ldr	r0, [sp, #496]
	add	r0, r0, #16
	str	r0, [sp, #2860]
	ldr	r0, [sp, #2860]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #2832
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #2816
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #192
	vst1.64	{d16, d17}, [r0]
	ldr	r3, [sp, #496]
	add	r3, r3, #32
	str	r3, [sp, #496]
	add	r3, sp, #480
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vld1.64	{d20, d21}, [r1]
	add	r4, sp, #1216
	vst1.64	{d20, d21}, [r4]
	vld1.64	{d20, d21}, [r4]
	vstr	d20, [lr, #160]
	vldr	d20, [lr, #160]
	add	r4, sp, #2800
	vst1.64	{d16, d17}, [r4]
	add	r5, sp, #2784
	vst1.64	{d18, d19}, [r5]
	vstr	d20, [r12, #688]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #2752
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #2736
	vst1.64	{d16, d17}, [r5]
	vldr	d16, [r12, #688]
	vstr	d16, [r12, #640]
	vld1.64	{d16, d17}, [r4]
	vld1.64	{d18, d19}, [r5]
	vldr	d0, [r12, #640]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r4, sp, #2704
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #2688
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r4]
	vst1.64	{d16, d17}, [r3]
	add	r3, sp, #448
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vld1.64	{d20, d21}, [r1]
	add	r4, sp, #1184
	vst1.64	{d20, d21}, [r4]
	vld1.64	{d20, d21}, [r4]
	vstr	d20, [lr, #128]
	vldr	d20, [lr, #128]
	add	r4, sp, #2672
	vst1.64	{d16, d17}, [r4]
	add	r5, sp, #2656
	vst1.64	{d18, d19}, [r5]
	vstr	d20, [r12, #560]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #2624
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #2608
	vst1.64	{d16, d17}, [r5]
	vldr	d16, [r12, #560]
	vstr	d16, [r12, #512]
	vld1.64	{d16, d17}, [r4]
	vld1.64	{d18, d19}, [r5]
	vldr	d0, [r12, #512]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r4, sp, #2576
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #2560
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r4]
	vst1.64	{d16, d17}, [r3]
	add	r3, sp, #416
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vld1.64	{d20, d21}, [r1]
	add	r4, sp, #1120
	vst1.64	{d20, d21}, [r4]
	vld1.64	{d20, d21}, [r4]
	vstr	d21, [lr, #64]
	vldr	d20, [lr, #64]
	add	r4, sp, #2544
	vst1.64	{d16, d17}, [r4]
	add	r5, sp, #2528
	vst1.64	{d18, d19}, [r5]
	vstr	d20, [r12, #432]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #2496
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #2480
	vst1.64	{d16, d17}, [r5]
	vldr	d16, [r12, #432]
	vstr	d16, [r12, #384]
	vld1.64	{d16, d17}, [r4]
	vld1.64	{d18, d19}, [r5]
	vldr	d0, [r12, #384]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r4, sp, #2448
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #2432
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r4]
	vst1.64	{d16, d17}, [r3]
	add	r3, sp, #384
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vld1.64	{d20, d21}, [r1]
	add	r4, sp, #1056
	vst1.64	{d20, d21}, [r4]
	vld1.64	{d20, d21}, [r4]
	vstr	d21, [lr]
	vldr	d20, [lr]
	add	r4, sp, #2416
	vst1.64	{d16, d17}, [r4]
	add	r5, sp, #2400
	vst1.64	{d18, d19}, [r5]
	vstr	d20, [r12, #304]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #2368
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #2352
	vst1.64	{d16, d17}, [r5]
	vldr	d16, [r12, #304]
	vstr	d16, [r12, #256]
	vld1.64	{d16, d17}, [r4]
	vld1.64	{d18, d19}, [r5]
	vldr	d0, [r12, #256]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r4, sp, #2320
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #2304
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r4]
	vst1.64	{d16, d17}, [r3]
	add	r3, sp, #352
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vldr	d20, [sp, #248]
	add	r4, sp, #2288
	vst1.64	{d16, d17}, [r4]
	add	r5, sp, #2272
	vst1.64	{d18, d19}, [r5]
	vstr	d20, [r12, #176]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #2240
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #2224
	vst1.64	{d16, d17}, [r5]
	vldr	d16, [r12, #176]
	vstr	d16, [r12, #128]
	vld1.64	{d16, d17}, [r4]
	vld1.64	{d18, d19}, [r5]
	vldr	d0, [r12, #128]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r4, sp, #2192
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #2176
	vst1.64	{d16, d17}, [r4]
	vld1.64	{d16, d17}, [r4]
	vst1.64	{d16, d17}, [r3]
	add	r3, sp, #320
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vldr	d20, [sp, #248]
	add	r2, sp, #2160
	vst1.64	{d16, d17}, [r2]
	add	r4, sp, #2144
	vst1.64	{d18, d19}, [r4]
	vstr	d20, [r12, #48]
	vld1.64	{d16, d17}, [r2]
	add	r2, sp, #2112
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #2096
	vst1.64	{d16, d17}, [r4]
	vldr	d16, [r12, #48]
	vstr	d16, [r12]
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r4]
	vldr	d0, [r12]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r2, sp, #2064
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r2]
	add	r2, sp, #2048
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r2]
	vst1.64	{d16, d17}, [r3]
	add	r2, sp, #464
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r0]
	vld1.64	{d20, d21}, [r1]
	add	r3, sp, #896
	vst1.64	{d20, d21}, [r3]
	vld1.64	{d20, d21}, [r3]
	vstr	d20, [sp, #888]
	vldr	d20, [sp, #888]
	add	r3, sp, #2032
	vst1.64	{d16, d17}, [r3]
	add	r4, sp, #2016
	vst1.64	{d18, d19}, [r4]
	vstr	d20, [lr, #960]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #1984
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #1968
	vst1.64	{d16, d17}, [r4]
	vldr	d16, [lr, #960]
	vstr	d16, [lr, #912]
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r4]
	vldr	d0, [lr, #912]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r3, sp, #1936
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #1920
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	vst1.64	{d16, d17}, [r2]
	add	r2, sp, #432
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r0]
	vld1.64	{d20, d21}, [r1]
	add	r3, sp, #832
	vst1.64	{d20, d21}, [r3]
	vld1.64	{d20, d21}, [r3]
	vstr	d20, [sp, #824]
	vldr	d20, [sp, #824]
	add	r3, sp, #1904
	vst1.64	{d16, d17}, [r3]
	add	r4, sp, #1888
	vst1.64	{d18, d19}, [r4]
	vstr	d20, [lr, #832]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #1856
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #1840
	vst1.64	{d16, d17}, [r4]
	vldr	d16, [lr, #832]
	vstr	d16, [lr, #784]
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r4]
	vldr	d0, [lr, #784]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r3, sp, #1808
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #1792
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	vst1.64	{d16, d17}, [r2]
	add	r2, sp, #400
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r0]
	vld1.64	{d20, d21}, [r1]
	add	r3, sp, #768
	vst1.64	{d20, d21}, [r3]
	vld1.64	{d20, d21}, [r3]
	vstr	d21, [sp, #760]
	vldr	d20, [sp, #760]
	add	r3, sp, #1776
	vst1.64	{d16, d17}, [r3]
	add	r4, sp, #1760
	vst1.64	{d18, d19}, [r4]
	vstr	d20, [lr, #704]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #1728
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r4]
	add	r4, sp, #1712
	vst1.64	{d16, d17}, [r4]
	vldr	d16, [lr, #704]
	vstr	d16, [lr, #656]
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r4]
	vldr	d0, [lr, #656]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r3, sp, #1680
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #1664
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	vst1.64	{d16, d17}, [r2]
	add	r2, sp, #368
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r0]
	vld1.64	{d20, d21}, [r1]
	add	r1, sp, #704
	vst1.64	{d20, d21}, [r1]
	vld1.64	{d20, d21}, [r1]
	vstr	d21, [sp, #696]
	vldr	d20, [sp, #696]
	add	r1, sp, #1648
	vst1.64	{d16, d17}, [r1]
	add	r3, sp, #1632
	vst1.64	{d18, d19}, [r3]
	vstr	d20, [lr, #576]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1600
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #1584
	vst1.64	{d16, d17}, [r3]
	vldr	d16, [lr, #576]
	vstr	d16, [lr, #528]
	vld1.64	{d16, d17}, [r1]
	vld1.64	{d18, d19}, [r3]
	vldr	d0, [lr, #528]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r1, sp, #1552
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1536
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	vst1.64	{d16, d17}, [r2]
	add	r1, sp, #336
	vld1.64	{d16, d17}, [r1]
	vld1.64	{d18, d19}, [r0]
	vldr	d20, [sp, #248]
	add	r2, sp, #1520
	vst1.64	{d16, d17}, [r2]
	add	r3, sp, #1504
	vst1.64	{d18, d19}, [r3]
	vstr	d20, [lr, #448]
	vld1.64	{d16, d17}, [r2]
	add	r2, sp, #1472
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #1456
	vst1.64	{d16, d17}, [r3]
	vldr	d16, [lr, #448]
	vstr	d16, [lr, #400]
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r3]
	vldr	d0, [lr, #400]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r2, sp, #1424
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r2]
	add	r2, sp, #1408
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r2]
	vst1.64	{d16, d17}, [r1]
	add	r1, sp, #304
	vld1.64	{d16, d17}, [r1]
	vld1.64	{d18, d19}, [r0]
	vldr	d20, [sp, #248]
	add	r0, sp, #1392
	vst1.64	{d16, d17}, [r0]
	add	r2, sp, #1376
	vst1.64	{d18, d19}, [r2]
	vstr	d20, [lr, #320]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #1344
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r2]
	add	r2, sp, #1328
	vst1.64	{d16, d17}, [r2]
	vldr	d16, [lr, #320]
	vstr	d16, [lr, #272]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r2]
	vldr	d0, [lr, #272]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #1296
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #1280
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	vst1.64	{d16, d17}, [r1]
	mvn	r0, #0
	ldr	r1, [sp, #508]
	add	r0, r1, r0
	str	r0, [sp, #508]
	cmp	r0, #0
	bne	.LBB4_1
	add	r0, sp, #480
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #176
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	add	r0, sp, #464
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #160
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #20]
	ldr	r1, [r11, #12]
	add	r0, r1, r0, lsl #2
	str	r0, [r11, #12]
	add	r0, sp, #448
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #144
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	add	r0, sp, #432
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #128
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #20]
	ldr	r1, [r11, #12]
	add	r0, r1, r0, lsl #2
	str	r0, [r11, #12]
	add	r0, sp, #416
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #112
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	add	r0, sp, #400
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #96
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #20]
	ldr	r1, [r11, #12]
	add	r0, r1, r0, lsl #2
	str	r0, [r11, #12]
	add	r0, sp, #384
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #80
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	add	r0, sp, #368
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #64
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #20]
	ldr	r1, [r11, #12]
	add	r0, r1, r0, lsl #2
	str	r0, [r11, #12]
	add	r0, sp, #352
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #48
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	add	r0, sp, #336
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #32
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #20]
	ldr	r1, [r11, #12]
	add	r0, r1, r0, lsl #2
	str	r0, [r11, #12]
	add	r0, sp, #320
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #16
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	add	r0, sp, #304
	vld1.64	{d16, d17}, [r0]
	mov	r0, sp
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	sub	sp, r11, #16
	pop	{r4, r5, r6, r7, r11, pc}
.Lfunc_end4:
	.size	sgemm_compute_6x8__neon, .Lfunc_end4-sgemm_compute_6x8__neon
	.cantunwind
	.fnend

	.globl	sgemm_reset_6x8__neon
	.p2align	2
	.type	sgemm_reset_6x8__neon,%function
	.code	32
sgemm_reset_6x8__neon:
	.fnstart
	.save	{r4, r5, r11, lr}
	push	{r4, r5, r11, lr}
	.setfp	r11, sp, #8
	add	r11, sp, #8
	.pad	#272
	sub	sp, sp, #272
	bfc	sp, #0, #4
	str	r0, [sp, #220]
	str	r1, [sp, #216]
	str	r2, [sp, #212]
	ldr	r0, [sp, #220]
	ldr	r1, [sp, #216]
	add	r0, r0, r1, lsl #2
	str	r0, [sp, #220]
	mov	r0, #0
	str	r0, [sp, #268]
	add	r0, sp, #268
	vld1.32	{d16[], d17[]}, [r0:32]
	add	r0, sp, #224
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #240
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #192
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #176
	vst1.64	{d16, d17}, [r1]
	ldr	r2, [sp, #220]
	vld1.64	{d16, d17}, [r1]
	vst1.32	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #160
	vst1.64	{d16, d17}, [r1]
	ldr	r2, [sp, #220]
	add	r2, r2, #16
	vld1.64	{d16, d17}, [r1]
	vst1.32	{d16, d17}, [r2]
	ldr	r1, [sp, #212]
	ldr	r2, [sp, #220]
	add	r1, r2, r1, lsl #2
	str	r1, [sp, #220]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #144
	vst1.64	{d16, d17}, [r1]
	ldr	r2, [sp, #220]
	vld1.64	{d16, d17}, [r1]
	vst1.32	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #128
	vst1.64	{d16, d17}, [r1]
	ldr	r2, [sp, #220]
	add	r2, r2, #16
	vld1.64	{d16, d17}, [r1]
	vst1.32	{d16, d17}, [r2]
	ldr	r1, [sp, #212]
	ldr	r2, [sp, #220]
	add	r1, r2, r1, lsl #2
	str	r1, [sp, #220]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #112
	vst1.64	{d16, d17}, [r1]
	ldr	r2, [sp, #220]
	vld1.64	{d16, d17}, [r1]
	vst1.32	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #96
	vst1.64	{d16, d17}, [r1]
	ldr	r2, [sp, #220]
	add	r2, r2, #16
	vld1.64	{d16, d17}, [r1]
	vst1.32	{d16, d17}, [r2]
	ldr	r1, [sp, #212]
	ldr	r2, [sp, #220]
	add	r1, r2, r1, lsl #2
	str	r1, [sp, #220]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #80
	vst1.64	{d16, d17}, [r1]
	ldr	r2, [sp, #220]
	vld1.64	{d16, d17}, [r1]
	vst1.32	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #64
	vst1.64	{d16, d17}, [r1]
	ldr	r2, [sp, #220]
	add	r2, r2, #16
	vld1.64	{d16, d17}, [r1]
	vst1.32	{d16, d17}, [r2]
	ldr	r1, [sp, #212]
	ldr	r2, [sp, #220]
	add	r1, r2, r1, lsl #2
	str	r1, [sp, #220]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #48
	vst1.64	{d16, d17}, [r1]
	ldr	r2, [sp, #220]
	vld1.64	{d16, d17}, [r1]
	vst1.32	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #32
	vst1.64	{d16, d17}, [r1]
	ldr	r2, [sp, #220]
	add	r2, r2, #16
	vld1.64	{d16, d17}, [r1]
	vst1.32	{d16, d17}, [r2]
	ldr	r1, [sp, #212]
	ldr	r2, [sp, #220]
	add	r1, r2, r1, lsl #2
	str	r1, [sp, #220]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #16
	vst1.64	{d16, d17}, [r1]
	ldr	r2, [sp, #220]
	vld1.64	{d16, d17}, [r1]
	vst1.32	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r0]
	mov	r0, sp
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [sp, #220]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	sub	sp, r11, #8
	pop	{r4, r5, r11, pc}
.Lfunc_end5:
	.size	sgemm_reset_6x8__neon, .Lfunc_end5-sgemm_reset_6x8__neon
	.cantunwind
	.fnend

	.globl	sgemm_update_6x8__neon
	.p2align	2
	.type	sgemm_update_6x8__neon,%function
	.code	32
sgemm_update_6x8__neon:
	.fnstart
	.save	{r4, r5, r6, r10, r11, lr}
	push	{r4, r5, r6, r10, r11, lr}
	.setfp	r11, sp, #16
	add	r11, sp, #16
	.pad	#760
	sub	sp, sp, #760
	.pad	#3072
	sub	sp, sp, #3072
	bfc	sp, #0, #4
	add	lr, sp, #3072
	add	r12, lr, #664
	add	lr, sp, #2048
	add	lr, lr, #664
	add	r4, sp, #1024
	add	r4, r4, #632
	ldr	r5, [r11, #20]
	ldr	r5, [r11, #16]
	ldr	r5, [r11, #12]
	ldr	r5, [r11, #8]
	str	r0, [sp, #892]
	str	r1, [sp, #888]
	str	r2, [sp, #884]
	str	r3, [sp, #880]
	ldr	r0, [sp, #888]
	ldr	r1, [sp, #884]
	add	r0, r0, r1, lsl #2
	str	r0, [sp, #888]
	ldr	r0, [sp, #880]
	ldr	r1, [r11, #8]
	add	r0, r0, r1, lsl #2
	str	r0, [sp, #880]
	ldr	r0, [r11, #12]
	ldr	r1, [r11, #16]
	add	r0, r0, r1, lsl #2
	str	r0, [r11, #12]
	mov	r0, #0
	str	r0, [sp, #940]
	add	r1, sp, #940
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #896
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #912
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #864
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1564]
	add	r1, sp, #1024
	add	r1, r1, #540
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #1520
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1536
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #848
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1612]
	add	r1, sp, #1024
	add	r1, r1, #588
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #1568
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1584
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #832
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1652]
	add	r1, sp, #1024
	add	r1, r1, #628
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #1616
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1632
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #816
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1716]
	add	r1, sp, #1024
	add	r1, r1, #692
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #1680
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1696
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #800
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1780]
	add	r1, sp, #1024
	add	r1, r1, #756
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #1744
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1760
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #784
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1844]
	add	r1, sp, #1024
	add	r1, r1, #820
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #1808
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1824
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #768
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1916]
	add	r1, sp, #1024
	add	r1, r1, #892
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #1872
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1888
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #752
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #1964]
	add	r1, sp, #1024
	add	r1, r1, #940
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #1920
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1936
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #736
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #2004]
	add	r1, sp, #1024
	add	r1, r1, #980
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #1968
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #1984
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #720
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #2068]
	add	r1, sp, #2048
	add	r1, r1, #20
	vld1.32	{d16[], d17[]}, [r1:32]
	add	r1, sp, #2032
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #2048
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #704
	vst1.64	{d16, d17}, [r1]
	str	r0, [sp, #2132]
	add	r0, sp, #2048
	add	r0, r0, #84
	vld1.32	{d16[], d17[]}, [r0:32]
	add	r0, sp, #2096
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #2112
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #688
	vst1.64	{d16, d17}, [r0]
.LBB6_1:
	ldr	r0, [sp, #888]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #656
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #640
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r1, sp, #672
	vst1.64	{d16, d17}, [r1]
	ldr	r0, [sp, #888]
	add	r0, r0, #16
	vld1.32	{d16}, [r0]
	vstr	d16, [sp, #624]
	vldr	d16, [sp, #624]
	vstr	d16, [sp, #616]
	vldr	d16, [sp, #616]
	vstr	d16, [sp, #632]
	ldr	r0, [sp, #888]
	add	r0, r0, #24
	str	r0, [sp, #888]
	ldr	r0, [sp, #880]
	str	r0, [sp, #2236]
	ldr	r0, [sp, #2236]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #2208
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #2192
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r2, sp, #592
	vst1.64	{d16, d17}, [r2]
	ldr	r0, [sp, #880]
	add	r0, r0, #16
	str	r0, [sp, #3820]
	ldr	r0, [sp, #3820]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #3792
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #3776
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #576
	vst1.64	{d16, d17}, [r0]
	ldr	r3, [sp, #880]
	add	r3, r3, #32
	str	r3, [sp, #880]
	add	r3, sp, #864
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vld1.64	{d20, d21}, [r1]
	add	r5, sp, #2176
	vst1.64	{d20, d21}, [r5]
	vld1.64	{d20, d21}, [r5]
	vstr	d20, [r4, #512]
	vldr	d20, [r4, #512]
	add	r5, sp, #3760
	vst1.64	{d16, d17}, [r5]
	add	r6, sp, #3744
	vst1.64	{d18, d19}, [r6]
	vstr	d20, [r12]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #3712
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r6]
	add	r6, sp, #3696
	vst1.64	{d16, d17}, [r6]
	vldr	d16, [r12]
	vstr	d16, [lr, #976]
	vld1.64	{d16, d17}, [r5]
	vld1.64	{d18, d19}, [r6]
	vldr	d0, [lr, #976]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r5, sp, #3664
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #3648
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r5]
	vst1.64	{d16, d17}, [r3]
	add	r3, sp, #832
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vld1.64	{d20, d21}, [r1]
	add	r5, sp, #2144
	vst1.64	{d20, d21}, [r5]
	vld1.64	{d20, d21}, [r5]
	vstr	d20, [r4, #480]
	vldr	d20, [r4, #480]
	add	r5, sp, #3632
	vst1.64	{d16, d17}, [r5]
	add	r6, sp, #3616
	vst1.64	{d18, d19}, [r6]
	vstr	d20, [lr, #896]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #3584
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r6]
	add	r6, sp, #3568
	vst1.64	{d16, d17}, [r6]
	vldr	d16, [lr, #896]
	vstr	d16, [lr, #848]
	vld1.64	{d16, d17}, [r5]
	vld1.64	{d18, d19}, [r6]
	vldr	d0, [lr, #848]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r5, sp, #3536
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #3520
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r5]
	vst1.64	{d16, d17}, [r3]
	add	r3, sp, #800
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vld1.64	{d20, d21}, [r1]
	add	r5, sp, #2080
	vst1.64	{d20, d21}, [r5]
	vld1.64	{d20, d21}, [r5]
	vstr	d21, [r4, #416]
	vldr	d20, [r4, #416]
	add	r5, sp, #3504
	vst1.64	{d16, d17}, [r5]
	add	r6, sp, #3488
	vst1.64	{d18, d19}, [r6]
	vstr	d20, [lr, #768]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #3456
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r6]
	add	r6, sp, #3440
	vst1.64	{d16, d17}, [r6]
	vldr	d16, [lr, #768]
	vstr	d16, [lr, #720]
	vld1.64	{d16, d17}, [r5]
	vld1.64	{d18, d19}, [r6]
	vldr	d0, [lr, #720]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r5, sp, #3408
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #3392
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r5]
	vst1.64	{d16, d17}, [r3]
	add	r3, sp, #768
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vld1.64	{d20, d21}, [r1]
	add	r5, sp, #2016
	vst1.64	{d20, d21}, [r5]
	vld1.64	{d20, d21}, [r5]
	vstr	d21, [r4, #352]
	vldr	d20, [r4, #352]
	add	r5, sp, #3376
	vst1.64	{d16, d17}, [r5]
	add	r6, sp, #3360
	vst1.64	{d18, d19}, [r6]
	vstr	d20, [lr, #640]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #3328
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r6]
	add	r6, sp, #3312
	vst1.64	{d16, d17}, [r6]
	vldr	d16, [lr, #640]
	vstr	d16, [lr, #592]
	vld1.64	{d16, d17}, [r5]
	vld1.64	{d18, d19}, [r6]
	vldr	d0, [lr, #592]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r5, sp, #3280
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #3264
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r5]
	vst1.64	{d16, d17}, [r3]
	add	r3, sp, #736
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vldr	d20, [sp, #632]
	add	r5, sp, #3248
	vst1.64	{d16, d17}, [r5]
	add	r6, sp, #3232
	vst1.64	{d18, d19}, [r6]
	vstr	d20, [lr, #512]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #3200
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r6]
	add	r6, sp, #3184
	vst1.64	{d16, d17}, [r6]
	vldr	d16, [lr, #512]
	vstr	d16, [lr, #464]
	vld1.64	{d16, d17}, [r5]
	vld1.64	{d18, d19}, [r6]
	vldr	d0, [lr, #464]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r5, sp, #3152
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #3136
	vst1.64	{d16, d17}, [r5]
	vld1.64	{d16, d17}, [r5]
	vst1.64	{d16, d17}, [r3]
	add	r3, sp, #704
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r2]
	vldr	d20, [sp, #632]
	add	r2, sp, #3120
	vst1.64	{d16, d17}, [r2]
	add	r5, sp, #3104
	vst1.64	{d18, d19}, [r5]
	vstr	d20, [lr, #384]
	vld1.64	{d16, d17}, [r2]
	add	r2, sp, #3072
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #3056
	vst1.64	{d16, d17}, [r5]
	vldr	d16, [lr, #384]
	vstr	d16, [lr, #336]
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r5]
	vldr	d0, [lr, #336]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r2, sp, #3024
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r2]
	add	r2, sp, #3008
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r2]
	vst1.64	{d16, d17}, [r3]
	add	r2, sp, #848
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r0]
	vld1.64	{d20, d21}, [r1]
	add	r3, sp, #1856
	vst1.64	{d20, d21}, [r3]
	vld1.64	{d20, d21}, [r3]
	vstr	d20, [r4, #192]
	vldr	d20, [r4, #192]
	add	r3, sp, #2992
	vst1.64	{d16, d17}, [r3]
	add	r5, sp, #2976
	vst1.64	{d18, d19}, [r5]
	vstr	d20, [lr, #256]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #2944
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #2928
	vst1.64	{d16, d17}, [r5]
	vldr	d16, [lr, #256]
	vstr	d16, [lr, #208]
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r5]
	vldr	d0, [lr, #208]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r3, sp, #2896
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #2880
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	vst1.64	{d16, d17}, [r2]
	add	r2, sp, #816
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r0]
	vld1.64	{d20, d21}, [r1]
	add	r3, sp, #1792
	vst1.64	{d20, d21}, [r3]
	vld1.64	{d20, d21}, [r3]
	vstr	d20, [r4, #128]
	vldr	d20, [r4, #128]
	add	r3, sp, #2864
	vst1.64	{d16, d17}, [r3]
	add	r5, sp, #2848
	vst1.64	{d18, d19}, [r5]
	vstr	d20, [lr, #128]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #2816
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #2800
	vst1.64	{d16, d17}, [r5]
	vldr	d16, [lr, #128]
	vstr	d16, [lr, #80]
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r5]
	vldr	d0, [lr, #80]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r3, sp, #2768
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #2752
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	vst1.64	{d16, d17}, [r2]
	add	r2, sp, #784
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r0]
	vld1.64	{d20, d21}, [r1]
	add	r3, sp, #1728
	vst1.64	{d20, d21}, [r3]
	vld1.64	{d20, d21}, [r3]
	vstr	d21, [r4, #64]
	vldr	d20, [r4, #64]
	add	r3, sp, #2736
	vst1.64	{d16, d17}, [r3]
	add	r5, sp, #2720
	vst1.64	{d18, d19}, [r5]
	vstr	d20, [lr]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #2688
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r5]
	add	r5, sp, #2672
	vst1.64	{d16, d17}, [r5]
	vldr	d16, [lr]
	vstr	d16, [r4, #1008]
	vld1.64	{d16, d17}, [r3]
	vld1.64	{d18, d19}, [r5]
	vldr	d0, [r4, #1008]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r3, sp, #2640
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #2624
	vst1.64	{d16, d17}, [r3]
	vld1.64	{d16, d17}, [r3]
	vst1.64	{d16, d17}, [r2]
	add	r2, sp, #752
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r0]
	vld1.64	{d20, d21}, [r1]
	add	r1, sp, #1664
	vst1.64	{d20, d21}, [r1]
	vld1.64	{d20, d21}, [r1]
	vstr	d21, [r4]
	vldr	d20, [r4]
	add	r1, sp, #2608
	vst1.64	{d16, d17}, [r1]
	add	r3, sp, #2592
	vst1.64	{d18, d19}, [r3]
	vstr	d20, [r4, #928]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #2560
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #2544
	vst1.64	{d16, d17}, [r3]
	vldr	d16, [r4, #928]
	vstr	d16, [r4, #880]
	vld1.64	{d16, d17}, [r1]
	vld1.64	{d18, d19}, [r3]
	vldr	d0, [r4, #880]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r1, sp, #2512
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	add	r1, sp, #2496
	vst1.64	{d16, d17}, [r1]
	vld1.64	{d16, d17}, [r1]
	vst1.64	{d16, d17}, [r2]
	add	r1, sp, #720
	vld1.64	{d16, d17}, [r1]
	vld1.64	{d18, d19}, [r0]
	vldr	d20, [sp, #632]
	add	r2, sp, #2480
	vst1.64	{d16, d17}, [r2]
	add	r3, sp, #2464
	vst1.64	{d18, d19}, [r3]
	vstr	d20, [r4, #800]
	vld1.64	{d16, d17}, [r2]
	add	r2, sp, #2432
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r3]
	add	r3, sp, #2416
	vst1.64	{d16, d17}, [r3]
	vldr	d16, [r4, #800]
	vstr	d16, [r4, #752]
	vld1.64	{d16, d17}, [r2]
	vld1.64	{d18, d19}, [r3]
	vldr	d0, [r4, #752]
	vmul.f32	q9, q9, d0[0]
	vadd.f32	q8, q8, q9
	add	r2, sp, #2384
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r2]
	add	r2, sp, #2368
	vst1.64	{d16, d17}, [r2]
	vld1.64	{d16, d17}, [r2]
	vst1.64	{d16, d17}, [r1]
	add	r1, sp, #688
	vld1.64	{d16, d17}, [r1]
	vld1.64	{d18, d19}, [r0]
	vldr	d20, [sp, #632]
	add	r0, sp, #2352
	vst1.64	{d16, d17}, [r0]
	add	r2, sp, #2336
	vst1.64	{d18, d19}, [r2]
	vstr	d20, [r4, #672]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #2304
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r2]
	add	r2, sp, #2288
	vst1.64	{d16, d17}, [r2]
	vldr	d16, [r4, #672]
	vstr	d16, [r4, #624]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r2]
	vldr	d0, [r4, #624]
	vmul.f32	q9, q9, d0[1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #2256
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #2240
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	vst1.64	{d16, d17}, [r1]
	mvn	r0, #0
	ldr	r1, [sp, #892]
	add	r0, r1, r0
	str	r0, [sp, #892]
	cmp	r0, #0
	bne	.LBB6_1
	ldr	r0, [r11, #12]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #544
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #528
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #864
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #1504
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #1488
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #1472
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #560
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #12]
	add	r0, r0, #16
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #496
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #480
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #848
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #1456
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #1440
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #1424
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #512
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #20]
	ldr	r1, [r11, #12]
	add	r0, r1, r0, lsl #2
	str	r0, [r11, #12]
	ldr	r0, [r11, #12]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #448
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #432
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #832
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #1408
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #1392
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #1376
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #464
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #12]
	add	r0, r0, #16
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #400
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #384
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #816
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #1360
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #1344
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #1328
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #416
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #20]
	ldr	r1, [r11, #12]
	add	r0, r1, r0, lsl #2
	str	r0, [r11, #12]
	ldr	r0, [r11, #12]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #352
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #336
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #800
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #1312
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #1296
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #1280
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #368
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #12]
	add	r0, r0, #16
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #304
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #288
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #784
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #1264
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #1248
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #1232
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #320
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #20]
	ldr	r1, [r11, #12]
	add	r0, r1, r0, lsl #2
	str	r0, [r11, #12]
	ldr	r0, [r11, #12]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #256
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #240
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #768
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #1216
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #1200
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #1184
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #272
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #12]
	add	r0, r0, #16
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #208
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #192
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #752
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #1168
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #1152
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #1136
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #224
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #20]
	ldr	r1, [r11, #12]
	add	r0, r1, r0, lsl #2
	str	r0, [r11, #12]
	ldr	r0, [r11, #12]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #160
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #144
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #736
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #1120
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #1104
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #1088
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #176
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #12]
	add	r0, r0, #16
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #112
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #96
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #720
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #1072
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #1056
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #1040
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #128
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #20]
	ldr	r1, [r11, #12]
	add	r0, r1, r0, lsl #2
	str	r0, [r11, #12]
	ldr	r0, [r11, #12]
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #64
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #48
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #704
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #1024
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #1008
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #992
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #80
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	ldr	r0, [r11, #12]
	add	r0, r0, #16
	vld1.32	{d16, d17}, [r0]
	add	r0, sp, #16
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	mov	r0, sp
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #688
	vld1.64	{d18, d19}, [r0]
	add	r0, sp, #976
	vst1.64	{d16, d17}, [r0]
	add	r1, sp, #960
	vst1.64	{d18, d19}, [r1]
	vld1.64	{d16, d17}, [r0]
	vld1.64	{d18, d19}, [r1]
	vadd.f32	q8, q8, q9
	add	r0, sp, #944
	vst1.64	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	add	r0, sp, #32
	vst1.64	{d16, d17}, [r0]
	ldr	r1, [r11, #12]
	add	r1, r1, #16
	vld1.64	{d16, d17}, [r0]
	vst1.32	{d16, d17}, [r1]
	sub	sp, r11, #16
	pop	{r4, r5, r6, r10, r11, pc}
.Lfunc_end6:
	.size	sgemm_update_6x8__neon, .Lfunc_end6-sgemm_update_6x8__neon
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

	.type	.L.str,%object
	.section	.rodata,"a",%progbits
.L.str:
	.asciz	"Assert fail: (num_args == 3), default_function: num_args should be 3"
	.size	.L.str, 69

	.type	.L.str.1,%object
.L.str.1:
	.asciz	"Assert fail: ((((1 == int32(arg0.strides[3])) && ((1*512) == int32(arg0.strides[2]))) && (((1*512)*28) == int32(arg0.strides[1]))) && ((((1*512)*28)*28) == int32(arg0.strides[0]))), arg0.strides: expected to be compact array"
	.size	.L.str.1, 225

	.type	.L.str.2,%object
.L.str.2:
	.asciz	"Assert fail: ((((1 == int32(arg1.strides[3])) && ((1*256) == int32(arg1.strides[2]))) && (((1*256)*512) == int32(arg1.strides[1]))) && ((((1*256)*512)*1) == int32(arg1.strides[0]))), arg1.strides: expected to be compact array"
	.size	.L.str.2, 226

	.type	.L.str.3,%object
.L.str.3:
	.asciz	"Assert fail: ((((1 == int32(arg2.strides[3])) && ((1*256) == int32(arg2.strides[2]))) && (((1*256)*14) == int32(arg2.strides[1]))) && ((((1*256)*14)*14) == int32(arg2.strides[0]))), arg2.strides: expected to be compact array"
	.size	.L.str.3, 225

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
	.asciz	"Assert fail: (int32(arg0.shape[1]) == 28), Argument arg0.shape[1] has an unsatisfied constraint"
	.size	.L.str.11, 96

	.type	.L.str.12,%object
.L.str.12:
	.asciz	"Assert fail: (int32(arg0.shape[2]) == 28), Argument arg0.shape[2] has an unsatisfied constraint"
	.size	.L.str.12, 96

	.type	.L.str.13,%object
.L.str.13:
	.asciz	"Assert fail: (int32(arg0.shape[3]) == 512), Argument arg0.shape[3] has an unsatisfied constraint"
	.size	.L.str.13, 97

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
	.asciz	"Assert fail: (int32(arg1.shape[0]) == 1), Argument arg1.shape[0] has an unsatisfied constraint"
	.size	.L.str.17, 95

	.type	.L.str.18,%object
.L.str.18:
	.asciz	"Assert fail: (int32(arg1.shape[1]) == 1), Argument arg1.shape[1] has an unsatisfied constraint"
	.size	.L.str.18, 95

	.type	.L.str.19,%object
.L.str.19:
	.asciz	"Assert fail: (int32(arg1.shape[2]) == 512), Argument arg1.shape[2] has an unsatisfied constraint"
	.size	.L.str.19, 97

	.type	.L.str.20,%object
.L.str.20:
	.asciz	"Assert fail: (int32(arg1.shape[3]) == 256), Argument arg1.shape[3] has an unsatisfied constraint"
	.size	.L.str.20, 97

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
	.asciz	"Assert fail: (int32(arg2.shape[1]) == 14), Argument arg2.shape[1] has an unsatisfied constraint"
	.size	.L.str.27, 96

	.type	.L.str.28,%object
.L.str.28:
	.asciz	"Assert fail: (int32(arg2.shape[2]) == 14), Argument arg2.shape[2] has an unsatisfied constraint"
	.size	.L.str.28, 96

	.type	.L.str.29,%object
.L.str.29:
	.asciz	"Assert fail: (int32(arg2.shape[3]) == 256), Argument arg2.shape[3] has an unsatisfied constraint"
	.size	.L.str.29, 97

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
