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
	cmp	r0, #7168
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
	vldr	d18, .LCPI0_67
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
	cmp	r0, #262144
	ldreq	r0, [r1]
	cmpeq	r0, #262144
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
	ldr	r1, [r12]
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
	cmp	r0, #7168
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
	cmp	r0, #14
	bne	.LBB0_64
	ldr	r0, [r7, #16]
	cmp	r0, #14
	bne	.LBB0_65
	ldr	r0, [r7, #24]
	cmp	r0, #512
	bne	.LBB0_66
	ldr	r0, [r6, #32]
	ldr	r3, [r6, #36]
	orrs	r0, r0, r3
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
	cmp	r0, #1
	bne	.LBB0_69
	ldr	r0, [r10, #8]
	cmp	r0, #1
	bne	.LBB0_70
	ldr	r0, [r10, #16]
	cmp	r0, #512
	bne	.LBB0_71
	ldr	r0, [r10, #24]
	cmp	r0, #512
	bne	.LBB0_72
	ldr	r0, [lr, #32]
	ldr	r3, [lr, #36]
	orrs	r0, r0, r3
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
	cmp	r0, #14
	bne	.LBB0_78
	ldr	r0, [r8, #16]
	cmp	r0, #14
	bne	.LBB0_79
	ldr	r0, [r8, #24]
	cmp	r0, #512
	bne	.LBB0_80
	ldr	r0, [r12, #32]
	ldr	r5, [r12, #36]
	orrs	r0, r0, r5
	bne	.LBB0_81
	ldr	r0, [r12, #4]
	cmp	r0, #1
	bne	.LBB0_82
	ldr	r0, [r12, #8]
	cmp	r3, r0
	bne	.LBB0_83
	ldr	r0, [sp, #4]
	ldr	r2, [sp]
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
	.long	512
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
	mov	r0, #32
	mov	r1, #2
	mov	r2, #401408
	mov	r3, #0
	str	r1, [sp]
	str	r0, [sp, #4]
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
	.save	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	.pad	#96
	sub	sp, sp, #96
	ldr	r1, [r1, #4]
	mov	r5, #14
	add	r3, r1, #13
	sdiv	r1, r3, r1
	add	r3, r0, #1
	mul	r3, r1, r3
	mul	r0, r1, r0
	cmp	r3, #14
	movge	r3, r5
	cmp	r0, #14
	movlt	r5, r0
	str	r3, [sp, #4]
	cmp	r5, r3
	bge	.LBB2_31
	mvn	r0, r0
	movw	r10, #58368
	ldr	lr, [r2]
	ldr	r8, [r2, #4]
	cmn	r0, #15
	movt	r10, #65535
	mvnle	r0, #14
	rsb	r3, r0, r0, lsl #3
	add	r0, r10, #6144
	sub	r0, r0, r3, lsl #10
	str	lr, [sp, #16]
	add	r6, r8, r0, lsl #2
	add	r7, lr, r0, lsl #2
	add	r0, r10, #5632
	sub	r0, r0, r3, lsl #10
	add	r4, r8, r0, lsl #2
	add	r2, lr, r0, lsl #2
	add	r0, r10, #5120
	str	r0, [sp]
	sub	r0, r0, r3, lsl #10
	add	r1, r8, r0, lsl #2
	add	r9, lr, r0, lsl #2
	add	r0, r10, #4608
	sub	r12, r0, r3, lsl #10
	add	r0, r8, r12, lsl #2
	str	r0, [sp, #92]
	add	r0, lr, r12, lsl #2
	str	r0, [sp, #88]
	add	r0, r10, #4096
	sub	r12, r0, r3, lsl #10
	add	r0, r8, r12, lsl #2
	str	r0, [sp, #84]
	add	r0, lr, r12, lsl #2
	str	r0, [sp, #80]
	add	r0, r10, #3584
	sub	r12, r0, r3, lsl #10
	add	r0, r8, r12, lsl #2
	str	r0, [sp, #76]
	add	r0, lr, r12, lsl #2
	str	r0, [sp, #72]
	add	r0, r10, #3072
	sub	r12, r0, r3, lsl #10
	add	r0, r8, r12, lsl #2
	str	r0, [sp, #28]
	add	r0, lr, r12, lsl #2
	str	r0, [sp, #68]
	add	r0, r10, #2560
	sub	r12, r0, r3, lsl #10
	add	r0, r8, r12, lsl #2
	str	r0, [sp, #64]
	add	r0, lr, r12, lsl #2
	str	r0, [sp, #60]
	add	r0, r10, #2048
	sub	r0, r0, r3, lsl #10
	add	r11, r8, r0, lsl #2
	add	r0, lr, r0, lsl #2
	str	r0, [sp, #56]
	add	r0, r10, #1536
	sub	r12, r0, r3, lsl #10
	add	r0, r8, r12, lsl #2
	str	r0, [sp, #52]
	add	r0, lr, r12, lsl #2
	str	r0, [sp, #48]
	add	r0, r10, #1024
	sub	r12, r0, r3, lsl #10
	add	r0, r8, r12, lsl #2
	str	r0, [sp, #24]
	add	r0, lr, r12, lsl #2
	str	r0, [sp, #44]
	add	r0, r10, #512
	mov	r10, r11
	ldr	r11, [sp, #28]
	sub	r12, r0, r3, lsl #10
	add	r0, r8, r12, lsl #2
	str	r0, [sp, #40]
	add	r0, lr, r12, lsl #2
	ldr	r12, [sp, #24]
	str	r0, [sp, #36]
	movw	r0, #58368
	movt	r0, #65535
	sub	r0, r0, r3, lsl #10
	str	r0, [sp, #20]
	add	r0, r8, r0, lsl #2
	ldr	r3, [sp, #20]
	add	r3, lr, r3, lsl #2
	ldr	lr, [sp, #20]
	str	r3, [sp, #32]
	movw	r3, #58368
	movt	r3, #65535
	sub	lr, lr, r3
	ldr	r3, [sp, #16]
	add	r8, r8, lr, lsl #2
	add	lr, r3, lr, lsl #2
.LBB2_2:
	str	r5, [sp, #20]
	ldr	r5, [sp, #32]
	str	r9, [sp, #16]
	mov	r9, #0
.LBB2_3:
	ldr	r3, [r0, r9, lsl #2]
	str	r3, [r5, r9, lsl #2]
	add	r9, r9, #1
	cmp	r9, #512
	bne	.LBB2_3
	str	r0, [sp, #8]
	ldr	r9, [sp, #40]
	ldr	r0, [sp, #36]
	mov	r3, #0
.LBB2_5:
	ldr	r5, [r9, r3, lsl #2]
	str	r5, [r0, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_5
	ldr	r9, [sp, #16]
	ldr	r0, [sp, #44]
	mov	r3, #0
.LBB2_7:
	ldr	r5, [r12, r3, lsl #2]
	str	r5, [r0, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_7
	str	r12, [sp, #24]
	ldr	r0, [sp, #52]
	ldr	r12, [sp, #48]
	mov	r3, #0
.LBB2_9:
	ldr	r5, [r0, r3, lsl #2]
	str	r5, [r12, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_9
	ldr	r0, [sp, #56]
	mov	r3, #0
.LBB2_11:
	ldr	r5, [r10, r3, lsl #2]
	str	r5, [r0, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_11
	str	r10, [sp, #12]
	ldr	r0, [sp, #64]
	ldr	r10, [sp, #60]
	ldr	r12, [sp, #24]
	mov	r3, #0
.LBB2_13:
	ldr	r5, [r0, r3, lsl #2]
	str	r5, [r10, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_13
	ldr	r0, [sp, #68]
	mov	r3, #0
.LBB2_15:
	ldr	r5, [r11, r3, lsl #2]
	str	r5, [r0, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_15
	str	r11, [sp, #28]
	ldr	r0, [sp, #76]
	ldr	r11, [sp, #72]
	ldr	r10, [sp, #12]
	mov	r3, #0
.LBB2_17:
	ldr	r5, [r0, r3, lsl #2]
	str	r5, [r11, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_17
	ldr	r11, [sp, #80]
	ldr	r0, [sp, #84]
	mov	r3, #0
.LBB2_19:
	ldr	r5, [r0, r3, lsl #2]
	str	r5, [r11, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_19
	ldr	r0, [sp, #92]
	ldr	r11, [sp, #88]
	mov	r3, #0
.LBB2_21:
	ldr	r5, [r0, r3, lsl #2]
	str	r5, [r11, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_21
	mov	r3, #0
.LBB2_23:
	ldr	r5, [r1, r3, lsl #2]
	str	r5, [r9, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_23
	ldr	r11, [sp, #28]
	mov	r3, #0
.LBB2_25:
	ldr	r5, [r4, r3, lsl #2]
	str	r5, [r2, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_25
	mov	r3, #0
.LBB2_27:
	ldr	r5, [r6, r3, lsl #2]
	str	r5, [r7, r3, lsl #2]
	add	r3, r3, #1
	cmp	r3, #512
	bne	.LBB2_27
	ldr	r3, [sp]
.LBB2_29:
	ldr	r5, [r8, r3]
	str	r5, [lr, r3]
	adds	r3, r3, #4
	bne	.LBB2_29
	ldr	r0, [sp, #92]
	ldr	r3, [sp, #80]
	ldr	r5, [sp, #20]
	add	r8, r8, #28672
	add	lr, lr, #28672
	add	r6, r6, #28672
	add	r7, r7, #28672
	add	r4, r4, #28672
	add	r2, r2, #28672
	add	r1, r1, #28672
	add	r9, r9, #28672
	add	r11, r11, #28672
	add	r10, r10, #28672
	add	r12, r12, #28672
	add	r0, r0, #28672
	add	r3, r3, #28672
	add	r5, r5, #1
	str	r0, [sp, #92]
	ldr	r0, [sp, #88]
	str	r3, [sp, #80]
	ldr	r3, [sp, #68]
	add	r0, r0, #28672
	add	r3, r3, #28672
	str	r0, [sp, #88]
	ldr	r0, [sp, #84]
	str	r3, [sp, #68]
	ldr	r3, [sp, #56]
	add	r0, r0, #28672
	add	r3, r3, #28672
	str	r0, [sp, #84]
	ldr	r0, [sp, #76]
	str	r3, [sp, #56]
	ldr	r3, [sp, #44]
	add	r0, r0, #28672
	add	r3, r3, #28672
	str	r0, [sp, #76]
	ldr	r0, [sp, #72]
	str	r3, [sp, #44]
	ldr	r3, [sp, #32]
	add	r0, r0, #28672
	add	r3, r3, #28672
	str	r0, [sp, #72]
	ldr	r0, [sp, #64]
	str	r3, [sp, #32]
	ldr	r3, [sp, #4]
	add	r0, r0, #28672
	cmp	r5, r3
	str	r0, [sp, #64]
	ldr	r0, [sp, #60]
	add	r0, r0, #28672
	str	r0, [sp, #60]
	ldr	r0, [sp, #52]
	add	r0, r0, #28672
	str	r0, [sp, #52]
	ldr	r0, [sp, #48]
	add	r0, r0, #28672
	str	r0, [sp, #48]
	ldr	r0, [sp, #40]
	add	r0, r0, #28672
	str	r0, [sp, #40]
	ldr	r0, [sp, #36]
	add	r0, r0, #28672
	str	r0, [sp, #36]
	ldr	r0, [sp, #8]
	add	r0, r0, #28672
	blt	.LBB2_2
.LBB2_31:
	mov	r0, #0
	add	sp, sp, #96
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
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
	.pad	#88
	sub	sp, sp, #88
	.pad	#2048
	sub	sp, sp, #2048
	ldr	r1, [r1, #4]
	mov	r7, #196
	add	r3, r1, #195
	sdiv	r1, r3, r1
	add	r3, r0, #1
	mul	r3, r1, r3
	mul	r0, r1, r0
	cmp	r3, #196
	movge	r3, r7
	cmp	r0, #196
	movlt	r7, r0
	str	r3, [sp, #16]
	cmp	r7, r3
	bge	.LBB3_5
	ldr	r1, [r2]
	mvn	r0, r0
	mov	r4, #32
	cmn	r0, #197
	mvnle	r0, #196
	str	r1, [sp, #12]
	ldmib	r2, {r1, r2}
	str	r2, [sp, #8]
	movw	r2, #63488
	movt	r2, #65535
	add	r2, r2, #1536
	mul	r0, r0, r2
	add	r1, r1, r0, lsl #2
.LBB3_2:
	ldr	r0, [sp, #12]
	str	r7, [sp, #252]
	vmov.i32	q10, #0x0
	vmov.i32	q8, #0x0
	movw	r8, #63488
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
	movt	r8, #65535
	add	r7, r0, r7, lsl #11
	add	r0, r7, #2000
	add	r2, r7, #512
	add	lr, r7, #384
	add	r11, r7, #320
	add	r6, r7, #256
	add	r3, r7, #192
	add	r9, r7, #128
	add	r12, r7, #64
	str	r0, [sp, #248]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1936
	str	r2, [sp, #24]
	vst1.32	{d20, d21}, [r2:128], r4
	str	r0, [sp, #244]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1872
	vst1.32	{d20, d21}, [r2:128]!
	str	r0, [sp, #240]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1808
	vst1.64	{d20, d21}, [r2:128]
	add	r2, r7, #448
	str	r0, [sp, #236]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1744
	str	r2, [sp, #20]
	vst1.32	{d20, d21}, [r2:128], r4
	str	r0, [sp, #232]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1680
	vst1.32	{d20, d21}, [r2:128]!
	str	r0, [sp, #228]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1616
	vst1.64	{d20, d21}, [r2:128]
	mov	r2, lr
	str	r0, [sp, #224]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1552
	vst1.32	{d20, d21}, [r2:128], r4
	str	r0, [sp, #220]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1488
	vst1.32	{d20, d21}, [r2:128]!
	str	r0, [sp, #216]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1424
	vst1.64	{d20, d21}, [r2:128]
	mov	r2, r11
	str	r0, [sp, #208]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1360
	vst1.32	{d20, d21}, [r2:128], r4
	str	r0, [sp, #200]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1296
	vst1.32	{d20, d21}, [r2:128]!
	str	r0, [sp, #196]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1232
	vst1.64	{d20, d21}, [r2:128]
	mov	r2, r6
	str	r0, [sp, #188]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1168
	vst1.32	{d20, d21}, [r2:128], r4
	str	r0, [sp, #180]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1104
	vst1.32	{d20, d21}, [r2:128]!
	str	r0, [sp, #176]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1040
	vst1.64	{d20, d21}, [r2:128]
	mov	r2, r3
	str	r0, [sp, #168]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #976
	vst1.32	{d20, d21}, [r2:128], r4
	str	r0, [sp, #160]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #912
	vst1.32	{d20, d21}, [r2:128]!
	str	r0, [sp, #156]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #848
	vst1.64	{d20, d21}, [r2:128]
	mov	r2, r9
	str	r0, [sp, #148]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #784
	vst1.32	{d20, d21}, [r2:128], r4
	str	r0, [sp, #140]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #720
	vst1.32	{d20, d21}, [r2:128]!
	str	r0, [sp, #136]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #656
	vst1.64	{d20, d21}, [r2:128]
	mov	r2, r12
	str	r0, [sp, #128]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #592
	vst1.32	{d20, d21}, [r2:128], r4
	str	r0, [sp, #120]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #528
	vst1.32	{d20, d21}, [r2:128]!
	str	r0, [sp, #116]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #464
	vst1.64	{d20, d21}, [r2:128]
	ldr	r2, [sp, #8]
	str	r0, [sp, #108]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #400
	str	r0, [sp, #100]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #336
	str	r0, [sp, #96]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #272
	str	r0, [sp, #88]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #208
	str	r0, [sp, #80]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #144
	str	r0, [sp, #76]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #80
	str	r0, [sp, #68]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #48
	str	r0, [sp, #64]
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #32
	str	r0, [sp, #56]
	vst1.64	{d20, d21}, [r0:128]
	mov	r0, r7
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1984
	str	r0, [sp, #212]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1920
	str	r0, [sp, #204]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1856
	str	r0, [sp, #192]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1792
	str	r0, [sp, #184]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1728
	str	r0, [sp, #172]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1664
	str	r0, [sp, #164]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1600
	str	r0, [sp, #152]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1536
	str	r0, [sp, #144]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1472
	str	r0, [sp, #132]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1408
	str	r0, [sp, #124]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1344
	str	r0, [sp, #112]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1280
	str	r0, [sp, #104]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1216
	str	r0, [sp, #92]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1152
	str	r0, [sp, #84]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1088
	str	r0, [sp, #72]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #1024
	str	r0, [sp, #60]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #960
	str	r0, [sp, #52]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #896
	str	r0, [sp, #48]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #832
	str	r0, [sp, #44]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #768
	str	r0, [sp, #40]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #704
	str	r0, [sp, #36]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #640
	str	r0, [sp, #32]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, r7, #576
	str	r0, [sp, #28]
	vst1.32	{d20, d21}, [r0:128], r4
	vst1.32	{d20, d21}, [r0:128]!
	vst1.64	{d20, d21}, [r0:128]
	add	r0, sp, #336
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #320
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #304
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #400
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #288
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #384
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #368
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #464
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #352
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #448
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #432
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #528
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #416
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #512
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #496
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #592
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #480
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #576
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #560
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #656
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #544
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #640
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #624
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #720
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #608
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #704
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #688
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #784
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #672
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #768
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #752
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #848
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #736
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #832
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #816
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #912
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #800
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #896
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #880
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #976
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #864
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #960
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #944
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1040
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #928
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1024
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1008
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1104
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #992
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1088
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1072
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1168
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1056
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1152
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1136
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1232
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1120
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1216
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1200
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1296
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1184
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1280
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1264
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1360
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1248
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1344
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1328
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1424
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1312
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1408
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1392
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1488
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1376
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1472
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1456
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1552
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1440
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1536
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1520
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1616
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1504
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1600
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1584
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1680
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1568
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1664
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1648
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1744
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1632
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1728
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1712
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1808
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1696
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1792
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1776
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1872
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1760
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1856
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1840
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1936
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1824
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1920
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1904
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #2000
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1888
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1984
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1968
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #2064
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #1952
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #2048
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #2032
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #2016
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #2096
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
	add	r0, sp, #2080
	vstmia	r0, {d16, d17}
	vmov.i32	q8, #0x0
.LBB3_3:
	add	r0, sp, #256
	add	r10, r1, r8
	add	r5, r2, #128
	adds	r8, r8, #4
	vstmia	r0, {d12, d13}
	add	r0, sp, #272
	vorr	q6, q4, q4
	vorr	q4, q1, q1
	vorr	q1, q15, q15
	vorr	q15, q13, q13
	vorr	q13, q11, q11
	vld1.32	{d22, d23}, [r5:128], r4
	vstmia	r0, {d14, d15}
	add	r0, sp, #2112
	vorr	q7, q5, q5
	vorr	q5, q2, q2
	vorr	q2, q0, q0
	vorr	q0, q14, q14
	vorr	q14, q12, q12
	vorr	q12, q10, q10
	vstmia	r0, {d16, d17}
	add	r0, r2, #64
	vld1.32	{d16[], d17[]}, [r10:32]
	add	r10, sp, #400
	vld1.32	{d18, d19}, [r0:128], r4
	vldmia	r10, {d6, d7}
	add	r10, sp, #400
	vfma.f32	q3, q8, q9
	vld1.32	{d20, d21}, [r0:128]!
	vstmia	r10, {d6, d7}
	add	r10, sp, #384
	vldmia	r10, {d18, d19}
	add	r10, sp, #384
	vfma.f32	q9, q8, q10
	vorr	q10, q12, q12
	vorr	q12, q14, q14
	vorr	q14, q0, q0
	vorr	q0, q2, q2
	vorr	q2, q5, q5
	vorr	q5, q7, q7
	vstmia	r10, {d18, d19}
	add	r10, sp, #272
	vldmia	r10, {d14, d15}
	add	r10, sp, #464
	vldmia	r10, {d18, d19}
	add	r10, sp, #464
	vfma.f32	q9, q8, q11
	vorr	q11, q13, q13
	vorr	q13, q15, q15
	vorr	q15, q1, q1
	vorr	q1, q4, q4
	vorr	q4, q6, q6
	vstmia	r10, {d18, d19}
	add	r10, sp, #256
	vld1.32	{d18, d19}, [r5:128]!
	vldmia	r10, {d12, d13}
	add	r10, sp, #448
	vldmia	r10, {d6, d7}
	add	r10, sp, #448
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #368
	vstmia	r10, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #368
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r5:128]
	add	r5, sp, #528
	vstmia	r0, {d6, d7}
	add	r0, sp, #432
	vldmia	r0, {d6, d7}
	add	r0, sp, #432
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #192
	vldmia	r5, {d6, d7}
	add	r5, sp, #528
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #512
	vldmia	r5, {d6, d7}
	add	r5, sp, #512
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #496
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #496
	add	r5, sp, #592
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #256
	vldmia	r5, {d6, d7}
	add	r5, sp, #592
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #576
	vldmia	r5, {d6, d7}
	add	r5, sp, #576
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #560
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #560
	add	r5, sp, #656
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #320
	vldmia	r5, {d6, d7}
	add	r5, sp, #656
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #640
	vldmia	r5, {d6, d7}
	add	r5, sp, #640
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #624
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #624
	add	r5, sp, #720
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #384
	vldmia	r5, {d6, d7}
	add	r5, sp, #720
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #704
	vldmia	r5, {d6, d7}
	add	r5, sp, #704
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #688
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #688
	add	r5, sp, #784
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #448
	vldmia	r5, {d6, d7}
	add	r5, sp, #784
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #768
	vldmia	r5, {d6, d7}
	add	r5, sp, #768
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #752
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #752
	add	r5, sp, #848
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #512
	vldmia	r5, {d6, d7}
	add	r5, sp, #848
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #832
	vldmia	r5, {d6, d7}
	add	r5, sp, #832
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #816
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #816
	add	r5, sp, #912
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #576
	vldmia	r5, {d6, d7}
	add	r5, sp, #912
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #896
	vldmia	r5, {d6, d7}
	add	r5, sp, #896
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #880
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #880
	add	r5, sp, #976
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #640
	vldmia	r5, {d6, d7}
	add	r5, sp, #976
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #960
	vldmia	r5, {d6, d7}
	add	r5, sp, #960
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #944
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #944
	add	r5, sp, #1040
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #704
	vldmia	r5, {d6, d7}
	add	r5, sp, #1040
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1024
	vldmia	r5, {d6, d7}
	add	r5, sp, #1024
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1008
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1008
	add	r5, sp, #1104
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #768
	vldmia	r5, {d6, d7}
	add	r5, sp, #1104
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1088
	vldmia	r5, {d6, d7}
	add	r5, sp, #1088
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1072
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1072
	add	r5, sp, #1168
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #832
	vldmia	r5, {d6, d7}
	add	r5, sp, #1168
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1152
	vldmia	r5, {d6, d7}
	add	r5, sp, #1152
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1136
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1136
	add	r5, sp, #1232
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #896
	vldmia	r5, {d6, d7}
	add	r5, sp, #1232
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1216
	vldmia	r5, {d6, d7}
	add	r5, sp, #1216
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1200
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1200
	add	r5, sp, #1296
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #960
	vldmia	r5, {d6, d7}
	add	r5, sp, #1296
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1280
	vldmia	r5, {d6, d7}
	add	r5, sp, #1280
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1264
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1264
	add	r5, sp, #1360
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1024
	vldmia	r5, {d6, d7}
	add	r5, sp, #1360
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1344
	vldmia	r5, {d6, d7}
	add	r5, sp, #1344
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1328
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1328
	add	r5, sp, #1424
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1088
	vldmia	r5, {d6, d7}
	add	r5, sp, #1424
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1408
	vldmia	r5, {d6, d7}
	add	r5, sp, #1408
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1392
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1392
	add	r5, sp, #1488
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1152
	vldmia	r5, {d6, d7}
	add	r5, sp, #1488
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1472
	vldmia	r5, {d6, d7}
	add	r5, sp, #1472
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1456
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1456
	add	r5, sp, #1552
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1216
	vldmia	r5, {d6, d7}
	add	r5, sp, #1552
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1536
	vldmia	r5, {d6, d7}
	add	r5, sp, #1536
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1520
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1520
	add	r5, sp, #1616
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1280
	vldmia	r5, {d6, d7}
	add	r5, sp, #1616
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1600
	vldmia	r5, {d6, d7}
	add	r5, sp, #1600
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1584
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1584
	add	r5, sp, #1680
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1344
	vldmia	r5, {d6, d7}
	add	r5, sp, #1680
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1664
	vldmia	r5, {d6, d7}
	add	r5, sp, #1664
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1648
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1648
	add	r5, sp, #1744
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1408
	vldmia	r5, {d6, d7}
	add	r5, sp, #1744
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1728
	vldmia	r5, {d6, d7}
	add	r5, sp, #1728
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1712
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1712
	add	r5, sp, #1808
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1472
	vldmia	r5, {d6, d7}
	add	r5, sp, #1808
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1792
	vldmia	r5, {d6, d7}
	add	r5, sp, #1792
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1776
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1776
	add	r5, sp, #1872
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1536
	vldmia	r5, {d6, d7}
	add	r5, sp, #1872
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1856
	vldmia	r5, {d6, d7}
	add	r5, sp, #1856
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1840
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1840
	add	r5, sp, #1936
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1600
	vldmia	r5, {d6, d7}
	add	r5, sp, #1936
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1920
	vldmia	r5, {d6, d7}
	add	r5, sp, #1920
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1904
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1904
	add	r5, sp, #2000
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1664
	vldmia	r5, {d6, d7}
	add	r5, sp, #2000
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #1984
	vldmia	r5, {d6, d7}
	add	r5, sp, #1984
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1968
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #1968
	add	r5, sp, #2064
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1728
	vldmia	r5, {d6, d7}
	add	r5, sp, #2064
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q3, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vstmia	r5, {d6, d7}
	add	r5, sp, #2048
	vldmia	r5, {d6, d7}
	add	r5, sp, #2048
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #2032
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #2032
	add	r5, r2, #80
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1792
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q6, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vfma.f32	q7, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #2096
	vldmia	r0, {d6, d7}
	add	r0, sp, #2096
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1856
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q1, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vfma.f32	q2, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, r2, #1920
	vfma.f32	q4, q8, q9
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q14, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vfma.f32	q15, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, r2, #1984
	vfma.f32	q0, q8, q9
	vld1.32	{d18, d19}, [r0:128], r4
	vfma.f32	q11, q8, q9
	vld1.32	{d18, d19}, [r0:128]!
	vfma.f32	q12, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, r2, #48
	vfma.f32	q13, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #304
	vldmia	r0, {d6, d7}
	add	r0, sp, #304
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #32
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #320
	vldmia	r0, {d6, d7}
	add	r0, sp, #320
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	mov	r0, r2
	vld1.32	{d18, d19}, [r0:128]!
	vfma.f32	q10, q8, q9
	vld1.64	{d18, d19}, [r5:128]
	add	r5, sp, #288
	vldmia	r5, {d6, d7}
	add	r5, sp, #288
	vfma.f32	q3, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #336
	vstmia	r5, {d6, d7}
	vldmia	r0, {d6, d7}
	add	r0, sp, #336
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #144
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #352
	vldmia	r0, {d6, d7}
	add	r0, sp, #352
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #208
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #416
	vldmia	r0, {d6, d7}
	add	r0, sp, #416
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #272
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #480
	vldmia	r0, {d6, d7}
	add	r0, sp, #480
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #336
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #544
	vldmia	r0, {d6, d7}
	add	r0, sp, #544
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #400
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #608
	vldmia	r0, {d6, d7}
	add	r0, sp, #608
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #464
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #672
	vldmia	r0, {d6, d7}
	add	r0, sp, #672
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #528
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #736
	vldmia	r0, {d6, d7}
	add	r0, sp, #736
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #592
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #800
	vldmia	r0, {d6, d7}
	add	r0, sp, #800
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #656
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #864
	vldmia	r0, {d6, d7}
	add	r0, sp, #864
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #720
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #928
	vldmia	r0, {d6, d7}
	add	r0, sp, #928
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #784
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #992
	vldmia	r0, {d6, d7}
	add	r0, sp, #992
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #848
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1056
	vldmia	r0, {d6, d7}
	add	r0, sp, #1056
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #912
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1120
	vldmia	r0, {d6, d7}
	add	r0, sp, #1120
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #976
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1184
	vldmia	r0, {d6, d7}
	add	r0, sp, #1184
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1040
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1248
	vldmia	r0, {d6, d7}
	add	r0, sp, #1248
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1104
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1312
	vldmia	r0, {d6, d7}
	add	r0, sp, #1312
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1168
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1376
	vldmia	r0, {d6, d7}
	add	r0, sp, #1376
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1232
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1440
	vldmia	r0, {d6, d7}
	add	r0, sp, #1440
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1296
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1504
	vldmia	r0, {d6, d7}
	add	r0, sp, #1504
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1360
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1568
	vldmia	r0, {d6, d7}
	add	r0, sp, #1568
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1424
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1632
	vldmia	r0, {d6, d7}
	add	r0, sp, #1632
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1488
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1696
	vldmia	r0, {d6, d7}
	add	r0, sp, #1696
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1552
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1760
	vldmia	r0, {d6, d7}
	add	r0, sp, #1760
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1616
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1824
	vldmia	r0, {d6, d7}
	add	r0, sp, #1824
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1680
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1888
	vldmia	r0, {d6, d7}
	add	r0, sp, #1888
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1744
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #1952
	vldmia	r0, {d6, d7}
	add	r0, sp, #1952
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1808
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #2016
	vldmia	r0, {d6, d7}
	add	r0, sp, #2016
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1872
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #2080
	vldmia	r0, {d6, d7}
	add	r0, sp, #2080
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, r2, #1936
	vld1.64	{d18, d19}, [r0:128]
	add	r0, r2, #2000
	add	r2, r2, #2048
	vfma.f32	q5, q8, q9
	vld1.64	{d18, d19}, [r0:128]
	add	r0, sp, #2112
	vldmia	r0, {d6, d7}
	add	r0, sp, #2112
	vfma.f32	q3, q8, q9
	vstmia	r0, {d6, d7}
	add	r0, sp, #2112
	vldmia	r0, {d16, d17}
	bne	.LBB3_3
	add	r0, sp, #336
	vst1.32	{d20, d21}, [r7:128]!
	add	r2, sp, #320
	add	r1, r1, #2048
	vldmia	r0, {d18, d19}
	ldr	r0, [sp, #56]
	vst1.64	{d18, d19}, [r7:128]
	vldmia	r2, {d18, d19}
	add	r2, sp, #304
	ldr	r7, [sp, #252]
	vst1.64	{d18, d19}, [r0:128]
	vldmia	r2, {d18, d19}
	ldr	r0, [sp, #64]
	add	r2, sp, #288
	add	r7, r7, #1
	vst1.64	{d18, d19}, [r0:128]
	vldmia	r2, {d18, d19}
	ldr	r0, [sp, #68]
	add	r2, sp, #352
	vst1.64	{d18, d19}, [r0:128]
	add	r0, sp, #400
	vldmia	r0, {d18, d19}
	add	r0, sp, #384
	vst1.32	{d18, d19}, [r12:128], r4
	vldmia	r0, {d18, d19}
	add	r0, sp, #368
	vst1.32	{d18, d19}, [r12:128]!
	vldmia	r0, {d18, d19}
	ldr	r0, [sp, #76]
	vst1.64	{d18, d19}, [r12:128]
	vldmia	r2, {d18, d19}
	add	r2, sp, #416
	vst1.64	{d18, d19}, [r0:128]
	add	r0, sp, #464
	vldmia	r0, {d18, d19}
	add	r0, sp, #448
	vst1.32	{d18, d19}, [r9:128], r4
	vldmia	r0, {d18, d19}
	add	r0, sp, #432
	vst1.32	{d18, d19}, [r9:128]!
	vldmia	r0, {d18, d19}
	ldr	r0, [sp, #80]
	vst1.64	{d18, d19}, [r9:128]
	vldmia	r2, {d18, d19}
	add	r2, sp, #480
	vst1.64	{d18, d19}, [r0:128]
	add	r0, sp, #528
	vldmia	r0, {d18, d19}
	add	r0, sp, #512
	vst1.32	{d18, d19}, [r3:128], r4
	vldmia	r0, {d18, d19}
	add	r0, sp, #496
	vst1.32	{d18, d19}, [r3:128]!
	vldmia	r0, {d18, d19}
	ldr	r0, [sp, #88]
	vst1.64	{d18, d19}, [r3:128]
	vldmia	r2, {d18, d19}
	add	r2, sp, #544
	vst1.64	{d18, d19}, [r0:128]
	add	r0, sp, #592
	vldmia	r0, {d18, d19}
	add	r0, sp, #576
	vst1.32	{d18, d19}, [r6:128], r4
	vldmia	r0, {d18, d19}
	add	r0, sp, #560
	vst1.32	{d18, d19}, [r6:128]!
	vldmia	r0, {d18, d19}
	ldr	r0, [sp, #96]
	vst1.64	{d18, d19}, [r6:128]
	vldmia	r2, {d18, d19}
	add	r2, sp, #608
	vst1.64	{d18, d19}, [r0:128]
	add	r0, sp, #656
	vldmia	r0, {d18, d19}
	add	r0, sp, #640
	vst1.32	{d18, d19}, [r11:128], r4
	vldmia	r0, {d18, d19}
	add	r0, sp, #624
	vst1.32	{d18, d19}, [r11:128]!
	vldmia	r0, {d18, d19}
	ldr	r0, [sp, #100]
	vst1.64	{d18, d19}, [r11:128]
	vldmia	r2, {d18, d19}
	vst1.64	{d18, d19}, [r0:128]
	add	r0, sp, #720
	vldmia	r0, {d18, d19}
	add	r0, sp, #704
	vst1.32	{d18, d19}, [lr:128], r4
	vldmia	r0, {d18, d19}
	add	r0, sp, #688
	vst1.32	{d18, d19}, [lr:128]!
	vldmia	r0, {d18, d19}
	ldr	r0, [sp, #108]
	vst1.64	{d18, d19}, [lr:128]
	add	lr, sp, #672
	vldmia	lr, {d18, d19}
	add	lr, sp, #784
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #20]
	add	lr, sp, #768
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #752
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #736
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #116]
	add	lr, sp, #848
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #24]
	add	lr, sp, #832
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #816
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #800
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #120]
	add	lr, sp, #912
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #28]
	add	lr, sp, #896
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #880
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #864
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #128]
	add	lr, sp, #976
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #32]
	add	lr, sp, #960
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #944
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #928
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #136]
	add	lr, sp, #1040
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #36]
	add	lr, sp, #1024
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1008
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #992
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #140]
	add	lr, sp, #1104
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #40]
	add	lr, sp, #1088
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1072
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1056
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #148]
	add	lr, sp, #1168
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #44]
	add	lr, sp, #1152
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1136
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1120
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #156]
	add	lr, sp, #1232
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #48]
	add	lr, sp, #1216
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1200
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1184
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #160]
	add	lr, sp, #1296
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #52]
	add	lr, sp, #1280
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1264
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1248
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #168]
	add	lr, sp, #1360
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #60]
	add	lr, sp, #1344
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1328
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1312
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #176]
	add	lr, sp, #1424
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #72]
	add	lr, sp, #1408
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1392
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1376
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #180]
	add	lr, sp, #1488
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #84]
	add	lr, sp, #1472
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1456
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1440
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #188]
	add	lr, sp, #1552
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #92]
	add	lr, sp, #1536
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1520
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1504
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #196]
	add	lr, sp, #1616
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #104]
	add	lr, sp, #1600
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1584
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1568
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #200]
	add	lr, sp, #1680
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #112]
	add	lr, sp, #1664
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1648
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1632
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #208]
	add	lr, sp, #1744
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #124]
	add	lr, sp, #1728
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1712
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1696
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #216]
	add	lr, sp, #1808
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #132]
	add	lr, sp, #1792
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1776
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1760
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #220]
	add	lr, sp, #1872
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #144]
	add	lr, sp, #1856
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1840
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1824
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #224]
	add	lr, sp, #1936
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #152]
	add	lr, sp, #1920
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1904
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1888
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #228]
	add	lr, sp, #2000
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #164]
	add	lr, sp, #1984
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #1968
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #1952
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #232]
	add	lr, sp, #2064
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #172]
	add	lr, sp, #2048
	vst1.32	{d18, d19}, [r0:128], r4
	vldmia	lr, {d18, d19}
	add	lr, sp, #2032
	vst1.32	{d18, d19}, [r0:128]!
	vldmia	lr, {d18, d19}
	add	lr, sp, #2016
	vst1.64	{d18, d19}, [r0:128]
	vldmia	lr, {d18, d19}
	ldr	r0, [sp, #236]
	add	lr, sp, #2096
	vst1.64	{d18, d19}, [r0:128]
	ldr	r0, [sp, #184]
	vldmia	lr, {d18, d19}
	add	lr, sp, #2080
	vst1.32	{d12, d13}, [r0:128], r4
	vst1.32	{d14, d15}, [r0:128]!
	vst1.64	{d18, d19}, [r0:128]
	ldr	r0, [sp, #240]
	vldmia	lr, {d18, d19}
	vst1.64	{d18, d19}, [r0:128]
	ldr	r0, [sp, #192]
	vst1.32	{d2, d3}, [r0:128], r4
	vst1.32	{d4, d5}, [r0:128]!
	vst1.64	{d8, d9}, [r0:128]
	ldr	r0, [sp, #244]
	vst1.64	{d10, d11}, [r0:128]
	ldr	r0, [sp, #204]
	vst1.32	{d28, d29}, [r0:128], r4
	vst1.32	{d30, d31}, [r0:128]!
	vst1.64	{d0, d1}, [r0:128]
	ldr	r0, [sp, #248]
	vst1.64	{d16, d17}, [r0:128]
	ldr	r0, [sp, #212]
	vst1.32	{d22, d23}, [r0:128], r4
	vst1.32	{d24, d25}, [r0:128]!
	vst1.64	{d26, d27}, [r0:128]
	ldr	r0, [sp, #16]
	cmp	r7, r0
	blt	.LBB3_2
.LBB3_5:
	mov	r0, #0
	add	sp, sp, #88
	add	sp, sp, #2048
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
	subs	r0, r0, #1
	bne	.Ltmp36
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
.Ltmp37:
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
	bne	.Ltmp37
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
	.asciz	"Assert fail: ((((1 == int32(arg0.strides[3])) && ((1*512) == int32(arg0.strides[2]))) && (((1*512)*14) == int32(arg0.strides[1]))) && ((((1*512)*14)*14) == int32(arg0.strides[0]))), arg0.strides: expected to be compact array"
	.size	.L.str.1, 225

	.type	.L.str.2,%object
.L.str.2:
	.asciz	"Assert fail: ((((1 == int32(arg1.strides[3])) && ((1*512) == int32(arg1.strides[2]))) && (((1*512)*512) == int32(arg1.strides[1]))) && ((((1*512)*512)*1) == int32(arg1.strides[0]))), arg1.strides: expected to be compact array"
	.size	.L.str.2, 226

	.type	.L.str.3,%object
.L.str.3:
	.asciz	"Assert fail: ((((1 == int32(arg2.strides[3])) && ((1*512) == int32(arg2.strides[2]))) && (((1*512)*14) == int32(arg2.strides[1]))) && ((((1*512)*14)*14) == int32(arg2.strides[0]))), arg2.strides: expected to be compact array"
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
	.asciz	"Assert fail: (int32(arg0.shape[1]) == 14), Argument arg0.shape[1] has an unsatisfied constraint"
	.size	.L.str.11, 96

	.type	.L.str.12,%object
.L.str.12:
	.asciz	"Assert fail: (int32(arg0.shape[2]) == 14), Argument arg0.shape[2] has an unsatisfied constraint"
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
	.asciz	"Assert fail: (int32(arg1.shape[3]) == 512), Argument arg1.shape[3] has an unsatisfied constraint"
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
	.asciz	"Assert fail: (int32(arg2.shape[3]) == 512), Argument arg2.shape[3] has an unsatisfied constraint"
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
	.eabi_attribute	30, 1
