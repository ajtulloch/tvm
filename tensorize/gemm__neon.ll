; ModuleID = 'tensorize/gemm__neon.c'
source_filename = "tensorize/gemm__neon.c"
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabihf"

; Function Attrs: noinline nounwind optnone
define void @sgemm_compute_6x8__neon(i32, float*, i32, float*, i32, float*, i32, i32) #0 {
  %9 = alloca i32, align 4
  %10 = alloca float*, align 4
  %11 = alloca i32, align 4
  %12 = alloca float*, align 4
  %13 = alloca i32, align 4
  %14 = alloca float*, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  store i32 %0, i32* %9, align 4
  store float* %1, float** %10, align 4
  store i32 %2, i32* %11, align 4
  store float* %3, float** %12, align 4
  store i32 %4, i32* %13, align 4
  store float* %5, float** %14, align 4
  store i32 %6, i32* %15, align 4
  store i32 %7, i32* %16, align 4
  %19 = load float*, float** %10, align 4
  %20 = load i32, i32* %11, align 4
  %21 = getelementptr inbounds float, float* %19, i32 %20
  store float* %21, float** %10, align 4
  %22 = load float*, float** %12, align 4
  %23 = load i32, i32* %13, align 4
  %24 = getelementptr inbounds float, float* %22, i32 %23
  store float* %24, float** %12, align 4
  %25 = load float*, float** %14, align 4
  %26 = load i32, i32* %15, align 4
  %27 = getelementptr inbounds float, float* %25, i32 %26
  store float* %27, float** %14, align 4
  %28 = load i32, i32* %9, align 4
  store i32 %28, i32* %17, align 4
  %29 = load i32, i32* %16, align 4
  store i32 %29, i32* %18, align 4
  %30 = load float*, float** %14, align 4
  %31 = load float*, float** %12, align 4
  %32 = load float*, float** %10, align 4
  %33 = load i32, i32* %17, align 4
  %34 = load i32, i32* %18, align 4
  %35 = call { float*, float*, float*, i32, i32 } asm sideeffect "\09VMOV.I32  q4, #0\0A\09\09VMOV.I32  q5, #0\0A\09\09VMOV.I32  q6, #0\0A\09\09VMOV.I32  q7, #0\0A\09\09VMOV.I32  q8, #0\0A\09\09VMOV.I32  q9, #0\0A\09\09VMOV.I32 q10, #0\0A\09\09VMOV.I32 q11, #0\0A\09\09VMOV.I32 q12, #0\0A\09\09VMOV.I32 q13, #0\0A\09\09VMOV.I32 q14, #0\0A\09\09VMOV.I32 q15, #0\0A\090:\0A\09\09VLD1.32 {d4-d7}, [$1]!\0A\09\09VLD1.32 {d0-d2}, [$2]!\0A\09\09VMLA.F32 q4, q2, d0[0]\0A\09\09VMLA.F32 q5, q3, d0[0]\0A\09\09VMLA.F32 q6, q2, d0[1]\0A\09\09VMLA.F32 q7, q3, d0[1]\0A\09\09VMLA.F32  q8, q2, d1[0]\0A\09\09VMLA.F32  q9, q3, d1[0]\0A\09\09VMLA.F32 q10, q2, d1[1]\0A\09\09VMLA.F32 q11, q3, d1[1]\0A\09\09VMLA.F32 q12, q2, d2[0]\0A\09\09VMLA.F32 q13, q3, d2[0]\0A\09\09VMLA.F32 q14, q2, d2[1]\0A\09\09VMLA.F32 q15, q3, d2[1]\0A\09\09SUBS $3, $3, #1\0A\09\09BNE 0b\0A\09\09LSL $4, $4, #2\0A\09\09VST1.32 {d8-d11}, [$0], $4\0A\09\09VST1.32 {d12-d15}, [$0], $4\0A\09\09VST1.32 {d16-d19}, [$0], $4\0A\09\09VST1.32 {d20-d23}, [$0], $4\0A\09\09VST1.32 {d24-d27}, [$0], $4\0A\09\09VST1.32 {d28-d31}, [$0]\0A\09", "=r,=r,=r,=r,=r,0,1,2,3,4,~{cc},~{memory},~{d0},~{d1},~{d2},~{d3},~{d4},~{d5},~{d6},~{d7},~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15},~{d16},~{d17},~{d18},~{d19},~{d20},~{d21},~{d22},~{d23},~{d24},~{d25},~{d26},~{d27},~{d28},~{d29},~{d30},~{d31}"(float* %30, float* %31, float* %32, i32 %33, i32 %34) #2, !srcloc !3
  %36 = extractvalue { float*, float*, float*, i32, i32 } %35, 0
  %37 = extractvalue { float*, float*, float*, i32, i32 } %35, 1
  %38 = extractvalue { float*, float*, float*, i32, i32 } %35, 2
  %39 = extractvalue { float*, float*, float*, i32, i32 } %35, 3
  %40 = extractvalue { float*, float*, float*, i32, i32 } %35, 4
  store float* %36, float** %14, align 4
  store float* %37, float** %12, align 4
  store float* %38, float** %10, align 4
  store i32 %39, i32* %17, align 4
  store i32 %40, i32* %18, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone
define void @sgemm_reset_6x8__neon(float*, i32, i32) #0 {
  %4 = alloca float, align 4
  %5 = alloca <4 x float>, align 8
  %6 = alloca <4 x float>, align 8
  %7 = alloca float*, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca <4 x float>, align 8
  %11 = alloca <4 x float>, align 8
  %12 = alloca <4 x float>, align 8
  %13 = alloca <4 x float>, align 8
  %14 = alloca <4 x float>, align 8
  %15 = alloca <4 x float>, align 8
  %16 = alloca <4 x float>, align 8
  %17 = alloca <4 x float>, align 8
  %18 = alloca <4 x float>, align 8
  %19 = alloca <4 x float>, align 8
  %20 = alloca <4 x float>, align 8
  %21 = alloca <4 x float>, align 8
  %22 = alloca <4 x float>, align 8
  store float* %0, float** %7, align 4
  store i32 %1, i32* %8, align 4
  store i32 %2, i32* %9, align 4
  %23 = load float*, float** %7, align 4
  %24 = load i32, i32* %8, align 4
  %25 = getelementptr inbounds float, float* %23, i32 %24
  store float* %25, float** %7, align 4
  store float 0.000000e+00, float* %4, align 4
  %26 = load float, float* %4, align 4
  %27 = insertelement <4 x float> undef, float %26, i32 0
  %28 = load float, float* %4, align 4
  %29 = insertelement <4 x float> %27, float %28, i32 1
  %30 = load float, float* %4, align 4
  %31 = insertelement <4 x float> %29, float %30, i32 2
  %32 = load float, float* %4, align 4
  %33 = insertelement <4 x float> %31, float %32, i32 3
  store <4 x float> %33, <4 x float>* %6, align 8
  %34 = load <4 x float>, <4 x float>* %6, align 8
  store <4 x float> %34, <4 x float>* %5, align 8
  %35 = load <4 x float>, <4 x float>* %5, align 8
  store <4 x float> %35, <4 x float>* %10, align 8
  %36 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %36, <4 x float>* %11, align 8
  %37 = load float*, float** %7, align 4
  %38 = getelementptr inbounds float, float* %37, i32 0
  %39 = bitcast float* %38 to i8*
  %40 = load <4 x float>, <4 x float>* %11, align 8
  %41 = bitcast <4 x float> %40 to <16 x i8>
  %42 = bitcast <16 x i8> %41 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %39, <4 x float> %42, i32 4)
  %43 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %43, <4 x float>* %12, align 8
  %44 = load float*, float** %7, align 4
  %45 = getelementptr inbounds float, float* %44, i32 4
  %46 = bitcast float* %45 to i8*
  %47 = load <4 x float>, <4 x float>* %12, align 8
  %48 = bitcast <4 x float> %47 to <16 x i8>
  %49 = bitcast <16 x i8> %48 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %46, <4 x float> %49, i32 4)
  %50 = load i32, i32* %9, align 4
  %51 = load float*, float** %7, align 4
  %52 = getelementptr inbounds float, float* %51, i32 %50
  store float* %52, float** %7, align 4
  %53 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %53, <4 x float>* %13, align 8
  %54 = load float*, float** %7, align 4
  %55 = getelementptr inbounds float, float* %54, i32 0
  %56 = bitcast float* %55 to i8*
  %57 = load <4 x float>, <4 x float>* %13, align 8
  %58 = bitcast <4 x float> %57 to <16 x i8>
  %59 = bitcast <16 x i8> %58 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %56, <4 x float> %59, i32 4)
  %60 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %60, <4 x float>* %14, align 8
  %61 = load float*, float** %7, align 4
  %62 = getelementptr inbounds float, float* %61, i32 4
  %63 = bitcast float* %62 to i8*
  %64 = load <4 x float>, <4 x float>* %14, align 8
  %65 = bitcast <4 x float> %64 to <16 x i8>
  %66 = bitcast <16 x i8> %65 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %63, <4 x float> %66, i32 4)
  %67 = load i32, i32* %9, align 4
  %68 = load float*, float** %7, align 4
  %69 = getelementptr inbounds float, float* %68, i32 %67
  store float* %69, float** %7, align 4
  %70 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %70, <4 x float>* %15, align 8
  %71 = load float*, float** %7, align 4
  %72 = getelementptr inbounds float, float* %71, i32 0
  %73 = bitcast float* %72 to i8*
  %74 = load <4 x float>, <4 x float>* %15, align 8
  %75 = bitcast <4 x float> %74 to <16 x i8>
  %76 = bitcast <16 x i8> %75 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %73, <4 x float> %76, i32 4)
  %77 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %77, <4 x float>* %16, align 8
  %78 = load float*, float** %7, align 4
  %79 = getelementptr inbounds float, float* %78, i32 4
  %80 = bitcast float* %79 to i8*
  %81 = load <4 x float>, <4 x float>* %16, align 8
  %82 = bitcast <4 x float> %81 to <16 x i8>
  %83 = bitcast <16 x i8> %82 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %80, <4 x float> %83, i32 4)
  %84 = load i32, i32* %9, align 4
  %85 = load float*, float** %7, align 4
  %86 = getelementptr inbounds float, float* %85, i32 %84
  store float* %86, float** %7, align 4
  %87 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %87, <4 x float>* %17, align 8
  %88 = load float*, float** %7, align 4
  %89 = getelementptr inbounds float, float* %88, i32 0
  %90 = bitcast float* %89 to i8*
  %91 = load <4 x float>, <4 x float>* %17, align 8
  %92 = bitcast <4 x float> %91 to <16 x i8>
  %93 = bitcast <16 x i8> %92 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %90, <4 x float> %93, i32 4)
  %94 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %94, <4 x float>* %18, align 8
  %95 = load float*, float** %7, align 4
  %96 = getelementptr inbounds float, float* %95, i32 4
  %97 = bitcast float* %96 to i8*
  %98 = load <4 x float>, <4 x float>* %18, align 8
  %99 = bitcast <4 x float> %98 to <16 x i8>
  %100 = bitcast <16 x i8> %99 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %97, <4 x float> %100, i32 4)
  %101 = load i32, i32* %9, align 4
  %102 = load float*, float** %7, align 4
  %103 = getelementptr inbounds float, float* %102, i32 %101
  store float* %103, float** %7, align 4
  %104 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %104, <4 x float>* %19, align 8
  %105 = load float*, float** %7, align 4
  %106 = getelementptr inbounds float, float* %105, i32 0
  %107 = bitcast float* %106 to i8*
  %108 = load <4 x float>, <4 x float>* %19, align 8
  %109 = bitcast <4 x float> %108 to <16 x i8>
  %110 = bitcast <16 x i8> %109 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %107, <4 x float> %110, i32 4)
  %111 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %111, <4 x float>* %20, align 8
  %112 = load float*, float** %7, align 4
  %113 = getelementptr inbounds float, float* %112, i32 4
  %114 = bitcast float* %113 to i8*
  %115 = load <4 x float>, <4 x float>* %20, align 8
  %116 = bitcast <4 x float> %115 to <16 x i8>
  %117 = bitcast <16 x i8> %116 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %114, <4 x float> %117, i32 4)
  %118 = load i32, i32* %9, align 4
  %119 = load float*, float** %7, align 4
  %120 = getelementptr inbounds float, float* %119, i32 %118
  store float* %120, float** %7, align 4
  %121 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %121, <4 x float>* %21, align 8
  %122 = load float*, float** %7, align 4
  %123 = getelementptr inbounds float, float* %122, i32 0
  %124 = bitcast float* %123 to i8*
  %125 = load <4 x float>, <4 x float>* %21, align 8
  %126 = bitcast <4 x float> %125 to <16 x i8>
  %127 = bitcast <16 x i8> %126 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %124, <4 x float> %127, i32 4)
  %128 = load <4 x float>, <4 x float>* %10, align 8
  store <4 x float> %128, <4 x float>* %22, align 8
  %129 = load float*, float** %7, align 4
  %130 = getelementptr inbounds float, float* %129, i32 4
  %131 = bitcast float* %130 to i8*
  %132 = load <4 x float>, <4 x float>* %22, align 8
  %133 = bitcast <4 x float> %132 to <16 x i8>
  %134 = bitcast <16 x i8> %133 to <4 x float>
  call void @llvm.arm.neon.vst1.p0i8.v4f32(i8* %131, <4 x float> %134, i32 4)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.arm.neon.vst1.p0i8.v4f32(i8*, <4 x float>, i32) #1

; Function Attrs: noinline nounwind optnone
define void @sgemm_update_6x8__neon(i32, float*, i32, float*, i32, float*, i32, i32) #0 {
  %9 = alloca i32, align 4
  %10 = alloca float*, align 4
  %11 = alloca i32, align 4
  %12 = alloca float*, align 4
  %13 = alloca i32, align 4
  %14 = alloca float*, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  store i32 %0, i32* %9, align 4
  store float* %1, float** %10, align 4
  store i32 %2, i32* %11, align 4
  store float* %3, float** %12, align 4
  store i32 %4, i32* %13, align 4
  store float* %5, float** %14, align 4
  store i32 %6, i32* %15, align 4
  store i32 %7, i32* %16, align 4
  %19 = load float*, float** %10, align 4
  %20 = load i32, i32* %11, align 4
  %21 = getelementptr inbounds float, float* %19, i32 %20
  store float* %21, float** %10, align 4
  %22 = load float*, float** %12, align 4
  %23 = load i32, i32* %13, align 4
  %24 = getelementptr inbounds float, float* %22, i32 %23
  store float* %24, float** %12, align 4
  %25 = load float*, float** %14, align 4
  %26 = load i32, i32* %15, align 4
  %27 = getelementptr inbounds float, float* %25, i32 %26
  store float* %27, float** %14, align 4
  %28 = load i32, i32* %9, align 4
  store i32 %28, i32* %17, align 4
  %29 = load i32, i32* %16, align 4
  store i32 %29, i32* %18, align 4
  %30 = load float*, float** %14, align 4
  %31 = load float*, float** %12, align 4
  %32 = load float*, float** %10, align 4
  %33 = load i32, i32* %17, align 4
  %34 = load i32, i32* %18, align 4
  %35 = call { float*, float*, float*, i32, i32 } asm sideeffect "\09VMOV.I32  q4, #0\0A\09\09VMOV.I32  q5, #0\0A\09\09VMOV.I32  q6, #0\0A\09\09VMOV.I32  q7, #0\0A\09\09VMOV.I32  q8, #0\0A\09\09VMOV.I32  q9, #0\0A\09\09VMOV.I32 q10, #0\0A\09\09VMOV.I32 q11, #0\0A\09\09VMOV.I32 q12, #0\0A\09\09VMOV.I32 q13, #0\0A\09\09VMOV.I32 q14, #0\0A\09\09VMOV.I32 q15, #0\0A\090:\0A\09\09VLD1.32 {d4-d7}, [$1]!\0A\09\09VLD1.32 {d0-d2}, [$2]!\0A\09\09VMLA.F32 q4, q2, d0[0]\0A\09\09VMLA.F32 q5, q3, d0[0]\0A\09\09VMLA.F32 q6, q2, d0[1]\0A\09\09VMLA.F32 q7, q3, d0[1]\0A\09\09VMLA.F32  q8, q2, d1[0]\0A\09\09VMLA.F32  q9, q3, d1[0]\0A\09\09VMLA.F32 q10, q2, d1[1]\0A\09\09VMLA.F32 q11, q3, d1[1]\0A\09\09VMLA.F32 q12, q2, d2[0]\0A\09\09VMLA.F32 q13, q3, d2[0]\0A\09\09VMLA.F32 q14, q2, d2[1]\0A\09\09VMLA.F32 q15, q3, d2[1]\0A\09\09SUBS $3, $3, #1\0A\09\09BNE 0b\0A\09\09LSL $4, $4, #2\0A\09  VLD1.32 {d0-d3}, [$0]\0A\09  VADD.F32 q0, q0, q4\0A\09  VADD.F32 q1, q1, q5\0A\09  VST1.32 {d0-d3}, [$0], $4\0A\09  VLD1.32 {d4-d7}, [$0]\0A\09  VADD.F32 q2, q2, q6\0A\09  VADD.F32 q3, q3, q7\0A\09  VST1.32 {d4-d7}, [$0], $4\0A\09  VLD1.32 {d0-d3}, [$0]\0A\09  VADD.F32 q0, q0, q8\0A\09  VADD.F32 q1, q1, q9\0A\09  VST1.32 {d0-d3}, [$0], $4\0A\09  VLD1.32 {d4-d7}, [$0]\0A\09  VADD.F32 q2, q2, q10\0A\09  VADD.F32 q3, q3, q11\0A\09  VST1.32 {d4-d7}, [$0], $4\0A\09  VLD1.32 {d0-d3}, [$0]\0A\09  VADD.F32 q0, q0, q12\0A\09  VADD.F32 q1, q1, q13\0A\09  VST1.32 {d0-d3}, [$0], $4\0A\09  VLD1.32 {d4-d7}, [$0]\0A\09  VADD.F32 q2, q2, q14\0A\09  VADD.F32 q3, q3, q15\0A\09  VST1.32 {d4-d7}, [$0]\0A\09", "=r,=r,=r,=r,=r,0,1,2,3,4,~{cc},~{memory},~{d0},~{d1},~{d2},~{d3},~{d4},~{d5},~{d6},~{d7},~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15},~{d16},~{d17},~{d18},~{d19},~{d20},~{d21},~{d22},~{d23},~{d24},~{d25},~{d26},~{d27},~{d28},~{d29},~{d30},~{d31}"(float* %30, float* %31, float* %32, i32 %33, i32 %34) #2, !srcloc !4
  %36 = extractvalue { float*, float*, float*, i32, i32 } %35, 0
  %37 = extractvalue { float*, float*, float*, i32, i32 } %35, 1
  %38 = extractvalue { float*, float*, float*, i32, i32 } %35, 2
  %39 = extractvalue { float*, float*, float*, i32, i32 } %35, 3
  %40 = extractvalue { float*, float*, float*, i32, i32 } %35, 4
  store float* %36, float** %14, align 4
  store float* %37, float** %12, align 4
  store float* %38, float** %10, align 4
  store i32 %39, i32* %17, align 4
  store i32 %40, i32* %18, align 4
  ret void
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a7" "target-features"="+dsp,+hwdiv,+hwdiv-arm,+neon,+vfp3,-crc,-crypto,-d16,-dotprod,-fp-armv8,-fp-only-sp,-fp16,-ras,-thumb-mode,-vfp4" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{!"clang version 6.0.0 (tags/RELEASE_600/final)"}
!3 = !{i32 420, i32 440, i32 479, i32 518, i32 557, i32 596, i32 635, i32 674, i32 713, i32 752, i32 791, i32 830, i32 869, i32 893, i32 940, i32 987, i32 1032, i32 1077, i32 1122, i32 1167, i32 1213, i32 1259, i32 1305, i32 1351, i32 1397, i32 1443, i32 1489, i32 1535, i32 1591, i32 1620, i32 1679, i32 1741, i32 1804, i32 1867, i32 1930, i32 1993, i32 2041}
!4 = !{i32 5749, i32 5769, i32 5808, i32 5847, i32 5886, i32 5925, i32 5964, i32 6003, i32 6042, i32 6081, i32 6120, i32 6159, i32 6198, i32 6222, i32 6269, i32 6316, i32 6361, i32 6406, i32 6451, i32 6496, i32 6542, i32 6588, i32 6634, i32 6680, i32 6726, i32 6772, i32 6818, i32 6864, i32 6920, i32 6949, i32 7008, i32 7055, i32 7098, i32 7141, i32 7203, i32 7250, i32 7293, i32 7336, i32 7398, i32 7445, i32 7488, i32 7531, i32 7593, i32 7640, i32 7684, i32 7728, i32 7790, i32 7837, i32 7881, i32 7925, i32 7987, i32 8034, i32 8078, i32 8122, i32 8169}
