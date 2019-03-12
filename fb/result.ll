; ModuleID = 'default_function'
source_filename = "default_function"
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7l-linux-gnueabihf"

%0 = type { i8*, %1, i32, %2, i64*, i64*, i64 }
%1 = type { i32, i32 }
%2 = type { i8, i8, i16 }

@__TVMAPISetLastError = linkonce dllexport local_unnamed_addr global void (i8*)* null, align 4
@.str = private constant [69 x i8] c"Assert fail: (num_args == 3), default_function: num_args should be 3\00", align 1
@.str.1 = private constant [123 x i8] c"Assert fail: ((1 == int32(arg0.strides[1])) && (32 == int32(arg0.strides[0]))), arg0.strides: expected to be compact array\00", align 1
@.str.2 = private constant [122 x i8] c"Assert fail: ((1 == int32(arg1.strides[1])) && (8 == int32(arg1.strides[0]))), arg1.strides: expected to be compact array\00", align 1
@.str.3 = private constant [122 x i8] c"Assert fail: ((1 == int32(arg2.strides[1])) && (8 == int32(arg2.strides[0]))), arg2.strides: expected to be compact array\00", align 1
@.str.4 = private constant [144 x i8] c"Assert fail: ((((arg0.code == 3) || (arg0.code == 13)) || (arg0.code == 7)) || (arg0.code == 4)), default_function: Expect arg[0] to be pointer\00", align 1
@.str.5 = private constant [144 x i8] c"Assert fail: ((((arg1.code == 3) || (arg1.code == 13)) || (arg1.code == 7)) || (arg1.code == 4)), default_function: Expect arg[1] to be pointer\00", align 1
@.str.6 = private constant [144 x i8] c"Assert fail: ((((arg2.code == 3) || (arg2.code == 13)) || (arg2.code == 7)) || (arg2.code == 4)), default_function: Expect arg[2] to be pointer\00", align 1
@.str.7 = private constant [55 x i8] c"Assert fail: (dev_type == 1), device_type need to be 1\00", align 1
@.str.8 = private constant [81 x i8] c"Assert fail: (2 == tvm_struct_get(arg0, 0, 4)), arg0.ndim is expected to equal 2\00", align 1
@.str.9 = private constant [182 x i8] c"Assert fail: (((tvm_struct_get(arg0, 0, 5) == (uint8)0) && (tvm_struct_get(arg0, 0, 6) == (uint8)8)) && (tvm_struct_get(arg0, 0, 7) == (uint16)1)), arg0.dtype is expected to be int8\00", align 1
@.str.10 = private constant [95 x i8] c"Assert fail: (int32(arg0.shape[0]) == 4), Argument arg0.shape[0] has an unsatisfied constraint\00", align 1
@.str.11 = private constant [96 x i8] c"Assert fail: (int32(arg0.shape[1]) == 32), Argument arg0.shape[1] has an unsatisfied constraint\00", align 1
@.str.12 = private constant [112 x i8] c"Assert fail: (tvm_struct_get(arg0, 0, 8) == (uint64)0), Argument arg0.byte_offset has an unsatisfied constraint\00", align 1
@.str.13 = private constant [81 x i8] c"Assert fail: (2 == tvm_struct_get(arg1, 0, 4)), arg1.ndim is expected to equal 2\00", align 1
@.str.14 = private constant [182 x i8] c"Assert fail: (((tvm_struct_get(arg1, 0, 5) == (uint8)0) && (tvm_struct_get(arg1, 0, 6) == (uint8)8)) && (tvm_struct_get(arg1, 0, 7) == (uint16)1)), arg1.dtype is expected to be int8\00", align 1
@.str.15 = private constant [96 x i8] c"Assert fail: (int32(arg1.shape[0]) == 32), Argument arg1.shape[0] has an unsatisfied constraint\00", align 1
@.str.16 = private constant [95 x i8] c"Assert fail: (int32(arg1.shape[1]) == 8), Argument arg1.shape[1] has an unsatisfied constraint\00", align 1
@.str.17 = private constant [112 x i8] c"Assert fail: (tvm_struct_get(arg1, 0, 8) == (uint64)0), Argument arg1.byte_offset has an unsatisfied constraint\00", align 1
@.str.18 = private constant [105 x i8] c"Assert fail: (1 == tvm_struct_get(arg1, 0, 10)), Argument arg1.device_type has an unsatisfied constraint\00", align 1
@.str.19 = private constant [107 x i8] c"Assert fail: (dev_id == tvm_struct_get(arg1, 0, 9)), Argument arg1.device_id has an unsatisfied constraint\00", align 1
@.str.20 = private constant [81 x i8] c"Assert fail: (2 == tvm_struct_get(arg2, 0, 4)), arg2.ndim is expected to equal 2\00", align 1
@.str.21 = private constant [184 x i8] c"Assert fail: (((tvm_struct_get(arg2, 0, 5) == (uint8)0) && (tvm_struct_get(arg2, 0, 6) == (uint8)32)) && (tvm_struct_get(arg2, 0, 7) == (uint16)1)), arg2.dtype is expected to be int32\00", align 1
@.str.22 = private constant [95 x i8] c"Assert fail: (int32(arg2.shape[0]) == 4), Argument arg2.shape[0] has an unsatisfied constraint\00", align 1
@.str.23 = private constant [95 x i8] c"Assert fail: (int32(arg2.shape[1]) == 8), Argument arg2.shape[1] has an unsatisfied constraint\00", align 1
@.str.24 = private constant [112 x i8] c"Assert fail: (tvm_struct_get(arg2, 0, 8) == (uint64)0), Argument arg2.byte_offset has an unsatisfied constraint\00", align 1
@.str.25 = private constant [105 x i8] c"Assert fail: (1 == tvm_struct_get(arg2, 0, 10)), Argument arg2.device_type has an unsatisfied constraint\00", align 1
@.str.26 = private constant [107 x i8] c"Assert fail: (dev_id == tvm_struct_get(arg2, 0, 9)), Argument arg2.device_id has an unsatisfied constraint\00", align 1
@__tvm_main__ = weak local_unnamed_addr constant [17 x i8] c"default_function\00", align 1

define dllexport i32 @default_function(i8* noalias nocapture readonly, i8* noalias nocapture readonly, i32) local_unnamed_addr {
entry:
  %3 = icmp eq i32 %2, 3
  br i1 %3, label %assert_end, label %assert_fail, !prof !1

assert_fail:                                      ; preds = %entry
  %4 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %4(i8* getelementptr inbounds ([69 x i8], [69 x i8]* @.str, i32 0, i32 0))
  ret i32 -1

assert_end:                                       ; preds = %entry
  %5 = bitcast i8* %0 to %0**
  %6 = load %0*, %0** %5, align 4
  %7 = bitcast i8* %1 to i32*
  %8 = load i32, i32* %7, align 4, !tbaa !5
  %9 = getelementptr inbounds i8, i8* %0, i32 8
  %10 = bitcast i8* %9 to %0**
  %11 = load %0*, %0** %10, align 4
  %12 = getelementptr inbounds i8, i8* %1, i32 4
  %13 = bitcast i8* %12 to i32*
  %14 = load i32, i32* %13, align 4, !tbaa !19
  %15 = getelementptr inbounds i8, i8* %0, i32 16
  %16 = bitcast i8* %15 to %0**
  %17 = load %0*, %0** %16, align 4
  %18 = getelementptr inbounds i8, i8* %1, i32 8
  %19 = bitcast i8* %18 to i32*
  %20 = load i32, i32* %19, align 4, !tbaa !21
  %21 = getelementptr inbounds %0, %0* %6, i32 0, i32 0
  %22 = load i8*, i8** %21, align 4
  %23 = getelementptr inbounds %0, %0* %6, i32 0, i32 4
  %24 = load i64*, i64** %23, align 4
  %25 = getelementptr inbounds %0, %0* %6, i32 0, i32 5
  %26 = load i64*, i64** %25, align 4
  %27 = icmp eq i64* %26, null
  br i1 %27, label %if_end, label %if_then, !prof !24

if_then:                                          ; preds = %assert_end
  %28 = getelementptr inbounds i64, i64* %26, i32 1
  %29 = load i64, i64* %28, align 8, !tbaa !25
  %30 = trunc i64 %29 to i32
  %31 = icmp eq i32 %30, 1
  %32 = load i64, i64* %26, align 8, !tbaa !39
  %33 = trunc i64 %32 to i32
  %34 = icmp eq i32 %33, 32
  %35 = and i1 %31, %34
  br i1 %35, label %if_end, label %assert_fail1, !prof !1

if_end:                                           ; preds = %assert_end, %if_then
  %36 = getelementptr inbounds %0, %0* %6, i32 0, i32 1, i32 0
  %37 = load i32, i32* %36, align 4
  %38 = getelementptr inbounds %0, %0* %6, i32 0, i32 1, i32 1
  %39 = load i32, i32* %38, align 4
  %40 = getelementptr inbounds %0, %0* %11, i32 0, i32 0
  %41 = load i8*, i8** %40, align 4
  %42 = getelementptr inbounds %0, %0* %11, i32 0, i32 4
  %43 = load i64*, i64** %42, align 4
  %44 = getelementptr inbounds %0, %0* %11, i32 0, i32 5
  %45 = load i64*, i64** %44, align 4
  %46 = icmp eq i64* %45, null
  br i1 %46, label %if_end4, label %if_then3, !prof !24

assert_fail1:                                     ; preds = %if_then
  %47 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %47(i8* getelementptr inbounds ([123 x i8], [123 x i8]* @.str.1, i32 0, i32 0))
  ret i32 -1

if_then3:                                         ; preds = %if_end
  %48 = getelementptr inbounds i64, i64* %45, i32 1
  %49 = load i64, i64* %48, align 8, !tbaa !41
  %50 = trunc i64 %49 to i32
  %51 = icmp eq i32 %50, 1
  %52 = load i64, i64* %45, align 8, !tbaa !55
  %53 = trunc i64 %52 to i32
  %54 = icmp eq i32 %53, 8
  %55 = and i1 %51, %54
  br i1 %55, label %if_end4, label %assert_fail5, !prof !1

if_end4:                                          ; preds = %if_end, %if_then3
  %56 = getelementptr inbounds %0, %0* %17, i32 0, i32 0
  %57 = load i8*, i8** %56, align 4
  %58 = getelementptr inbounds %0, %0* %17, i32 0, i32 4
  %59 = load i64*, i64** %58, align 4
  %60 = getelementptr inbounds %0, %0* %17, i32 0, i32 5
  %61 = load i64*, i64** %60, align 4
  %62 = icmp eq i64* %61, null
  br i1 %62, label %if_end8, label %if_then7, !prof !24

assert_fail5:                                     ; preds = %if_then3
  %63 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %63(i8* getelementptr inbounds ([122 x i8], [122 x i8]* @.str.2, i32 0, i32 0))
  ret i32 -1

if_then7:                                         ; preds = %if_end4
  %64 = getelementptr inbounds i64, i64* %61, i32 1
  %65 = load i64, i64* %64, align 8, !tbaa !57
  %66 = trunc i64 %65 to i32
  %67 = icmp eq i32 %66, 1
  %68 = load i64, i64* %61, align 8, !tbaa !71
  %69 = trunc i64 %68 to i32
  %70 = icmp eq i32 %69, 8
  %71 = and i1 %67, %70
  br i1 %71, label %if_end8, label %assert_fail9, !prof !1

if_end8:                                          ; preds = %if_end4, %if_then7
  switch i32 %8, label %assert_fail11 [
    i32 13, label %assert_end12
    i32 7, label %assert_end12
    i32 4, label %assert_end12
    i32 3, label %assert_end12
  ]

assert_fail9:                                     ; preds = %if_then7
  %72 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %72(i8* getelementptr inbounds ([122 x i8], [122 x i8]* @.str.3, i32 0, i32 0))
  ret i32 -1

assert_fail11:                                    ; preds = %if_end8
  %73 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %73(i8* getelementptr inbounds ([144 x i8], [144 x i8]* @.str.4, i32 0, i32 0))
  ret i32 -1

assert_end12:                                     ; preds = %if_end8, %if_end8, %if_end8, %if_end8
  switch i32 %14, label %assert_fail13 [
    i32 13, label %assert_end14
    i32 7, label %assert_end14
    i32 4, label %assert_end14
    i32 3, label %assert_end14
  ]

assert_fail13:                                    ; preds = %assert_end12
  %74 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %74(i8* getelementptr inbounds ([144 x i8], [144 x i8]* @.str.5, i32 0, i32 0))
  ret i32 -1

assert_end14:                                     ; preds = %assert_end12, %assert_end12, %assert_end12, %assert_end12
  switch i32 %20, label %assert_fail15 [
    i32 13, label %assert_end16
    i32 7, label %assert_end16
    i32 4, label %assert_end16
    i32 3, label %assert_end16
  ]

assert_fail15:                                    ; preds = %assert_end14
  %75 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %75(i8* getelementptr inbounds ([144 x i8], [144 x i8]* @.str.6, i32 0, i32 0))
  ret i32 -1

assert_end16:                                     ; preds = %assert_end14, %assert_end14, %assert_end14, %assert_end14
  %76 = icmp eq i32 %37, 1
  br i1 %76, label %assert_end18, label %assert_fail17, !prof !1

assert_fail17:                                    ; preds = %assert_end16
  %77 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %77(i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.7, i32 0, i32 0))
  ret i32 -1

assert_end18:                                     ; preds = %assert_end16
  %78 = getelementptr inbounds %0, %0* %6, i32 0, i32 2
  %79 = load i32, i32* %78, align 4
  %80 = icmp eq i32 %79, 2
  br i1 %80, label %assert_end20, label %assert_fail19, !prof !1

assert_fail19:                                    ; preds = %assert_end18
  %81 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %81(i8* getelementptr inbounds ([81 x i8], [81 x i8]* @.str.8, i32 0, i32 0))
  ret i32 -1

assert_end20:                                     ; preds = %assert_end18
  %82 = getelementptr inbounds %0, %0* %6, i32 0, i32 3, i32 0
  %83 = load i8, i8* %82, align 1
  %84 = icmp eq i8 %83, 0
  %85 = getelementptr inbounds %0, %0* %6, i32 0, i32 3, i32 1
  %86 = load i8, i8* %85, align 1
  %87 = icmp eq i8 %86, 8
  %88 = and i1 %84, %87
  %89 = getelementptr inbounds %0, %0* %6, i32 0, i32 3, i32 2
  %90 = load i16, i16* %89, align 2
  %91 = icmp eq i16 %90, 1
  %92 = and i1 %88, %91
  br i1 %92, label %assert_end22, label %assert_fail21, !prof !1

assert_fail21:                                    ; preds = %assert_end20
  %93 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %93(i8* getelementptr inbounds ([182 x i8], [182 x i8]* @.str.9, i32 0, i32 0))
  ret i32 -1

assert_end22:                                     ; preds = %assert_end20
  %94 = load i64, i64* %24, align 8, !tbaa !73
  %95 = trunc i64 %94 to i32
  %96 = icmp eq i32 %95, 4
  br i1 %96, label %assert_end24, label %assert_fail23, !prof !1

assert_fail23:                                    ; preds = %assert_end22
  %97 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %97(i8* getelementptr inbounds ([95 x i8], [95 x i8]* @.str.10, i32 0, i32 0))
  ret i32 -1

assert_end24:                                     ; preds = %assert_end22
  %98 = getelementptr inbounds i64, i64* %24, i32 1
  %99 = load i64, i64* %98, align 8, !tbaa !87
  %100 = trunc i64 %99 to i32
  %101 = icmp eq i32 %100, 32
  br i1 %101, label %assert_end26, label %assert_fail25, !prof !1

assert_fail25:                                    ; preds = %assert_end24
  %102 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %102(i8* getelementptr inbounds ([96 x i8], [96 x i8]* @.str.11, i32 0, i32 0))
  ret i32 -1

assert_end26:                                     ; preds = %assert_end24
  %103 = getelementptr inbounds %0, %0* %6, i32 0, i32 6
  %104 = load i64, i64* %103, align 8
  %105 = icmp eq i64 %104, 0
  br i1 %105, label %assert_end28, label %assert_fail27, !prof !1

assert_fail27:                                    ; preds = %assert_end26
  %106 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %106(i8* getelementptr inbounds ([112 x i8], [112 x i8]* @.str.12, i32 0, i32 0))
  ret i32 -1

assert_end28:                                     ; preds = %assert_end26
  %107 = getelementptr inbounds %0, %0* %11, i32 0, i32 2
  %108 = load i32, i32* %107, align 4
  %109 = icmp eq i32 %108, 2
  br i1 %109, label %assert_end30, label %assert_fail29, !prof !1

assert_fail29:                                    ; preds = %assert_end28
  %110 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %110(i8* getelementptr inbounds ([81 x i8], [81 x i8]* @.str.13, i32 0, i32 0))
  ret i32 -1

assert_end30:                                     ; preds = %assert_end28
  %111 = getelementptr inbounds %0, %0* %11, i32 0, i32 3, i32 0
  %112 = load i8, i8* %111, align 1
  %113 = icmp eq i8 %112, 0
  %114 = getelementptr inbounds %0, %0* %11, i32 0, i32 3, i32 1
  %115 = load i8, i8* %114, align 1
  %116 = icmp eq i8 %115, 8
  %117 = and i1 %113, %116
  %118 = getelementptr inbounds %0, %0* %11, i32 0, i32 3, i32 2
  %119 = load i16, i16* %118, align 2
  %120 = icmp eq i16 %119, 1
  %121 = and i1 %117, %120
  br i1 %121, label %assert_end32, label %assert_fail31, !prof !1

assert_fail31:                                    ; preds = %assert_end30
  %122 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %122(i8* getelementptr inbounds ([182 x i8], [182 x i8]* @.str.14, i32 0, i32 0))
  ret i32 -1

assert_end32:                                     ; preds = %assert_end30
  %123 = load i64, i64* %43, align 8, !tbaa !89
  %124 = trunc i64 %123 to i32
  %125 = icmp eq i32 %124, 32
  br i1 %125, label %assert_end34, label %assert_fail33, !prof !1

assert_fail33:                                    ; preds = %assert_end32
  %126 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %126(i8* getelementptr inbounds ([96 x i8], [96 x i8]* @.str.15, i32 0, i32 0))
  ret i32 -1

assert_end34:                                     ; preds = %assert_end32
  %127 = getelementptr inbounds i64, i64* %43, i32 1
  %128 = load i64, i64* %127, align 8, !tbaa !103
  %129 = trunc i64 %128 to i32
  %130 = icmp eq i32 %129, 8
  br i1 %130, label %assert_end36, label %assert_fail35, !prof !1

assert_fail35:                                    ; preds = %assert_end34
  %131 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %131(i8* getelementptr inbounds ([95 x i8], [95 x i8]* @.str.16, i32 0, i32 0))
  ret i32 -1

assert_end36:                                     ; preds = %assert_end34
  %132 = getelementptr inbounds %0, %0* %11, i32 0, i32 6
  %133 = load i64, i64* %132, align 8
  %134 = icmp eq i64 %133, 0
  br i1 %134, label %assert_end38, label %assert_fail37, !prof !1

assert_fail37:                                    ; preds = %assert_end36
  %135 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %135(i8* getelementptr inbounds ([112 x i8], [112 x i8]* @.str.17, i32 0, i32 0))
  ret i32 -1

assert_end38:                                     ; preds = %assert_end36
  %136 = getelementptr inbounds %0, %0* %11, i32 0, i32 1, i32 0
  %137 = load i32, i32* %136, align 4
  %138 = icmp eq i32 %137, 1
  br i1 %138, label %assert_end40, label %assert_fail39, !prof !1

assert_fail39:                                    ; preds = %assert_end38
  %139 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %139(i8* getelementptr inbounds ([105 x i8], [105 x i8]* @.str.18, i32 0, i32 0))
  ret i32 -1

assert_end40:                                     ; preds = %assert_end38
  %140 = getelementptr inbounds %0, %0* %11, i32 0, i32 1, i32 1
  %141 = load i32, i32* %140, align 4
  %142 = icmp eq i32 %39, %141
  br i1 %142, label %assert_end42, label %assert_fail41, !prof !1

assert_fail41:                                    ; preds = %assert_end40
  %143 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %143(i8* getelementptr inbounds ([107 x i8], [107 x i8]* @.str.19, i32 0, i32 0))
  ret i32 -1

assert_end42:                                     ; preds = %assert_end40
  %144 = getelementptr inbounds %0, %0* %17, i32 0, i32 2
  %145 = load i32, i32* %144, align 4
  %146 = icmp eq i32 %145, 2
  br i1 %146, label %assert_end44, label %assert_fail43, !prof !1

assert_fail43:                                    ; preds = %assert_end42
  %147 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %147(i8* getelementptr inbounds ([81 x i8], [81 x i8]* @.str.20, i32 0, i32 0))
  ret i32 -1

assert_end44:                                     ; preds = %assert_end42
  %148 = getelementptr inbounds %0, %0* %17, i32 0, i32 3, i32 0
  %149 = load i8, i8* %148, align 1
  %150 = icmp eq i8 %149, 0
  %151 = getelementptr inbounds %0, %0* %17, i32 0, i32 3, i32 1
  %152 = load i8, i8* %151, align 1
  %153 = icmp eq i8 %152, 32
  %154 = and i1 %150, %153
  %155 = getelementptr inbounds %0, %0* %17, i32 0, i32 3, i32 2
  %156 = load i16, i16* %155, align 2
  %157 = icmp eq i16 %156, 1
  %158 = and i1 %154, %157
  br i1 %158, label %assert_end46, label %assert_fail45, !prof !1

assert_fail45:                                    ; preds = %assert_end44
  %159 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %159(i8* getelementptr inbounds ([184 x i8], [184 x i8]* @.str.21, i32 0, i32 0))
  ret i32 -1

assert_end46:                                     ; preds = %assert_end44
  %160 = load i64, i64* %59, align 8, !tbaa !105
  %161 = trunc i64 %160 to i32
  %162 = icmp eq i32 %161, 4
  br i1 %162, label %assert_end48, label %assert_fail47, !prof !1

assert_fail47:                                    ; preds = %assert_end46
  %163 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %163(i8* getelementptr inbounds ([95 x i8], [95 x i8]* @.str.22, i32 0, i32 0))
  ret i32 -1

assert_end48:                                     ; preds = %assert_end46
  %164 = getelementptr inbounds i64, i64* %59, i32 1
  %165 = load i64, i64* %164, align 8, !tbaa !119
  %166 = trunc i64 %165 to i32
  %167 = icmp eq i32 %166, 8
  br i1 %167, label %assert_end50, label %assert_fail49, !prof !1

assert_fail49:                                    ; preds = %assert_end48
  %168 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %168(i8* getelementptr inbounds ([95 x i8], [95 x i8]* @.str.23, i32 0, i32 0))
  ret i32 -1

assert_end50:                                     ; preds = %assert_end48
  %169 = getelementptr inbounds %0, %0* %17, i32 0, i32 6
  %170 = load i64, i64* %169, align 8
  %171 = icmp eq i64 %170, 0
  br i1 %171, label %assert_end52, label %assert_fail51, !prof !1

assert_fail51:                                    ; preds = %assert_end50
  %172 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %172(i8* getelementptr inbounds ([112 x i8], [112 x i8]* @.str.24, i32 0, i32 0))
  ret i32 -1

assert_end52:                                     ; preds = %assert_end50
  %173 = getelementptr inbounds %0, %0* %17, i32 0, i32 1, i32 0
  %174 = load i32, i32* %173, align 4
  %175 = icmp eq i32 %174, 1
  br i1 %175, label %assert_end54, label %assert_fail53, !prof !1

assert_fail53:                                    ; preds = %assert_end52
  %176 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %176(i8* getelementptr inbounds ([105 x i8], [105 x i8]* @.str.25, i32 0, i32 0))
  ret i32 -1

assert_end54:                                     ; preds = %assert_end52
  %177 = getelementptr inbounds %0, %0* %17, i32 0, i32 1, i32 1
  %178 = load i32, i32* %177, align 4
  %179 = icmp eq i32 %39, %178
  br i1 %179, label %assert_end56, label %assert_fail55, !prof !1

assert_fail55:                                    ; preds = %assert_end54
  %180 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %180(i8* getelementptr inbounds ([107 x i8], [107 x i8]* @.str.26, i32 0, i32 0))
  ret i32 -1

assert_end56:                                     ; preds = %assert_end54
  tail call fastcc void @default_function_compute_(i8* %57, i8* %41, i8* %22)
  ret i32 0
}

; Function Attrs: noinline norecurse nounwind
define private fastcc void @default_function_compute_(i8* noalias nocapture, i8* noalias nocapture readonly, i8* noalias nocapture readonly) unnamed_addr #0 {
entry:
  %3 = bitcast i8* %1 to <8 x i8>*
  %4 = load <8 x i8>, <8 x i8>* %3, align 64, !tbaa !121
  %5 = sext <8 x i8> %4 to <8 x i16>
  %6 = shufflevector <8 x i16> %5, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %7 = sext <4 x i16> %6 to <4 x i32>
  %8 = bitcast i8* %2 to <8 x i8>*
  %9 = load <8 x i8>, <8 x i8>* %8, align 64, !tbaa !132
  %10 = sext <8 x i8> %9 to <8 x i16>
  %11 = shufflevector <8 x i16> %10, <8 x i16> undef, <4 x i32> zeroinitializer
  %12 = sext <4 x i16> %11 to <4 x i32>
  %13 = mul nsw <4 x i32> %12, %7
  %14 = getelementptr inbounds i8, i8* %1, i32 64
  %15 = bitcast i8* %14 to <8 x i8>*
  %16 = load <8 x i8>, <8 x i8>* %15, align 64, !tbaa !121
  %17 = sext <8 x i8> %16 to <8 x i16>
  %18 = shufflevector <8 x i16> %17, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %19 = sext <4 x i16> %18 to <4 x i32>
  %20 = getelementptr inbounds i8, i8* %2, i32 8
  %21 = bitcast i8* %20 to <8 x i8>*
  %22 = load <8 x i8>, <8 x i8>* %21, align 8, !tbaa !132
  %23 = sext <8 x i8> %22 to <8 x i16>
  %24 = shufflevector <8 x i16> %23, <8 x i16> undef, <4 x i32> zeroinitializer
  %25 = sext <4 x i16> %24 to <4 x i32>
  %26 = mul nsw <4 x i32> %25, %19
  %27 = add nsw <4 x i32> %26, %13
  %28 = getelementptr inbounds i8, i8* %1, i32 128
  %29 = bitcast i8* %28 to <8 x i8>*
  %30 = load <8 x i8>, <8 x i8>* %29, align 64, !tbaa !121
  %31 = sext <8 x i8> %30 to <8 x i16>
  %32 = shufflevector <8 x i16> %31, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %33 = sext <4 x i16> %32 to <4 x i32>
  %34 = getelementptr inbounds i8, i8* %2, i32 16
  %35 = bitcast i8* %34 to <8 x i8>*
  %36 = load <8 x i8>, <8 x i8>* %35, align 16, !tbaa !132
  %37 = sext <8 x i8> %36 to <8 x i16>
  %38 = shufflevector <8 x i16> %37, <8 x i16> undef, <4 x i32> zeroinitializer
  %39 = sext <4 x i16> %38 to <4 x i32>
  %40 = mul nsw <4 x i32> %39, %33
  %41 = add nsw <4 x i32> %27, %40
  %42 = getelementptr inbounds i8, i8* %1, i32 192
  %43 = bitcast i8* %42 to <8 x i8>*
  %44 = load <8 x i8>, <8 x i8>* %43, align 64, !tbaa !121
  %45 = sext <8 x i8> %44 to <8 x i16>
  %46 = shufflevector <8 x i16> %45, <8 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %47 = sext <4 x i16> %46 to <4 x i32>
  %48 = getelementptr inbounds i8, i8* %2, i32 24
  %49 = bitcast i8* %48 to <8 x i8>*
  %50 = load <8 x i8>, <8 x i8>* %49, align 8, !tbaa !132
  %51 = sext <8 x i8> %50 to <8 x i16>
  %52 = shufflevector <8 x i16> %51, <8 x i16> undef, <4 x i32> zeroinitializer
  %53 = sext <4 x i16> %52 to <4 x i32>
  %54 = mul nsw <4 x i32> %53, %47
  %55 = add nsw <4 x i32> %41, %54
  %56 = bitcast i8* %0 to <4 x i32>*
  store <4 x i32> %55, <4 x i32>* %56, align 64, !tbaa !143
  %57 = shufflevector <8 x i16> %5, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %58 = sext <4 x i16> %57 to <4 x i32>
  %59 = mul nsw <4 x i32> %12, %58
  %60 = shufflevector <8 x i16> %17, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %61 = sext <4 x i16> %60 to <4 x i32>
  %62 = mul nsw <4 x i32> %25, %61
  %63 = add nsw <4 x i32> %62, %59
  %64 = shufflevector <8 x i16> %31, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %65 = sext <4 x i16> %64 to <4 x i32>
  %66 = mul nsw <4 x i32> %39, %65
  %67 = add nsw <4 x i32> %63, %66
  %68 = shufflevector <8 x i16> %45, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %69 = sext <4 x i16> %68 to <4 x i32>
  %70 = mul nsw <4 x i32> %53, %69
  %71 = add nsw <4 x i32> %67, %70
  %72 = getelementptr inbounds i8, i8* %0, i32 16
  %73 = bitcast i8* %72 to <4 x i32>*
  store <4 x i32> %71, <4 x i32>* %73, align 16, !tbaa !143
  %74 = getelementptr inbounds i8, i8* %2, i32 32
  %75 = bitcast i8* %74 to <8 x i8>*
  %76 = load <8 x i8>, <8 x i8>* %75, align 32, !tbaa !132
  %77 = sext <8 x i8> %76 to <8 x i16>
  %78 = shufflevector <8 x i16> %77, <8 x i16> undef, <4 x i32> zeroinitializer
  %79 = sext <4 x i16> %78 to <4 x i32>
  %80 = mul nsw <4 x i32> %79, %7
  %81 = getelementptr inbounds i8, i8* %2, i32 40
  %82 = bitcast i8* %81 to <8 x i8>*
  %83 = load <8 x i8>, <8 x i8>* %82, align 8, !tbaa !132
  %84 = sext <8 x i8> %83 to <8 x i16>
  %85 = shufflevector <8 x i16> %84, <8 x i16> undef, <4 x i32> zeroinitializer
  %86 = sext <4 x i16> %85 to <4 x i32>
  %87 = mul nsw <4 x i32> %86, %19
  %88 = add nsw <4 x i32> %87, %80
  %89 = getelementptr inbounds i8, i8* %2, i32 48
  %90 = bitcast i8* %89 to <8 x i8>*
  %91 = load <8 x i8>, <8 x i8>* %90, align 16, !tbaa !132
  %92 = sext <8 x i8> %91 to <8 x i16>
  %93 = shufflevector <8 x i16> %92, <8 x i16> undef, <4 x i32> zeroinitializer
  %94 = sext <4 x i16> %93 to <4 x i32>
  %95 = mul nsw <4 x i32> %94, %33
  %96 = add nsw <4 x i32> %88, %95
  %97 = getelementptr inbounds i8, i8* %2, i32 56
  %98 = bitcast i8* %97 to <8 x i8>*
  %99 = load <8 x i8>, <8 x i8>* %98, align 8, !tbaa !132
  %100 = sext <8 x i8> %99 to <8 x i16>
  %101 = shufflevector <8 x i16> %100, <8 x i16> undef, <4 x i32> zeroinitializer
  %102 = sext <4 x i16> %101 to <4 x i32>
  %103 = mul nsw <4 x i32> %102, %47
  %104 = add nsw <4 x i32> %96, %103
  %105 = getelementptr inbounds i8, i8* %0, i32 32
  %106 = bitcast i8* %105 to <4 x i32>*
  store <4 x i32> %104, <4 x i32>* %106, align 32, !tbaa !143
  %107 = mul nsw <4 x i32> %79, %58
  %108 = mul nsw <4 x i32> %86, %61
  %109 = add nsw <4 x i32> %108, %107
  %110 = mul nsw <4 x i32> %94, %65
  %111 = add nsw <4 x i32> %109, %110
  %112 = mul nsw <4 x i32> %102, %69
  %113 = add nsw <4 x i32> %111, %112
  %114 = getelementptr inbounds i8, i8* %0, i32 48
  %115 = bitcast i8* %114 to <4 x i32>*
  store <4 x i32> %113, <4 x i32>* %115, align 16, !tbaa !143
  %116 = getelementptr inbounds i8, i8* %2, i32 64
  %117 = bitcast i8* %116 to <8 x i8>*
  %118 = load <8 x i8>, <8 x i8>* %117, align 64, !tbaa !132
  %119 = sext <8 x i8> %118 to <8 x i16>
  %120 = shufflevector <8 x i16> %119, <8 x i16> undef, <4 x i32> zeroinitializer
  %121 = sext <4 x i16> %120 to <4 x i32>
  %122 = mul nsw <4 x i32> %121, %7
  %123 = getelementptr inbounds i8, i8* %2, i32 72
  %124 = bitcast i8* %123 to <8 x i8>*
  %125 = load <8 x i8>, <8 x i8>* %124, align 8, !tbaa !132
  %126 = sext <8 x i8> %125 to <8 x i16>
  %127 = shufflevector <8 x i16> %126, <8 x i16> undef, <4 x i32> zeroinitializer
  %128 = sext <4 x i16> %127 to <4 x i32>
  %129 = mul nsw <4 x i32> %128, %19
  %130 = add nsw <4 x i32> %129, %122
  %131 = getelementptr inbounds i8, i8* %2, i32 80
  %132 = bitcast i8* %131 to <8 x i8>*
  %133 = load <8 x i8>, <8 x i8>* %132, align 16, !tbaa !132
  %134 = sext <8 x i8> %133 to <8 x i16>
  %135 = shufflevector <8 x i16> %134, <8 x i16> undef, <4 x i32> zeroinitializer
  %136 = sext <4 x i16> %135 to <4 x i32>
  %137 = mul nsw <4 x i32> %136, %33
  %138 = add nsw <4 x i32> %130, %137
  %139 = getelementptr inbounds i8, i8* %2, i32 88
  %140 = bitcast i8* %139 to <8 x i8>*
  %141 = load <8 x i8>, <8 x i8>* %140, align 8, !tbaa !132
  %142 = sext <8 x i8> %141 to <8 x i16>
  %143 = shufflevector <8 x i16> %142, <8 x i16> undef, <4 x i32> zeroinitializer
  %144 = sext <4 x i16> %143 to <4 x i32>
  %145 = mul nsw <4 x i32> %144, %47
  %146 = add nsw <4 x i32> %138, %145
  %147 = getelementptr inbounds i8, i8* %0, i32 64
  %148 = bitcast i8* %147 to <4 x i32>*
  store <4 x i32> %146, <4 x i32>* %148, align 64, !tbaa !143
  %149 = mul nsw <4 x i32> %121, %58
  %150 = mul nsw <4 x i32> %128, %61
  %151 = add nsw <4 x i32> %150, %149
  %152 = mul nsw <4 x i32> %136, %65
  %153 = add nsw <4 x i32> %151, %152
  %154 = mul nsw <4 x i32> %144, %69
  %155 = add nsw <4 x i32> %153, %154
  %156 = getelementptr inbounds i8, i8* %0, i32 80
  %157 = bitcast i8* %156 to <4 x i32>*
  store <4 x i32> %155, <4 x i32>* %157, align 16, !tbaa !143
  %158 = getelementptr inbounds i8, i8* %2, i32 96
  %159 = bitcast i8* %158 to <8 x i8>*
  %160 = load <8 x i8>, <8 x i8>* %159, align 32, !tbaa !132
  %161 = sext <8 x i8> %160 to <8 x i16>
  %162 = shufflevector <8 x i16> %161, <8 x i16> undef, <4 x i32> zeroinitializer
  %163 = sext <4 x i16> %162 to <4 x i32>
  %164 = mul nsw <4 x i32> %163, %7
  %165 = getelementptr inbounds i8, i8* %2, i32 104
  %166 = bitcast i8* %165 to <8 x i8>*
  %167 = load <8 x i8>, <8 x i8>* %166, align 8, !tbaa !132
  %168 = sext <8 x i8> %167 to <8 x i16>
  %169 = shufflevector <8 x i16> %168, <8 x i16> undef, <4 x i32> zeroinitializer
  %170 = sext <4 x i16> %169 to <4 x i32>
  %171 = mul nsw <4 x i32> %170, %19
  %172 = add nsw <4 x i32> %171, %164
  %173 = getelementptr inbounds i8, i8* %2, i32 112
  %174 = bitcast i8* %173 to <8 x i8>*
  %175 = load <8 x i8>, <8 x i8>* %174, align 16, !tbaa !132
  %176 = sext <8 x i8> %175 to <8 x i16>
  %177 = shufflevector <8 x i16> %176, <8 x i16> undef, <4 x i32> zeroinitializer
  %178 = sext <4 x i16> %177 to <4 x i32>
  %179 = mul nsw <4 x i32> %178, %33
  %180 = add nsw <4 x i32> %172, %179
  %181 = getelementptr inbounds i8, i8* %2, i32 120
  %182 = bitcast i8* %181 to <8 x i8>*
  %183 = load <8 x i8>, <8 x i8>* %182, align 8, !tbaa !132
  %184 = sext <8 x i8> %183 to <8 x i16>
  %185 = shufflevector <8 x i16> %184, <8 x i16> undef, <4 x i32> zeroinitializer
  %186 = sext <4 x i16> %185 to <4 x i32>
  %187 = mul nsw <4 x i32> %186, %47
  %188 = add nsw <4 x i32> %180, %187
  %189 = getelementptr inbounds i8, i8* %0, i32 96
  %190 = bitcast i8* %189 to <4 x i32>*
  store <4 x i32> %188, <4 x i32>* %190, align 32, !tbaa !143
  %191 = mul nsw <4 x i32> %163, %58
  %192 = mul nsw <4 x i32> %170, %61
  %193 = add nsw <4 x i32> %192, %191
  %194 = mul nsw <4 x i32> %178, %65
  %195 = add nsw <4 x i32> %193, %194
  %196 = mul nsw <4 x i32> %186, %69
  %197 = add nsw <4 x i32> %195, %196
  %198 = getelementptr inbounds i8, i8* %0, i32 112
  %199 = bitcast i8* %198 to <4 x i32>*
  store <4 x i32> %197, <4 x i32>* %199, align 16, !tbaa !143
  ret void
}

attributes #0 = { noinline norecurse nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"tvm_target", !"llvm -device=arm_cpu -model=bcm2837 -target=armv7l-linux-gnueabihf -mattr=+neon"}
!1 = !{!"branch_weights", i32 1048576, i32 1}
!2 = !{!3, !3, i64 0}
!3 = !{!"ctx_ptr", !4, i64 0}
!4 = !{!"tvm-tbaa"}
!5 = !{!6, !6, i64 0}
!6 = !{!"0x7fe9fce8ea50.w1.b0", !7, i64 0}
!7 = !{!"0x7fe9fce8ea50.w2.b0", !8, i64 0}
!8 = !{!"0x7fe9fce8ea50.w4.b0", !9, i64 0}
!9 = !{!"0x7fe9fce8ea50.w8.b0", !10, i64 0}
!10 = !{!"0x7fe9fce8ea50.w16.b0", !11, i64 0}
!11 = !{!"0x7fe9fce8ea50.w32.b0", !12, i64 0}
!12 = !{!"0x7fe9fce8ea50.w64.b0", !13, i64 0}
!13 = !{!"0x7fe9fce8ea50.w128.b0", !14, i64 0}
!14 = !{!"0x7fe9fce8ea50.w256.b0", !15, i64 0}
!15 = !{!"0x7fe9fce8ea50.w512.b0", !16, i64 0}
!16 = !{!"0x7fe9fce8ea50.w1024.b0", !17, i64 0}
!17 = !{!"int32", !18, i64 0}
!18 = !{!"0x7fe9fce8ea50", !4, i64 0}
!19 = !{!20, !20, i64 0}
!20 = !{!"0x7fe9fce8ea50.w1.b1", !7, i64 0}
!21 = !{!22, !22, i64 0}
!22 = !{!"0x7fe9fce8ea50.w1.b2", !23, i64 0}
!23 = !{!"0x7fe9fce8ea50.w2.b2", !8, i64 0}
!24 = !{!"branch_weights", i32 1, i32 1048576}
!25 = !{!26, !26, i64 0}
!26 = !{!"0x7fe9fce94410.w1.b1", !27, i64 0}
!27 = !{!"0x7fe9fce94410.w2.b0", !28, i64 0}
!28 = !{!"0x7fe9fce94410.w4.b0", !29, i64 0}
!29 = !{!"0x7fe9fce94410.w8.b0", !30, i64 0}
!30 = !{!"0x7fe9fce94410.w16.b0", !31, i64 0}
!31 = !{!"0x7fe9fce94410.w32.b0", !32, i64 0}
!32 = !{!"0x7fe9fce94410.w64.b0", !33, i64 0}
!33 = !{!"0x7fe9fce94410.w128.b0", !34, i64 0}
!34 = !{!"0x7fe9fce94410.w256.b0", !35, i64 0}
!35 = !{!"0x7fe9fce94410.w512.b0", !36, i64 0}
!36 = !{!"0x7fe9fce94410.w1024.b0", !37, i64 0}
!37 = !{!"int64", !38, i64 0}
!38 = !{!"0x7fe9fce94410", !4, i64 0}
!39 = !{!40, !40, i64 0}
!40 = !{!"0x7fe9fce94410.w1.b0", !27, i64 0}
!41 = !{!42, !42, i64 0}
!42 = !{!"0x7fe9fce96930.w1.b1", !43, i64 0}
!43 = !{!"0x7fe9fce96930.w2.b0", !44, i64 0}
!44 = !{!"0x7fe9fce96930.w4.b0", !45, i64 0}
!45 = !{!"0x7fe9fce96930.w8.b0", !46, i64 0}
!46 = !{!"0x7fe9fce96930.w16.b0", !47, i64 0}
!47 = !{!"0x7fe9fce96930.w32.b0", !48, i64 0}
!48 = !{!"0x7fe9fce96930.w64.b0", !49, i64 0}
!49 = !{!"0x7fe9fce96930.w128.b0", !50, i64 0}
!50 = !{!"0x7fe9fce96930.w256.b0", !51, i64 0}
!51 = !{!"0x7fe9fce96930.w512.b0", !52, i64 0}
!52 = !{!"0x7fe9fce96930.w1024.b0", !53, i64 0}
!53 = !{!"int64", !54, i64 0}
!54 = !{!"0x7fe9fce96930", !4, i64 0}
!55 = !{!56, !56, i64 0}
!56 = !{!"0x7fe9fce96930.w1.b0", !43, i64 0}
!57 = !{!58, !58, i64 0}
!58 = !{!"0x7fe9fce98f60.w1.b1", !59, i64 0}
!59 = !{!"0x7fe9fce98f60.w2.b0", !60, i64 0}
!60 = !{!"0x7fe9fce98f60.w4.b0", !61, i64 0}
!61 = !{!"0x7fe9fce98f60.w8.b0", !62, i64 0}
!62 = !{!"0x7fe9fce98f60.w16.b0", !63, i64 0}
!63 = !{!"0x7fe9fce98f60.w32.b0", !64, i64 0}
!64 = !{!"0x7fe9fce98f60.w64.b0", !65, i64 0}
!65 = !{!"0x7fe9fce98f60.w128.b0", !66, i64 0}
!66 = !{!"0x7fe9fce98f60.w256.b0", !67, i64 0}
!67 = !{!"0x7fe9fce98f60.w512.b0", !68, i64 0}
!68 = !{!"0x7fe9fce98f60.w1024.b0", !69, i64 0}
!69 = !{!"int64", !70, i64 0}
!70 = !{!"0x7fe9fce98f60", !4, i64 0}
!71 = !{!72, !72, i64 0}
!72 = !{!"0x7fe9fce98f60.w1.b0", !59, i64 0}
!73 = !{!74, !74, i64 0}
!74 = !{!"0x7fe9fce93f00.w1.b0", !75, i64 0}
!75 = !{!"0x7fe9fce93f00.w2.b0", !76, i64 0}
!76 = !{!"0x7fe9fce93f00.w4.b0", !77, i64 0}
!77 = !{!"0x7fe9fce93f00.w8.b0", !78, i64 0}
!78 = !{!"0x7fe9fce93f00.w16.b0", !79, i64 0}
!79 = !{!"0x7fe9fce93f00.w32.b0", !80, i64 0}
!80 = !{!"0x7fe9fce93f00.w64.b0", !81, i64 0}
!81 = !{!"0x7fe9fce93f00.w128.b0", !82, i64 0}
!82 = !{!"0x7fe9fce93f00.w256.b0", !83, i64 0}
!83 = !{!"0x7fe9fce93f00.w512.b0", !84, i64 0}
!84 = !{!"0x7fe9fce93f00.w1024.b0", !85, i64 0}
!85 = !{!"int64", !86, i64 0}
!86 = !{!"0x7fe9fce93f00", !4, i64 0}
!87 = !{!88, !88, i64 0}
!88 = !{!"0x7fe9fce93f00.w1.b1", !75, i64 0}
!89 = !{!90, !90, i64 0}
!90 = !{!"0x7fe9fce94d70.w1.b0", !91, i64 0}
!91 = !{!"0x7fe9fce94d70.w2.b0", !92, i64 0}
!92 = !{!"0x7fe9fce94d70.w4.b0", !93, i64 0}
!93 = !{!"0x7fe9fce94d70.w8.b0", !94, i64 0}
!94 = !{!"0x7fe9fce94d70.w16.b0", !95, i64 0}
!95 = !{!"0x7fe9fce94d70.w32.b0", !96, i64 0}
!96 = !{!"0x7fe9fce94d70.w64.b0", !97, i64 0}
!97 = !{!"0x7fe9fce94d70.w128.b0", !98, i64 0}
!98 = !{!"0x7fe9fce94d70.w256.b0", !99, i64 0}
!99 = !{!"0x7fe9fce94d70.w512.b0", !100, i64 0}
!100 = !{!"0x7fe9fce94d70.w1024.b0", !101, i64 0}
!101 = !{!"int64", !102, i64 0}
!102 = !{!"0x7fe9fce94d70", !4, i64 0}
!103 = !{!104, !104, i64 0}
!104 = !{!"0x7fe9fce94d70.w1.b1", !91, i64 0}
!105 = !{!106, !106, i64 0}
!106 = !{!"0x7fe9fce98aa0.w1.b0", !107, i64 0}
!107 = !{!"0x7fe9fce98aa0.w2.b0", !108, i64 0}
!108 = !{!"0x7fe9fce98aa0.w4.b0", !109, i64 0}
!109 = !{!"0x7fe9fce98aa0.w8.b0", !110, i64 0}
!110 = !{!"0x7fe9fce98aa0.w16.b0", !111, i64 0}
!111 = !{!"0x7fe9fce98aa0.w32.b0", !112, i64 0}
!112 = !{!"0x7fe9fce98aa0.w64.b0", !113, i64 0}
!113 = !{!"0x7fe9fce98aa0.w128.b0", !114, i64 0}
!114 = !{!"0x7fe9fce98aa0.w256.b0", !115, i64 0}
!115 = !{!"0x7fe9fce98aa0.w512.b0", !116, i64 0}
!116 = !{!"0x7fe9fce98aa0.w1024.b0", !117, i64 0}
!117 = !{!"int64", !118, i64 0}
!118 = !{!"0x7fe9fce98aa0", !4, i64 0}
!119 = !{!120, !120, i64 0}
!120 = !{!"0x7fe9fce98aa0.w1.b1", !107, i64 0}
!121 = !{!122, !122, i64 0}
!122 = !{!"0x7fe9ff300cc0.w8.b0", !123, i64 0}
!123 = !{!"0x7fe9ff300cc0.w16.b0", !124, i64 0}
!124 = !{!"0x7fe9ff300cc0.w32.b0", !125, i64 0}
!125 = !{!"0x7fe9ff300cc0.w64.b0", !126, i64 0}
!126 = !{!"0x7fe9ff300cc0.w128.b0", !127, i64 0}
!127 = !{!"0x7fe9ff300cc0.w256.b0", !128, i64 0}
!128 = !{!"0x7fe9ff300cc0.w512.b0", !129, i64 0}
!129 = !{!"0x7fe9ff300cc0.w1024.b0", !130, i64 0}
!130 = !{!"int8", !131, i64 0}
!131 = !{!"0x7fe9ff300cc0", !4, i64 0}
!132 = !{!133, !133, i64 0}
!133 = !{!"0x7fe9fcdcd430.w8.b0", !134, i64 0}
!134 = !{!"0x7fe9fcdcd430.w16.b0", !135, i64 0}
!135 = !{!"0x7fe9fcdcd430.w32.b0", !136, i64 0}
!136 = !{!"0x7fe9fcdcd430.w64.b0", !137, i64 0}
!137 = !{!"0x7fe9fcdcd430.w128.b0", !138, i64 0}
!138 = !{!"0x7fe9fcdcd430.w256.b0", !139, i64 0}
!139 = !{!"0x7fe9fcdcd430.w512.b0", !140, i64 0}
!140 = !{!"0x7fe9fcdcd430.w1024.b0", !141, i64 0}
!141 = !{!"int8", !142, i64 0}
!142 = !{!"0x7fe9fcdcd430", !4, i64 0}
!143 = !{!144, !144, i64 0}
!144 = !{!"0x7fe9ff300e80.w4.b0", !145, i64 0}
!145 = !{!"0x7fe9ff300e80.w8.b0", !146, i64 0}
!146 = !{!"0x7fe9ff300e80.w16.b0", !147, i64 0}
!147 = !{!"0x7fe9ff300e80.w32.b0", !148, i64 0}
!148 = !{!"0x7fe9ff300e80.w64.b0", !149, i64 0}
!149 = !{!"0x7fe9ff300e80.w128.b0", !150, i64 0}
!150 = !{!"0x7fe9ff300e80.w256.b0", !151, i64 0}
!151 = !{!"0x7fe9ff300e80.w512.b0", !152, i64 0}
!152 = !{!"0x7fe9ff300e80.w1024.b0", !153, i64 0}
!153 = !{!"int32", !154, i64 0}
!154 = !{!"0x7fe9ff300e80", !4, i64 0}