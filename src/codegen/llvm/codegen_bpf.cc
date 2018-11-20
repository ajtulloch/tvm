/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_arm.cc
 * \brief ARM specific code generator
 */
#ifdef TVM_LLVM_VERSION
#include "codegen_cpu.h"

namespace tvm {
namespace codegen {

// ARM specific code generator, this is used as an example on
// how to override behavior llvm code generator for specific target
class CodeGenBPF final : public CodeGenLLVM {
public:
  void Init(const std::string &module_name, llvm::TargetMachine *tm,
            llvm::LLVMContext *ctx, bool system_lib, bool dynamic_lookup) {
    CodeGenLLVM::Init(module_name, tm, ctx, system_lib, dynamic_lookup);
    t_tvm_shape_index_ =
        llvm::Type::getIntNTy(*ctx, TVMShapeIndexType().bits());
    t_tvm_context_ = llvm::StructType::create({t_int_, t_int_});
    t_tvm_type_ = llvm::StructType::create({t_int8_, t_int8_, t_int16_});
    t_tvm_value_ = llvm::StructType::create({t_float64_});
    t_tvm_array_ = llvm::StructType::create(
        {t_void_p_, t_tvm_context_, t_int_, t_tvm_type_,
         t_tvm_shape_index_->getPointerTo(), t_tvm_shape_index_->getPointerTo(),
         t_int64_});
  }

  llvm::Value *CreateIntrinsic(const Call *op) {
    if (op->is_intrinsic(intrinsic::tvm_call_packed_lowered)) {
      return nullptr; // CreateCallPacked(op);
    // } else if (op->is_intrinsic(intrinsic::tvm_static_handle)) {
    //   return CreateStaticHandle();
    // } else if (op->is_intrinsic(intrinsic::tvm_throw_last_error)) {
    //   builder_->CreateRet(ConstInt32(-1));
    //   return ConstInt32(-1);
    } else if (op->is_intrinsic(intrinsic::tvm_struct_get)) {
      CHECK_EQ(op->args.size(), 3U);
      int kind = op->args[2].as<IntImm>()->value;
      llvm::Value *ref = this->CreateStructRefPtr(
          op->type, MakeValue(op->args[0]), MakeValue(op->args[1]), kind);
      if (kind == intrinsic::kArrAddr) {
        return builder_->CreatePointerCast(ref, t_void_p_);
      } else {
        return builder_->CreateLoad(ref);
      }
    } else if (op->is_intrinsic(intrinsic::tvm_struct_set)) {
      CHECK_EQ(op->args.size(), 4U);
      int kind = op->args[2].as<IntImm>()->value;
      llvm::Value *value = MakeValue(op->args[3]);
      llvm::Value *ref =
          this->CreateStructRefPtr(op->args[3].type(), MakeValue(op->args[0]),
                                   MakeValue(op->args[1]), kind);
      CHECK(kind != intrinsic::kArrAddr);
      if (value->getType()->isPointerTy()) {
        value = builder_->CreatePointerCast(
            value, ref->getType()->getPointerElementType());
      }
      builder_->CreateStore(value, ref);
      return ConstInt32(0);
    } else if (op->is_intrinsic(intrinsic::tvm_stack_alloca)) {
      CHECK_EQ(op->args.size(), 2U);
      const std::string &type = op->args[0].as<StringImm>()->value;
      return WithFunctionEntry([&]() -> llvm::AllocaInst * {
        const int64_t *pval = as_const_int(op->args[1]);
        CHECK(pval) << "require stack alloca to contain constant value";
        llvm::Value *num = ConstInt32(pval[0]);
        if (type == "shape") {
          return builder_->CreateAlloca(t_tvm_shape_index_, num);
        } else if (type == "arg_value") {
          return builder_->CreateAlloca(t_tvm_value_, num);
        } else if (type == "arg_tcode") {
          return builder_->CreateAlloca(t_int_, num);
        } else if (type == "array") {
          return builder_->CreateAlloca(t_tvm_array_, num);
        } else {
          LOG(FATAL) << "Unknown stack alloca type " << type;
          return nullptr;
        }
      });
    } else {
      return CodeGenLLVM::CreateIntrinsic(op);
    }
  }
  llvm::Value *CreateStructRefPtr(Type t, llvm::Value *buf, llvm::Value *index,
                                  int kind) {
    if (kind < intrinsic::kArrKindBound_) {
      if (buf->getType() == t_void_p_) {
        buf = builder_->CreatePointerCast(buf, t_tvm_array_->getPointerTo());
      } else {
        CHECK_EQ(buf->getType(), t_tvm_array_->getPointerTo());
      }
    }
    switch (kind) {
    case intrinsic::kArrAddr: {
      return builder_->CreateInBoundsGEP(buf, index);
    }
    case intrinsic::kArrData: {
      return builder_->CreateInBoundsGEP(buf, {index, ConstInt32(0)});
    }
    case intrinsic::kArrShape: {
      return builder_->CreateInBoundsGEP(buf, {index, ConstInt32(4)});
    }
    case intrinsic::kArrStrides: {
      return builder_->CreateInBoundsGEP(buf, {index, ConstInt32(5)});
    }
    case intrinsic::kArrNDim: {
      return builder_->CreateInBoundsGEP(buf, {index, ConstInt32(2)});
    }
    case intrinsic::kArrTypeCode: {
      return builder_->CreateInBoundsGEP(buf,
                                         {index, ConstInt32(3), ConstInt32(0)});
    }
    case intrinsic::kArrTypeBits: {
      return builder_->CreateInBoundsGEP(buf,
                                         {index, ConstInt32(3), ConstInt32(1)});
    }
    case intrinsic::kArrTypeLanes: {
      return builder_->CreateInBoundsGEP(buf,
                                         {index, ConstInt32(3), ConstInt32(2)});
    }
    case intrinsic::kArrByteOffset: {
      return builder_->CreateInBoundsGEP(buf, {index, ConstInt32(6)});
    }
    case intrinsic::kArrDeviceId: {
      return builder_->CreateInBoundsGEP(buf,
                                         {index, ConstInt32(1), ConstInt32(1)});
    }
    case intrinsic::kArrDeviceType: {
      return builder_->CreateInBoundsGEP(buf,
                                         {index, ConstInt32(1), ConstInt32(0)});
    }
    case intrinsic::kTVMValueContent: {
      CHECK_EQ(t.lanes(), 1);
      CHECK(t.is_handle() || t.bits() == 64);
      if (t.is_int()) {
        buf = builder_->CreatePointerCast(buf, t_int64_->getPointerTo());
        return builder_->CreateInBoundsGEP(buf, index);
      } else if (t.is_float()) {
        buf = builder_->CreatePointerCast(buf, t_float64_->getPointerTo());
        return builder_->CreateInBoundsGEP(buf, index);
      } else {
        CHECK(t.is_handle());
        buf = builder_->CreatePointerCast(buf, t_tvm_value_->getPointerTo());
        buf = builder_->CreateInBoundsGEP(buf, index);
        return builder_->CreatePointerCast(buf, t_void_p_->getPointerTo());
      }
    }
    default:
      LOG(FATAL) << "unknown field code";
      return nullptr;
    }
  }
  void AddMainFunction(const std::string &entry_func_name) {
    llvm::Function *f = module_->getFunction(entry_func_name);
    CHECK(f) << "Function " << entry_func_name << "does not in module";
    llvm::Type *type =
        llvm::ArrayType::get(t_char_, entry_func_name.length() + 1);
    llvm::GlobalVariable *global = new llvm::GlobalVariable(
        *module_, type, true, llvm::GlobalValue::WeakAnyLinkage, 0,
        runtime::symbol::tvm_module_main);
    global->setAlignment(1);
    global->setInitializer(
        llvm::ConstantDataArray::getString(*ctx_, entry_func_name));
  }

  llvm::Type *t_tvm_shape_index_{nullptr};
  llvm::StructType *t_tvm_context_{nullptr};
  llvm::StructType *t_tvm_type_{nullptr};
  llvm::StructType *t_tvm_value_{nullptr};
  llvm::StructType *t_tvm_array_{nullptr};
};

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_bpfel")
    .set_body([](const TVMArgs &targs, TVMRetValue *rv) {
      CodeGenLLVM *cg = new CodeGenBPF();
      LOG(ERROR) << "Creating BPF target";
      *rv = static_cast<void *>(cg);
    });

} // namespace codegen
} // namespace tvm
#endif // TVM_LLVM_VERSION
