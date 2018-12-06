#include "topi/nn.h"
#include "topi/nn/dense.h"
#include "topi/nn/sparse.h"
#include "topi/nn/softmax.h"
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/util.h>
#include <nnvm/top/nn.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/layout.h>
#include <nnvm/node.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <tvm/tvm.h>

#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {

using tvm::Var;
using tvm::Expr;
using tvm::Tensor;
using tvm::Array;
using nnvm::compiler::FTVMCompute;

inline bool SparseLengthsSumInferType(const NodeAttrs &attrs,
                                     std::vector<int> *in_attrs,
                                     std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  // TODO: is this always true?
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, static_cast<int>(kFloat32));
  return true;
}

inline bool SparseLengthsSumInferShape(const nnvm::NodeAttrs &attrs,
                                      std::vector<TShape> *in_shape,
                                      std::vector<TShape> *out_shape) {
  CHECK_EQ(in_shape->size(), 3);
  CHECK_EQ(out_shape->size(), 1);
  TShape dshape = (*in_shape)[0];
  TShape lshape = (*in_shape)[2];
  CHECK_EQ(dshape.ndim(), 2);
  TShape oshape = dshape;
  CHECK_EQ(lshape.ndim(), 1);
  oshape[0] = lshape[0];
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

inline bool SparseLengthsSumFused8BitRowwiseInferShape(const nnvm::NodeAttrs &attrs,
                                      std::vector<TShape> *in_shape,
                                      std::vector<TShape> *out_shape) {
  CHECK_EQ(in_shape->size(), 3);
  CHECK_EQ(out_shape->size(), 1);
  TShape dshape = (*in_shape)[0];
  TShape lshape = (*in_shape)[2];
  CHECK_EQ(dshape.ndim(), 2);
  CHECK_EQ(lshape.ndim(), 1);
  TShape oshape = {lshape[0], dshape[1] - 8};
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(sparse_lengths_sum)
    .describe(R"code(Applies the sparse length sum transformation.

- **data**: `(M, K)`
- **indices**: `(sum(lengths),)`
- **lengths**: `(L,)`
- **out**: `(L, K)`

The reference implementation (in Python) is simply:

    def sparse_length_sum_ref(D, I, L):
        R = np.zeros(shape=(L.size, ) + D.shape[1:], dtype=D.dtype)
        Lsum = np.cumsum([0] + L.tolist())
        for g in range(L.size):
            for gg in range(L[g]):
                R[g, :] += D[I[Lsum[g] + gg], :]
        return R

)code" NNVM_ADD_FILELINE)
    .add_argument("data", "2D Tensor", "Input data.")
    .add_argument("indices", "1D Tensor", "Indices for lookups.")
    .add_argument("lengths", "1D Tensor", "Segment lengths.")
    // .set_attr_parser(ParamParser<DenseParam>)
    // .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<DenseParam>)
    .set_num_outputs(1)
    .set_num_inputs(3)
    .set_attr<FInferShape>("FInferShape", SparseLengthsSumInferShape)
    .set_attr<FInferType>("FInferType", SparseLengthsSumInferType)
    .set_support_level(1);

NNVM_REGISTER_OP(sparse_lengths_sum_fused_8bit_rowwise)
    .describe(R"code(Applies the sparse length sum transformation, with 8-bit rowwise
quantization.

- **data**: `(M, K)`
- **indices**: `(sum(lengths),)`
- **lengths**: `(L,)`
- **out**: `(L, K)`

The reference implementation (in Python) is simply:

    def sparse_length_sum_ref(D, I, L):
        R = np.zeros(shape=[L.size, D.shape[1] - 8], dtype=np.float32)
        Lsum = np.cumsum([0] + L.tolist())
        for g in range(L.size):
            for gg in range(L[g]):
                data_idx = I[Lsum[g] + gg]
                scale = D[data_idx, :].view(np.float32)[0]
                bias = D[data_idx, :].view(np.float32)[1]
                R[g, :] += D[data_idx, 8:].astype(np.float32) * scale + bias
        return R

)code" NNVM_ADD_FILELINE)
    .add_argument("data", "2D Tensor", "Input data.")
    .add_argument("indices", "1D Tensor", "Indices for lookups.")
    .add_argument("lengths", "1D Tensor", "Segment lengths.")
    // .set_attr_parser(ParamParser<DenseParam>)
    // .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<DenseParam>)
    .set_num_outputs(1)
    .set_num_inputs(3)
    .set_attr<FInferShape>("FInferShape", SparseLengthsSumFused8BitRowwiseInferShape)
    .set_attr<FInferType>("FInferType", SparseLengthsSumInferType)
    .set_support_level(1);

// batch_gather_nd
inline bool BatchGatherNDInferShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape>* in_attrs,
                               std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TShape& data_shape = in_attrs->at(0);
  const TShape& indices_shape = in_attrs->at(1);
  CHECK_GE(indices_shape.ndim(), 1) << "indices must have at least 1 dimensions";
  CHECK_GE(data_shape.ndim(), 2) << "data must have at least 2 dimensions";
  std::vector<dim_t> oshape;
  oshape.push_back(data_shape[0]);
  for (size_t i = 0; i < indices_shape.ndim(); ++i) {
    oshape.push_back(indices_shape[i]);
  }
  for (size_t i = 2; i < data_shape.ndim(); ++i) {
    oshape.push_back(data_shape[i]);
  }
  if (oshape.size() == 0) {
    oshape.push_back(1);
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0,
                           TShape(oshape.begin(), oshape.end()));
  return true;
}

inline bool BatchGatherNDInferType(const NodeAttrs &attrs,
                              std::vector<int> *in_attrs,
                              std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, (*in_attrs)[0]);
  return true;
}

inline bool BatchGatherNDCorrectLayout(const NodeAttrs& attrs,
                                  std::vector<Layout> *ilayouts,
                                  const std::vector<Layout> *last_ilayouts,
                                  std::vector<Layout> *olayouts) {
  CHECK_EQ(ilayouts->size(), last_ilayouts->size());
  CHECK_EQ(olayouts->size(), 1U);

  for (size_t i = 0; i < ilayouts->size(); ++i) {
    const Layout& input = last_ilayouts->at(i).defined() ?
                          last_ilayouts->at(i) : ilayouts->at(i);
    NNVM_ASSIGN_LAYOUT(*ilayouts, i, input);
  }

  return true;
}

NNVM_REGISTER_OP(batch_gather_nd)
.describe(R"code(

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "Input data.")
.add_argument("indices", "Tensor", "Indices of data")
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", BatchGatherNDInferShape)
.set_attr<FInferType>("FInferType", BatchGatherNDInferType)
.set_attr<FCorrectLayout>("FCorrectLayout", BatchGatherNDCorrectLayout)
.set_attr<FTVMCompute>(
    "FTVMCompute", [](const NodeAttrs& attrs,
                      const Array<Tensor>& inputs,
                      const Array<Tensor>& out_info) {
      return Array<Tensor>{
        topi::batch_gather_nd(inputs[0], inputs[1]) };
  })
.set_attr<FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "indices"};
})
.set_support_level(3);

} // namespace top
} // namespace nnvm
