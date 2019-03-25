/*!
 *  Copyright (c) 2018 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */

#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <vector>

#include "../../pass/alter_op_layout.h"

namespace tvm {
namespace relay {

// relay.nn.dense
TVM_REGISTER_NODE_TYPE(GRUGatesAttrs);

bool GRUGatesRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* input_transform = types[0].as<TensorTypeNode>();
  Array<IndexExpr> oshape({input_transform->shape[0], input_transform->shape[1] / 3});
  reporter->Assign(types[2], TensorTypeNode::make(oshape, input_transform->dtype));
  return false;
}

// Positional relay function to create dense operator used by frontend FFI.
Expr MakeGRUGates(Expr input_transform, Expr hidden_transform) {
  auto attrs = make_node<GRUGatesAttrs>();
  static const Op& op = Op::Get("nn.gru_gates");
  return CallNode::make(op, {input_transform, hidden_transform}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.gru_gates")
    .set_body([](const TVMArgs& args, TVMRetValue* rv) {
      runtime::detail::unpack_call<Expr, 2>(MakeGRUGates, args, rv);
    });

RELAY_REGISTER_OP("nn.gru_gates")
    .describe(R"code()code" TVM_ADD_FILELINE)
    .set_attrs_type_key("relay.attrs.GRUGatesAttrs")
    .set_num_inputs(2)
    .add_argument("input_transform", "2D Tensor", "")
    .add_argument("hidden_transform", "2D Tensor", "")
    .set_support_level(1)
    .add_type_rel("GRUGates", GRUGatesRel);

}  // namespace relay
}  // namespace tvm
