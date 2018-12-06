#pragma once

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "topi/detail/constant_utils.h"
#include "topi/detail/ravel_unravel.h"
#include "topi/tags.h"
#include "tvm/tvm.h"

namespace topi {
using namespace tvm;
using namespace topi::detail;

inline Tensor batch_gather_nd(const Tensor &data, const Tensor &indices,
                        std::string name = "tensor",
                        std::string tag = kInjective) {
  size_t ndim_d = data->shape.size();
  size_t ndim_i = indices->shape.size();
  CHECK_GE(ndim_i, 1) << "indices tensor must have at least 1 dimensions";
  CHECK_GE(ndim_d, 2) << "data tensor must have at least 2 dimensions";
  CHECK_EQ(ndim_i, 1);
  Array<Expr> out_shape;
  out_shape.push_back(data->shape[0]);
  for (size_t i = 0; i < ndim_i; ++i) {
    out_shape.push_back(indices->shape[i]);
  }
  for (size_t i = 2; i < ndim_d; ++i) {
    out_shape.push_back(indices->shape[i]);
  }
  return compute(out_shape,
                 [&](const Array<Var> &out_index) {
                   Array<Expr> indices_position;
                   for (size_t i = 0; i < ndim_i; ++i) {
                     indices_position.push_back(out_index[i + 1]);
                   }
                   Array<Expr> real_indices;
                   real_indices.push_back(out_index[0]);
                   real_indices.push_back(indices(indices_position));
                   for (size_t i = ndim_i + 1; i < out_index.size(); ++i) {
                     real_indices.push_back(out_index[i]);
                   }
                   return data(real_indices);
                 },
                 name, tag);
}

} // namespace topi
