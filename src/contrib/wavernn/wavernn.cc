/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use standard C library call.
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dlpack/dlpack.h>
#include <algorithm>
#include <vector>

#include "../../runtime/graph/graph_runtime.h"

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.wavernn.frame")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor *a1 = args[0];
  DLTensor *a2 = args[1];
  DLTensor *m = args[2];
  DLTensor *x_0 = args[3];
  DLTensor *h1_0 = args[4];
  Module sample_module = args[5];
  auto* graph_runtime = static_cast<GraphRuntime*>(sample_module.operator->());

});

}  // namespace contrib
}  // namespace tvm
