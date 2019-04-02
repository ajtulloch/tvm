/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use standard C library call.
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dlpack/dlpack.h>
#include <algorithm>
#include <vector>
#include <random>
#include "../../runtime/graph/graph_runtime.h"

namespace tvm {
namespace contrib {

using namespace runtime;


// TODO: seed consistently.
static std::unique_ptr<std::mt19937> gen; //Standard mersenne_twister_engine seeded with rd()

TVM_REGISTER_GLOBAL("tvm.contrib.wavernn.set_seed")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int seed = args[0];
  gen.reset(new std::mt19937(seed));
});

TVM_REGISTER_GLOBAL("tvm.contrib.wavernn.random_uniform")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  std::uniform_real_distribution<float> dis;
  *ret = dis(*gen);
});

TVM_REGISTER_GLOBAL("tvm.contrib.wavernn.frame")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor* I_residual = args[0];
  DLTensor* fc1_residual = args[1];
  DLTensor* x_0 = args[2];
  DLTensor* h1_0 = args[3];
  Module sample_module = args[4];
  DLTensor* outs = args[5];
  DLTensor* h1_out = args[6];
  DLTensor* x_proba = args[7];
  DLTensor* I_residual_t = args[8];
  DLTensor* fc1_residual_t = args[9];
  auto* gr = static_cast<GraphRuntime*>(sample_module.operator->());

  const size_t T = I_residual->shape[0];

  DLTensor* x = x_0;
  DLTensor* h1 = h1_0;

  auto sample_proba = [&](const float* p) -> float {
    std::uniform_real_distribution<float> dis;
    const float rand = dis(*gen);
    size_t rand_sample = 0;
    float prob_sum = p[0];
    while (prob_sum < rand) {
      rand_sample += 1;
      prob_sum += p[rand_sample];
    }
    return static_cast<float>(rand_sample) / 256;
  };

  for (size_t t = 0; t < T; ++t) {
    gr->SetInput(gr->GetInputIndex("x"), x);
    gr->SetInput(gr->GetInputIndex("h1"), h1);
    std::copy_n(static_cast<float*>(I_residual->data) + t * I_residual->shape[1],
                I_residual->shape[1], static_cast<float*>(I_residual_t->data));
    gr->SetInput(gr->GetInputIndex("I_residual"), const_cast<DLTensor*>(I_residual_t));
    std::copy_n(static_cast<float*>(fc1_residual->data) + t * fc1_residual->shape[1],
                fc1_residual->shape[1], static_cast<float*>(fc1_residual_t->data));

    gr->SetInput(gr->GetInputIndex("fc1_residual"), const_cast<DLTensor*>(fc1_residual_t));
    gr->Run();

    // Update h1
    gr->CopyOutputTo(1, h1);


    // Compute and update new sampled x values
    gr->CopyOutputTo(0, x_proba);


    const auto x_t = sample_proba(static_cast<const float*>(x_proba->data));
    static_cast<float*>(outs->data)[t] = x_t;
    static_cast<float*>(x->data)[0] = x_t;
  }

  // Copy out final hidden state.
  std::copy_n(static_cast<const float*>(h1->data), h1->shape[1], static_cast<float*>(h1_out->data));

});

}  // namespace contrib
}  // namespace tvm
