#include <dmlc/logging.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>

#include <qnnpack.h>

namespace tvm {
namespace contrib {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.qnnpack.convolution")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      static std::once_flag flag;
      std::call_once(
          flag, []() { CHECK_EQ(qnnp_initialize(), qnnp_status_success); });
      DLTensor* input = args[0];
      DLTensor* kernel = args[1];
      DLTensor* bias = nullptr;
      if (args[2].type_code() == kArrayHandle) {
        bias = args[2];
      }
      DLTensor* output = args[3];
      uint64_t pad_top = args[4], pad_right = args[5], pad_bottom = args[6],
               pad_left = args[7];
      uint64_t stride_width = args[8], stride_height = args[9];

      CHECK_EQ(input->ndim, 4);
      CHECK_EQ(kernel->ndim, 4);
      if (bias) {
        CHECK_EQ(bias->ndim, 1);
      }
      CHECK_EQ(output->ndim, 4);
      // Input: N, H, W, C
      // Kernel: F, KH, KW, C
      CHECK_EQ(input->shape[3], kernel->shape[3]);
      CHECK_EQ(output->shape[3], kernel->shape[0]);
      if (bias) {
        CHECK_EQ(output->shape[3], bias->shape[0]);
      }
      CHECK(input->strides == nullptr);
      CHECK(kernel->strides == nullptr);
      if (bias) {
        CHECK(bias->strides == nullptr);
      }

      CHECK(TypeMatch(input->dtype, kDLUInt, 8));
      CHECK(TypeMatch(kernel->dtype, kDLUInt, 8));
      if (bias) {
        CHECK(TypeMatch(bias->dtype, kDLInt, 32));
      }
      CHECK(TypeMatch(output->dtype, kDLUInt, 8));

      // Allocate a zero-bias if we don't pass one in.
      std::unique_ptr<std::vector<int32_t>> zero_bias;
      if (!bias) {
        zero_bias.reset(new std::vector<int32_t>(output->shape[3], 0.0));
      }

      uint8_t input_zero_point = 127, weight_zero_point = 127, output_zero_point = 127;
      float input_scale = 0.5f, weight_scale = 0.5f, output_scale = 0.5f;
      uint8_t output_min = 0, output_max = 255;
      qnnp_operator_t convolutionObject = nullptr;
      auto status = qnnp_create_convolution2d_nhwc_q8(
          pad_top, pad_right, pad_bottom, pad_left, kernel->shape[1],
          kernel->shape[2], stride_height, stride_width,
          1 /* dilation_h */, 1 /* dilation_w */, 1 /* groups */,
          kernel->shape[3], kernel->shape[0], input_zero_point, input_scale,
          weight_zero_point, weight_scale, (const uint8_t*)kernel->data, (const int32_t*)bias->data,
          output_zero_point, output_scale, output_min, output_max,
          0 /* flags */, &convolutionObject);
      CHECK_EQ(status, qnnp_status_success);
      status = qnnp_setup_convolution2d_nhwc_q8(
          convolutionObject, input->shape[0], input->shape[1], input->shape[2],
          (const uint8_t*)input->data, input->shape[3], (uint8_t*)output->data, output->shape[3],
          nullptr /* thread pool */);
      CHECK_EQ(status, qnnp_status_success);
      qnnp_run_operator(convolutionObject, nullptr /* thread pool */);
      status = qnnp_delete_operator(convolutionObject);
      CHECK_EQ(status, qnnp_status_success);
    });
}  // namespace contrib
}  // namespace tvm
