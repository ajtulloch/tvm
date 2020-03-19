#include <xnnpack.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/data_type.h>
#include <dmlc/logging.h>
#include <mutex>

#include <functional>
#include <unordered_map>

#include <tuple>
// function has to live in the std namespace 
// so that it is picked up by argument-dependent name lookup (ADL).
namespace std{
    namespace
    {

        // Code from boost
        // Reciprocal of the golden ratio helps spread entropy
        //     and handles duplicates.
        // See Mike Seymour in magic-numbers-in-boosthash-combine:
        //     https://stackoverflow.com/questions/4948780

        template <class T>
        inline void hash_combine(std::size_t& seed, T const& v)
        {
            seed ^= hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }

        // Recursive template code derived from Matthieu M.
        template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
        struct HashValueImpl
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
            hash_combine(seed, get<Index>(tuple));
          }
        };

        template <class Tuple>
        struct HashValueImpl<Tuple,0>
        {
          static void apply(size_t& seed, Tuple const& tuple)
          {
            hash_combine(seed, get<0>(tuple));
          }
        };
    }

    template <typename ... TT>
    struct hash<std::tuple<TT...>> 
    {
        size_t
        operator()(std::tuple<TT...> const& tt) const
        {                                              
            size_t seed = 0;                             
            HashValueImpl<std::tuple<TT...> >::apply(seed, tt);    
            return seed;                                 
        }                                              

    };
}


namespace tvm {
namespace contrib {
using namespace runtime;

using XNNOp = std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>;

static thread_local std::unordered_map<
    std::tuple<void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>, 
    XNNOp> opcache;

TVM_REGISTER_GLOBAL("tvm.contrib.xnnpack.conv1d")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      static std::once_flag flag;
      std::call_once(flag,
                     []() { CHECK_EQ(xnn_initialize(nullptr), xnn_status_success); });
      DLTensor *input = args[0];
      DLTensor *kernel = args[1];
      DLTensor *output = args[2];
      uint64_t stride_width = args[3];
      uint64_t padding_width = args[4];
      uint64_t dilation_width = args[5];
      CHECK_EQ(input->ndim, 3);
      CHECK_EQ(kernel->ndim, 3);
      CHECK_EQ(output->ndim, 3);

      int64_t batch_size = input->shape[0];
      int64_t data_width = input->shape[1];
      int64_t input_channels = input->shape[2];
      
      int64_t output_channels = kernel->shape[0];
      int64_t kernel_width = kernel->shape[1];
      int64_t input_channels_ = kernel->shape[2];

      CHECK_EQ(input_channels_, input_channels);
      CHECK(input->strides == nullptr);
      CHECK(kernel->strides == nullptr);
      CHECK(TypeMatch(input->dtype, kDLFloat, 32));
      CHECK(TypeMatch(kernel->dtype, kDLFloat, 32));
      CHECK(TypeMatch(output->dtype, kDLFloat, 32));

      auto setup = [&](){
        xnn_operator_t op_t = nullptr;
        CHECK_EQ(xnn_create_convolution2d_nhwc_f32(
            0, padding_width, 0, padding_width, 1, kernel_width, 1, stride_width, 1, dilation_width,
            1, input_channels, output_channels, input_channels, output_channels, 
            (const float*)(kernel->data), 
            nullptr, 
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::max(),
            XNN_FLAG_INPUT_NHWC,
            &op_t
        ), xnn_status_success);
        return XNNOp(op_t, xnn_delete_operator);
      };

      const auto key = std::make_tuple(kernel->data, padding_width, kernel_width, stride_width, dilation_width, input_channels, output_channels);

      if (opcache.find(key) == opcache.end()) {
          opcache.emplace(key, setup());
      }
      const auto& op = opcache.find(key)->second;
      CHECK_EQ(xnn_setup_convolution2d_nhwc_f32(op.get(), batch_size, 1, data_width, (const float*)input->data, (float*)output->data, nullptr), xnn_status_success);
      CHECK_EQ(xnn_run_operator(op.get(), nullptr), xnn_status_success);
    });



TVM_REGISTER_GLOBAL("tvm.contrib.xnnpack.conv1d_transpose")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      static std::once_flag flag;
      std::call_once(flag,
                     []() { CHECK_EQ(xnn_initialize(nullptr), xnn_status_success); });
      DLTensor *input = args[0];
      DLTensor *kernel = args[1];
      DLTensor *output = args[2];
      uint64_t stride_width = args[3];
      uint64_t padding_width = args[4];
      uint64_t output_adjustment_width = args[5];

      CHECK_EQ(input->ndim, 3);
      CHECK_EQ(kernel->ndim, 3);
      CHECK_EQ(output->ndim, 3);

      int64_t batch_size = input->shape[0];
      int64_t data_width = input->shape[1];
      int64_t input_channels = input->shape[2];
      
      int64_t output_channels = kernel->shape[0];
      int64_t kernel_width = kernel->shape[1];
      int64_t input_channels_ = kernel->shape[2];

      CHECK_EQ(input_channels_, input_channels);
      CHECK(input->strides == nullptr);
      CHECK(kernel->strides == nullptr);
      CHECK(TypeMatch(input->dtype, kDLFloat, 32));
      CHECK(TypeMatch(kernel->dtype, kDLFloat, 32));
      CHECK(TypeMatch(output->dtype, kDLFloat, 32));

      auto setup = [&](){
        xnn_operator_t op_t = nullptr;
        CHECK_EQ(xnn_create_deconvolution2d_nhwc_f32(
            0, padding_width, 0, padding_width, 1, kernel_width, 1, stride_width, 1, 1,
            1, input_channels, output_channels, input_channels, output_channels, 
            (const float*)(kernel->data), 
            nullptr, 
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::max(),
            XNN_FLAG_INPUT_NHWC,
            &op_t
        ), xnn_status_success);
        return XNNOp(op_t, xnn_delete_operator);
      };

      const auto key = std::make_tuple(kernel->data, padding_width, kernel_width, stride_width, output_adjustment_width, input_channels, output_channels);

      if (opcache.find(key) == opcache.end()) {
          opcache.emplace(key, setup());
      }
      const auto& op = opcache.find(key)->second;
      CHECK_EQ(xnn_setup_deconvolution2d_nhwc_f32(op.get(), batch_size, 1, data_width, 0, output_adjustment_width, (const float*)input->data, (float*)output->data, nullptr), xnn_status_success);
      CHECK_EQ(xnn_run_operator(op.get(), nullptr), xnn_status_success);
    });

}  // namespace contrib
}  // namespace tvm
