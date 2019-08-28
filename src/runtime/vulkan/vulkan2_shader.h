#pragma once

#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>

namespace tvm {
namespace runtime {
namespace vulkan {

struct VulkanShader {
  /*! \brief header flag */
  uint32_t flag{0};
  /*! \brief Data segment */
  std::vector<uint32_t> data;

  void Save(dmlc::Stream* writer) const {
    writer->Write(flag);
    writer->Write(data);
  }
  bool Load(dmlc::Stream* reader) {
    if (!reader->Read(&flag)) return false;
    if (!reader->Read(&data)) return false;
    return true;
  }
};

}  // namespace vulkan

using vulkan::VulkanShader;
}  // namespace runtime
}  // namespace tvm

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, ::tvm::runtime::vulkan::VulkanShader, true);
}  // namespace dmlc
