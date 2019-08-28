#pragma once

#include "../meta_data.h"
#include "vulkan2_shader.h"

namespace tvm {
namespace runtime {
namespace vulkan {
Module VulkanModuleCreate(std::unordered_map<std::string, VulkanShader> smap,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source);

}  // namespace vulkan

using vulkan::VulkanModuleCreate;
}  // namespace runtime
}  // namespace tvm
