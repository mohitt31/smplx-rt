#include "smplxrt/interop/cuda_vulkan.hpp"
namespace smplxrt {
InteropCapabilities CudaVulkanBridge::probe() {
#if defined(SMPLXRT_WITH_CUDA) && defined(SMPLXRT_WITH_VULKAN)
  return {true, true, true, "SDK integration enabled; verify VK_KHR_external_memory_fd at runtime."};
#else
  return {false, false, false, "Build with SMPLXRT_WITH_CUDA=ON and SMPLXRT_WITH_VULKAN=ON on Linux to enable external-memory interop."};
#endif
}
bool CudaVulkanBridge::import_vk_allocation(const ExternalMemoryHandle& handle, std::string* error) {
  const auto caps=probe(); if (!caps.external_memory_fd || handle.opaque_fd < 0 || handle.allocation_size == 0) { if(error) *error=caps.external_memory_fd ? "invalid external-memory handle" : caps.diagnostic; return false; }
  ready_=true; return true;
}
}  // namespace smplxrt
