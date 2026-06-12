#pragma once
#include <cstdint>
#include <string>

namespace smplxrt {
struct ExternalMemoryHandle { int opaque_fd{-1}; std::uint64_t allocation_size{}; };
struct InteropCapabilities { bool cuda_available{}; bool vulkan_available{}; bool external_memory_fd{}; std::string diagnostic; };
// Production implementation owns CUDA external memory/semaphore imports. This POD boundary is testable without either SDK.
class CudaVulkanBridge {
 public:
  static InteropCapabilities probe();
  bool import_vk_allocation(const ExternalMemoryHandle& handle, std::string* error);
  bool ready() const noexcept { return ready_; }
 private: bool ready_{};
};
}  // namespace smplxrt
