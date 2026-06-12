#pragma once
#include <cstdint>
#include <string>

namespace smplxrt {
// Non-owning device frame token. Backends retain ownership; CPU bytes are never implied.
struct GpuFrame {
  std::uint64_t sequence{};
  void* device_ptr{};
  int width{};
  int height{};
  std::size_t pitch_bytes{};
  std::int64_t capture_time_ns{};
  bool is_device_resident() const noexcept { return device_ptr != nullptr; }
};

enum class CaptureBackend { kV4L2Dmabuf, kGstreamerNvmm, kPinnedHostFallback };
inline const char* capture_backend_name(CaptureBackend backend) {
  switch (backend) { case CaptureBackend::kV4L2Dmabuf: return "v4l2-dmabuf"; case CaptureBackend::kGstreamerNvmm: return "gstreamer-nvmm"; default: return "pinned-host-fallback"; }
}
}  // namespace smplxrt
