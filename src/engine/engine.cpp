#include "smplxrt/engine/engine.hpp"

namespace smplxrt {

const char* device_name(Device device) {
  switch (device) {
    case Device::kCoreMl:
      return "coreml";
    case Device::kCpu:
    default:
      return "cpu";
  }
}

#ifndef SMPLXRT_WITH_ORT
std::unique_ptr<Engine> Engine::create(const EngineOptions&) {
  return nullptr;
}
#endif

}  // namespace smplxrt
