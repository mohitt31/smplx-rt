#include "smplxrt/pipeline/pipeline.hpp"
namespace smplxrt {
Tensor Pipeline::run_preprocess(const Frame& frame, StageTimings* timings) const {
  const auto start = std::chrono::steady_clock::now();
  auto output = preprocess_cpu(frame, preprocess_config_);
  const auto end = std::chrono::steady_clock::now();
  if (timings) {
    timings->total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    timings->milliseconds["preprocess_cpu"] = timings->total_ms;
  }
  return output;
}
}  // namespace smplxrt
