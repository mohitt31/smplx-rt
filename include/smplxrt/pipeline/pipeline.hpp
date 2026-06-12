#pragma once
#include <chrono>
#include <map>
#include <string_view>
#include "smplxrt/common.hpp"
#include "smplxrt/preprocess/preprocess.hpp"
namespace smplxrt {
struct StageTimings { std::map<std::string, double> milliseconds; double total_ms{}; };
struct LatencyBudget {
  double end_to_end_ms{16.6};
  double inference_ms{10.0};
  double p99_ms{16.6};
  bool accepts(const StageTimings& timings) const { return timings.total_ms <= end_to_end_ms; }
};
struct PipelineSlot { std::uint64_t sequence{}; int slot{}; };  // Triple-buffer index: sequence % 3.
inline PipelineSlot schedule_slot(std::uint64_t sequence) { return {sequence, static_cast<int>(sequence % 3)}; }
class Pipeline {
 public:
  explicit Pipeline(PreprocessConfig config = {}): preprocess_config_(config) {}
  Tensor run_preprocess(const Frame& frame, StageTimings* timings = nullptr) const;
 private: PreprocessConfig preprocess_config_;
};
}  // namespace smplxrt
