#pragma once
#include <chrono>
#include <map>
#include "smplxrt/common.hpp"
#include "smplxrt/preprocess/preprocess.hpp"
namespace smplxrt {
struct StageTimings { std::map<std::string, double> milliseconds; double total_ms{}; };
class Pipeline {
 public:
  explicit Pipeline(PreprocessConfig config = {}): preprocess_config_(config) {}
  Tensor run_preprocess(const Frame& frame, StageTimings* timings = nullptr) const;
 private: PreprocessConfig preprocess_config_;
};
}  // namespace smplxrt
