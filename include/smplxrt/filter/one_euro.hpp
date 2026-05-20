#pragma once
#include <optional>
namespace smplxrt {
class OneEuroFilter {
 public:
  OneEuroFilter(float min_cutoff = 1.F, float beta = 0.02F, float derivative_cutoff = 1.F);
  float update(float value, double timestamp_seconds);
  void reset();
 private:
  float alpha(float cutoff, float dt) const;
  float min_cutoff_, beta_, derivative_cutoff_;
  std::optional<float> previous_value_, previous_derivative_;
  std::optional<double> previous_time_;
};
}  // namespace smplxrt
