#pragma once
#include "smplxrt/common.hpp"
#include "smplxrt/filter/one_euro.hpp"
namespace smplxrt {
// Filters a continuous 6D rotation representation, then Gram-Schmidt projects it to SO(3).
// This avoids discontinuities inherent in filtering axis-angle coordinates directly.
class RotationFilter {
 public:
  RotationFilter(float min_cutoff = 1.F, float beta = 0.02F);
  Rotation6D update(const Rotation6D& input, double timestamp_seconds);
 private: std::array<OneEuroFilter, 6> filters_;
};
Quaternion quaternion_from_rotation6d(const Rotation6D& rotation);
}  // namespace smplxrt
