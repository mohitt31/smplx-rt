#pragma once
#include "smplxrt/common.hpp"
namespace smplxrt {
struct SparseWeights {
  std::array<std::uint16_t, 4> joints{};
  std::array<float, 4> weights{};
};
struct LbsModel {
  std::vector<Vec3> template_vertices;
  std::vector<SparseWeights> skinning;
};
// Applies 3x4 skinning transforms to vertices. Production CUDA path mirrors this reference.
std::vector<Vec3> lbs_cpu(const LbsModel& model, const std::vector<Mat34>& joint_transforms);
float max_abs_error(const std::vector<Vec3>& lhs, const std::vector<Vec3>& rhs);
}  // namespace smplxrt
