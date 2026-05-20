#include "smplxrt/smplx/lbs.hpp"
#include <algorithm>
#include <cmath>
namespace smplxrt {
std::vector<Vec3> lbs_cpu(const LbsModel& model, const std::vector<Mat34>& transforms) {
  require(model.template_vertices.size() == model.skinning.size(), "vertex/weight count mismatch");
  std::vector<Vec3> output(model.template_vertices.size());
  for (size_t i = 0; i < output.size(); ++i) {
    const auto& v = model.template_vertices[i]; const auto& s = model.skinning[i]; Vec3 result{};
    for (int n = 0; n < 4; ++n) { require(s.joints[n] < transforms.size(), "skinning joint outside transform array"); const auto& m = transforms[s.joints[n]].value; const float w = s.weights[n];
      result.x += w * (m[0]*v.x + m[1]*v.y + m[2]*v.z + m[3]);
      result.y += w * (m[4]*v.x + m[5]*v.y + m[6]*v.z + m[7]);
      result.z += w * (m[8]*v.x + m[9]*v.y + m[10]*v.z + m[11]); }
    output[i] = result;
  } return output;
}
float max_abs_error(const std::vector<Vec3>& lhs, const std::vector<Vec3>& rhs) {
  require(lhs.size() == rhs.size(), "different vertex counts"); float result = 0.F;
  for (size_t i = 0; i < lhs.size(); ++i) for (float d : {lhs[i].x-rhs[i].x, lhs[i].y-rhs[i].y, lhs[i].z-rhs[i].z}) result = std::max(result, std::abs(d));
  return result;
}
}  // namespace smplxrt
