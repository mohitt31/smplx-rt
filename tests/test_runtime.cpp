#include <cmath>
#include <iostream>

#include "smplxrt/filter/rotation_filter.hpp"
#include "smplxrt/pipeline/pipeline.hpp"
#include "smplxrt/runtime/spsc_ring.hpp"
#include "smplxrt/smplx/kinematics.hpp"

int main() {
  smplxrt::SpscRing<int, 4> queue;
  if (!queue.try_push(7) || !queue.try_push(8) || queue.try_pop().value() != 7 ||
      queue.try_pop().value() != 8 || queue.try_pop())
    return 1;
  std::vector<smplxrt::Vec3> joints = {{0, 0, 0}, {3, 4, 0}};
  std::vector<smplxrt::Bone> skeleton = {{-1, 0}, {0, 2}};
  if (!smplxrt::enforce_bone_lengths(&joints, skeleton) || std::abs(joints[1].x - 1.2F) > 1e-5F ||
      std::abs(joints[1].y - 1.6F) > 1e-5F)
    return 2;
  smplxrt::RotationFilter filter;
  const auto q = smplxrt::quaternion_from_rotation6d(filter.update({{1, 0, 0}, {0, 1, 0}}, 1.0));
  if (std::abs(q.w - 1.F) > 1e-4F) return 3;
  if (smplxrt::schedule_slot(5).slot != 2 || !smplxrt::LatencyBudget{}.accepts({{}, 16.6}))
    return 4;
  std::cout << "runtime contracts passed\n";
}
