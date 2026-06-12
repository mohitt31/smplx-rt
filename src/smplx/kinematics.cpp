#include "smplxrt/smplx/kinematics.hpp"
#include <cmath>
namespace smplxrt {
bool enforce_bone_lengths(std::vector<Vec3>* joints, const std::vector<Bone>& skeleton, float epsilon) {
  if (!joints || joints->size()!=skeleton.size()) return false;
  for (size_t i=0;i<skeleton.size();++i) { const auto bone=skeleton[i]; if (bone.parent < 0) continue; if (bone.parent >= static_cast<int>(i) || bone.rest_length < 0.F) return false;
    const Vec3 parent=(*joints)[bone.parent], child=(*joints)[i]; const float dx=child.x-parent.x, dy=child.y-parent.y, dz=child.z-parent.z; const float norm=std::sqrt(dx*dx+dy*dy+dz*dz);
    if (norm < epsilon) return false; const float scale=bone.rest_length/norm; (*joints)[i]={parent.x+dx*scale,parent.y+dy*scale,parent.z+dz*scale}; }
  return true;
}
}  // namespace smplxrt
