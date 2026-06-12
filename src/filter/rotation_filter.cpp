#include "smplxrt/filter/rotation_filter.hpp"
#include <cmath>

namespace smplxrt {
namespace {
float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec3 subtract(const Vec3& a, const Vec3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
Vec3 scale(const Vec3& v, float s) { return {v.x*s, v.y*s, v.z*s}; }
Vec3 normalize(const Vec3& v) { const float n=std::sqrt(dot(v,v)); return n > 1e-8F ? scale(v,1.F/n) : Vec3{1.F,0.F,0.F}; }
Vec3 cross(const Vec3& a, const Vec3& b) { return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
}  // namespace

RotationFilter::RotationFilter(float min_cutoff, float beta)
    : filters_{OneEuroFilter(min_cutoff,beta), OneEuroFilter(min_cutoff,beta), OneEuroFilter(min_cutoff,beta), OneEuroFilter(min_cutoff,beta), OneEuroFilter(min_cutoff,beta), OneEuroFilter(min_cutoff,beta)} {}
Rotation6D RotationFilter::update(const Rotation6D& input, double t) {
  return {{filters_[0].update(input.first_column.x,t), filters_[1].update(input.first_column.y,t), filters_[2].update(input.first_column.z,t)},
          {filters_[3].update(input.second_column.x,t), filters_[4].update(input.second_column.y,t), filters_[5].update(input.second_column.z,t)}};
}
Quaternion quaternion_from_rotation6d(const Rotation6D& r) {
  const Vec3 c0=normalize(r.first_column); const Vec3 c1=normalize(subtract(r.second_column,scale(c0,dot(c0,r.second_column)))); const Vec3 c2=cross(c0,c1);
  const float trace=c0.x+c1.y+c2.z; Quaternion q;
  if (trace > 0.F) { const float s=std::sqrt(trace+1.F)*2.F; q={.25F*s,(c1.z-c2.y)/s,(c2.x-c0.z)/s,(c0.y-c1.x)/s}; }
  else if (c0.x > c1.y && c0.x > c2.z) { const float s=std::sqrt(1.F+c0.x-c1.y-c2.z)*2.F; q={(c1.z-c2.y)/s,.25F*s,(c1.x+c0.y)/s,(c2.x+c0.z)/s}; }
  else if (c1.y > c2.z) { const float s=std::sqrt(1.F+c1.y-c0.x-c2.z)*2.F; q={(c2.x-c0.z)/s,(c1.x+c0.y)/s,.25F*s,(c2.y+c1.z)/s}; }
  else { const float s=std::sqrt(1.F+c2.z-c0.x-c1.y)*2.F; q={(c0.y-c1.x)/s,(c2.x+c0.z)/s,(c2.y+c1.z)/s,.25F*s}; }
  return q;
}
}  // namespace smplxrt
