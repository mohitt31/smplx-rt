#pragma once
#include <vector>
#include "smplxrt/common.hpp"
namespace smplxrt {
struct Bone { int parent{-1}; float rest_length{}; };
// Projects joint locations to their rest bone lengths while retaining the current parent->child direction.
bool enforce_bone_lengths(std::vector<Vec3>* joints, const std::vector<Bone>& skeleton, float epsilon = 1e-6F);
}  // namespace smplxrt
