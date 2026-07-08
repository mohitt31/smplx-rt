#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace smplxrt {
constexpr int kVertexCount = 10475;
constexpr int kJointCount = 55;
constexpr int kShapeCoeffs = 10;
constexpr int kExpressionCoeffs = 10;

struct Vec3 {
  float x{}, y{}, z{};
};
struct Mat34 {
  std::array<float, 12> value{};
};
struct Quaternion {
  float w{1.F}, x{}, y{}, z{};
};
struct Rotation6D {
  Vec3 first_column{1.F, 0.F, 0.F};
  Vec3 second_column{0.F, 1.F, 0.F};
};
struct Frame {
  int width{};
  int height{};
  std::vector<std::uint8_t> bgr;
};
struct Tensor {
  std::vector<float> values;
  std::vector<int> shape;
};

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}
}  // namespace smplxrt
