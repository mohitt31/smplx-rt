#pragma once
#include "smplxrt/common.hpp"

namespace smplxrt {
struct Letterbox { float scale{}; int pad_x{}; int pad_y{}; int resized_width{}; int resized_height{}; };
struct PreprocessConfig {
  int output_width{640}; int output_height{640};
  std::array<float, 3> mean{0.F, 0.F, 0.F};
  std::array<float, 3> stddev{255.F, 255.F, 255.F};
  std::uint8_t padding_value{114};
};
Letterbox compute_letterbox(int input_w, int input_h, int output_w, int output_h);
Tensor preprocess_cpu(const Frame& frame, const PreprocessConfig& config, Letterbox* transform = nullptr);
}  // namespace smplxrt
