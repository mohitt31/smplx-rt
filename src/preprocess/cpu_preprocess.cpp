#include <algorithm>
#include <cmath>

#include "smplxrt/preprocess/preprocess.hpp"

namespace smplxrt {
Letterbox compute_letterbox(int input_w, int input_h, int output_w, int output_h) {
  require(input_w > 0 && input_h > 0 && output_w > 0 && output_h > 0,
          "image dimensions must be positive");
  const float scale =
      std::min(static_cast<float>(output_w) / input_w, static_cast<float>(output_h) / input_h);
  const int resized_w = static_cast<int>(std::round(input_w * scale));
  const int resized_h = static_cast<int>(std::round(input_h * scale));
  return {scale, (output_w - resized_w) / 2, (output_h - resized_h) / 2, resized_w, resized_h};
}

Tensor preprocess_cpu(const Frame& frame, const PreprocessConfig& config, Letterbox* transform) {
  require(frame.width > 0 && frame.height > 0 &&
              frame.bgr.size() == static_cast<size_t>(frame.width * frame.height * 3),
          "invalid BGR frame");
  auto letterbox =
      compute_letterbox(frame.width, frame.height, config.output_width, config.output_height);
  if (transform) *transform = letterbox;
  Tensor out;
  out.shape = {1, 3, config.output_height, config.output_width};
  out.values.assign(static_cast<size_t>(3 * config.output_height * config.output_width), 0.F);
  const auto at = [&](int c, int y, int x) -> float& {
    return out.values[(c * config.output_height + y) * config.output_width + x];
  };
  for (int y = 0; y < config.output_height; ++y)
    for (int x = 0; x < config.output_width; ++x) {
      const float sx = (static_cast<float>(x - letterbox.pad_x) + .5F) / letterbox.scale - .5F;
      const float sy = (static_cast<float>(y - letterbox.pad_y) + .5F) / letterbox.scale - .5F;
      for (int c = 0; c < 3; ++c) {
        float pixel = config.padding_value;
        if (sx >= 0 && sy >= 0 && sx < frame.width && sy < frame.height) {
          const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, frame.width - 1);
          const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, frame.height - 1);
          const int x1 = std::min(x0 + 1, frame.width - 1), y1 = std::min(y0 + 1, frame.height - 1);
          const float dx = sx - std::floor(sx), dy = sy - std::floor(sy);
          const auto p = [&](int px, int py) {
            return static_cast<float>(frame.bgr[(py * frame.width + px) * 3 + c]);
          };
          pixel = (1 - dx) * (1 - dy) * p(x0, y0) + dx * (1 - dy) * p(x1, y0) +
                  (1 - dx) * dy * p(x0, y1) + dx * dy * p(x1, y1);
        }
        at(c, y, x) = (pixel - config.mean[c]) / config.stddev[c];
      }
    }
  return out;
}
}  // namespace smplxrt
