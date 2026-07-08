#include <cmath>
#include <iostream>

#include "smplxrt/preprocess/preprocess.hpp"
int main() {
  smplxrt::Frame frame{2, 1, {0, 10, 20, 100, 110, 120}};
  smplxrt::PreprocessConfig cfg;
  cfg.output_width = 2;
  cfg.output_height = 2;
  cfg.mean = {0, 0, 0};
  cfg.stddev = {1, 1, 1};
  cfg.padding_value = 114;
  smplxrt::Letterbox letterbox;
  const auto output = smplxrt::preprocess_cpu(frame, cfg, &letterbox);
  if (letterbox.pad_y != 0 || letterbox.resized_height != 1 || output.values.size() != 12) return 1;
  // First output row samples the source exactly; CHW layout must preserve B, G, R channels.
  if (std::abs(output.values[0] - 0.F) > 1e-4F || std::abs(output.values[1] - 100.F) > 1e-4F ||
      std::abs(output.values[4] - 10.F) > 1e-4F)
    return 2;
  std::cout << "preprocess reference checks passed\n";
}
