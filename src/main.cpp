#include <iostream>
#include "smplxrt/pipeline/pipeline.hpp"
int main() {
  smplxrt::Frame frame{1280, 720, std::vector<std::uint8_t>(1280 * 720 * 3, 127)};
  smplxrt::Pipeline pipeline; smplxrt::StageTimings timings;
  const auto tensor = pipeline.run_preprocess(frame, &timings);
  std::cout << "smplx-rt bootstrap: preprocessed " << tensor.shape[3] << "x" << tensor.shape[2]
            << " in " << timings.total_ms << " ms\n";
  std::cout << "Next: set MODEL_DIR, run tools/download_models.sh, then enable a backend.\n";
}
