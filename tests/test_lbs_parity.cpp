#include <iostream>

#include "smplxrt/smplx/lbs.hpp"
int main() {
  smplxrt::LbsModel model;
  model.template_vertices = {{1, 2, 3}};
  model.skinning.resize(1);
  model.skinning[0].joints = {0, 0, 0, 0};
  model.skinning[0].weights = {1, 0, 0, 0};
  std::vector<smplxrt::Mat34> transforms(1);
  transforms[0].value = {1, 0, 0, 5, 0, 1, 0, 6, 0, 0, 1, 7};
  const auto result = smplxrt::lbs_cpu(model, transforms);
  const std::vector<smplxrt::Vec3> expected = {{6, 8, 10}};
  const auto error = smplxrt::max_abs_error(result, expected);
  if (error >= 1e-6F) return 1;
  std::cout << "sparse LBS reference checks passed; error=" << error << "\n";
}
