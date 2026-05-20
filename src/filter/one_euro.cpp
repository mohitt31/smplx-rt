#include "smplxrt/filter/one_euro.hpp"
#include <cmath>
#include <stdexcept>
namespace smplxrt {
OneEuroFilter::OneEuroFilter(float min_cutoff, float beta, float derivative_cutoff): min_cutoff_(min_cutoff), beta_(beta), derivative_cutoff_(derivative_cutoff) {
  if (min_cutoff <= 0 || derivative_cutoff <= 0 || beta < 0) throw std::invalid_argument("invalid One Euro configuration");
}
float OneEuroFilter::alpha(float cutoff, float dt) const { const float tau = 1.F / (2.F * static_cast<float>(M_PI) * cutoff); return 1.F / (1.F + tau / dt); }
float OneEuroFilter::update(float value, double time) {
  if (!previous_time_) { previous_time_=time; previous_value_=value; previous_derivative_=0.F; return value; }
  const float dt = static_cast<float>(time - *previous_time_); if (dt <= 0) throw std::invalid_argument("timestamps must increase");
  const float derivative = (value - *previous_value_) / dt;
  const float filtered_derivative = *previous_derivative_ + alpha(derivative_cutoff_, dt) * (derivative - *previous_derivative_);
  const float filtered = *previous_value_ + alpha(min_cutoff_ + beta_ * std::abs(filtered_derivative), dt) * (value - *previous_value_);
  previous_time_=time; previous_value_=filtered; previous_derivative_=filtered_derivative; return filtered;
}
void OneEuroFilter::reset() { previous_value_.reset(); previous_derivative_.reset(); previous_time_.reset(); }
}  // namespace smplxrt
