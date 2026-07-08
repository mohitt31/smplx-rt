#pragma once

#include <memory>
#include <string>
#include <vector>

#include "smplxrt/common.hpp"

namespace smplxrt {

enum class Device { kCpu, kCoreMl };

struct EngineOptions {
  std::string model_path;
  Device device{Device::kCpu};
  int intra_op_threads{0};  // 0 lets the runtime decide.
};

// One model, one float input, any number of float outputs.
class Engine {
 public:
  virtual ~Engine() = default;

  // Runs the model. `outputs` is resized on the first call and reused afterwards, so a steady
  // frame loop does not allocate for output storage.
  virtual void run(const Tensor& input, std::vector<Tensor>* outputs) = 0;
  virtual std::vector<int> input_shape() const = 0;
  virtual std::string describe() const = 0;

  // Returns nullptr when the library was built without a backend (SMPLXRT_WITH_ORT=OFF).
  static std::unique_ptr<Engine> create(const EngineOptions& options);
};

const char* device_name(Device device);

}  // namespace smplxrt
