#pragma once
#include <memory>
#include "smplxrt/common.hpp"
namespace smplxrt {
enum class Backend { kStub, kOnnxRuntime, kTensorRt };
struct EngineOptions { Backend backend{Backend::kStub}; std::string plan_path; int max_batch{1}; };
class Engine {
 public:
  virtual ~Engine() = default;
  virtual void load(const EngineOptions& options) = 0;
  virtual Tensor enqueue(const Tensor& input) = 0;
  virtual const char* name() const = 0;
  static std::unique_ptr<Engine> create(Backend backend);
};
}  // namespace smplxrt
