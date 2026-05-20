#include "smplxrt/engine/engine.hpp"
namespace smplxrt {
namespace { class StubEngine final : public Engine {
 public:
  void load(const EngineOptions& options) override { options_ = options; }
  Tensor enqueue(const Tensor& input) override { require(!input.values.empty(), "empty inference input"); return input; }
  const char* name() const override { return "stub"; }
 private: EngineOptions options_; }; }
std::unique_ptr<Engine> Engine::create(Backend) { return std::make_unique<StubEngine>(); }
}  // namespace smplxrt
