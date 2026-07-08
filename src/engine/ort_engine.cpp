#include <onnxruntime_cxx_api.h>

#include <sstream>
#include <unordered_map>

#include "smplxrt/engine/engine.hpp"

namespace smplxrt {
namespace {

Ort::Env& env() {
  static Ort::Env instance(ORT_LOGGING_LEVEL_ERROR, "smplx-rt");
  return instance;
}

std::vector<int> to_int_shape(const std::vector<int64_t>& dims) {
  return {dims.begin(), dims.end()};
}

class OrtEngine final : public Engine {
 public:
  explicit OrtEngine(const EngineOptions& options) : device_(options.device) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (options.intra_op_threads > 0) so.SetIntraOpNumThreads(options.intra_op_threads);
    if (device_ == Device::kCoreMl) {
      // MLProgram is the current CoreML format; ALL lets CoreML pick CPU, GPU or Neural Engine.
      // Nodes CoreML cannot run fall back to the CPU EP; the graph may be split into partitions.
      std::unordered_map<std::string, std::string> coreml{{"ModelFormat", "MLProgram"},
                                                          {"MLComputeUnits", "ALL"},
                                                          {"RequireStaticInputShapes", "0"}};
      so.AppendExecutionProvider("CoreML", coreml);
    }
    session_ = Ort::Session(env(), options.model_path.c_str(), so);

    Ort::AllocatorWithDefaultOptions alloc;
    require(session_.GetInputCount() == 1, "expected a single-input model");
    input_name_ = session_.GetInputNameAllocated(0, alloc).get();
    input_shape_ = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    // Only float outputs are fetched. The detector also returns int64 class labels, which a
    // person-only detector does not need.
    for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
      const auto type = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
      if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        output_names_.push_back(session_.GetOutputNameAllocated(i, alloc).get());
      }
    }
    require(!output_names_.empty(), "model has no float outputs");
    for (const auto& n : output_names_)
      output_name_ptrs_.push_back(n.c_str());
  }

  void run(const Tensor& input, std::vector<Tensor>* outputs) override {
    const std::vector<int64_t> shape(input.shape.begin(), input.shape.end());
    auto value = Ort::Value::CreateTensor<float>(memory_, const_cast<float*>(input.values.data()),
                                                 input.values.size(), shape.data(), shape.size());
    const char* in_name = input_name_.c_str();
    auto results = session_.Run(Ort::RunOptions{nullptr}, &in_name, &value, 1,
                                output_name_ptrs_.data(), output_name_ptrs_.size());

    outputs->resize(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
      const auto info = results[i].GetTensorTypeAndShapeInfo();
      require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
              "only float outputs are supported, output " + output_names_[i]);
      auto& out = (*outputs)[i];
      out.shape = to_int_shape(info.GetShape());
      const float* data = results[i].GetTensorData<float>();
      out.values.assign(data, data + info.GetElementCount());
    }
  }

  std::vector<int> input_shape() const override { return to_int_shape(input_shape_); }

  std::string describe() const override {
    std::ostringstream s;
    s << device_name(device_) << " input " << input_name_ << "[";
    for (size_t i = 0; i < input_shape_.size(); ++i)
      s << (i ? "," : "") << input_shape_[i];
    s << "] outputs";
    for (const auto& n : output_names_)
      s << ' ' << n;
    return s.str();
  }

 private:
  Device device_;
  Ort::Session session_{nullptr};
  Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::string input_name_;
  std::vector<int64_t> input_shape_;
  std::vector<std::string> output_names_;
  std::vector<const char*> output_name_ptrs_;
};

}  // namespace

std::unique_ptr<Engine> Engine::create(const EngineOptions& options) {
  return std::make_unique<OrtEngine>(options);
}

}  // namespace smplxrt
