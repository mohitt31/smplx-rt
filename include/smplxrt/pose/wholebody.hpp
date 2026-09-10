#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "smplxrt/engine/engine.hpp"
#include "smplxrt/pose/geometry.hpp"

namespace smplxrt {

constexpr int kWholebodyKeypoints = 133;  // COCO-WholeBody: 17 body, 6 feet, 68 face, 2x21 hands.

// YOLOX person detector. Accepts the mmdeploy end-to-end export (NMS inside the graph) or the
// pre-NMS person-only graph written by tools/prepare_models.py (NMS done here).
class PersonDetector {
 public:
  static constexpr float kScoreThreshold = 0.3F;
  explicit PersonDetector(std::unique_ptr<Engine> engine);
  void preprocess(const cv::Mat& bgr);
  void infer();
  std::vector<Box> postprocess() const;

 private:
  std::unique_ptr<Engine> engine_;
  int input_w_, input_h_;
  float ratio_{1.F};
  cv::Mat canvas_;
  Tensor input_;
  std::vector<Tensor> outputs_;
};

// RTMW / RTMPose top-down whole-body keypoints with SimCC heads.
class WholebodyPose {
 public:
  explicit WholebodyPose(std::unique_ptr<Engine> engine);
  void preprocess(const cv::Mat& bgr, const Box& person);
  void infer();
  void postprocess(std::vector<Keypoint>* keypoints) const;

 private:
  std::unique_ptr<Engine> engine_;
  int input_w_, input_h_;
  CropTransform crop_;
  cv::Mat crop_bgr_;
  Tensor input_;
  std::vector<Tensor> outputs_;
};

void draw_wholebody(cv::Mat* image, const std::vector<Keypoint>& keypoints, float threshold);

}  // namespace smplxrt
