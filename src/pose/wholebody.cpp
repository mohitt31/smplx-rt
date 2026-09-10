#include "smplxrt/pose/wholebody.hpp"

#include <array>
#include <opencv2/imgproc.hpp>
#include <utility>

namespace smplxrt {
namespace {

// Writes an 8-bit BGR image into a 1x3xHxW float tensor. With `to_rgb` the channel order is
// swapped; mean and std are given in the output channel order.
void pack_chw(const cv::Mat& bgr, bool to_rgb, const std::array<float, 3>& mean,
              const std::array<float, 3>& stddev, Tensor* out) {
  const int h = bgr.rows;
  const int w = bgr.cols;
  out->shape = {1, 3, h, w};
  out->values.resize(static_cast<size_t>(3) * h * w);
  float* planes[3] = {out->values.data(), out->values.data() + h * w,
                      out->values.data() + 2 * h * w};
  const float inv[3] = {1.F / stddev[0], 1.F / stddev[1], 1.F / stddev[2]};
  for (int y = 0; y < h; ++y) {
    const auto* row = bgr.ptr<std::uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < 3; ++c) {
        const int src = to_rgb ? 2 - c : c;
        planes[c][y * w + x] = (static_cast<float>(row[3 * x + src]) - mean[c]) * inv[c];
      }
    }
  }
}

std::pair<int, int> model_size(const Engine& engine, int fallback_w, int fallback_h) {
  const auto shape = engine.input_shape();
  require(shape.size() == 4 && (shape[1] == 3 || shape[1] < 0), "expected an NCHW image input");
  const int h = shape[2] > 0 ? shape[2] : fallback_h;
  const int w = shape[3] > 0 ? shape[3] : fallback_w;
  return {w, h};
}

}  // namespace

PersonDetector::PersonDetector(std::unique_ptr<Engine> engine) : engine_(std::move(engine)) {
  // The mmdeploy YOLOX export has a dynamic spatial size; 416 matches rtmlib's lightweight mode.
  std::tie(input_w_, input_h_) = model_size(*engine_, 416, 416);
  canvas_ = cv::Mat(input_h_, input_w_, CV_8UC3);
}

void PersonDetector::preprocess(const cv::Mat& bgr) {
  ratio_ = detector_ratio(bgr.cols, bgr.rows, input_w_, input_h_);
  const int w = static_cast<int>(bgr.cols * ratio_);
  const int h = static_cast<int>(bgr.rows * ratio_);
  canvas_.setTo(cv::Scalar(114, 114, 114));
  cv::resize(bgr, canvas_(cv::Rect(0, 0, w, h)), cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
  // mmdeploy pipeline.json: BGR, mean 0, std 1.
  pack_chw(canvas_, false, {0.F, 0.F, 0.F}, {1.F, 1.F, 1.F}, &input_);
}

void PersonDetector::infer() {
  engine_->run(input_, &outputs_);
}

std::vector<Box> PersonDetector::postprocess() const {
  std::vector<Box> boxes;
  if (outputs_.size() == 1) {
    // End-to-end export: dets [1, N, 5] as x1, y1, x2, y2, score, NMS already applied.
    const auto& dets = outputs_[0];
    require(dets.shape.size() == 3 && dets.shape[2] == 5, "unexpected detector output shape");
    boxes.reserve(dets.shape[1]);
    for (int i = 0; i < dets.shape[1]; ++i) {
      const float* d = dets.values.data() + 5 * i;
      boxes.push_back({d[0] / ratio_, d[1] / ratio_, d[2] / ratio_, d[3] / ratio_, d[4]});
    }
    return boxes;
  }
  // Pre-NMS export from tools/prepare_models.py: boxes [1, N, 4] and person_scores [1, N, 1].
  require(outputs_.size() == 2, "unexpected detector outputs");
  const auto& xyxy = outputs_[0];
  const auto& scores = outputs_[1];
  require(xyxy.shape.size() == 3 && xyxy.shape[2] == 4 && scores.shape.size() == 3 &&
              scores.shape[1] == xyxy.shape[1] && scores.shape[2] == 1,
          "unexpected pre-NMS detector output shapes");
  for (int i = 0; i < xyxy.shape[1]; ++i) {
    const float score = scores.values[i];
    if (score <= kScoreThreshold) continue;
    const float* d = xyxy.values.data() + 4 * i;
    boxes.push_back({d[0] / ratio_, d[1] / ratio_, d[2] / ratio_, d[3] / ratio_, score});
  }
  // IoU 0.5 as in the export's own post-processing config (detail.json).
  return nms(std::move(boxes), kScoreThreshold, 0.5F);
}

WholebodyPose::WholebodyPose(std::unique_ptr<Engine> engine) : engine_(std::move(engine)) {
  std::tie(input_w_, input_h_) = model_size(*engine_, 192, 256);
}

void WholebodyPose::preprocess(const cv::Mat& bgr, const Box& person) {
  crop_ = crop_transform(person, input_w_, input_h_);
  const auto m = crop_.matrix();
  const cv::Mat warp = (cv::Mat_<float>(2, 3) << m[0], m[1], m[2], m[3], m[4], m[5]);
  cv::warpAffine(bgr, crop_bgr_, warp, cv::Size(input_w_, input_h_), cv::INTER_LINEAR);
  // ImageNet statistics in RGB order, as in the mmpose data preprocessor.
  pack_chw(crop_bgr_, true, {123.675F, 116.28F, 103.53F}, {58.395F, 57.12F, 57.375F}, &input_);
}

void WholebodyPose::infer() {
  engine_->run(input_, &outputs_);
}

void WholebodyPose::postprocess(std::vector<Keypoint>* keypoints) const {
  require(outputs_.size() == 2, "expected simcc_x and simcc_y outputs");
  const auto& sx = outputs_[0];
  const auto& sy = outputs_[1];
  require(sx.shape.size() == 3 && sy.shape.size() == 3 && sx.shape[1] == sy.shape[1],
          "unexpected SimCC output shape");
  decode_simcc(sx.values.data(), sx.shape[2], sy.values.data(), sy.shape[2], sx.shape[1], 2.F,
               keypoints);
  uncrop(crop_, keypoints);
}

namespace {

using Link = std::pair<int, int>;

std::vector<Link> wholebody_links() {
  // clang-format off
  std::vector<Link> links = {
      // body
      {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
      {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6},
      // feet
      {15, 17}, {15, 18}, {15, 19}, {16, 20}, {16, 21}, {16, 22},
      // wrists to hand roots
      {9, 91}, {10, 112}};
  // clang-format on
  // Each hand: root, then 5 fingers of 4 points.
  for (int root : {91, 112}) {
    for (int finger = 0; finger < 5; ++finger) {
      int prev = root;
      for (int j = 1; j <= 4; ++j) {
        const int cur = root + finger * 4 + j;
        links.emplace_back(prev, cur);
        prev = cur;
      }
    }
  }
  return links;
}

}  // namespace

void draw_wholebody(cv::Mat* image, const std::vector<Keypoint>& kps, float threshold) {
  static const std::vector<Link> links = wholebody_links();
  const auto visible = [&](int i) { return kps[i].score > threshold; };
  const auto pt = [&](int i) { return cv::Point(cvRound(kps[i].x), cvRound(kps[i].y)); };
  for (const auto& [a, b] : links) {
    if (a < static_cast<int>(kps.size()) && b < static_cast<int>(kps.size()) && visible(a) &&
        visible(b)) {
      const cv::Scalar color = a >= 91 ? cv::Scalar(255, 160, 0) : cv::Scalar(0, 200, 255);
      cv::line(*image, pt(a), pt(b), color, 2, cv::LINE_AA);
    }
  }
  for (int i = 0; i < static_cast<int>(kps.size()); ++i) {
    if (!visible(i)) continue;
    const bool face = i >= 23 && i < 91;
    cv::circle(*image, pt(i), face ? 1 : 3,
               face ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
  }
}

}  // namespace smplxrt
