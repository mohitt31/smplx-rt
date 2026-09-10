#include "smplxrt/pose/geometry.hpp"

#include <algorithm>

#include "smplxrt/common.hpp"

namespace smplxrt {

float detector_ratio(int image_w, int image_h, int input_w, int input_h) {
  require(image_w > 0 && image_h > 0 && input_w > 0 && input_h > 0, "sizes must be positive");
  return std::min(static_cast<float>(input_w) / image_w, static_cast<float>(input_h) / image_h);
}

CropTransform crop_transform(const Box& box, int input_w, int input_h, float padding) {
  require(input_w > 0 && input_h > 0, "input size must be positive");
  const float cx = 0.5F * (box.x1 + box.x2);
  const float cy = 0.5F * (box.y1 + box.y2);
  float w = (box.x2 - box.x1) * padding;
  float h = (box.y2 - box.y1) * padding;
  require(w > 0 && h > 0, "degenerate box");

  const float aspect = static_cast<float>(input_w) / input_h;
  if (w > h * aspect) {
    h = w / aspect;
  } else {
    w = h * aspect;
  }

  CropTransform t;
  t.scale = input_w / w;
  t.offset_x = 0.5F * input_w - t.scale * cx;
  t.offset_y = 0.5F * input_h - t.scale * cy;
  return t;
}

Box pick_person(const std::vector<Box>& boxes, int image_w, int image_h, float threshold) {
  Box best{0.F, 0.F, static_cast<float>(image_w), static_cast<float>(image_h), 0.F};
  for (const auto& b : boxes) {
    if (b.score > threshold && b.score > best.score && b.x2 > b.x1 && b.y2 > b.y1) best = b;
  }
  return best;
}

float iou(const Box& a, const Box& b) {
  const float w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  const float h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (w <= 0 || h <= 0) return 0.F;
  const float inter = w * h;
  const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return uni > 0 ? inter / uni : 0.F;
}

std::vector<Box> nms(std::vector<Box> boxes, float score_threshold, float iou_threshold) {
  boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                             [&](const Box& b) { return b.score <= score_threshold; }),
              boxes.end());
  std::sort(boxes.begin(), boxes.end(),
            [](const Box& a, const Box& b) { return a.score > b.score; });
  std::vector<Box> kept;
  for (const auto& b : boxes) {
    bool overlaps = false;
    for (const auto& k : kept) {
      if (iou(b, k) > iou_threshold) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) kept.push_back(b);
  }
  return kept;
}

bool box_from_keypoints(const std::vector<Keypoint>& keypoints, float threshold, int min_visible,
                        int image_w, int image_h, Box* out, float expansion) {
  Box b{static_cast<float>(image_w), static_cast<float>(image_h), 0.F, 0.F, 0.F};
  int visible = 0;
  float score_sum = 0.F;
  for (const auto& kp : keypoints) {
    if (kp.score <= threshold) continue;
    b.x1 = std::min(b.x1, kp.x);
    b.y1 = std::min(b.y1, kp.y);
    b.x2 = std::max(b.x2, kp.x);
    b.y2 = std::max(b.y2, kp.y);
    score_sum += kp.score;
    ++visible;
  }
  if (visible < min_visible) return false;
  const float cx = 0.5F * (b.x1 + b.x2), cy = 0.5F * (b.y1 + b.y2);
  const float hw = 0.5F * expansion * (b.x2 - b.x1), hh = 0.5F * expansion * (b.y2 - b.y1);
  b = {cx - hw, cy - hh, cx + hw, cy + hh, b.score};
  b.x1 = std::max(b.x1, 0.F);
  b.y1 = std::max(b.y1, 0.F);
  b.x2 = std::min(b.x2, static_cast<float>(image_w));
  b.y2 = std::min(b.y2, static_cast<float>(image_h));
  if (b.x2 <= b.x1 || b.y2 <= b.y1) return false;
  b.score = score_sum / visible;
  *out = b;
  return true;
}

void decode_simcc(const float* simcc_x, int bins_x, const float* simcc_y, int bins_y, int keypoints,
                  float split_ratio, std::vector<Keypoint>* out) {
  out->resize(keypoints);
  for (int k = 0; k < keypoints; ++k) {
    const float* row_x = simcc_x + static_cast<std::ptrdiff_t>(k) * bins_x;
    const float* row_y = simcc_y + static_cast<std::ptrdiff_t>(k) * bins_y;
    const auto max_x = std::max_element(row_x, row_x + bins_x);
    const auto max_y = std::max_element(row_y, row_y + bins_y);
    const float score = 0.5F * (*max_x + *max_y);
    auto& kp = (*out)[k];
    kp.score = score;
    if (score <= 0.F) {
      kp.x = kp.y = -1.F;
    } else {
      kp.x = static_cast<float>(max_x - row_x) / split_ratio;
      kp.y = static_cast<float>(max_y - row_y) / split_ratio;
    }
  }
}

void uncrop(const CropTransform& t, std::vector<Keypoint>* keypoints) {
  for (auto& kp : *keypoints) {
    if (kp.score <= 0.F) continue;
    kp.x = (kp.x - t.offset_x) / t.scale;
    kp.y = (kp.y - t.offset_y) / t.scale;
  }
}

}  // namespace smplxrt
