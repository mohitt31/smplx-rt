#include <cmath>
#include <iostream>
#include <vector>

#include "smplxrt/pose/geometry.hpp"

namespace {
int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}
bool near(float a, float b, float tol = 1e-4F) {
  return std::abs(a - b) <= tol;
}
}  // namespace

int main() {
  using namespace smplxrt;

  // 640x480 into 416x416: width limits the scale.
  check(near(detector_ratio(640, 480, 416, 416), 0.65F), "detector ratio");

  // A tall box: height wins, width is widened to the 192:256 aspect ratio.
  const Box tall{100, 50, 200, 450, 0.9F};
  const auto t = crop_transform(tall, 192, 256);
  // Box height 400 * 1.25 = 500 maps to 256 pixels.
  check(near(t.scale, 256.F / 500.F), "crop scale");
  // The box centre lands in the centre of the crop.
  check(near(t.scale * 150.F + t.offset_x, 96.F), "crop centre x");
  check(near(t.scale * 250.F + t.offset_y, 128.F), "crop centre y");

  // uncrop is the exact inverse of the forward mapping.
  std::vector<Keypoint> kps = {{96.F, 128.F, 1.F}, {0.F, 0.F, 1.F}, {5.F, 5.F, 0.F}};
  uncrop(t, &kps);
  check(near(kps[0].x, 150.F) && near(kps[0].y, 250.F), "uncrop centre");
  check(near(t.scale * kps[1].x + t.offset_x, 0.F), "uncrop corner");
  check(near(kps[2].x, 5.F), "zero-score keypoint left untouched");

  // SimCC: argmax per axis divided by the split ratio, score is the mean of the maxima.
  const std::vector<float> sx = {0.1F, 0.8F, 0.2F, 0.0F, /* k1 */ 0.F, 0.F, 0.F, 0.F};
  const std::vector<float> sy = {0.0F, 0.1F, 0.6F, /* k1 */ 0.F, 0.F, 0.F};
  std::vector<Keypoint> decoded;
  decode_simcc(sx.data(), 4, sy.data(), 3, 2, 2.F, &decoded);
  check(decoded.size() == 2, "decoded count");
  check(near(decoded[0].x, 0.5F) && near(decoded[0].y, 1.F), "simcc location");
  check(near(decoded[0].score, 0.7F), "simcc score");
  check(decoded[1].x == -1.F && decoded[1].score == 0.F,
        "non-positive score marks keypoint missing");

  // Person selection: best score above threshold, full frame when nothing qualifies.
  const auto best =
      pick_person({{0, 0, 10, 10, 0.4F}, {5, 5, 50, 90, 0.8F}, {1, 1, 2, 2, 0.2F}}, 640, 480);
  check(near(best.score, 0.8F), "pick highest score");
  const auto none = pick_person({{0, 0, 10, 10, 0.1F}}, 640, 480);
  check(near(none.x2, 640.F) && near(none.y2, 480.F), "fallback to full frame");

  // NMS: the lower-scoring overlapping box goes, the separate one stays, low scores are dropped.
  const auto kept = nms(
      {{0, 0, 10, 10, 0.6F}, {1, 1, 10, 10, 0.9F}, {50, 50, 60, 60, 0.5F}, {70, 70, 80, 80, 0.1F}},
      0.3F, 0.5F);
  check(kept.size() == 2 && near(kept[0].score, 0.9F) && near(kept[1].x1, 50.F), "nms");
  check(near(iou({0, 0, 10, 10, 0}, {5, 0, 15, 10, 0}), 50.F / 150.F), "iou");

  // Tracking box: tight around visible keypoints, clipped; too few visible means lost.
  Box tracked;
  const std::vector<Keypoint> body = {
      {10, 20, 0.9F}, {-5, 40, 0.8F}, {30, 700, 0.7F}, {99, 99, 0.1F}};
  check(box_from_keypoints(body, 0.3F, 3, 640, 480, &tracked) && near(tracked.x1, 0.F) &&
            near(tracked.y2, 480.F) && near(tracked.x2, 30.F),
        "box from keypoints");
  check(!box_from_keypoints(body, 0.3F, 4, 640, 480, &tracked), "tracking lost");
  check(box_from_keypoints({{100, 100, 1.F}, {200, 300, 1.F}}, 0.3F, 2, 640, 480, &tracked, 1.5F) &&
            near(tracked.x1, 75.F) && near(tracked.y2, 350.F),
        "expanded tracking box");

  if (failures == 0) std::cout << "pose geometry checks passed\n";
  return failures == 0 ? 0 : 1;
}
