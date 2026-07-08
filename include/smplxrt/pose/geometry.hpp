#pragma once

#include <array>
#include <vector>

namespace smplxrt {

struct Box {
  float x1{}, y1{}, x2{}, y2{};
  float score{};
};

struct Keypoint {
  float x{}, y{};
  float score{};
};

// Scale-only affine used to crop a person box to the pose model input (no rotation).
// Maps image coordinates to crop coordinates: crop = scale * image + offset.
struct CropTransform {
  float scale{1.F};
  float offset_x{}, offset_y{};
  std::array<float, 6> matrix() const { return {scale, 0.F, offset_x, 0.F, scale, offset_y}; }
};

// Letterbox used by the YOLOX detector: resize by `ratio`, pad right and bottom.
float detector_ratio(int image_w, int image_h, int input_w, int input_h);

// Expands the box by `padding`, fixes its aspect ratio to input_w / input_h and returns the
// transform that places it in the centre of the model input. Same as RTMPose's top-down affine
// with zero rotation.
CropTransform crop_transform(const Box& box, int input_w, int input_h, float padding = 1.25F);

// Highest-scoring box above `threshold`, or a box covering the whole image if there is none.
Box pick_person(const std::vector<Box>& boxes, int image_w, int image_h, float threshold = 0.3F);

// Decodes SimCC outputs (K x bins_x and K x bins_y, row major) to keypoints in crop coordinates.
// The score is the mean of the two per-axis maxima, as in rtmlib.
void decode_simcc(const float* simcc_x, int bins_x, const float* simcc_y, int bins_y, int keypoints,
                  float split_ratio, std::vector<Keypoint>* out);

// Maps keypoints from crop coordinates back to image coordinates in place.
void uncrop(const CropTransform& transform, std::vector<Keypoint>* keypoints);

}  // namespace smplxrt
