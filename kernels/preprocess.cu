#include <cuda_fp16.h>

// Fused letterbox + bilinear resize + normalize + HWC->CHW, one thread per output pixel.
// Mirrors preprocess_cpu() in src/preprocess/cpu_preprocess.cpp, which is the parity reference.
// Not yet compiled or run: this machine has no NVIDIA GPU.
extern "C" __global__ void smplxrt_preprocess_kernel(const unsigned char* input, half* output,
                                                     int input_w, int input_h, int output_w,
                                                     int output_h, float scale, int pad_x,
                                                     int pad_y, float3 mean, float3 inv_std,
                                                     float padding_value) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= output_w || y >= output_h) return;

  const float sx = ((x - pad_x) + .5f) / scale - .5f;
  const float sy = ((y - pad_y) + .5f) / scale - .5f;
  float px[3] = {padding_value, padding_value, padding_value};
  if (sx >= 0.f && sy >= 0.f && sx < input_w && sy < input_h) {
    const int x0 = min(static_cast<int>(floorf(sx)), input_w - 1);
    const int y0 = min(static_cast<int>(floorf(sy)), input_h - 1);
    const int x1 = min(x0 + 1, input_w - 1);
    const int y1 = min(y0 + 1, input_h - 1);
    const float dx = sx - floorf(sx);
    const float dy = sy - floorf(sy);
    const unsigned char* p00 = input + (y0 * input_w + x0) * 3;
    const unsigned char* p01 = input + (y0 * input_w + x1) * 3;
    const unsigned char* p10 = input + (y1 * input_w + x0) * 3;
    const unsigned char* p11 = input + (y1 * input_w + x1) * 3;
    for (int c = 0; c < 3; ++c) {
      px[c] = (1 - dx) * (1 - dy) * p00[c] + dx * (1 - dy) * p01[c] + (1 - dx) * dy * p10[c] +
              dx * dy * p11[c];
    }
  }

  const int plane = output_w * output_h;
  const int i = y * output_w + x;
  output[i] = __float2half((px[0] - mean.x) * inv_std.x);
  output[plane + i] = __float2half((px[1] - mean.y) * inv_std.y);
  output[2 * plane + i] = __float2half((px[2] - mean.z) * inv_std.z);
}
