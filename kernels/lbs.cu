#include <cuda_runtime.h>
struct Vec3f {
  float x, y, z;
};
struct Skin4 {
  unsigned short joints[4];
  float weights[4];
};
// Not yet compiled or run: this machine has no NVIDIA GPU. lbs_cpu() is the parity reference.
// Transforms are passed as 55 contiguous row-major 3x4 matrices; each vertex reads only four
// joints.
extern "C" __global__ void smplxrt_lbs_kernel(const Vec3f* rest, const Skin4* skin,
                                              const float* transforms, Vec3f* output,
                                              int vertices) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= vertices) return;
  const Vec3f v = rest[i];
  Vec3f r{};
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const float* m = transforms + skin[i].joints[n] * 12;
    const float w = skin[i].weights[n];
    r.x += w * (m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3]);
    r.y += w * (m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7]);
    r.z += w * (m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11]);
  }
  output[i] = r;
}
