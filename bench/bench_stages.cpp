#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "smplxrt/pipeline/pipeline.hpp"
int main() {
  constexpr int warmup=20, samples=100; smplxrt::Frame frame{1280,720,std::vector<std::uint8_t>(1280*720*3,128)}; smplxrt::Pipeline pipeline; std::vector<double> times;
  for (int i=0;i<warmup+samples;++i) { smplxrt::StageTimings t; pipeline.run_preprocess(frame,&t); if (i>=warmup) times.push_back(t.total_ms); }
  std::sort(times.begin(),times.end()); const auto p50=times[times.size()/2], p99=times[static_cast<size_t>(times.size()*.99)];
  std::cout << "stage,p50_ms,p99_ms,fps\npreprocess_cpu," << p50 << ',' << p99 << ',' << 1000.0/p50 << "\n";
}
