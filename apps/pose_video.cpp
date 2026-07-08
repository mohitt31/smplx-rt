// Video -> person detector -> whole-body pose -> overlay, with per-stage timing.
//
//   pose_video --video data/jumping_jacks.mp4 --det models/downloads/yolox_tiny_humanart.onnx \
//              --pose models/downloads/rtmw_l_m_256x192.onnx --device coreml \
//              --warmup 100 --frames 1000 --out out.mp4 --csv results.csv

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

#include "smplxrt/filter/one_euro.hpp"
#include "smplxrt/pose/wholebody.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
  std::string video, det, pose, out, csv, keypoints;
  smplxrt::Device device{smplxrt::Device::kCpu};
  int warmup{100};
  int frames{1000};
  int threads{0};
  bool smooth{false};
};

[[noreturn]] void usage() {
  std::cerr << "usage: pose_video --video FILE --det ONNX --pose ONNX [--device cpu|coreml]\n"
               "                  [--warmup N] [--frames N] [--threads N] [--smooth]\n"
               "                  [--out annotated.mp4] [--csv stages.csv] [--keypoints kps.csv]\n";
  std::exit(2);
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (k == "--video")
      a.video = next();
    else if (k == "--det")
      a.det = next();
    else if (k == "--pose")
      a.pose = next();
    else if (k == "--out")
      a.out = next();
    else if (k == "--csv")
      a.csv = next();
    else if (k == "--keypoints")
      a.keypoints = next();
    else if (k == "--warmup")
      a.warmup = std::stoi(next());
    else if (k == "--frames")
      a.frames = std::stoi(next());
    else if (k == "--threads")
      a.threads = std::stoi(next());
    else if (k == "--smooth")
      a.smooth = true;
    else if (k == "--device") {
      const auto d = next();
      if (d == "cpu")
        a.device = smplxrt::Device::kCpu;
      else if (d == "coreml")
        a.device = smplxrt::Device::kCoreMl;
      else
        usage();
    } else
      usage();
  }
  if (a.video.empty() || a.det.empty() || a.pose.empty() || a.frames <= 0) usage();
  return a;
}

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  const auto idx = static_cast<size_t>(p * static_cast<double>(v.size() - 1) + 0.5);
  return v[std::min(idx, v.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) try {
  const Args args = parse(argc, argv);

  smplxrt::EngineOptions det_opts{args.det, args.device, args.threads};
  smplxrt::EngineOptions pose_opts{args.pose, args.device, args.threads};
  auto det_engine = smplxrt::Engine::create(det_opts);
  auto pose_engine = smplxrt::Engine::create(pose_opts);
  if (!det_engine || !pose_engine) {
    std::cerr << "built without an inference backend; configure with -DSMPLXRT_WITH_ORT=ON\n";
    return 1;
  }
  std::cerr << "detector: " << det_engine->describe() << "\npose:     " << pose_engine->describe()
            << '\n';
  smplxrt::PersonDetector detector(std::move(det_engine));
  smplxrt::WholebodyPose pose(std::move(pose_engine));

  cv::VideoCapture cap(args.video);
  if (!cap.isOpened()) throw std::runtime_error("cannot open " + args.video);
  const double fps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 30.0;

  cv::VideoWriter writer;
  std::ofstream kp_out;
  if (!args.keypoints.empty()) {
    kp_out.open(args.keypoints);
    kp_out << "frame,keypoint,x,y,score\n";
  }

  const char* stage_names[] = {"decode",   "det_pre",    "det_infer", "det_post",
                               "pose_pre", "pose_infer", "pose_post", "draw"};
  constexpr int kStages = 8;
  std::vector<double> samples[kStages];
  std::vector<double> total;

  std::vector<smplxrt::OneEuroFilter> filters;
  if (args.smooth)
    filters.assign(2 * smplxrt::kWholebodyKeypoints, smplxrt::OneEuroFilter(1.F, 0.05F));

  cv::Mat frame;
  std::vector<smplxrt::Keypoint> keypoints;
  const int total_frames = args.warmup + args.frames;
  for (int n = 0; n < total_frames; ++n) {
    double t[kStages];
    const auto frame_start = Clock::now();

    auto s = Clock::now();
    if (!cap.read(frame)) {  // Loop short clips so the measured window is always full.
      cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      if (!cap.read(frame)) throw std::runtime_error("cannot decode " + args.video);
    }
    t[0] = ms_since(s);

    s = Clock::now();
    detector.preprocess(frame);
    t[1] = ms_since(s);
    s = Clock::now();
    detector.infer();
    t[2] = ms_since(s);
    s = Clock::now();
    const auto person = smplxrt::pick_person(detector.postprocess(), frame.cols, frame.rows);
    t[3] = ms_since(s);

    s = Clock::now();
    pose.preprocess(frame, person);
    t[4] = ms_since(s);
    s = Clock::now();
    pose.infer();
    t[5] = ms_since(s);
    s = Clock::now();
    pose.postprocess(&keypoints);
    if (args.smooth) {
      const double ts = n / fps;
      for (size_t k = 0; k < keypoints.size(); ++k) {
        keypoints[k].x = filters[2 * k].update(keypoints[k].x, ts);
        keypoints[k].y = filters[2 * k + 1].update(keypoints[k].y, ts);
      }
    }
    t[6] = ms_since(s);

    s = Clock::now();
    smplxrt::draw_wholebody(&frame, keypoints, 0.3F);
    t[7] = ms_since(s);
    const double frame_ms = ms_since(frame_start);

    // Writing the output video and keypoints is excluded from the timed window.
    if (!args.out.empty()) {
      if (!writer.isOpened()) {
        writer.open(args.out, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, frame.size());
        if (!writer.isOpened()) throw std::runtime_error("cannot write " + args.out);
      }
      writer.write(frame);
    }
    if (kp_out.is_open()) {
      for (size_t k = 0; k < keypoints.size(); ++k) {
        kp_out << n << ',' << k << ',' << keypoints[k].x << ',' << keypoints[k].y << ','
               << keypoints[k].score << '\n';
      }
    }

    if (n >= args.warmup) {
      for (int i = 0; i < kStages; ++i)
        samples[i].push_back(t[i]);
      total.push_back(frame_ms);
    }
  }

  std::printf("device=%s warmup=%d frames=%d input=%dx%d\n", smplxrt::device_name(args.device),
              args.warmup, args.frames, frame.cols, frame.rows);
  std::printf("%-11s %9s %9s %9s\n", "stage", "p50_ms", "p99_ms", "mean_ms");
  std::ofstream csv;
  if (!args.csv.empty()) {
    csv.open(args.csv);
    csv << "device,stage,p50_ms,p99_ms,mean_ms\n";
  }
  const auto report = [&](const char* name, const std::vector<double>& v) {
    double mean = 0;
    for (double x : v)
      mean += x;
    mean /= static_cast<double>(v.size());
    const double p50 = percentile(v, 0.50);
    const double p99 = percentile(v, 0.99);
    std::printf("%-11s %9.3f %9.3f %9.3f\n", name, p50, p99, mean);
    if (csv.is_open()) {
      csv << smplxrt::device_name(args.device) << ',' << name << ',' << p50 << ',' << p99 << ','
          << mean << '\n';
    }
  };
  for (int i = 0; i < kStages; ++i)
    report(stage_names[i], samples[i]);
  report("total", total);
  std::printf("fps_from_p50=%.1f\n", 1000.0 / percentile(total, 0.50));
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << '\n';
  return 1;
}
