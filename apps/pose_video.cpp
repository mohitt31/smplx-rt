// Video -> person detector -> whole-body pose -> overlay, with per-stage timing.
//
//   pose_video --video data/jumping_jacks.mp4 \
//              --det models/downloads/yolox_tiny_humanart_person.onnx \
//              --pose models/downloads/rtmw_l_m_256x192_static.onnx --device coreml \
//              --warmup 100 --frames 1000 [--source-fps 60] [--det-every 3 | --pipeline |
//              --async-det] --csv results.csv
//
// Sequential mode runs every stage for a frame before reading the next one. --det-every N runs
// the detector on every Nth frame and tracks the person in between with a box around the
// previous frame's keypoints. --pipeline moves decode and detection to a second thread that
// hands frames to the pose thread through a lock-free SPSC ring, so the two halves overlap.
// --async-det keeps only pose on the frame's critical path: the person is tracked from the
// previous keypoints every frame, and a detector thread re-checks the latest frame in the
// background (every Nth frame with --det-every N), replacing the tracked box when tracking is
// lost or has drifted from it.
//
// --source-fps F delivers frames at F per second like a camera, and latency is measured from
// when a frame was due, so a pipeline that falls behind shows it. Without it frames are read
// as fast as the pipeline takes them.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <pthread/qos.h>
#endif

#include "smplxrt/filter/one_euro.hpp"
#include "smplxrt/pose/wholebody.hpp"
#include "smplxrt/runtime/spsc_ring.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
  std::string video, det, pose, out, csv, keypoints;
  smplxrt::Device device{smplxrt::Device::kCpu};
  int warmup{100};
  int frames{1000};
  int threads{0};
  int det_every{1};
  double source_fps{0};
  bool pipeline{false};
  bool async_det{false};
  bool smooth{false};
  bool interactive_qos{true};
  bool spin_wait{false};
};

const char* mode_name(const Args& a) {
  return a.pipeline ? "pipeline" : a.async_det ? "async-det" : "sequential";
}

[[noreturn]] void usage() {
  std::cerr << "usage: pose_video --video FILE --det ONNX --pose ONNX [--device cpu|coreml]\n"
               "                  [--warmup N] [--frames N] [--threads N] [--det-every N]\n"
               "                  [--source-fps F] [--pipeline | --async-det] [--smooth]\n"
               "                  [--no-qos] [--spin] [--out annotated.mp4]\n"
               "                  [--csv stages.csv] [--keypoints kps.csv]\n";
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
    if (k == "--video") {
      a.video = next();
    } else if (k == "--det") {
      a.det = next();
    } else if (k == "--pose") {
      a.pose = next();
    } else if (k == "--out") {
      a.out = next();
    } else if (k == "--csv") {
      a.csv = next();
    } else if (k == "--keypoints") {
      a.keypoints = next();
    } else if (k == "--warmup") {
      a.warmup = std::stoi(next());
    } else if (k == "--frames") {
      a.frames = std::stoi(next());
    } else if (k == "--threads") {
      a.threads = std::stoi(next());
    } else if (k == "--det-every") {
      a.det_every = std::stoi(next());
    } else if (k == "--source-fps") {
      a.source_fps = std::stod(next());
    } else if (k == "--pipeline") {
      a.pipeline = true;
    } else if (k == "--async-det") {
      a.async_det = true;
    } else if (k == "--spin") {
      a.spin_wait = true;
    } else if (k == "--no-qos") {
      a.interactive_qos = false;
    } else if (k == "--smooth") {
      a.smooth = true;
    } else if (k == "--device") {
      const auto d = next();
      if (d == "cpu") {
        a.device = smplxrt::Device::kCpu;
      } else if (d == "coreml") {
        a.device = smplxrt::Device::kCoreMl;
      } else {
        usage();
      }
    } else {
      usage();
    }
  }
  if (a.video.empty() || a.det.empty() || a.pose.empty() || a.frames < 2 || a.warmup < 0 ||
      a.det_every < 1) {
    usage();
  }
  if (a.pipeline && a.async_det) {
    std::cerr << "--pipeline and --async-det are separate modes\n";
    std::exit(2);
  }
  if (a.pipeline && a.det_every != 1) {
    // Tracking feeds the pose result of frame n into the crop of frame n+1, which a two-stage
    // pipeline cannot do without waiting for the pose thread.
    std::cerr << "--pipeline runs the detector on every frame; drop --det-every\n";
    std::exit(2);
  }
  return a;
}

// Asks the scheduler to treat the calling thread as latency-critical. On macOS a thread that
// sleeps between camera frames is otherwise free to be moved to efficiency cores and slower
// clocks, which roughly doubled every stage at 30 FPS in this pipeline.
void mark_latency_critical(bool enable) {
#ifdef __APPLE__
  if (enable) pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#else
  (void)enable;
#endif
}

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  const auto idx = static_cast<size_t>(p * static_cast<double>(v.size() - 1) + 0.5);
  return v[std::min(idx, v.size() - 1)];
}

// Keypoint box scale used for tracking, as in rtmlib's PoseTracker.
constexpr float kTrackExpansion = 1.25F;

enum Stage { kDecode, kDetPre, kDetInfer, kDetPost, kPosePre, kPoseInfer, kPosePost, kDraw };
constexpr int kStages = 8;
constexpr const char* kStageNames[kStages] = {"decode",   "det_pre",    "det_infer", "det_post",
                                              "pose_pre", "pose_infer", "pose_post", "draw"};

// One frame's measurements. `start` is when decoding began and `done` when drawing ended, so
// done - start is that frame's capture-to-overlay latency. Stages that did not run stay 0.
struct FrameRecord {
  std::array<double, kStages> ms{};
  Clock::time_point start, done;
  bool detected{false};
};

template <typename F>
void timed(double* out, F&& f) {
  const auto s = Clock::now();
  f();
  *out = ms_between(s, Clock::now());
}

class Runner {
 public:
  explicit Runner(const Args& args)
      : args_(args),
        detector_(make_engine(args.det)),
        pose_(make_engine(args.pose)),
        cap_(args.video) {
    if (!cap_.isOpened()) throw std::runtime_error("cannot open " + args.video);
    fps_ = cap_.get(cv::CAP_PROP_FPS) > 0 ? cap_.get(cv::CAP_PROP_FPS) : 30.0;
    if (args.smooth) {
      filters_.assign(2 * smplxrt::kWholebodyKeypoints, smplxrt::OneEuroFilter(1.F, 0.05F));
    }
    if (!args.keypoints.empty()) {
      kp_out_.open(args.keypoints);
      kp_out_ << "frame,keypoint,x,y,score\n";
    }
    records_.reserve(args.warmup + args.frames);
  }

  void run() {
    mark_latency_critical(args_.interactive_qos);
    t0_ = Clock::now();
    if (args_.pipeline) {
      run_pipelined();
    } else if (args_.async_det) {
      run_async_detect();
    } else {
      run_sequential();
    }
  }

  void report() const {
    const int n0 = args_.warmup;
    const int n = static_cast<int>(records_.size()) - n0;
    std::vector<double> stage[kStages], latency;
    int detections = 0;
    for (int i = n0; i < n0 + n; ++i) {
      const auto& r = records_[i];
      for (int s = 0; s < kStages; ++s)
        stage[s].push_back(r.ms[s]);
      latency.push_back(ms_between(r.start, r.done));
      detections += r.detected;
    }
    // Frames completed per second of wall time across the measured window.
    const auto from = n0 > 0 ? records_[n0 - 1].done : records_[0].done;
    const int counted = n0 > 0 ? n : n - 1;
    const double throughput = counted * 1000.0 / ms_between(from, records_.back().done);

    const char* mode = mode_name(args_);
    std::printf(
        "device=%s mode=%s det_every=%d source_fps=%g warmup=%d frames=%d detector_runs=%d\n",
        smplxrt::device_name(args_.device), mode, args_.det_every, args_.source_fps, n0, n,
        detections);
    std::printf("%-11s %9s %9s %9s\n", "stage", "p50_ms", "p99_ms", "mean_ms");
    std::ofstream csv;
    if (!args_.csv.empty()) {
      csv.open(args_.csv);
      csv << "device,mode,det_every,stage,p50_ms,p99_ms,mean_ms\n";
    }
    const auto line = [&](const char* name, const std::vector<double>& v) {
      double mean = 0;
      for (double x : v)
        mean += x;
      mean /= static_cast<double>(v.size());
      const double p50 = percentile(v, 0.50), p99 = percentile(v, 0.99);
      std::printf("%-11s %9.3f %9.3f %9.3f\n", name, p50, p99, mean);
      if (csv.is_open()) {
        csv << smplxrt::device_name(args_.device) << ',' << mode << ',' << args_.det_every << ','
            << name << ',' << p50 << ',' << p99 << ',' << mean << '\n';
      }
    };
    for (int s = 0; s < kStages; ++s)
      line(kStageNames[s], stage[s]);
    line("latency", latency);
    std::printf("throughput_fps=%.1f\n", throughput);
    if (csv.is_open()) {
      csv << smplxrt::device_name(args_.device) << ',' << mode << ',' << args_.det_every
          << ",throughput_fps," << throughput << ",," << '\n';
    }
  }

 private:
  std::unique_ptr<smplxrt::Engine> make_engine(const std::string& path) {
    auto e = smplxrt::Engine::create({path, args_.device, args_.threads});
    if (!e) throw std::runtime_error("built without an inference backend (SMPLXRT_WITH_ORT=OFF)");
    std::cerr << path << ": " << e->describe() << '\n';
    return e;
  }

  // When frame n is due. With --source-fps this waits for it, like a camera would deliver it.
  Clock::time_point arrive(int n) {
    if (args_.source_fps <= 0) return Clock::now();
    const auto due = t0_ + std::chrono::duration_cast<Clock::duration>(
                               std::chrono::duration<double>(n / args_.source_fps));
    if (args_.spin_wait) {
      // Busy-wait keeps the core, and its clock, up between frames.
      while (Clock::now() < due) {
      }
    } else {
      std::this_thread::sleep_until(due);
    }
    return due;
  }

  // Reads the next frame, rewinding at the end so short clips still fill the measured window.
  void decode(cv::Mat* frame) {
    if (!cap_.read(*frame)) {
      cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
      if (!cap_.read(*frame)) throw std::runtime_error("cannot decode " + args_.video);
    }
  }

  smplxrt::Box detect(const cv::Mat& frame, FrameRecord* r) {
    timed(&r->ms[kDetPre], [&] { detector_.preprocess(frame); });
    timed(&r->ms[kDetInfer], [&] { detector_.infer(); });
    smplxrt::Box person;
    timed(&r->ms[kDetPost], [&] {
      person = smplxrt::pick_person(detector_.postprocess(), frame.cols, frame.rows,
                                    smplxrt::PersonDetector::kScoreThreshold);
    });
    r->detected = true;
    return person;
  }

  // Pose, smoothing and drawing for a frame whose person box is known; output writing happens
  // after the frame's clock stops.
  void finish(int n, cv::Mat* frame, const smplxrt::Box& person, FrameRecord* r) {
    timed(&r->ms[kPosePre], [&] { pose_.preprocess(*frame, person); });
    timed(&r->ms[kPoseInfer], [&] { pose_.infer(); });
    timed(&r->ms[kPosePost], [&] {
      pose_.postprocess(&keypoints_);
      if (args_.smooth) {
        const double ts = n / fps_;
        for (size_t k = 0; k < keypoints_.size(); ++k) {
          keypoints_[k].x = filters_[2 * k].update(keypoints_[k].x, ts);
          keypoints_[k].y = filters_[2 * k + 1].update(keypoints_[k].y, ts);
        }
      }
    });
    timed(&r->ms[kDraw], [&] { smplxrt::draw_wholebody(frame, keypoints_, 0.3F); });
    r->done = Clock::now();
    records_.push_back(*r);
    write_outputs(n, *frame);
  }

  void write_outputs(int n, const cv::Mat& frame) {
    if (!args_.out.empty()) {
      if (!writer_.isOpened()) {
        writer_.open(args_.out, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps_, frame.size());
        if (!writer_.isOpened()) throw std::runtime_error("cannot write " + args_.out);
      }
      writer_.write(frame);
    }
    if (kp_out_.is_open()) {
      for (size_t k = 0; k < keypoints_.size(); ++k) {
        kp_out_ << n << ',' << k << ',' << keypoints_[k].x << ',' << keypoints_[k].y << ','
                << keypoints_[k].score << '\n';
      }
    }
  }

  void run_sequential() {
    cv::Mat frame;
    const int total = args_.warmup + args_.frames;
    for (int n = 0; n < total; ++n) {
      FrameRecord r;
      r.start = arrive(n);
      timed(&r.ms[kDecode], [&] { decode(&frame); });
      // Between detector runs, track with a box around the previous frame's keypoints; fall back
      // to the detector whenever too few keypoints are confident.
      smplxrt::Box person;
      const bool tracked = n % args_.det_every != 0 &&
                           smplxrt::box_from_keypoints(keypoints_, 0.3F, 10, frame.cols, frame.rows,
                                                       &person, kTrackExpansion);
      if (!tracked) person = detect(frame, &r);
      finish(n, &frame, person, &r);
    }
  }

  void run_pipelined() {
    // Frames live in a fixed pool so the steady state does not allocate. The ring holds at most
    // kRing - 1 items and the pose thread works on one more, so a pool slot is reused only after
    // the frame that last used it, kPool frames earlier, is finished.
    constexpr std::size_t kRing = 2;
    constexpr int kPool = 8;
    static_assert(kPool > static_cast<int>(kRing) + 1, "pool must outlive in-flight frames");
    struct Item {
      int n{-1};
      smplxrt::Box person;
      FrameRecord record;
    };
    std::array<cv::Mat, kPool> pool;
    smplxrt::SpscRing<Item, kRing> ring;
    std::atomic<bool> failed{false};
    const int total = args_.warmup + args_.frames;

    // Decode and detection on their own thread; cap_ and detector_ are touched only here.
    std::jthread producer([&](std::stop_token stop) {
      mark_latency_critical(args_.interactive_qos);
      try {
        for (int n = 0; n < total && !stop.stop_requested(); ++n) {
          Item item;
          item.n = n;
          item.record.start = arrive(n);
          cv::Mat& frame = pool[n % kPool];
          timed(&item.record.ms[kDecode], [&] { decode(&frame); });
          item.person = detect(frame, &item.record);
          while (!ring.try_push(item)) {
            if (stop.stop_requested()) return;
            std::this_thread::yield();
          }
        }
      } catch (const std::exception& e) {
        std::cerr << "producer: " << e.what() << '\n';
        failed = true;
      }
    });

    // Pose, drawing and output on this thread.
    for (int done = 0; done < total;) {
      auto item = ring.try_pop();
      if (!item) {
        if (failed) throw std::runtime_error("producer thread failed");
        std::this_thread::yield();
        continue;
      }
      finish(item->n, &pool[item->n % kPool], item->person, &item->record);
      ++done;
    }
  }

  void run_async_detect() {
    // Mailbox between this thread and the detector thread. The pose thread writes det_frame only
    // in kIdle, the detector reads it only in kRequest and publishes det_box before kDone, so
    // the acquire/release pairs on `state` order every access to the shared frame and box.
    // Transitions: pose thread Idle->Request and Done->Idle, detector Request->Done, and
    // anything->Stop at shutdown. The detector's Request->Done is a compare-exchange so that a
    // Stop arriving mid-detection is not overwritten, which would leave the join hanging.
    enum : int { kIdle, kRequest, kDone, kStop };
    std::atomic<int> state{kIdle};
    cv::Mat det_frame;
    smplxrt::Box det_box;
    std::atomic<bool> failed{false};

    std::jthread detector_thread([&] {
      mark_latency_critical(args_.interactive_qos);
      try {
        for (;;) {
          int s = state.load(std::memory_order_acquire);
          while (s != kRequest && s != kStop) {
            state.wait(s, std::memory_order_acquire);
            s = state.load(std::memory_order_acquire);
          }
          if (s == kStop) return;
          FrameRecord unused;
          det_box = detect(det_frame, &unused);
          int expected = kRequest;
          if (!state.compare_exchange_strong(expected, kDone, std::memory_order_acq_rel)) return;
          state.notify_all();
        }
      } catch (const std::exception& e) {
        std::cerr << "detector: " << e.what() << '\n';
        failed = true;
        int expected = kRequest;
        state.compare_exchange_strong(expected, kDone, std::memory_order_acq_rel);
        state.notify_all();
      }
    });
    const auto stop = [&] {
      state.store(kStop, std::memory_order_release);
      state.notify_all();
    };
    const auto wait_done = [&] {
      int s = state.load(std::memory_order_acquire);
      while (s == kRequest) {
        state.wait(s, std::memory_order_acquire);
        s = state.load(std::memory_order_acquire);
      }
      if (failed) throw std::runtime_error("detector thread failed");
    };
    const auto submit = [&](const cv::Mat& frame) {
      frame.copyTo(det_frame);  // same size every frame, so this reuses the buffer
      state.store(kRequest, std::memory_order_release);
      state.notify_all();
    };

    try {
      cv::Mat frame;
      const int total = args_.warmup + args_.frames;
      for (int n = 0; n < total; ++n) {
        FrameRecord r;
        r.start = arrive(n);
        timed(&r.ms[kDecode], [&] { decode(&frame); });

        smplxrt::Box person;
        bool detected_this_frame = false;
        const bool tracked = smplxrt::box_from_keypoints(keypoints_, 0.3F, 10, frame.cols,
                                                         frame.rows, &person, kTrackExpansion);
        if (state.load(std::memory_order_acquire) == kDone) {
          if (failed) throw std::runtime_error("detector thread failed");
          // A background result for an earlier frame. Keep tracking unless the detector disagrees
          // with it, which means the track has drifted onto something else.
          if (!tracked || smplxrt::iou(person, det_box) < 0.3F) person = det_box;
          r.detected = true;
          state.store(kIdle, std::memory_order_release);
        } else if (!tracked) {
          // No track (first frame, or the person was lost): this frame has to wait for a fresh
          // detection of itself. The wait is booked as detector inference.
          timed(&r.ms[kDetInfer], [&] {
            wait_done();
            state.store(kIdle, std::memory_order_release);
            submit(frame);
            wait_done();
          });
          person = det_box;
          r.detected = true;
          detected_this_frame = true;
          state.store(kIdle, std::memory_order_release);
        }
        // With --det-every N the background detector is started at most every N frames, so it
        // competes less with pose for the accelerator.
        if (!detected_this_frame && n % args_.det_every == 0 &&
            state.load(std::memory_order_acquire) == kIdle) {
          submit(frame);
        }
        finish(n, &frame, person, &r);
      }
    } catch (...) {
      stop();
      throw;
    }
    stop();
  }

  Args args_;
  Clock::time_point t0_;
  smplxrt::PersonDetector detector_;
  smplxrt::WholebodyPose pose_;
  cv::VideoCapture cap_;
  cv::VideoWriter writer_;
  std::ofstream kp_out_;
  double fps_{30.0};
  std::vector<smplxrt::OneEuroFilter> filters_;
  std::vector<smplxrt::Keypoint> keypoints_;
  std::vector<FrameRecord> records_;
};

}  // namespace

int main(int argc, char** argv) try {
  Runner runner(parse(argc, argv));
  runner.run();
  runner.report();
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << '\n';
  return 1;
}
