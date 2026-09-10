# smplx-rt

Real-time whole-body human pose in C++: camera or video -> person detector -> 133-keypoint
whole-body pose (body, feet, face, hands) -> overlay, with per-stage p50/p99 latency. It is the
first stage of a monocular SMPL-X body-mesh pipeline aimed at 60 FPS on an NVIDIA RTX GPU.

On an Apple M4 MacBook Air, against a simulated 60 FPS camera, it keeps up at 60 FPS with
8.8 ms p50 and 16.8 ms p99 capture-to-overlay latency.

![whole-body keypoints on a jumping-jacks clip](docs/demo.gif)

## Status

| Part | State |
|---|---|
| C++ pipeline: OpenCV decode, YOLOX-tiny detector, RTMW-l-m whole-body pose, SimCC decode, overlay | working, ONNX Runtime on CPU and CoreML |
| Graph surgery so both models run as one CoreML partition, NMS moved to C++ | working, `tools/prepare_models.py` |
| Sequential, detect-every-N, two-thread pipeline, and async background detector modes | working, `--det-every`, `--pipeline`, `--async-det` |
| Camera-paced benchmarking with latency measured from when each frame was due | working, `--source-fps` |
| Parity with the Python reference (rtmlib) on the same frames | mean 0.001 px |
| One Euro keypoint smoothing (`--smooth`), 6D rotation filter, bone-length constraint | working and unit-tested; the last two are for the SMPL-X stage |
| SMPL-X regressor and full SMPL-X forward pass | not started; only the sparse LBS step exists (CPU) |
| CUDA preprocess and LBS kernels, TensorRT | written, never compiled or run (no NVIDIA GPU here) |

## Build and run (macOS, Apple Silicon)

Needs an unpacked [ONNX Runtime release](https://github.com/microsoft/onnxruntime/releases)
(1.30.0 here, it includes the CoreML EP) and OpenCV 4 with core, imgproc, imgcodecs, videoio.

```bash
cmake -S . -B build -G Ninja -DSMPLXRT_WITH_ORT=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-osx-arm64-1.30.0 \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
cmake --build build && ctest --test-dir build

python -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/prepare_models.py models/downloads

./build/pose_video --video data/jumping_jacks.mp4 \
  --det models/downloads/yolox_tiny_humanart_person.onnx \
  --pose models/downloads/rtmw_l_m_256x192_static.onnx \
  --device coreml --source-fps 60 --spin --async-det --det-every 4 --out runs/out.mp4
```

`bench/run_ladder.sh` reproduces every row below. Without `-DSMPLXRT_WITH_ORT=ON` only the
dependency-free core and its tests build (this is what CI runs).

Models are the OpenMMLab mmdeploy ONNX exports that rtmlib uses in its lightweight whole-body mode:
[YOLOX-tiny HumanArt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip)
and [RTMW-l-m 256x192](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip).
Unzip `end2end.onnx` from each into `models/downloads/` as `yolox_tiny_humanart.onnx` and
`rtmw_l_m_256x192.onnx` (gitignored), then run `tools/prepare_models.py`.

Test clip: [Jumping jacks and burpees](https://commons.wikimedia.org/wiki/File:Jumping_jacks_and_burpees.webm)
by Taco fleur, CC BY-SA 4.0, converted to H.264, 640x480, 1356 frames.

## Results (Apple M4, 10-core, fanless MacBook Air, macOS 26.5, on mains power)

Every row: 100 warm-up frames, then 1,000 measured frames, one run, 60 s cooldown before it
(`bench/run_ladder.sh`). Latency is from when decoding of a frame started (or, with a paced
source, when the frame was due) to when its overlay was drawn. Throughput is completed frames
per second of wall time. Output writing is outside the timed window.

### Frames as fast as the pipeline takes them

| # | Configuration | detector p50 | pose p50 | latency p50 | latency p99 | FPS |
|---|---|---:|---:|---:|---:|---:|
| 0 | CPU, original models | 16.6 | 35.0 | 52.8 | 60.9 | 18.9 |
| 1 | CoreML, original models | 15.4 | 19.8 | 38.3 | 59.8 | 24.9 |
| 2 | CoreML, prepared models | 8.1 | 11.9 | 21.4 | 25.3 | 46.7 |
| 3 | + detector every 3rd frame, track in between | 0 (p99 12.8) | 9.7 | 13.6 | 32.7 | 68.6 |
| 4 | two-thread pipeline instead | 7.3 | 10.7 | 33.2 | 34.9 | 90.2 |
| 5 | async background detector instead | off path | 11.1 | **12.0** | **14.8** | 82.0 |

### A 60 FPS camera (a frame due every 16.7 ms)

| Configuration | latency p50 | latency p99 | FPS |
|---|---:|---:|---:|
| sequential, detector every frame | 6958 | 12138 | 36.2 (falls behind) |
| two-thread pipeline | 16.7 | 18.0 | 60.0 |
| async detector on every frame | 19.6 | 404.6 | 58.9 |
| async detector every 4th frame | **8.8** | **16.8** | **60.0** |

What each step taught me:

- **Rows 1 to 2, graph surgery (38 to 21 ms).** Under CoreML the detector split into 5 partitions:
  the NMS block of the mmdeploy export (NonMaxSuppression, TopK and friends) runs on the CPU, and
  every frame crossed between CPU and CoreML several times. The pose model has a dynamic batch
  dimension, which CoreML's compiler rejects ("unbounded dimension"), so part of it fell back.
  `tools/prepare_models.py` cuts the detector before NMS and keeps only the person class (NMS is
  now 20 lines of C++ in `src/pose/geometry.cpp`), and fixes the pose input to 1x3x256x192. Both
  models are now one CoreML partition each, and keypoints are unchanged (mean 0.00008 px against
  the original models).
- **Row 3, detect less.** Tracking the person with a box around the previous frame's keypoints
  (scaled 1.25, as rtmlib's tracker does) removes the detector from two frames in three. The p50
  drops, but the p99 stays at 32.7 ms because every third frame still pays for detection.
- **Row 4, overlap.** Decode and detection on one thread, pose on another, handing frames over a
  lock-free SPSC ring from a fixed pool of frames. Throughput goes up to 90 FPS, and latency goes
  up to 33 ms: an unthrottled producer keeps the ring full, so every frame waits for the one ahead
  of it. More throughput is not lower latency.
- **Row 5, take the detector off the critical path.** The pose thread tracks every frame; a
  detector thread re-checks the latest frame in the background through a single-slot mailbox
  (one atomic state word with C++20 `wait`/`notify`), and its box replaces the track only when the
  track is lost or disagrees with it (IoU below 0.3). Best latency and best tail: 12.0 / 14.8 ms.
- **Camera pacing changes everything.** With frames arriving every 16.7 ms there is idle time
  between frames. When the thread slept through it, every stage got slower, even plain CPU work
  (decode 0.19 ms unpaced vs 0.59 ms at 30 FPS), and pose inference roughly doubled at 30 FPS.
  Raising the thread QoS to user-interactive barely helped, which points at clock scaling rather
  than core placement. Busy-waiting for the next frame (`--spin`) keeps the clocks up; the paced
  async run with sleeping instead of spinning came out at 225 ms p50 in the ladder.
- **Contention on the accelerator.** At 60 FPS, running the background detector on every frame
  slows pose down enough to push some frames past 16.7 ms; with no frame dropping the backlog never
  clears (404 ms p99). Re-detecting every 4th frame (`--det-every 4`) removes that and holds 60 FPS
  at 8.8 / 16.8 ms.

Caveats: a MacBook Air has no fan, so back-to-back runs throttle (the script's cooldown exists for
that). The camera row with the detector on every frame is unstable: repeated runs gave 12 ms
and 284 ms p50, depending on whether a backlog started. The async every-4th-frame row is the one
to quote, and it too sits right at the budget at p99. Tracking changes the pose
crop, so tracked keypoints are not identical to detecting every frame: against detect-every-frame
the difference is 1.3 px median and 3.5 px mean (5.0 px on hands). There is no ground truth on this
clip, so that is a difference, not an error measurement. The pipeline does not drop late frames
yet, which is why the rows that fall behind show latency in seconds.

## Correctness: parity with rtmlib

`tools/reference_keypoints.py` runs rtmlib (Python) on the same video with the same ONNX files
and compares keypoints above score 0.3 (200 frames, about 25k keypoints):

| Setup | mean | p99 |
|---|---:|---:|
| Same mp4, C++ decodes with AVFoundation, Python with FFmpeg, OpenCV 4.12 vs 5.0 | 0.67 px | 3.74 px |
| Same PNG frames, OpenCV 4.12 vs 5.0 | 0.33 px | 1.31 px |
| Same PNG frames, both OpenCV 4.12 | **0.001 px** | **0.001 px** |

So the remaining gap was the video decoder and the OpenCV version, not the C++ pre/post-processing.

One real discrepancy found on the way: the model's own mmdeploy `pipeline.json` says `to_rgb: True`
for the pose model, but rtmlib feeds it a BGR image with RGB mean/std. This project follows the deploy
config. Reproducing rtmlib's behaviour (`--rtmlib-bgr`) doubles the disagreement (1.38 px vs 0.67 px mean).

## Next

1. Drop late frames instead of queueing them, and report drops next to latency.
2. SMPL-X: pick a regressor with a clean ONNX export, verify PyTorch vs ONNX Runtime, implement
   the full forward pass (blendshapes, Rodrigues, kinematic chain) with a parity test against the
   official `smplx` package.
3. NVIDIA: TensorRT FP16 engines, compile and validate the CUDA kernels against the CPU references,
   then fill the optimization ladder in [docs/RESULTS.md](docs/RESULTS.md).
   The planned GPU design is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

SMPL-X model files are licensed by MPI and are never committed; see
[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/).
