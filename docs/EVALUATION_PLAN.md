# RTX validation protocol

This is the evidence required before claiming the 60 FPS / sub-16.6 ms target.

## Reproducible run

1. Pin the git commit, ONNX SHA-256, TensorRT version, CUDA version, driver, GPU, clocks, power limit, input-video SHA-256, and camera mode.
2. Build each engine on that same GPU. TensorRT plans are not portable across arbitrary GPU architectures.
3. Warm up exactly 100 frames; record the next 1,000 frames. Do not include initialization, engine build, model load, or first inference in frame-time statistics.
4. Measure CUDA stages with CUDA events on their owning streams. Measure capture-to-present latency with `steady_clock` plus presentation/fence timestamps.
5. Report p50 and p99 for each stage, p50/p99 end-to-end latency, and sustained FPS separately.

## Required ablations

| Row | Change | Evidence |
|---|---|---|
| 0 | PyTorch baseline | same model and video |
| 1 | ONNX Runtime CUDA EP | model-output parity + timings |
| 2 | TensorRT FP16 | model-output parity + timings |
| 3 | fused preprocess | pixel parity against CPU/OpenCV reference |
| 4 | custom CUDA LBS | official SMPL-X vertex parity |
| 5 | stream overlap | throughput **and** capture-to-present latency |
| 6 | INT8 | calibration set, accuracy delta, and timings |

## Acceptance gates

- No `cudaDeviceSynchronize` in the frame loop.
- LBS error < `1e-4` maximum absolute vertex error across fixed random parameter dumps.
- Quantization report identifies affected layers and retains higher precision where the accuracy budget requires it.
- p99 end-to-end latency <= 16.6 ms on the declared target configuration.

Until every gate is evidenced in [RESULTS.md](RESULTS.md), describe the project as an **in-progress systems implementation**, not a 60 FPS deployment.
