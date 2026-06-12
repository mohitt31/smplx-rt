# smplx-rt

An honest, performance-oriented scaffold for real-time single-person monocular SMPL-X reconstruction. The target is a 10,475-vertex SMPL-X mesh overlay at 720p and 60 FPS on an RTX GPU—measured, not assumed.

> Status: Gate 0 / core LLD. The portable C++ reference and runtime contracts work now; detector, RTMPose, chosen SMPL-X regressor, TensorRT bindings, and the full CUDA parity path remain explicit milestones. No RTX performance claim is made yet.

## Pipeline

```text
OpenCV capture -> fused CUDA letterbox/preprocess -> detector (TensorRT)
  -> RTMPose wholebody (TensorRT) -> SMPL-X regressor -> custom LBS -> One Euro -> overlay/Vulkan
```

The repository is deliberately backend-neutral: macOS can exercise the CPU/reference and ONNX Runtime/CoreML route; CUDA/TensorRT is enabled only on an NVIDIA benchmark system.

## Fast start

```bash
cmake -S . -B build -DSMPLXRT_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/smplxrt_demo
./build/bench_stages
```

Optional CUDA build:

```bash
cmake -S . -B build-cuda -DSMPLXRT_WITH_CUDA=ON -DSMPLXRT_WITH_TRT=ON
cmake --build build-cuda --parallel
```

`SMPLXRT_WITH_TRT` defines the integration surface only today—enable it after adding the supported TensorRT SDK and selected model bindings. TensorRT plans must be rebuilt on each target GPU.

## What is already protected

- CPU fused-reference semantics for letterbox, normalization, and HWC-to-CHW output.
- Sparse top-4 LBS reference implementation and a deterministic sanity test.
- One Euro scalar filter with increasing-timestamp validation.
- CUDA kernel entry points for fused preprocess and sparse LBS, guarded behind `SMPLXRT_WITH_CUDA`.
- Reproducible CI for the no-GPU reference layer and a results template that forbids guessed figures.
- A bounded lock-free SPSC handoff, triple-buffer scheduling contract, and an explicit non-owning GPU-frame token—no hidden heap allocation in those control paths.
- 6D rotation filtering followed by SO(3) projection, plus a rest-length bone-constraint pass.
- A CUDA–Vulkan external-memory capability boundary that can be enabled and validated on supported Linux/NVIDIA systems.

## Gate 0: do this before model integration

Choose a pretrained SMPL-X regressor with legal weights and a clean ONNX opset-17 export. Verify PyTorch versus ONNX Runtime on a fixed input batch and record max absolute difference. If export/parity fails, replace the regressor before building bindings or pipeline logic.

## Model assets and licensing

SMPL-X model files are not included and must never be committed. Obtain them after accepting the [SMPL-X license](https://smpl-x.is.tue.mpg.de/), place them outside the repository or in ignored `models/downloads/`, and keep a checksum in experiment notes. `tools/download_models.sh` intentionally only explains this process.

## Project map

| Area | Purpose |
|---|---|
| `kernels/` | Custom CUDA fused preprocess and sparse LBS kernels |
| `src/preprocess/` | Portable correctness reference |
| `src/engine/` | Backend interface: Stub now, ORT/TensorRT adapters next |
| `src/smplx/` | CPU sparse skinning oracle used for CUDA parity |
| `src/filter/` | One Euro temporal filtering |
| `tests/` | Reference and future official-SMPL-X parity contracts |
| `bench/` | Stage timing entry point |
| `docs/RESULTS.md` | Optimization-ladder run records |

## Definition of done

1. CUDA LBS output matches official Python `smplx` dumps to max absolute error below `1e-4` across fixed random parameters.
2. A detector, whole-body pose model, and SMPL-X regressor each have verified model/ONNX/TensorRT parity.
3. The hot path uses CUDA events and stream dependencies—no `cudaDeviceSynchronize`.
4. `docs/RESULTS.md` contains p50/p99, hardware, accuracy, input hash, and precision for 1,000 post-warmup frames.

See [docs/RESULTS.md](docs/RESULTS.md) for the intentionally blank optimization ladder.

## Engineering evidence

- [Architecture and ownership model](docs/ARCHITECTURE.md)
- [RTX validation protocol](docs/EVALUATION_PLAN.md)
- [Résumé-safe project wording](docs/RESUME.md)
