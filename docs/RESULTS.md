# Results

Never enter estimated results here. Benchmark after 100 warm-up frames and 1,000 measured frames, using CUDA events for GPU stages.

| # | Change | p50 ms | p99 ms | FPS | Accuracy |
|---|---|---:|---:|---:|---|
| 0 | Python / PyTorch baseline | | | | |
| 1 | C++ + ONNX Runtime CUDA EP | | | | |
| 2 | TensorRT FP16 | | | | |
| 3 | GPU preprocessing kernel | | | | |
| 4 | Custom CUDA LBS vs CPU LBS | | | | |
| 5 | Async multi-stream pipeline | | | | |
| 6 | INT8 | | | | |

## Run record

| Field | Value |
|---|---|
| GPU / clocks / power mode | |
| NVIDIA driver | |
| CUDA / TensorRT | |
| Input video SHA-256 | |
| Model / commit | |
| Precision | |
