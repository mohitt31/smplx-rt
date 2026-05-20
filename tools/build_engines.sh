#!/usr/bin/env bash
set -euo pipefail
: "${ONNX_MODEL:?Set ONNX_MODEL to a verified ONNX model}"
: "${ENGINE_OUT:?Set ENGINE_OUT to the target TensorRT plan path}"
trtexec --onnx="$ONNX_MODEL" --saveEngine="$ENGINE_OUT" --fp16 --workspace=4096
echo "Built GPU-specific FP16 engine: $ENGINE_OUT"
