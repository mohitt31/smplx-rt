#!/usr/bin/env bash
# Runs the optimization ladder behind docs/RESULTS.md, one configuration at a time with a
# cooldown in between. The development machine is a fanless MacBook Air, which throttles under
# back-to-back runs; without the pause the later rows come out slower than they are.
#
#   bench/run_ladder.sh [build dir] [output dir]
set -euo pipefail

BIN=${1:-build-ort}/pose_video
OUT=${2:-runs/ladder}
COOL=${COOL:-60}
VIDEO=data/jumping_jacks.mp4
M=models/downloads
mkdir -p "$OUT"

run() {
  local name=$1
  shift
  sleep "$COOL"
  echo "== $name"
  "$BIN" --video "$VIDEO" --warmup 100 --frames 1000 --csv "$OUT/$name.csv" "$@" 2>/dev/null |
    grep -E '^(device|det_infer|pose_infer|latency|throughput)'
}

ORIG=(--det $M/yolox_tiny_humanart.onnx --pose $M/rtmw_l_m_256x192.onnx)
PREP=(--det $M/yolox_tiny_humanart_person.onnx --pose $M/rtmw_l_m_256x192_static.onnx)

# Frames read as fast as the pipeline takes them.
run cpu_original          "${ORIG[@]}" --device cpu
run coreml_original       "${ORIG[@]}" --device coreml
run coreml_prepared       "${PREP[@]}" --device coreml
run coreml_det_every3     "${PREP[@]}" --device coreml --det-every 3
run coreml_pipeline       "${PREP[@]}" --device coreml --pipeline
run coreml_async_det      "${PREP[@]}" --device coreml --async-det

# A 60 FPS camera: frames are due every 16.7 ms and latency counts from when a frame was due.
run cam60_sequential      "${PREP[@]}" --device coreml --source-fps 60 --spin
run cam60_pipeline        "${PREP[@]}" --device coreml --source-fps 60 --spin --pipeline
run cam60_async_det       "${PREP[@]}" --device coreml --source-fps 60 --spin --async-det
run cam60_async_det4      "${PREP[@]}" --device coreml --source-fps 60 --spin --async-det --det-every 4
# Same as cam60_async_det but sleeping between frames instead of spinning.
run cam60_async_det_sleep "${PREP[@]}" --device coreml --source-fps 60 --async-det
