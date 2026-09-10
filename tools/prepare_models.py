#!/usr/bin/env python3
"""Rewrite the two mmdeploy ONNX exports so CoreML can take each one as a single partition.

    python tools/prepare_models.py models/downloads

1. Detector: cut the graph before its NonMaxSuppression block and keep only the person class.
   The end-to-end export splits into 5 CoreML partitions because NMS, TopK and friends run on
   the CPU; the cut graph is one partition, and NMS moves to C++ (src/pose/geometry.cpp).
   Outputs: boxes [1, 3549, 4] as x1 y1 x2 y2 in input pixels, person_scores [1, 3549, 1].
2. Pose: fix the dynamic batch dimension to 1. CoreML's compiler rejects the unbounded
   dimension and part of the graph falls back to the CPU.

The tensor names 1189 (decoded boxes) and 1191 (class x objectness scores) are specific to
yolox_tiny_8xb8-300e_humanart-6f3252f9; check them with Netron before using another export.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.utils import extract_model


def cut_detector(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp.onnx")
    extract_model(str(src), str(tmp), ["input"], ["1189", "1191"])
    m = onnx.load(str(tmp))
    g = m.graph
    for name, val in [("ps_starts", [0]), ("ps_ends", [1]), ("ps_axes", [2])]:
        g.initializer.append(numpy_helper.from_array(np.array(val, np.int64), name))
    g.node.append(helper.make_node("Slice", ["1191", "ps_starts", "ps_ends", "ps_axes"],
                                   ["person_scores"]))
    g.node.append(helper.make_node("Identity", ["1189"], ["boxes"]))
    del g.output[:]
    g.output.extend([
        helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 3549, 4]),
        helper.make_tensor_value_info("person_scores", TensorProto.FLOAT, [1, 3549, 1]),
    ])
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    tmp.unlink()


def fix_pose(src: Path, dst: Path) -> None:
    run = [sys.executable, "-m", "onnxruntime.tools.make_dynamic_shape_fixed"]
    subprocess.run(run + ["--input_name", "input", "--input_shape", "1,3,256,192",
                          str(src), str(dst)], check=True)
    subprocess.run(run + ["--dim_param", "MatMulsimcc_x_dim_1", "--dim_value", "133",
                          str(dst), str(dst)], check=True)


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "models/downloads")
    cut_detector(d / "yolox_tiny_humanart.onnx", d / "yolox_tiny_humanart_person.onnx")
    fix_pose(d / "rtmw_l_m_256x192.onnx", d / "rtmw_l_m_256x192_static.onnx")
    print("wrote", d / "yolox_tiny_humanart_person.onnx", "and", d / "rtmw_l_m_256x192_static.onnx")


if __name__ == "__main__":
    main()
