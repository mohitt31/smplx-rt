#!/usr/bin/env python3
"""Run the rtmlib Python reference on the same video and compare with pose_video's keypoints.

    python tools/reference_keypoints.py --video data/jumping_jacks.mp4 \
        --det models/downloads/yolox_tiny_humanart.onnx \
        --pose models/downloads/rtmw_l_m_256x192.onnx \
        --cpp cpp_keypoints.csv --frames 200

Both sides use onnxruntime on the CPU with the same ONNX files, so differences come from
pre/post-processing.

rtmlib feeds the pose model a BGR image with RGB mean/std, while the model's own mmdeploy
pipeline.json says to_rgb=True. By default this script passes an RGB frame to the pose model so the
reference matches the deploy config; --rtmlib-bgr reproduces rtmlib's default behaviour instead.
Also reports the Python pipeline's per-frame latency as a baseline.
"""
import argparse
import csv
import time
from collections import defaultdict

import cv2
import numpy as np
from rtmlib import YOLOX, RTMPose


def load_cpp(path):
    kps = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            kps[int(row["frame"])][int(row["keypoint"])] = (
                float(row["x"]), float(row["y"]), float(row["score"]))
    return kps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--cpp", help="keypoints CSV written by pose_video --keypoints")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--rtmlib-bgr", action="store_true",
                    help="feed BGR to the pose model like rtmlib does by default")
    args = ap.parse_args()

    det = YOLOX(args.det, model_input_size=(416, 416), backend="onnxruntime", device="cpu")
    pose = RTMPose(args.pose, model_input_size=(192, 256), backend="onnxruntime", device="cpu")
    cpp = load_cpp(args.cpp) if args.cpp else None

    cap = cv2.VideoCapture(args.video)
    times, errors, missing = [], [], 0
    for n in range(args.warmup + args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        boxes = det(frame)
        if len(boxes):
            # Same rule as the C++ side: one person, the first (highest-scoring) box.
            boxes = boxes[:1]
        pose_input = frame if args.rtmlib_bgr else np.ascontiguousarray(frame[..., ::-1])
        kps, scores = pose(pose_input, bboxes=boxes)
        dt = (time.perf_counter() - t0) * 1000.0
        if n < args.warmup:
            continue
        times.append(dt)
        if cpp is None:
            continue
        if n not in cpp:
            missing += 1
            continue
        for k in range(kps.shape[1]):
            if scores[0, k] <= args.score:
                continue
            x, y, s = cpp[n][k]
            if s <= args.score:
                continue
            errors.append(np.hypot(x - kps[0, k, 0], y - kps[0, k, 1]))

    t = np.array(times)
    print(f"python rtmlib cpu: frames={len(t)} p50={np.percentile(t, 50):.2f} ms "
          f"p99={np.percentile(t, 99):.2f} ms")
    if cpp is not None:
        e = np.array(errors)
        print(f"keypoint agreement vs C++ ({len(e)} keypoints above score {args.score}, "
              f"{missing} frames missing): mean={e.mean():.3f} px p99={np.percentile(e, 99):.3f} px "
              f"max={e.max():.3f} px")


if __name__ == "__main__":
    main()
