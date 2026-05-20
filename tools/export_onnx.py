#!/usr/bin/env python3
"""Export a selected PyTorch regressor and record PyTorch/ONNX Runtime parity.

Usage: python tools/export_onnx.py --checkpoint model.pt --output models/regressor.onnx
The model-specific import is deliberately a required argument: select an exporter that passes Gate 0.
"""
import argparse, hashlib
from pathlib import Path
parser=argparse.ArgumentParser(); parser.add_argument('--checkpoint',required=True); parser.add_argument('--output',required=True); parser.add_argument('--opset',type=int,default=17)
args=parser.parse_args(); source=Path(args.checkpoint); output=Path(args.output)
if not source.exists(): raise SystemExit(f'checkpoint not found: {source}')
raise SystemExit('Exporter scaffold: add the chosen regressor adapter only after its PyTorch/ONNX parity contract is defined. Do not claim an export without logged max-abs-diff.')
