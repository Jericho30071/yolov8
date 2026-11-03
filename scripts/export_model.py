"""Step 6: export the trained model for the Raspberry Pi 4.

NCNN is the fastest CPU format on the Pi 4 (roughly 3–4x faster than PyTorch for YOLOv8n).
A smaller input size (320–416) trades a little accuracy for speed; presence detection
at the stop line does not need full resolution.

  python scripts/export_model.py --model models/trafficsense_best.pt --format ncnn --imgsz 416
Then on the Pi set  model.weights: models/trafficsense_best_ncnn_model  and  model.imgsz: 416
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/trafficsense_best.pt")
    ap.add_argument("--format", default="ncnn", choices=["ncnn", "onnx", "openvino", "tflite"])
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--half", action="store_true", help="FP16 weights (smaller file)")
    args = ap.parse_args()
    path = YOLO(args.model).export(format=args.format, imgsz=args.imgsz, half=args.half)
    print(f"exported -> {path}")


if __name__ == "__main__":
    main()
