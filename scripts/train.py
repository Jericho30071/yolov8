"""Step 4: fine-tune YOLOv8 from COCO weights on the TrafficSense dataset (Section 3.8).

Defaults are sized for a 4 GB GPU (GTX 960): YOLOv8n/s, batch 8, 640 px. Augmentation
follows the paper: small rotation, brightness (HSV value) changes, horizontal flips.

  python scripts/train.py --data data/dataset/data.yaml --model yolov8s.pt --epochs 100
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt (Pi-friendly) or yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="lower to 4 if CUDA runs out of memory")
    ap.add_argument("--device", default="", help="'' auto, 'cpu', or '0'")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="trafficsense")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device or None, workers=args.workers, project=str(ROOT / "runs"), name=args.name,
        exist_ok=True, resume=args.resume, patience=25, seed=42, plots=True,
        # augmentation (Section 3.8): rotation, brightness, horizontal flip + YOLO defaults (mosaic etc.)
        degrees=5.0, hsv_v=0.4, hsv_s=0.6, fliplr=0.5, translate=0.1, scale=0.5,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    target = ROOT / "models" / "trafficsense_best.pt"
    shutil.copy2(best, target)
    print(f"\nbest weights copied to {target}")
    print("Set model.weights in config.yaml to models/trafficsense_best.pt, then run scripts/evaluate.py")


if __name__ == "__main__":
    main()
