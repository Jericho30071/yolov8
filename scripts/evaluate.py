"""Step 5: evaluate the trained model on the held-out test split (Section 3.11.1).

Reports precision, recall, mAP@0.5 and mAP@0.5:0.95 overall and per class, saves the
confusion matrix / PR curves, writes reports/model_metrics.csv, and stores the result
in the evaluation_results table.

  python scripts/evaluate.py --model models/trafficsense_best.pt --split test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from trafficsense.config import load_config
from trafficsense.db import Database, EvaluationResult


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/trafficsense_best.pt")
    ap.add_argument("--data", default="data/dataset/data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    model = YOLO(args.model)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device or None,
                  project=str(ROOT / "runs"), name=f"eval_{args.split}", exist_ok=True, plots=True)
    box = m.box
    rows = [{"class": "all", "precision": box.mp, "recall": box.mr, "mAP50": box.map50, "mAP50-95": box.map}]
    for i, cid in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        rows.append({"class": model.names[int(cid)], "precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap})

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    with open(out / "model_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    print(f"\n{'class':12s} {'P':>7s} {'R':>7s} {'mAP50':>7s} {'mAP50-95':>9s}")
    for r in rows:
        print(f"{r['class']:12s} {r['precision']:7.3f} {r['recall']:7.3f} {r['mAP50']:7.3f} {r['mAP50-95']:9.3f}")
    print(f"\nmetrics -> {out / 'model_metrics.csv'}; plots -> {m.save_dir}")

    Database(load_config().db_path).add(EvaluationResult(
        test_name=f"model {Path(args.model).name} on {args.split} split", mode="detection",
        precision=float(box.mp), recall=float(box.mr), map50=float(box.map50), map50_95=float(box.map),
        notes=f"data={args.data} imgsz={args.imgsz}"))


if __name__ == "__main__":
    main()
