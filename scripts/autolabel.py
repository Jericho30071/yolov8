"""Step 2: pre-label extracted frames so annotators only correct boxes instead of drawing them.

Round 1 (no custom model yet): a large COCO model draws the boxes, which is most of the
work. COCO only knows car / motorcycle / bus / truck, so each box gets a best-guess local
class (COCO_TO_LOCAL) and review.csv lists what to check:
  motorcycle -> motorcycle   check: is it a motorela?
  car        -> car          check: suv_van or multicab?
  truck      -> jeepney      check: jeepney, multicab or truck?
  bus        -> bus
Later rounds: pass your own trained model with --model models/trafficsense_best.pt; it
predicts all 8 classes directly, so each round needs fewer corrections.

Writes a YOLO dataset folder (images/, labels/, classes.txt, data.yaml) plus review.csv
listing the frames that most need a human check. Correct the boxes locally with
scripts/label_tool.py (no online service needed).

  python scripts/autolabel.py --images data/frames --out data/autolabel
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from ultralytics import YOLO

from trafficsense.config import load_config

# COCO class -> best-guess local class for pre-labeling (bicycle and others are ignored),
# and what the annotator must check for that guess.
COCO_TO_LOCAL = {"car": "car", "motorcycle": "motorcycle", "bus": "bus", "truck": "jeepney"}
COCO_CHECK = {"motorcycle": "motorcycle: motorela?", "car": "car: suv_van / multicab?",
              "truck": "truck->jeepney: jeepney / multicab / truck?"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/frames")
    ap.add_argument("--out", default="data/autolabel")
    ap.add_argument("--model", default="yolov8l.pt", help="COCO model (round 1) or your trained model")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="")
    ap.add_argument("--review-conf", type=float, default=0.5, help="flag frames with boxes below this")
    args = ap.parse_args()

    classes = load_config().classes
    images = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"no images in {args.images}")
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    is_coco = "person" in model.names.values()
    id_map = {}
    for cid, name in model.names.items():
        target = COCO_TO_LOCAL.get(name) if is_coco else name
        if target in classes:
            id_map[cid] = classes.index(target)
    print(f"model {args.model}: mapping {[(model.names[k], classes[v]) for k, v in id_map.items()]}")

    review = []
    totals = {c: 0 for c in classes}
    for i, path in enumerate(images, 1):
        r = model.predict(str(path), conf=args.conf, imgsz=args.imgsz, device=args.device or None,
                          classes=list(id_map), verbose=False)[0]
        lines, reasons = [], set()
        for box, cid, conf in zip(r.boxes.xywhn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            target = id_map[int(cid)]
            lines.append(f"{target} " + " ".join(f"{v:.6f}" for v in box))
            totals[classes[target]] += 1
            if conf < args.review_conf:
                reasons.add("low confidence")
            if is_coco and model.names[int(cid)] in COCO_CHECK:
                reasons.add(COCO_CHECK[model.names[int(cid)]])
        if not lines:
            reasons.add("no vehicles (keep a few as negatives)")
        shutil.copy2(path, out / "images" / path.name)
        (out / "labels" / f"{path.stem}.txt").write_text("\n".join(lines))
        if reasons:
            review.append({"image": path.name, "boxes": len(lines), "check": "; ".join(sorted(reasons))})
        if i % 50 == 0 or i == len(images):
            print(f"  {i}/{len(images)}")

    (out / "classes.txt").write_text("\n".join(classes))
    (out / "data.yaml").write_text(yaml.safe_dump({"path": str(out.resolve()), "train": "images",
                                                   "val": "images", "names": dict(enumerate(classes))},
                                                  sort_keys=False))
    with open(out / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "boxes", "check"])
        w.writeheader()
        w.writerows(review)
    print(f"labels: {totals}")
    print(f"{len(review)} of {len(images)} frames flagged in {out / 'review.csv'}")
    print(f"Next: python scripts/label_tool.py --dataset {out} --review   (set the local classes, add missed vehicles),")
    print("then run scripts/split_dataset.py.")


if __name__ == "__main__":
    main()
