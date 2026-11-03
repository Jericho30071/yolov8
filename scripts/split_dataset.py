"""Step 3: split corrected YOLO labels into train / val / test (70 / 20 / 10, Section 3.8).

--group-by-video keeps every frame of one video in the same split, so near-identical
frames do not leak from training into validation/test (gives honest mAP numbers;
needs several videos).

  python scripts/split_dataset.py --images data/autolabel/images --labels data/autolabel/labels
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from trafficsense.config import load_config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/autolabel/images")
    ap.add_argument("--labels", default="data/autolabel/labels")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    ap.add_argument("--group-by-video", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    classes = load_config().classes
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXT)
    labels = Path(args.labels)
    pairs = [(img, labels / f"{img.stem}.txt") for img in images]
    missing = [img.name for img, lbl in pairs if not lbl.exists()]
    if missing:
        print(f"warning: {len(missing)} images have no label file and are treated as background")

    rng = random.Random(args.seed)
    if args.group_by_video:
        # frames from extract_frames.py are named <video>_<frameindex>.jpg
        groups = defaultdict(list)
        for pair in pairs:
            groups[re.sub(r"_\d+$", "", pair[0].stem)].append(pair)
        keys = sorted(groups)
        rng.shuffle(keys)
        units = [groups[k] for k in keys]
        if len(units) < 3:
            print("warning: fewer than 3 videos — some splits will be empty; drop --group-by-video")
    else:
        rng.shuffle(pairs)
        units = [[p] for p in pairs]

    total = sum(len(u) for u in units)
    targets = [r * total for r in args.ratios]
    splits = {"train": [], "val": [], "test": []}
    names = list(splits)
    for unit in units:  # greedy fill toward each split's target size
        name = max(names, key=lambda n: targets[names.index(n)] - len(splits[n]))
        splits[name].extend(unit)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for split, items in splits.items():
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        per_class = [0] * len(classes)
        for img, lbl in items:
            shutil.copy2(img, out / split / "images" / img.name)
            text = lbl.read_text() if lbl.exists() else ""
            (out / split / "labels" / lbl.name).write_text(text)
            for line in text.splitlines():
                if line.strip():
                    per_class[int(line.split()[0])] += 1
        counts[split] = (len(items), dict(zip(classes, per_class)))

    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out.resolve()), "train": "train/images", "val": "val/images", "test": "test/images",
        "names": dict(enumerate(classes))}, sort_keys=False))
    for split, (n, pc) in counts.items():
        print(f"{split:5s}: {n:5d} images  {pc}")
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
