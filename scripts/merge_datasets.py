"""Optional: merge downloaded public YOLO-format datasets (e.g. Kaggle) into TrafficSense classes.

Each source dataset names its classes differently ("Car", "jeep", "motorbike", ...).
Labels are remapped through ALIASES; classes without an alias (pedestrian, bicycle, ...)
are dropped. The merged images + labels land in one folder ready for split_dataset.py
(alongside your own corrected frames).

  python scripts/merge_datasets.py --sources downloads/kaggle_jeepney downloads/kaggle_vehicles \
      --out data/merged
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from trafficsense.config import load_config

ALIASES = {
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "motor": "motorcycle", "scooter": "motorcycle",
    "motorela": "motorela", "tricycle": "motorela", "trike": "motorela", "e-trike": "motorela",
    "tuktuk": "motorela", "tuk-tuk": "motorela", "three-wheeler": "motorela",
    "multicab": "multicab", "multi-cab": "multicab", "minicab": "multicab", "kei truck": "multicab",
    "jeepney": "jeepney", "jeep": "jeepney", "modern jeepney": "jeepney", "e-jeepney": "jeepney",
    "puj": "jeepney", "puv": "jeepney",
    "car": "car", "sedan": "car", "hatchback": "car", "taxi": "car",
    "suv": "suv_van", "suv_van": "suv_van", "van": "suv_van", "minivan": "suv_van", "mpv": "suv_van",
    "pickup": "suv_van", "pickup truck": "suv_van", "uv express": "suv_van", "fx": "suv_van",
    "bus": "bus", "minibus": "bus", "coaster": "bus",
    "truck": "truck", "lorry": "truck", "bongo": "truck", "closed van": "truck", "box truck": "truck",
    "heavy truck": "truck", "dump truck": "truck", "trailer": "truck",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_names(root: Path) -> list[str]:
    for f in [root / "data.yaml", *root.rglob("data.yaml")]:
        if f.exists():
            names = yaml.safe_load(f.read_text())["names"]
            return list(names.values()) if isinstance(names, dict) else list(names)
    if (root / "classes.txt").exists():
        return (root / "classes.txt").read_text().split("\n")
    raise SystemExit(f"{root}: no data.yaml or classes.txt found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="folders of YOLO-format datasets")
    ap.add_argument("--out", default="data/merged")
    args = ap.parse_args()

    classes = load_config().classes
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    for src in map(Path, args.sources):
        names = [n.strip() for n in load_names(src)]
        remap = {i: classes.index(ALIASES[n.lower()]) for i, n in enumerate(names)
                 if ALIASES.get(n.lower()) in classes}
        dropped = [n for i, n in enumerate(names) if i not in remap]
        print(f"{src.name}: {len(remap)} classes kept, dropped {dropped}")
        kept = 0
        for img in (p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT):
            # YOLO layout: .../images/x.jpg <-> .../labels/x.txt
            lbl = Path(str(img.parent).replace("images", "labels")) / f"{img.stem}.txt"
            if not lbl.exists():
                continue
            lines = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5 and int(parts[0]) in remap:  # boxes only (skip polygons)
                    lines.append(" ".join([str(remap[int(parts[0])]), *parts[1:]]))
            if not lines:
                continue
            name = f"{src.name}_{img.stem}"
            shutil.copy2(img, out / "images" / f"{name}{img.suffix}")
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines))
            kept += 1
        print(f"  {kept} images merged")
    (out / "classes.txt").write_text("\n".join(classes))
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
