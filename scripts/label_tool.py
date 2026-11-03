"""Local bounding-box editor for correcting YOLOv8 auto-labels (no online service needed).

  python scripts/label_tool.py --dataset data/autolabel            # every image
  python scripts/label_tool.py --dataset data/autolabel --review   # only frames flagged in review.csv

Mouse
  left-drag          draw a new box with the current class
  left-click a box   select it (selected box is drawn thick)
  right-click a box  delete it
Keys
  1-8                set current class (see legend); also changes the selected box's class
  x / Delete         delete selected box
  d / Space / Right  save and go to next image
  a / Left           save and go to previous image
  c                  clear all boxes in this image (true negative frame)
  q / Esc            save and quit (progress is remembered in .label_progress)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trafficsense.config import load_config  # noqa: E402

# one BGR color per class: motorcycle, motorela, multicab, jeepney, car, suv_van, bus, truck
COLORS = [(0, 200, 255), (0, 120, 255), (200, 60, 200), (60, 200, 60), (255, 160, 60), (255, 255, 0),
          (60, 60, 230), (160, 160, 160)]
KEY_RIGHT, KEY_LEFT, KEY_DELETE = 2555904, 2424832, 3014656   # cv2.waitKeyEx codes on Windows
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_W, MAX_H = 1280, 720


class Editor:
    def __init__(self, images: list[Path], labels_dir: Path, classes: list[str], start: int):
        self.images, self.labels_dir, self.classes = images, labels_dir, classes
        self.i = start
        self.cls = 0
        self.boxes: list[list[float]] = []   # [cls, x1, y1, x2, y2] in original pixels
        self.selected: int | None = None
        self.drag_start = None
        self.drag_now = None

    # ------------------------------------------------------------- file io
    def label_path(self) -> Path:
        return self.labels_dir / f"{self.images[self.i].stem}.txt"

    def load(self) -> None:
        self.img = cv2.imread(str(self.images[self.i]))
        h, w = self.img.shape[:2]
        self.scale = min(MAX_W / w, MAX_H / h, 1.0)
        self.boxes, self.selected = [], None
        if self.label_path().exists():
            for line in self.label_path().read_text().splitlines():
                p = line.split()
                if len(p) == 5:
                    c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:])
                    self.boxes.append([c, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])

    def save(self) -> None:
        h, w = self.img.shape[:2]
        lines = []
        for c, x1, y1, x2, y2 in self.boxes:
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        self.label_path().write_text("\n".join(lines))

    # ---------------------------------------------------------------- mouse
    def box_at(self, x: float, y: float) -> int | None:
        hits = [(abs((b[3] - b[1]) * (b[4] - b[2])), i) for i, b in enumerate(self.boxes)
                if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
        return min(hits)[1] if hits else None   # smallest box wins when boxes overlap

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        ox, oy = x / self.scale, y / self.scale
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start, self.drag_now = (ox, oy), (ox, oy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_now = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            (x1, y1), (x2, y2) = self.drag_start, (ox, oy)
            if abs(x2 - x1) * self.scale > 5 and abs(y2 - y1) * self.scale > 5:
                self.boxes.append([self.cls, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                self.selected = len(self.boxes) - 1
            else:
                self.selected = self.box_at(ox, oy)
            self.drag_start = self.drag_now = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = self.box_at(ox, oy)
            if idx is not None:
                self.boxes.pop(idx)
                self.selected = None

    # ----------------------------------------------------------------- draw
    def render(self):
        view = cv2.resize(self.img, None, fx=self.scale, fy=self.scale) if self.scale < 1 else self.img.copy()
        s = self.scale
        for i, (c, x1, y1, x2, y2) in enumerate(self.boxes):
            color = COLORS[c % len(COLORS)]
            cv2.rectangle(view, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), color,
                          3 if i == self.selected else 1)
            cv2.putText(view, self.classes[c], (int(x1 * s), max(12, int(y1 * s) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if self.drag_start and self.drag_now:
            (x1, y1), (x2, y2) = self.drag_start, self.drag_now
            cv2.rectangle(view, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), COLORS[self.cls], 1)
        status = [(f"{self.i + 1}/{len(self.images)}  boxes:{len(self.boxes)}", (255, 255, 255)),
                  (f"drawing: [{self.cls + 1}] {self.classes[self.cls]}", COLORS[self.cls % len(COLORS)])]
        # legend: one class per line in its box color, so it fits narrow (portrait) videos too
        legend = [(f"[{k + 1}] {n}", COLORS[k % len(COLORS)]) for k, n in enumerate(self.classes)]
        for row, (text, color) in enumerate(status + legend):
            y = 20 + row * 19 + (6 if row >= len(status) else 0)
            cv2.putText(view, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return view


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/autolabel", help="folder with images/ and labels/")
    ap.add_argument("--review", action="store_true", help="only images listed in review.csv")
    ap.add_argument("--start", type=int, default=None, help="image index to start at")
    args = ap.parse_args()

    root = Path(args.dataset)
    classes = load_config().classes
    images = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    if args.review and (root / "review.csv").exists():
        with open(root / "review.csv") as f:
            flagged = {r["image"] for r in csv.DictReader(f)}
        images = [p for p in images if p.name in flagged]
    if not images:
        raise SystemExit("no images to label")
    (root / "labels").mkdir(exist_ok=True)

    progress = root / (".label_progress_review" if args.review else ".label_progress")
    start = args.start if args.start is not None else (int(progress.read_text()) if progress.exists() else 0)
    ed = Editor(images, root / "labels", classes, min(start, len(images) - 1))
    ed.load()

    win = "TrafficSense label tool"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, ed.on_mouse)
    while True:
        cv2.imshow(win, ed.render())
        key = cv2.waitKeyEx(20)
        if key == -1:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:   # window closed with X
                ed.save()
                break
            continue
        ch = chr(key) if 0 <= key < 256 else ""
        if ch and ch in "12345678"[:len(classes)]:
            ed.cls = int(ch) - 1
            if ed.selected is not None:
                ed.boxes[ed.selected][0] = ed.cls
        elif ch == "x" or key == KEY_DELETE:
            if ed.selected is not None:
                ed.boxes.pop(ed.selected)
                ed.selected = None
        elif ch == "c":
            ed.boxes, ed.selected = [], None
        elif ch in ("d", " ") or key == KEY_RIGHT:
            ed.save()
            if ed.i < len(images) - 1:
                ed.i += 1
                ed.load()
            else:
                print("last image reached")
        elif ch == "a" or key == KEY_LEFT:
            ed.save()
            if ed.i > 0:
                ed.i -= 1
                ed.load()
        elif ch == "q" or key == 27:
            ed.save()
            break
        progress.write_text(str(ed.i))
    cv2.destroyAllWindows()
    print(f"saved; stopped at image {ed.i + 1}/{len(images)}")


if __name__ == "__main__":
    main()
