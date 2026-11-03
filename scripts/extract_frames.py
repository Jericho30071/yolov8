"""Step 1 of the dataset workflow: turn traffic videos into a clean set of images.

Samples one frame every --every seconds, drops blurry frames (Laplacian variance)
and near-duplicates (consecutive frames that barely changed, e.g. an empty road).

  python scripts/extract_frames.py --videos data/raw_videos --out data/frames --every 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def iter_videos(paths: list[str]):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXT)
        elif p.suffix.lower() in VIDEO_EXT:
            yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["data/raw_videos"], help="video files or folders")
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between sampled frames")
    ap.add_argument("--blur", type=float, default=60.0, help="min Laplacian variance (0 disables)")
    ap.add_argument("--dedupe", type=float, default=4.0, help="min mean pixel change vs. last kept frame (0 disables)")
    ap.add_argument("--max-side", type=int, default=1280, help="resize so the longest side is at most this")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = list(iter_videos(args.videos))
    if not videos:
        raise SystemExit(f"no videos found in {args.videos}")

    grand = 0
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, round(fps * args.every))
        kept = blurry = dupes = idx = 0
        last_small = None
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                h, w = frame.shape[:2]
                scale = args.max_side / max(h, w)
                if scale < 1:
                    frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if args.blur and cv2.Laplacian(gray, cv2.CV_64F).var() < args.blur:
                    blurry += 1
                else:
                    small = cv2.resize(gray, (64, 36)).astype(np.float32)
                    if args.dedupe and last_small is not None and np.abs(small - last_small).mean() < args.dedupe:
                        dupes += 1
                    else:
                        last_small = small
                        name = f"{video.stem}_{idx:07d}.jpg"
                        cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        kept += 1
            idx += 1
        cap.release()
        grand += kept
        print(f"{video.name}: kept {kept}, blurry {blurry}, near-duplicate {dupes}")
    print(f"done: {grand} frames in {out}")


if __name__ == "__main__":
    main()
