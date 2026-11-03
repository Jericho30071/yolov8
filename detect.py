"""Watch YOLOv8 detect, track and count vehicles in a video — no web server needed.

  python detect.py                          # pick a video with a file dialog
  python detect.py data/raw_videos/a.mp4    # a specific video
  python detect.py 0                        # webcam
  python detect.py a.mp4 --save             # also write the annotated video to reports/
  python detect.py a.mp4 --model models/trafficsense_best.pt

Keys: o = open (upload) another video, z = draw the lane zone + count line for this video,
Space = pause/resume, q / Esc / X button = quit.
Every video you open is copied into data/raw_videos/ so it is ready for training
(scripts/extract_frames.py); use --no-copy to skip that. A zone drawn with z is saved as
<video>.zone.json and used automatically next time (also by run.py).
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from trafficsense.config import ROOT, geometry_for, load_config, resolve_path, save_geometry
from trafficsense.detector import Scene, build_class_map, load_model, make_labels, reset_tracker, track, warmup

VIDEO_DIR = ROOT / "data" / "raw_videos"
KEY_HINT = "[o] open  [z] zone  [space] pause  [q] quit"


def pick_video() -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)   # show the dialog above the OpenCV window
    path = filedialog.askopenfilename(
        title="Choose a traffic video", initialdir=VIDEO_DIR,
        filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"), ("All files", "*.*")])
    root.destroy()
    return path or None


def import_video(path: str) -> str:
    """Copy a video into data/raw_videos (the project's video library) and return the new path."""
    src = Path(path).resolve()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    if src.parent == VIDEO_DIR.resolve():
        return str(src)
    dst = VIDEO_DIR / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return str(dst)
    print(f"copying {src.name} into {VIDEO_DIR} ...")
    shutil.copy2(src, dst)
    return str(dst)


def edit_geometry(win: str, frame: np.ndarray, hud_scale: float):
    """Click the zone corners (Enter), then the two count-line ends (Enter). Returns normalized
    (zone, count_line), or None if cancelled with Esc."""
    h, w = frame.shape[:2]
    stages = [("Click the LANE ZONE corners, then Enter", 3), ("Click the 2 ends of the COUNT LINE, then Enter", 2)]
    pts: list[tuple[int, int]] = []
    done: list[list[tuple[int, int]]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and not (len(done) == 1 and len(pts) >= 2):
            pts.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))

    cv2.setMouseCallback(win, on_mouse)
    while len(done) < 2:
        msg, minimum = stages[len(done)]
        img = frame.copy()
        if done:
            cv2.polylines(img, [np.array(done[0])], True, Scene.ZONE_IDLE, 2, cv2.LINE_AA)
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], len(done) == 0 and len(pts) > 2, (0, 255, 255), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 4, (0, 255, 255), -1, cv2.LINE_AA)
        for i, text in enumerate((msg, "[backspace] undo  [r] restart  [esc] cancel")):
            y = int(30 * hud_scale / 0.6) * (i + 1)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, hud_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, hud_scale, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cv2.setMouseCallback(win, lambda *a: None)
            return None
        if key == 8 and pts:
            pts.pop()
        elif key == ord("r"):
            pts.clear()
            done.clear()
        elif key in (13, 10) and len(pts) >= minimum:
            done.append(list(pts))
            pts.clear()
    cv2.setMouseCallback(win, lambda *a: None)
    norm = lambda ps: [[round(x / w, 4), round(y / h, 4)] for x, y in ps]  # noqa: E731
    return norm(done[0]), norm(done[1][:2])


class Viewer:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config()
        if args.model:
            self.cfg.model.weights = args.model
        if args.imgsz:
            self.cfg.model.imgsz = args.imgsz
        self.lane = self.cfg.lane(args.lane)
        self.model = load_model(self.cfg.model.weights)
        warmup(self.model, self.cfg)
        self.class_map = build_class_map(self.model.names, self.cfg.classes, self.cfg.model.coco_class_map)
        self.win = "TrafficSense detection"
        print(f"model {self.cfg.model.weights} @ {self.cfg.model.imgsz}px, tracker {self.cfg.model.tracker}")

    def play(self, source) -> str | None:
        """Play one source. Returns the next video to play (key o), or None to quit."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"cannot open {source}")
            return None
        is_file = isinstance(source, str)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        name = Path(source).name if is_file else f"webcam {source}"
        print(f"\nplaying {name} ({w}x{h}, {fps:.0f} FPS)")

        reset_tracker(self.model)
        scene = Scene(w, h, *geometry_for(source, self.lane), self.lane.id)
        writer = out_path = None
        if self.args.save:
            (ROOT / "reports").mkdir(exist_ok=True)
            out_path = ROOT / "reports" / f"annotated_{Path(name).stem.replace(' ', '_')}.mp4"
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        scale = min(1280 / w, 820 / h, 1.6)
        cv2.resizeWindow(self.win, int(w * scale), int(h * scale))
        cv2.setWindowTitle(self.win, f"TrafficSense - {name}")

        seen: dict[int, str] = {}       # unique vehicles: track id -> class
        ms_avg, skipped = 0.0, 0
        paused, shown, next_source, frame = False, False, None, None
        clock = time.perf_counter()     # video time 0 in wall-clock terms (real-time playback)
        while True:
            if not paused:
                if is_file and not self.args.fast:
                    # Real time: if detection fell behind the video clock, skip frames to catch up
                    # instead of playing in slow motion.
                    due = int((time.perf_counter() - clock) * fps)
                    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    for _ in range(min(max(0, due - pos), 10)):
                        cap.grab()
                        skipped += 1
                ok, frame = cap.read()
                if not ok:
                    print("end of video — press o to open another, q to quit")
                    paused = True
                else:
                    t0 = time.perf_counter()
                    det = track(self.model, frame, self.cfg, self.class_map, scene)
                    ms = (time.perf_counter() - t0) * 1000
                    ms_avg = ms if not ms_avg else 0.9 * ms_avg + 0.1 * ms
                    in_zone = scene.update(det)
                    if det.tracker_id is not None:
                        for tid, cid in zip(det.tracker_id[in_zone], det.class_id[in_zone]):
                            seen[int(tid)] = self.class_map.get(int(cid), "?")   # latest voted class
                    counts: dict[str, int] = {}
                    for c in seen.values():
                        counts[c] = counts.get(c, 0) + 1
                    present = scene.occupied
                    hud = [
                        (("VEHICLE PRESENT" if present else "LANE EMPTY") + f"   {ms_avg:.0f} ms/frame",
                         (90, 230, 90) if present else (255, 255, 255)),
                        ("vehicles: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"),
                         (255, 255, 255)),
                        (KEY_HINT, (200, 200, 200)),
                    ]
                    img = scene.draw(frame, det, make_labels(det, self.class_map), in_zone, hud)
                    if writer:
                        writer.write(img)
                    cv2.imshow(self.win, img)
                    shown = True

            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key in (ord("q"), 27) or (shown and cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1):
                break
            if key == ord(" "):
                paused = not paused
                clock = time.perf_counter() - cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
            if key == ord("z") and frame is not None:
                geo = edit_geometry(self.win, frame, scene.hud)
                if geo:
                    f = save_geometry(source, *geo)
                    scene = Scene(w, h, *geo, self.lane.id)
                    reset_tracker(self.model)   # tracks were in the old zone's coordinates
                    seen.clear()
                    print(f"zone saved{' to ' + str(f) if f else ''}:\n  zone: {geo[0]}\n  count_line: {geo[1]}")
                clock = time.perf_counter() - cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
            if key == ord("o"):
                picked = pick_video()
                if picked:
                    next_source = picked
                    break
                clock = time.perf_counter() - cap.get(cv2.CAP_PROP_POS_FRAMES) / fps

        cap.release()
        if writer:
            writer.release()
            print(f"annotated video saved to {out_path}")
        counts_txt = ", ".join(f"{c}={list(seen.values()).count(c)}" for c in sorted(set(seen.values()))) or "none"
        print(f"{name}: unique vehicles {len(seen)} ({counts_txt})"
              + (f"; count line in {scene.line.in_count}, out {scene.line.out_count}" if scene.line else "")
              + (f"; {skipped} frames skipped to stay real-time" if skipped else ""))
        return next_source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="video path or webcam index (omit for a file dialog)")
    ap.add_argument("--model", help="weights (default: config.yaml)")
    ap.add_argument("--imgsz", type=int, help="inference size (default: config.yaml)")
    ap.add_argument("--lane", default="S1", help="lane whose default zone / count line to use")
    ap.add_argument("--save", action="store_true", help="save annotated video to reports/")
    ap.add_argument("--fast", action="store_true", help="process every frame as fast as possible")
    ap.add_argument("--no-copy", action="store_true", help="don't copy opened videos into data/raw_videos")
    args = ap.parse_args()

    source = args.source or pick_video()
    if not source:
        raise SystemExit("no video chosen")
    viewer = Viewer(args)
    while source:
        if source.isdigit():
            source = int(source)
        else:
            source = str(resolve_path(source)) if args.no_copy else import_video(str(resolve_path(source)))
        source = viewer.play(source)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
