"""Draw a lane's monitoring zone and count line on a frame of the video/webcam.

  python scripts/draw_zone.py --source data/raw_videos/lane1.mp4
  python scripts/draw_zone.py --source 0          # webcam

Left-click to add zone corners (the area in front of the stop line), press Enter.
Then click two points for the count line, press Enter. The normalized coordinates are
printed — paste them into the lane's `zone:` and `count_line:` in config.yaml.
Keys: z = undo last point, r = restart, Esc = quit.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame", type=int, default=30, help="frame index to show (videos)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not args.source.isdigit():
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read a frame from {args.source}")
    h, w = frame.shape[:2]
    scale = min(1280 / w, 720 / h, 1.0)
    base = cv2.resize(frame, None, fx=scale, fy=scale)

    stages = [("zone", 3), ("count_line", 2)]
    stage, points, result = 0, [], {}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and stage < len(stages):
            if stages[stage][0] == "count_line" and len(points) == 2:
                return
            points.append((x, y))

    win = "Draw zone"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while stage < len(stages):
        name, minimum = stages[stage]
        view = base.copy()
        for pts, color, closed in ((result.get("zone_px", []), (0, 180, 255), True), (points, (0, 255, 0), name == "zone")):
            if len(pts) > 1:
                cv2.polylines(view, [np.array(pts)], closed, color, 2)
            for p in pts:
                cv2.circle(view, p, 4, color, -1)
        msg = f"{name}: click points ({len(points)}), Enter = done, z = undo, r = restart"
        cv2.putText(view, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(view, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            raise SystemExit("cancelled")
        if key == ord("z") and points:
            points.pop()
        if key == ord("r"):
            points.clear()
        if key in (13, 10) and len(points) >= minimum:
            if name == "zone":
                result["zone_px"] = list(points)
            result[name] = [[round(x / base.shape[1], 3), round(y / base.shape[0], 3)] for x, y in points]
            points, stage = [], stage + 1
    cv2.destroyAllWindows()
    print("\nPaste into config.yaml under the lane:")
    print(f"    zone: {result['zone']}")
    print(f"    count_line: {result['count_line']}")


if __name__ == "__main__":
    main()
