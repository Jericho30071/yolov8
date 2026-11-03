"""Start TrafficSense: detection + adaptive signal control + web dashboard.

Examples
  python run.py                                  # everything from config.yaml
  python run.py --source data/raw_videos/a.mp4   # override lane S1's video
  python run.py --source 0                       # lane S1 from webcam 0
  python run.py --mode fixed                     # conventional fixed-time baseline
  python run.py --model models/trafficsense_best.pt
  python run.py --show                           # also open a live window with the detections
"""
from __future__ import annotations

import argparse
import logging
import signal
import threading
import time

import cv2
import numpy as np
import uvicorn

from trafficsense.config import load_config, resolve_path
from trafficsense.system import TrafficSystem
from trafficsense.web.app import create_app


def show_windows(system: TrafficSystem) -> None:
    """One OpenCV window per monitored lane showing the annotated detections.

    Closing a window (X button or q) closes it for good; the system and dashboard keep running.
    """
    opened, closed = set(), set()
    while True:
        for lane_id, mon in system.monitors.items():
            win = f"TrafficSense {lane_id}"
            if lane_id in closed:
                continue
            if lane_id in opened and cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                closed.add(lane_id)   # user clicked X — don't recreate it
                continue
            jpg = mon.jpeg()
            if jpg:
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                light = system.controller.lights()[lane_id]
                cv2.putText(frame, f"signal: {light.upper()}", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(frame, f"signal: {light.upper()}", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(win, frame)
                opened.add(lane_id)
        if (cv2.waitKey(30) & 0xFF == ord("q")) or (system.monitors and closed >= set(system.monitors)):
            cv2.destroyAllWindows()
            return
        time.sleep(0.01)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="video file or webcam index for the first monitored lane")
    ap.add_argument("--lane", default=None, help="lane id for --source (default: first lane)")
    ap.add_argument("--mode", choices=["adaptive", "fixed"])
    ap.add_argument("--model", help="YOLO weights (overrides config)")
    ap.add_argument("--port", type=int)
    ap.add_argument("--show", action="store_true", help="also open a window with the annotated live feed")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.model:
        cfg.model.weights = args.model
    if args.source is not None:
        lane = cfg.lane(args.lane) if args.lane else cfg.lanes[0]
        lane.source = int(args.source) if args.source.isdigit() else str(resolve_path(args.source))
    if args.port:
        cfg.web.port = args.port
    for lane in cfg.lanes:
        if isinstance(lane.source, str) and "://" not in lane.source and not resolve_path(lane.source).exists():
            logging.warning("lane %s: video %s not found — lane will enter failsafe (fixed timing)",
                            lane.id, lane.source)

    system = TrafficSystem(cfg, mode=args.mode).start()
    app = create_app(system)
    signal.signal(signal.SIGTERM, lambda *_: system.stop())
    if args.show:
        threading.Thread(target=show_windows, args=(system,), daemon=True, name="viewer").start()
    logging.info("dashboard: http://localhost:%d  (login %s / %s on first run)", cfg.web.port,
                 cfg.web.default_admin["username"], cfg.web.default_admin["password"])
    try:
        uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, log_level="warning")
    finally:
        system.stop()


if __name__ == "__main__":
    main()
