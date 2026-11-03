"""TrafficSense signal simulation: live YOLOv8 detection driving a 4-way traffic light.

  python simulate.py                     # pick a video with a file dialog
  python simulate.py data/raw_videos/SAMPLE1.mp4

Left: the camera lane S1 with detection (only the lane zone is analysed).
Right: the S1-S4 traffic lights, run by the same SignalController as the real system,
following the research paper:
  * Objective 2 / Flowchart (Sec. 3.12.5): during green, if no vehicle is detected for
    `empty_timeout` (5 s) the green is stopped -> yellow (3 s) -> red, and control passes to
    the next lane in the sequence S1 -> S2 -> S3 -> S4. While vehicles keep coming the green
    follows normal timing (max_green).
  * Sec. 3.9, Table 2 (failsafe): camera / detection failure -> that lane reverts to the
    fixed-time cycle.
  * Sec. 3.10.6: fixed-time mode, the conventional baseline the adaptive mode is compared to.
S2-S4 have no camera yet, so they are simulated: press 2 / 3 / 4 to switch "vehicles waiting"
on or off for that lane.

Keys: 2/3/4 = toggle vehicles on S2/S3/S4   f = simulate S1 camera failure   m = adaptive/fixed
      e = emergency stop (all red) / resume   o = open video   z = draw zone
      space = pause   q / Esc = quit
Every signal event is written to reports/simulation_<date>_<time>.csv for Chapter 4.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detect import edit_geometry, import_video, pick_video
from trafficsense.config import ROOT, geometry_for, load_config, resolve_path, save_geometry
from trafficsense.controller import LaneInput, Mode, Phase, SignalController
from trafficsense.detector import Scene, build_class_map, load_model, make_labels, reset_tracker, track, warmup

PANEL_W, MIN_H = 430, 760
FONT = cv2.FONT_HERSHEY_SIMPLEX
LAMP = {"red": (60, 60, 235), "yellow": (40, 200, 250), "green": (80, 210, 80)}
LAMP_OFF = (55, 55, 55)
WHITE, GREY, DIM = (240, 240, 240), (170, 170, 170), (120, 120, 120)
ORANGE, BG, CARD = (40, 160, 255), (28, 28, 30), (44, 44, 48)

# Why a green phase ended, in the words of the research paper.
END_REASON = {
    "stop_empty": "empty {t:.0f}s -> stopped (Obj.2)",
    "max_green": "max green, vehicles present",
    "fixed": "fixed-time cycle",
    "failsafe": "fault -> fixed time (Table 2)",
}


def put(img, text, org, scale=0.45, color=WHITE, thick=1):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, frac, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), -1)
    if frac > 0:
        cv2.rectangle(img, (x, y), (x + int(w * min(1.0, frac)), y + h), color, -1)


class Simulation:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config()
        self.sig = self.cfg.signal
        self.lane_ids = [l.id for l in self.cfg.lanes]
        self.cam_lane = self.lane_ids[0]
        self.model = load_model(self.cfg.model.weights)
        warmup(self.model, self.cfg)
        self.class_map = build_class_map(self.model.names, self.cfg.classes, self.cfg.model.coco_class_map)
        self.ctl = SignalController(self.sig, self.lane_ids, self.sig.mode)
        self.sim_vehicles = {l: False for l in self.lane_ids[1:]}   # simulated lanes S2-S4
        self.camera_fault = False
        self.log: deque[tuple[str, tuple]] = deque(maxlen=9)
        self.stats = {"stops": 0, "saved": 0.0, "cycles": 0, "greens": 0}
        self.t = 0.0                                                 # simulation clock (s)
        self.win = "TrafficSense - signal simulation"
        reports = ROOT / "reports"
        reports.mkdir(exist_ok=True)
        self.csv_path = reports / f"simulation_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["time_s", "lane", "event", "color", "duration_s", "reason", "mode",
                           "vehicles_in_zone"])

    # ------------------------------------------------------------- events
    def note(self, text: str, color=WHITE) -> None:
        m, s = divmod(int(self.t), 60)
        self.log.appendleft((f"{m:02d}:{s:02d} {text}", color))

    def handle(self, events, count: int) -> None:
        for ev in events:
            d, mode = ev.data, self.ctl.mode.value
            if ev.kind == "phase_start" and d["color"] == "green":
                self.stats["greens"] += 1
                if ev.lane == self.lane_ids[0]:
                    self.stats["cycles"] += 1
                basis = {"adaptive": f"adaptive, max {d['planned']:.0f}s", "fixed": f"fixed {d['planned']:.0f}s",
                         "failsafe": f"FAILSAFE fixed {d['planned']:.0f}s"}.get(d["basis"], d["basis"])
                self.note(f"{ev.lane} GREEN ({basis})", LAMP["green"])
                self.csv.writerow([f"{ev.time:.1f}", ev.lane, "green_start", "green", "", d["basis"], mode, count])
            elif ev.kind == "phase_end" and d["color"] == "green":
                reason = d["reason"]
                msg = END_REASON.get(reason, reason).format(t=self.sig.empty_timeout)
                self.note(f"{ev.lane} green {d['duration']:.1f}s: {msg}",
                          ORANGE if reason == "stop_empty" else GREY)
                self.csv.writerow([f"{ev.time:.1f}", ev.lane, "green_end", "green", f"{d['duration']:.1f}",
                                   reason, mode, count])
            elif ev.kind == "phase_start" and d["color"] == "yellow":
                self.note(f"{ev.lane} YELLOW {self.sig.yellow:.0f}s -> RED", LAMP["yellow"])
                self.csv.writerow([f"{ev.time:.1f}", ev.lane, "yellow_start", "yellow", f"{self.sig.yellow:.1f}",
                                   "", mode, count])
            elif ev.kind == "stop":
                self.stats["stops"] += 1
                self.stats["saved"] += d["saved_green"]
                self.csv.writerow([f"{ev.time:.1f}", ev.lane, "stop_empty", "green", f"{d['green_elapsed']:.1f}",
                                   f"idle green avoided {d['saved_green']:.1f}s", mode, count])
            elif ev.kind == "skip":
                self.note(f"{ev.lane} skipped: lane empty", ORANGE)
                self.csv.writerow([f"{ev.time:.1f}", ev.lane, "skip", "", "", d["reason"], mode, count])
        if events:
            self.csv_file.flush()

    def key(self, k: int) -> None:
        c = chr(k) if 0 <= k < 256 else ""
        if c in "234" and c and f"S{c}" in self.sim_vehicles:
            lane = f"S{c}"
            self.sim_vehicles[lane] = not self.sim_vehicles[lane]
            self.note(f"{lane}: {'vehicles waiting' if self.sim_vehicles[lane] else 'lane empty'} (simulated)", DIM)
        elif c == "f":
            self.camera_fault = not self.camera_fault
            self.note(f"{self.cam_lane} camera {'FAILURE -> fixed-time (Table 2)' if self.camera_fault else 'restored'}",
                      LAMP["red"] if self.camera_fault else LAMP["green"])
            self.csv.writerow([f"{self.t:.1f}", self.cam_lane, "camera_fault" if self.camera_fault else "camera_ok",
                               "", "", "simulated by operator", self.ctl.mode.value, ""])
        elif c == "m" and self.ctl.mode in (Mode.ADAPTIVE, Mode.FIXED):
            self.ctl.mode = Mode.FIXED if self.ctl.mode == Mode.ADAPTIVE else Mode.ADAPTIVE
            self.note(f"mode -> {self.ctl.mode.value.upper()} (Sec.3.10.6)", WHITE)
            self.csv.writerow([f"{self.t:.1f}", "", "mode", "", "", self.ctl.mode.value, self.ctl.mode.value, ""])
        elif c == "e":
            if self.ctl.mode == Mode.MANUAL:
                self.handle(self.ctl.resume(self.t, "adaptive"), 0)
                self.note("RESUMED adaptive control", LAMP["green"])
            else:
                self.ctl.emergency_stop()
                self.note("EMERGENCY STOP: all red (Sec.3.9 override)", LAMP["red"])
            self.csv.writerow([f"{self.t:.1f}", "", "override", "", "", self.ctl.mode.value, self.ctl.mode.value, ""])

    # ------------------------------------------------------------- drawing
    def draw_panel(self, h: int, count: int, occupied: bool, ms: float) -> np.ndarray:
        p = np.full((h, PANEL_W, 3), BG, np.uint8)
        ctl, sig = self.ctl, self.sig
        lights = ctl.lights()
        elapsed = self.t - ctl.phase_start if ctl.phase_start is not None else 0.0
        put(p, "TrafficSense - Adaptive Signal", (14, 28), 0.62, WHITE, 2)
        mode_col = {Mode.ADAPTIVE: LAMP["green"], Mode.FIXED: LAMP["yellow"]}.get(ctl.mode, LAMP["red"])
        put(p, f"mode {ctl.mode.value.upper()}", (14, 52), 0.5, mode_col, 1)
        put(p, f"t {int(self.t) // 60:02d}:{int(self.t) % 60:02d}   {ms:.0f} ms/frame", (220, 52), 0.45, GREY)

        # four signal heads (2 x 2)
        cw, ch = 197, 178
        for i, lane in enumerate(self.lane_ids):
            x0, y0 = 12 + (i % 2) * (cw + 12), 66 + (i // 2) * (ch + 10)
            active = lane == ctl.current_lane and ctl.mode in (Mode.ADAPTIVE, Mode.FIXED)
            cv2.rectangle(p, (x0, y0), (x0 + cw, y0 + ch), CARD, -1)
            if active:
                cv2.rectangle(p, (x0, y0), (x0 + cw, y0 + ch), LAMP.get(lights[lane], WHITE), 2)
            # housing + lamps
            cv2.rectangle(p, (x0 + 10, y0 + 10), (x0 + 52, y0 + 132), (20, 20, 20), -1)
            for j, col in enumerate(("red", "yellow", "green")):
                on = lights[lane] == col
                center = (x0 + 31, y0 + 31 + j * 40)
                cv2.circle(p, center, 16, LAMP[col] if on else LAMP_OFF, -1, cv2.LINE_AA)
                if on:
                    cv2.circle(p, center, 19, LAMP[col], 1, cv2.LINE_AA)
            tx = x0 + 62
            name = next(l.name for l in self.cfg.lanes if l.id == lane)
            put(p, lane, (tx, y0 + 24), 0.6, WHITE, 2)
            put(p, name, (tx + 34, y0 + 24), 0.4, GREY)
            put(p, lights[lane].upper(), (tx, y0 + 48), 0.55, LAMP.get(lights[lane], WHITE), 2)

            if lane == self.cam_lane:
                src = "camera: FAULT" if self.camera_fault else "camera (YOLOv8)"
                has_vehicle, detail = occupied, f"{count} in zone" if occupied else "no vehicle"
                if self.camera_fault:
                    has_vehicle, detail = False, "detection ignored"
            else:
                src = f"simulated [{lane[1]}]"
                has_vehicle = self.sim_vehicles[lane]
                detail = "vehicles waiting" if has_vehicle else "empty"
            put(p, src, (tx, y0 + 70), 0.38, LAMP["red"] if "FAULT" in src else DIM)
            put(p, detail, (tx, y0 + 90), 0.42, LAMP["green"] if has_vehicle else GREY)

            if active and ctl.phase == Phase.GREEN:
                put(p, f"green {elapsed:4.1f}/{ctl.planned_green:.0f}s", (tx, y0 + 112), 0.4, WHITE)
                bar(p, tx, y0 + 118, cw - 72, 6, elapsed / max(ctl.planned_green, 0.1), LAMP["green"])
                if ctl.empty_since is not None and ctl.green_reason == "adaptive" and ctl.mode == Mode.ADAPTIVE:
                    empty = self.t - ctl.empty_since
                    put(p, f"empty {empty:3.1f}/{sig.empty_timeout:.0f}s", (tx, y0 + 142), 0.4, ORANGE)
                    bar(p, tx, y0 + 148, cw - 72, 6, empty / sig.empty_timeout, ORANGE)
            elif active and ctl.phase == Phase.YELLOW:
                put(p, f"yellow {elapsed:3.1f}/{sig.yellow:.0f}s", (tx, y0 + 112), 0.4, WHITE)
                bar(p, tx, y0 + 118, cw - 72, 6, elapsed / sig.yellow, LAMP["yellow"])
            elif active and ctl.phase == Phase.ALL_RED:
                put(p, f"all-red {elapsed:3.1f}/{sig.all_red:.0f}s", (tx, y0 + 112), 0.4, WHITE)
            # position in the sequence S1 -> S2 -> S3 -> S4
            put(p, f"turn {i + 1} of {len(self.lane_ids)}", (x0 + 10, y0 + 160), 0.38, DIM)

        # research-paper rules
        y = 66 + 2 * (ch + 10) + 16
        put(p, "Rules (research paper)", (14, y), 0.5, WHITE, 1)
        rules = [
            f"No vehicle {sig.empty_timeout:.0f}s on green -> yellow -> red  (Obj.2)",
            f"Vehicles present -> normal green, max {sig.max_green:.0f}s",
            f"Yellow {sig.yellow:.0f}s, all-red {sig.all_red:.0f}s, then next lane",
            "Sequence S1 > S2 > S3 > S4  (Flowchart 3.12.5)",
            f"Camera/model fault -> fixed {sig.fixed_green:.0f}s  (Table 2)",
        ]
        for r in rules:
            y += 19
            put(p, "- " + r, (16, y), 0.4, GREY)
        y += 26
        put(p, f"green stops {self.stats['stops']}   idle green avoided {self.stats['saved']:.0f}s"
               f"   cycles {self.stats['cycles']}", (14, y), 0.42, ORANGE)

        # event log
        y += 24
        put(p, "Events", (14, y), 0.5, WHITE)
        for text, color in self.log:
            y += 18
            if y > h - 52:
                break
            put(p, text[:58], (16, y), 0.38, color)

        put(p, "[2][3][4] vehicles S2-S4  [f] camera fault  [m] mode", (12, h - 30), 0.38, DIM)
        put(p, "[e] emergency/resume  [o] open  [z] zone  [space] pause  [q] quit", (12, h - 12), 0.38, DIM)
        return p

    # --------------------------------------------------------------- loop
    def run(self, source) -> str | None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"cannot open {source}")
            return None
        is_file = isinstance(source, str)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        name = Path(source).name if is_file else f"webcam {source}"
        print(f"simulating with {name} ({w}x{h}); events -> {self.csv_path}")
        reset_tracker(self.model)
        lane_cfg = self.cfg.lane(self.cam_lane)
        scene = Scene(w, h, *geometry_for(source, lane_cfg), self.cam_lane)
        scale = min(MIN_H / h, 900 / w)
        vw, vh = int(w * scale), int(h * scale)
        H = max(vh, MIN_H)
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, vw + PANEL_W, H)
        self.note(f"start: {name}", WHITE)

        paused, frame, view, next_source = False, None, None, None
        count, occupied, ms_avg = 0, False, 0.0
        last_wall = time.perf_counter()
        video_clock = 0.0
        while True:
            now_wall = time.perf_counter()
            dt, last_wall = now_wall - last_wall, now_wall
            if not paused:
                self.t += dt
                if is_file:
                    # keep the video in real time with the signal clock (skip frames if behind)
                    video_clock += dt
                    behind = int(video_clock * fps) - int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    for _ in range(min(max(0, behind - 1), 10)):
                        cap.grab()
                ok, new = cap.read()
                if not ok and is_file:              # loop the recording so the simulation keeps going
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    video_clock = 0.0
                    reset_tracker(self.model)
                    ok, new = cap.read()
                if ok:
                    frame = new
                    t0 = time.perf_counter()
                    try:
                        det = track(self.model, frame, self.cfg, self.class_map, scene)
                        detection_ok = True
                    except Exception as exc:        # detection failure -> failsafe (Table 2)
                        print(f"detection error: {exc}")
                        det, detection_ok = None, False
                    ms = (time.perf_counter() - t0) * 1000
                    ms_avg = ms if not ms_avg else 0.9 * ms_avg + 0.1 * ms
                    if det is not None:
                        in_zone = scene.update(det)
                        count, occupied = int(in_zone.sum()), scene.occupied
                        light = self.ctl.lights()[self.cam_lane]
                        hud = [(f"{self.cam_lane} signal: {light.upper()}", LAMP.get(light, WHITE)),
                               ("VEHICLE PRESENT" if occupied else "LANE EMPTY",
                                (90, 230, 90) if occupied else (255, 255, 255))]
                        view = scene.draw(frame, det, make_labels(det, self.class_map), in_zone, hud)
                else:
                    detection_ok = False

                healthy = detection_ok and not self.camera_fault
                inputs = {self.cam_lane: LaneInput(monitored=True, healthy=healthy,
                                                   occupied=occupied and healthy, count=count if healthy else 0)}
                for lane, has in self.sim_vehicles.items():
                    inputs[lane] = LaneInput(monitored=True, healthy=True, occupied=has, count=3 if has else 0)
                self.handle(self.ctl.tick(self.t, inputs), count)

            if view is not None:
                video = cv2.resize(view, (vw, vh))
                if self.camera_fault:
                    cv2.putText(video, "CAMERA FAULT (simulated)", (12, vh - 16), FONT, 0.7, LAMP["red"], 2, cv2.LINE_AA)
                canvas = np.zeros((H, vw + PANEL_W, 3), np.uint8)
                y0 = (H - vh) // 2
                canvas[y0:y0 + vh, :vw] = video
                canvas[:, vw:] = self.draw_panel(H, count, occupied, ms_avg)
                if paused:
                    put(canvas, "PAUSED", (vw // 2 - 50, 40), 1.0, WHITE, 2)
                cv2.imshow(self.win, canvas)

            k = cv2.waitKeyEx(1 if not paused else 30)
            if k in (ord("q"), 27) or (frame is not None and cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1):
                break
            if k == ord(" "):
                paused = not paused
                last_wall = time.perf_counter()
            elif k == ord("o"):
                picked = pick_video()
                last_wall = time.perf_counter()
                if picked:
                    next_source = picked
                    break
            elif k == ord("z") and frame is not None:
                geo = edit_geometry(self.win, frame, scene.hud)
                if geo:
                    save_geometry(source, *geo)
                    scene = Scene(w, h, *geo, self.cam_lane)
                    reset_tracker(self.model)
                    self.note("zone redrawn and saved", WHITE)
                last_wall = time.perf_counter()
            elif k != -1:
                self.key(k & 0xFF if k < 256 else k)
        cap.release()
        return next_source

    def close(self) -> None:
        self.csv_file.close()
        print(f"\ngreen phases {self.stats['greens']}, stopped early (empty lane) {self.stats['stops']}, "
              f"idle green avoided {self.stats['saved']:.0f}s, cycles {self.stats['cycles']}")
        print(f"event log: {self.csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="video path or webcam index (omit for a file dialog)")
    ap.add_argument("--model", help="weights (default: config.yaml)")
    ap.add_argument("--no-copy", action="store_true", help="don't copy opened videos into data/raw_videos")
    args = ap.parse_args()

    source = args.source or pick_video()
    if not source:
        raise SystemExit("no video chosen")
    sim = Simulation(args)
    if args.model:
        sim.cfg.model.weights = args.model
        sim.model = load_model(args.model)
        sim.class_map = build_class_map(sim.model.names, sim.cfg.classes, sim.cfg.model.coco_class_map)
    try:
        while source:
            if source.isdigit():
                source = int(source)
            else:
                source = str(resolve_path(source)) if args.no_copy else import_video(str(resolve_path(source)))
            source = sim.run(source)
    finally:
        sim.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
