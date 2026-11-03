"""Vehicle detection for one lane: camera -> YOLOv8 + ByteTrack/BoT-SORT -> Supervision zone / count line.

* CameraStream reads frames on its own thread and always keeps only the newest frame,
  so slow inference (e.g. on a Raspberry Pi) drops frames instead of lagging behind.
  Video files are played back at their native FPS so timings match real traffic.
* LaneMonitor runs detection on the newest frame and publishes a LaneState with
  occupancy, per-class counts, line-crossing totals, and an annotated JPEG.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from .config import Config, LaneConfig, geometry_for, resolve_path

log = logging.getLogger(__name__)


class CameraStream:
    def __init__(self, source, loop_video: bool = True):
        self.source = source
        self.loop_video = loop_video
        self.is_file = isinstance(source, str) and "://" not in source
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self.last_frame_time = 0.0
        self.fps = 30.0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{source}")

    def start(self) -> "CameraStream":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def read(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._frame_id, self._frame

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if 1 <= fps <= 120 else 30.0
        return cap

    def _run(self) -> None:
        cap = self._open()
        while not self._stop.is_set():
            if not cap.isOpened():
                self.error = f"cannot open source {self.source}"
                time.sleep(1.0)
                cap = self._open()  # keep retrying so a reconnected webcam recovers
                continue
            t0 = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                if self.is_file and self.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.error = "frame read failed"
                cap.release()
                time.sleep(0.5)
                cap = self._open()
                continue
            self.error = None
            with self._lock:
                self._frame, self._frame_id = frame, self._frame_id + 1
                self.last_frame_time = time.time()
            if self.is_file:  # real-time playback
                time.sleep(max(0.0, 1.0 / self.fps - (time.monotonic() - t0)))
        cap.release()


def build_class_map(model_names: dict[int, str], classes: list[str], coco_map: dict[str, str]) -> dict[int, str]:
    """Map model class ids to TrafficSense class names; unmapped classes are ignored.

    A custom-trained model already uses the thesis class names. COCO weights are
    detected by the presence of COCO-only names and mapped through `coco_class_map`.
    """
    is_coco = "person" in model_names.values()
    out = {}
    for cid, name in model_names.items():
        if is_coco:
            if coco_map.get(name):
                out[cid] = coco_map[name]
        elif name in classes:
            out[cid] = name
    return out


def load_model(weights: str) -> YOLO:
    path = resolve_path(weights)
    return YOLO(str(path) if path.exists() else weights)


def track(model: YOLO, frame: np.ndarray, cfg: Config, class_map: dict[int, str],
          scene: "Scene | None" = None) -> sv.Detections:
    """YOLOv8 detection + Ultralytics' built-in tracker (ByteTrack or BoT-SORT).

    With a scene, only the lane zone is analysed (see Scene.zone_view) and the boxes are
    shifted back to full-frame coordinates.

    Detections down to 0.1 confidence are handed to the tracker: ByteTrack only *starts*
    tracks from confident boxes (new_track_thresh in the tracker yaml) but uses the weak
    ones to keep following vehicles that are small, far away or partly hidden.
    """
    tracker = resolve_path(cfg.model.tracker)
    image = scene.zone_view(frame) if scene is not None else frame
    result = model.track(image, persist=True, tracker=str(tracker) if tracker.exists() else cfg.model.tracker,
                         imgsz=cfg.model.imgsz, conf=min(0.1, cfg.model.conf), iou=cfg.model.iou,
                         device=cfg.model.device or None, classes=list(class_map) or None,
                         agnostic_nms=True,   # one box per vehicle (no "truck" + "motorcycle" on a jeepney)
                         verbose=False)[0]
    det = sv.Detections.from_ultralytics(result)
    if det.tracker_id is None:   # no active tracks this frame: keep only confident raw detections
        det = det[det.confidence >= cfg.model.conf]
    if scene is not None and len(det):
        x1, y1 = scene.crop[:2]
        det.xyxy = det.xyxy + np.array([x1, y1, x1, y1], dtype=det.xyxy.dtype)
    return det


def warmup(model: YOLO, cfg: Config) -> None:
    """Run one dummy inference so CUDA/cuDNN set-up does not stall the first real frames."""
    model.predict(np.full((cfg.model.imgsz, cfg.model.imgsz, 3), 114, np.uint8), imgsz=cfg.model.imgsz,
                  device=cfg.model.device or None, verbose=False)


def reset_tracker(model: YOLO) -> None:
    if model.predictor is not None:
        for tracker in getattr(model.predictor, "trackers", []):
            tracker.reset()


class Scene:
    """Lane zone + count line at one video resolution, and all on-frame drawing.

    Sizes scale with the resolution so labels stay small on phone videos and readable on 1080p.
    """

    ZONE_IDLE = (216, 180, 0)      # BGR cyan
    ZONE_BUSY = (80, 200, 60)      # BGR green

    def __init__(self, w: int, h: int, zone, count_line, lane_id: str = "S1", hold: float = 0.3):
        self.w, self.h, self.lane_id = w, h, lane_id
        self.hold, self.last_hit, self.occupied = hold, float("-inf"), False
        self.zone_pts = (np.array(zone or [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float) * [w, h]).astype(np.int32)
        self.zone = sv.PolygonZone(polygon=self.zone_pts)
        self.line = None
        if count_line:
            (x1, y1), (x2, y2) = count_line
            self.line_pts = ((int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)))
            self.line = sv.LineZone(start=sv.Point(*self.line_pts[0]), end=sv.Point(*self.line_pts[1]))
        short = min(w, h)
        self.text = float(np.clip(short / 1100, 0.33, 0.6))     # box labels
        self.hud = float(np.clip(short / 800, 0.42, 0.8))       # status panel
        self.thick = 1 if short < 720 else 2
        self.box_ann = sv.BoxAnnotator(thickness=self.thick)
        self.label_ann = sv.LabelAnnotator(text_scale=self.text, text_thickness=1, text_padding=3,
                                           border_radius=2, smart_position=True)
        # Detection runs only on the zone: crop to its bounding box and blank everything outside
        # the polygon, so vehicles in other lanes / at the horizon are never detected at all.
        self.votes: dict[int, dict[int, float]] = {}   # track id -> {class id: summed confidence}
        self.last_seen: dict[int, int] = {}
        self.frame_no = 0
        x, y, bw, bh = cv2.boundingRect(self.zone_pts)
        self.crop = (x, y, x + bw, y + bh)
        self.mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(self.mask, [self.zone_pts - [x, y]], 255)

    def zone_view(self, frame: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.crop
        view = frame[y1:y2, x1:x2].copy()
        view[self.mask == 0] = 114   # neutral grey, same as YOLO letterbox padding
        return view

    def vote(self, det: sv.Detections) -> None:
        """Stabilise classes per tracked vehicle: YOLO classifies every frame independently, so an
        uncertain vehicle flickers between e.g. car and jeepney. Each track keeps a confidence-
        weighted tally over all its frames and is always shown with its winning class."""
        self.frame_no += 1
        if det.tracker_id is None or not len(det):
            return
        voted = det.class_id.copy()
        for i, (tid, cid, conf) in enumerate(zip(det.tracker_id, det.class_id, det.confidence)):
            tally = self.votes.setdefault(int(tid), {})
            tally[int(cid)] = tally.get(int(cid), 0.0) + float(conf)
            self.last_seen[int(tid)] = self.frame_no
            voted[i] = max(tally, key=tally.get)
        det.class_id = voted
        if self.frame_no % 300 == 0:   # forget tracks gone for 10 s
            for tid in [t for t, f in self.last_seen.items() if self.frame_no - f > 300]:
                self.votes.pop(tid, None)
                self.last_seen.pop(tid, None)

    def update(self, det: sv.Detections) -> np.ndarray:
        """Stabilises classes, updates line counts and `self.occupied`, and returns a bool mask
        of detections inside the lane zone."""
        in_zone = np.zeros(0, dtype=bool)
        if len(det):
            self.vote(det)
            if self.line is not None and det.tracker_id is not None:   # line counting needs track IDs
                self.line.trigger(det)
            in_zone = self.zone.trigger(det)
        # Occupancy hold: a vehicle missed for a frame or two (blur, occlusion) does not flip the
        # lane to "empty". Removed all brief false-empty drops on SAMPLE1 (8 -> 0).
        now = time.monotonic()
        if in_zone.any():
            self.last_hit = now
        self.occupied = now - self.last_hit <= self.hold
        return in_zone

    def _text(self, img, text, org, scale, color=(255, 255, 255), bg=(0, 0, 0), alpha=0.55, pad=4):
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x, y = org
        x1, y1, x2, y2 = x - pad, y - th - pad, x + tw + pad, y + base + pad
        roi = img[max(0, y1):y2, max(0, x1):x2]
        roi[:] = (roi * (1 - alpha) + np.array(bg) * alpha).astype(np.uint8)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        return th + base + 2 * pad

    def draw(self, frame: np.ndarray, det: sv.Detections, labels: list[str], in_zone: np.ndarray,
             hud: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
        img = frame.copy()
        busy = self.occupied
        color = self.ZONE_BUSY if busy else self.ZONE_IDLE
        overlay = img.copy()
        cv2.fillPoly(overlay, [self.zone_pts], color)
        img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
        cv2.polylines(img, [self.zone_pts], True, color, self.thick + 1, cv2.LINE_AA)

        if self.line is not None:
            a, b = self.line_pts
            cv2.line(img, a, b, (255, 255, 255), self.thick, cv2.LINE_AA)
            for p in (a, b):
                cv2.circle(img, p, self.thick + 2, (255, 255, 255), -1, cv2.LINE_AA)
            left = min(a, b)
            self._text(img, f"passed: {self.line.in_count + self.line.out_count}",
                       (left[0] + 6, left[1] - 8), self.text)

        if len(det) and in_zone.any():
            near = det[in_zone]
            img = self.box_ann.annotate(img, near)
            img = self.label_ann.annotate(img, near, labels=[l for l, z in zip(labels, in_zone) if z])

        y = 8
        for text, col in hud:
            (_, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.hud, 1)
            y += self._text(img, text, (10, y + th + 4), self.hud, color=col)
        return img


def make_labels(det: sv.Detections, class_map: dict[int, str]) -> list[str]:
    ids = det.tracker_id if det.tracker_id is not None else [None] * len(det)
    return [(f"{t} " if t is not None else "") + f"{class_map.get(int(c), '?')} {p:.0%}"
            for t, c, p in zip(ids, det.class_id, det.confidence)]


@dataclass
class LaneState:
    lane_id: str
    occupied: bool = False
    count: int = 0                                   # vehicles currently inside the zone
    class_counts: dict[str, int] = field(default_factory=dict)
    line_in: int = 0                                 # vehicles that crossed the count line
    line_out: int = 0
    avg_conf: float = 0.0
    fps: float = 0.0
    infer_ms: float = 0.0
    last_update: float = 0.0                         # wall time of the last processed frame
    consecutive_errors: int = 0
    error: str | None = None
    zone_tracks: dict[int, float] = field(default_factory=dict)  # track_id -> first time seen in zone


class LaneMonitor:
    """Runs detection for one lane on a background thread."""

    def __init__(self, cfg: Config, lane: LaneConfig,
                 on_vehicle: Callable[[str, dict], None] | None = None,
                 on_status: Callable[[str, bool, int], None] | None = None):
        self.cfg, self.lane = cfg, lane
        self.on_vehicle = on_vehicle      # called once per new tracked vehicle entering the zone
        self.on_status = on_status        # called when the lane switches occupied <-> empty
        self.camera = CameraStream(lane.source, lane.loop_video)
        self.model = load_model(cfg.model.weights)
        warmup(self.model, cfg)
        self.class_map = build_class_map(self.model.names, cfg.classes, cfg.model.coco_class_map)
        self.scene: Scene | None = None
        self.state = LaneState(lane.id)
        self._jpeg: bytes | None = None
        self._reset = False
        self.source_since = time.time()   # watchdog grace period starts here
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"monitor-{lane.id}")

    # --------------------------------------------------------------- public
    def start(self) -> "LaneMonitor":
        self.camera.start()
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.camera.stop()

    def set_source(self, source) -> None:
        """Switch to another video file / webcam while running (dashboard upload)."""
        old = self.camera
        self.camera = CameraStream(source, self.lane.loop_video).start()
        old.stop()
        self.lane.source = source
        self._reset = True   # picked up by the detection thread: new geometry, tracker, counts
        self.source_since = time.time()

    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def get_state(self) -> LaneState:
        with self._lock:
            s = self.state
            return LaneState(**{**s.__dict__, "class_counts": dict(s.class_counts),
                                "zone_tracks": dict(s.zone_tracks)})

    # ------------------------------------------------------------- internal
    def setup_geometry(self, w: int, h: int) -> None:
        zone, line = geometry_for(self.lane.source, self.lane)
        self.scene = Scene(w, h, zone, line, self.lane.id)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        return track(self.model, frame, self.cfg, self.class_map, self.scene)

    def _run(self) -> None:
        last_id, t_prev = -1, time.time()
        while not self._stop.is_set():
            if self._reset:
                self._reset, last_id, self.scene = False, -1, None
                reset_tracker(self.model)
                with self._lock:
                    self.state = LaneState(self.lane.id)
            frame_id, frame = self.camera.read()
            if frame is None or frame_id == last_id:
                time.sleep(0.005)
                continue
            last_id = frame_id
            if self.scene is None:
                self.setup_geometry(frame.shape[1], frame.shape[0])
            try:
                t0 = time.perf_counter()
                det = self.detect(frame)
                infer_ms = (time.perf_counter() - t0) * 1000
                self._update(frame, det, infer_ms, t_prev)
                t_prev = time.time()
            except Exception as exc:  # detection failure -> watchdog switches to failsafe
                log.exception("lane %s detection error", self.lane.id)
                with self._lock:
                    self.state.consecutive_errors += 1
                    self.state.error = str(exc)
                time.sleep(0.2)

    def _update(self, frame: np.ndarray, det: sv.Detections, infer_ms: float, t_prev: float) -> None:
        now = time.time()
        scene = self.scene
        in_zone = scene.update(det)
        zdet = det[in_zone] if len(det) else det
        names = [self.class_map.get(int(c), "?") for c in zdet.class_id] if len(zdet) else []

        state = self.state
        was_occupied = state.occupied
        zone_tracks = {}
        new_vehicles = []
        for i, tid in enumerate(zdet.tracker_id if zdet.tracker_id is not None else []):
            tid = int(tid)
            first = state.zone_tracks.get(tid)
            if first is None:
                first = now
                new_vehicles.append({"track_id": tid, "vehicle_type": names[i],
                                     "confidence": float(zdet.confidence[i]), "vehicle_count": len(zdet)})
            zone_tracks[tid] = first

        counts: dict[str, int] = {}
        for n in names:
            counts[n] = counts.get(n, 0) + 1

        hud = [(("VEHICLE PRESENT" if scene.occupied else "LANE EMPTY") + f"  {infer_ms:.0f} ms",
                (90, 230, 90) if scene.occupied else (255, 255, 255))]
        img = scene.draw(frame, det, make_labels(det, self.class_map), in_zone, hud)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])

        with self._lock:
            state.occupied = scene.occupied
            state.count = len(zdet)
            state.class_counts = counts
            state.zone_tracks = zone_tracks
            state.avg_conf = float(np.mean(zdet.confidence)) if len(zdet) else 0.0
            state.line_in = scene.line.in_count if scene.line else 0
            state.line_out = scene.line.out_count if scene.line else 0
            state.infer_ms = infer_ms
            dt = now - t_prev
            state.fps = 0.9 * state.fps + 0.1 * (1.0 / dt) if dt > 0 else state.fps
            state.last_update = now
            state.consecutive_errors = 0
            state.error = None
            if ok:
                self._jpeg = buf.tobytes()

        for v in new_vehicles:
            if self.on_vehicle:
                self.on_vehicle(self.lane.id, v)
        if was_occupied != state.occupied and self.on_status:
            self.on_status(self.lane.id, state.occupied, state.count)
