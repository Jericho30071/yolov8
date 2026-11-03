"""TrafficSense runtime: lane monitors + watchdog + signal controller + relays + logging."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, fields
from datetime import datetime

from .config import Config, ROOT, SignalConfig
from .controller import Event, LaneInput, SignalController
from .db import (Database, DetectionRecord, FaultRecord, Lane, ManualOverride, RunSession,
                 SignalTimingLog, StopSkipEvent)
from .signals import make_driver

log = logging.getLogger(__name__)
RUNTIME_SETTINGS = ROOT / "data" / "runtime_settings.json"
EDITABLE_SIGNAL_FIELDS = [f.name for f in fields(SignalConfig) if f.name != "mode"]


def ts(t: float) -> datetime:
    return datetime.fromtimestamp(t)


class LaneHealth:
    """Watchdog state for one monitored lane, with hysteresis on recovery."""

    def __init__(self):
        self.healthy = True
        self.reason = ""
        self.ok_since: float | None = None
        self.fault_id: int | None = None


class TrafficSystem:
    def __init__(self, cfg: Config, mode: str | None = None, with_detection: bool = True):
        self.cfg = cfg
        self._load_runtime_settings()
        self.db = Database(cfg.db_path)
        self.db.sync_lanes_and_devices(cfg)
        self.db.ensure_admin(cfg.web.default_admin["username"], cfg.web.default_admin["password"])
        self.controller = SignalController(cfg.signal, [l.id for l in cfg.lanes], mode)
        self.driver = make_driver(cfg.hardware)
        self.session = self.db.add(RunSession(
            mode=self.controller.mode.value, model=cfg.model.weights,
            source=json.dumps({l.id: l.source for l in cfg.lanes})))
        self.monitors = {}
        if with_detection:
            from .detector import LaneMonitor  # heavy import (torch) only when needed
            for lane in cfg.lanes:
                if lane.monitored:
                    self.monitors[lane.id] = LaneMonitor(cfg, lane, self._on_vehicle, self._on_status)
        self.health = {lid: LaneHealth() for lid in self.monitors}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._log_ids: dict[str, int | None] = {}      # lane -> open SignalTimingLog id
        self._last_status_record: dict[str, int] = {}
        self._last_lights: dict[str, str] = {}
        self.stats = {"stops": 0, "skips": 0, "saved_green": 0.0, "vehicles": {}, "waited": 0, "total_wait": 0.0,
                      "faults": 0}
        self.started = time.time()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "TrafficSystem":
        for m in self.monitors.values():
            m.start()
        threading.Thread(target=self._loop, daemon=True, name="controller").start()
        return self

    def stop(self) -> None:
        self._stop.set()
        for m in self.monitors.values():
            m.stop()
        self._close_open_logs(time.time(), "shutdown")
        self.driver.close()
        self.db.update(RunSession, self.session.id, ended_at=datetime.now())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    now = time.time()
                    inputs = self._inputs(now)
                    self._handle(self.controller.tick(now, inputs))
                    self._apply_lights()
            except Exception:
                log.exception("controller loop error")
            time.sleep(0.1)

    # --------------------------------------------------------------- inputs
    def _inputs(self, now: float) -> dict[str, LaneInput]:
        inputs = {}
        for lane in self.cfg.lanes:
            mon = self.monitors.get(lane.id)
            if mon is None:
                inputs[lane.id] = LaneInput(monitored=False)
                continue
            st = mon.get_state()
            healthy = self._watchdog(lane.id, mon, st, now)
            inputs[lane.id] = LaneInput(monitored=True, healthy=healthy, occupied=st.occupied, count=st.count)
        return inputs

    def _watchdog(self, lane_id, mon, st, now) -> bool:
        fs, h = self.cfg.failsafe, self.health[lane_id]
        last_frame = max(mon.camera.last_frame_time, 0.0)
        if st.last_update == 0 and st.consecutive_errors == 0 and now - mon.source_since < 60:
            return h.healthy  # grace period until the model has processed its first frame
        problem = ""
        if now - last_frame > fs.frame_timeout:
            problem = f"camera: no frames for {now - last_frame:.1f}s ({mon.camera.error or 'feed lost'})"
        elif st.consecutive_errors >= fs.error_limit:
            problem = f"detection: {st.error}"
        elif now - st.last_update > fs.frame_timeout:
            problem = "detection: model stalled"

        if problem:
            h.ok_since = None
            if h.healthy:
                h.healthy, h.reason = False, problem
                self.stats["faults"] += 1
                log.warning("lane %s FAILSAFE: %s", lane_id, problem)
                h.fault_id = self.db.add(FaultRecord(
                    session_id=self.session.id, lane_id=lane_id, fault_type=problem.split(":")[0],
                    message=problem, response="lane reverted to fixed-time cycle")).id
        elif not h.healthy:
            h.ok_since = h.ok_since or now
            if now - h.ok_since >= fs.recover_after:
                h.healthy, h.reason = True, ""
                log.info("lane %s recovered, adaptive control resumed", lane_id)
                self.db.update(FaultRecord, h.fault_id, resolved_at=datetime.now())
        return h.healthy

    # --------------------------------------------------------------- events
    def _handle(self, events: list[Event]) -> None:
        for ev in events:
            d = ev.data
            if ev.kind == "phase_start":
                waited, total_wait = 0, 0.0
                if d["color"] == "green" and ev.lane in self.monitors:
                    tracks = self.monitors[ev.lane].get_state().zone_tracks
                    waits = [ev.time - first for first in tracks.values() if first < ev.time]
                    waited, total_wait = len(waits), float(sum(waits))
                    self.stats["waited"] += waited
                    self.stats["total_wait"] += total_wait
                row = self.db.add(SignalTimingLog(
                    session_id=self.session.id, lane_id=ev.lane, signal_color=d["color"],
                    mode=self.controller.mode.value, basis=d.get("basis", ""), planned_duration=d["planned"],
                    vehicles_at_start=d.get("count", 0), vehicles_waited=waited, total_wait=total_wait,
                    started_at=ts(ev.time)))
                self._log_ids[ev.lane] = row.id
            elif ev.kind == "phase_end":
                self.db.update(SignalTimingLog, self._log_ids.pop(ev.lane, None), actual_duration=d["duration"],
                               action=d.get("reason", "complete"), ended_at=ts(ev.time))
            elif ev.kind == "stop":
                self.stats["stops"] += 1
                self.stats["saved_green"] += d["saved_green"]
                response = max(0.0, ev.time - (d["empty_since"] + self.cfg.signal.empty_timeout))
                self.db.add(StopSkipEvent(
                    session_id=self.session.id, lane_id=ev.lane, signal_log_id=self._log_ids.get(ev.lane),
                    detection_record_id=self._last_status_record.get(ev.lane), event_type="stop",
                    reason=d["reason"], green_elapsed=d["green_elapsed"], saved_green=d["saved_green"],
                    response_time=response, occurred_at=ts(ev.time)))
            elif ev.kind == "skip":
                self.stats["skips"] += 1
                self.db.add(StopSkipEvent(
                    session_id=self.session.id, lane_id=ev.lane, event_type="skip", reason=d["reason"],
                    detection_record_id=self._last_status_record.get(ev.lane), occurred_at=ts(ev.time)))

    def _apply_lights(self) -> None:
        lights = self.controller.lights()
        self.driver.apply(lights)
        for lane, color in lights.items():
            if self._last_lights.get(lane) != color:
                self.db.update(Lane, lane, status=color)
        self._last_lights = lights

    def _close_open_logs(self, now: float, action: str) -> None:
        for lane in list(self._log_ids):
            self.db.update(SignalTimingLog, self._log_ids.pop(lane), action=action, ended_at=ts(now))

    # ----------------------------------------------------- monitor callbacks
    def _on_vehicle(self, lane_id: str, v: dict) -> None:
        counts = self.stats["vehicles"].setdefault(lane_id, {})
        counts[v["vehicle_type"]] = counts.get(v["vehicle_type"], 0) + 1
        self.db.add(DetectionRecord(session_id=self.session.id, lane_id=lane_id, detection_status="vehicle", **v))

    def _on_status(self, lane_id: str, occupied: bool, count: int) -> None:
        row = self.db.add(DetectionRecord(session_id=self.session.id, lane_id=lane_id, vehicle_count=count,
                                          detection_status="occupied" if occupied else "empty"))
        self._last_status_record[lane_id] = row.id

    # ------------------------------------------------------ operator commands
    def command(self, user: str, action: str, lane: str | None = None, color: str | None = None,
                settings: dict | None = None) -> None:
        """Manual override / control panel entry point (dashboard, WebSocket)."""
        with self._lock:
            now = time.time()
            c = self.controller
            if action == "manual":
                self._close_open_logs(now, "manual")
                c.set_manual(lane, color)
            elif action == "emergency_stop":
                self._close_open_logs(now, "emergency_stop")
                c.emergency_stop()
            elif action == "off":
                self._close_open_logs(now, "off")
                c.switch_off()
            elif action in ("resume", "adaptive", "fixed"):
                self._close_open_logs(now, f"switch_{action}")
                self._handle(c.resume(now, "adaptive" if action == "resume" else action))
            elif action == "settings":
                self._apply_settings(settings or {})
            else:
                raise ValueError(f"unknown action {action}")
            self._apply_lights()
            self.db.add(ManualOverride(username=user, action=action, lane_id=lane,
                                       details=json.dumps(settings if action == "settings" else {"color": color})))

    def set_source(self, user: str, lane_id: str, source) -> None:
        """Feed a lane from another video / webcam without restarting (dashboard upload)."""
        mon = self.monitors.get(lane_id)
        if mon is None:
            raise ValueError(f"lane {lane_id} has no camera configured")
        mon.set_source(source)
        self.db.add(ManualOverride(username=user, action="source", lane_id=lane_id,
                                   details=json.dumps({"source": str(source)})))

    def _apply_settings(self, settings: dict) -> None:
        for k, v in settings.items():
            if k in EDITABLE_SIGNAL_FIELDS:
                cur = getattr(self.cfg.signal, k)
                setattr(self.cfg.signal, k, (str(v).lower() in ("1", "true", "on")) if isinstance(cur, bool)
                        else type(cur)(v))
        RUNTIME_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        RUNTIME_SETTINGS.write_text(json.dumps({k: getattr(self.cfg.signal, k) for k in EDITABLE_SIGNAL_FIELDS},
                                               indent=2))

    def _load_runtime_settings(self) -> None:
        """Re-apply the last settings saved from the dashboard ("last valid configuration")."""
        if RUNTIME_SETTINGS.exists():
            try:
                for k, v in json.loads(RUNTIME_SETTINGS.read_text()).items():
                    if k in EDITABLE_SIGNAL_FIELDS:
                        setattr(self.cfg.signal, k, v)
            except (ValueError, TypeError):
                log.warning("ignoring unreadable %s", RUNTIME_SETTINGS)

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        with self._lock:
            now = time.time()
            lanes = []
            for lane in self.cfg.lanes:
                item = {"id": lane.id, "name": lane.name, "direction": lane.direction,
                        "monitored": lane.monitored, "light": self.controller.lights()[lane.id]}
                mon = self.monitors.get(lane.id)
                if mon:
                    st = mon.get_state()
                    h = self.health[lane.id]
                    item.update(occupied=st.occupied, count=st.count, class_counts=st.class_counts,
                                line_in=st.line_in, line_out=st.line_out, avg_conf=round(st.avg_conf, 2),
                                fps=round(st.fps, 1), infer_ms=round(st.infer_ms, 1), healthy=h.healthy,
                                fault=h.reason, seen=self.stats["vehicles"].get(lane.id, {}))
                lanes.append(item)
            waited = self.stats["waited"]
            return {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session": self.session.id,
                "controller": self.controller.snapshot(now),
                "lanes": lanes,
                "stats": {"stops": self.stats["stops"], "skips": self.stats["skips"],
                          "saved_green": round(self.stats["saved_green"], 1), "faults": self.stats["faults"],
                          "avg_wait": round(self.stats["total_wait"] / waited, 1) if waited else 0.0},
                "signal": asdict(self.cfg.signal),
            }
