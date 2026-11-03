"""Fixed-time vs. adaptive evaluation (Section 3.10.6) on recorded footage.

Vehicles in recorded video do not obey our signal, so waiting time cannot be measured
from the footage directly. Instead:
  1. `extract_arrivals` runs YOLOv8 + ByteTrack over a lane video and records the time
     each unique vehicle enters the lane zone (real arrival pattern of that road).
  2. `simulate` replays those arrivals through a queue model at the intersection, driven
     by the *same* SignalController used in the live system, once per signal mode.
Queued vehicles discharge one per `headway` seconds during green (saturation flow),
after a start-up lost time.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, replace

import cv2

from .config import Config, LaneConfig, SignalConfig
from .controller import LaneInput, Phase, SignalController


def extract_arrivals(cfg: Config, lane: LaneConfig, video: str, stride: int = 2,
                     progress: bool = True) -> tuple[list[float], float]:
    """Return (arrival times in video seconds, video duration)."""
    from .detector import LaneMonitor

    lane = replace(lane, source=video)
    mon = LaneMonitor(cfg, lane)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seen: set[int] = set()
    arrivals: list[float] = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if mon.scene is None:
                mon.setup_geometry(frame.shape[1], frame.shape[0])
            det = mon.detect(frame)
            if len(det) and det.tracker_id is not None:
                for tid in det.tracker_id[mon.scene.update(det)]:
                    if int(tid) not in seen:
                        seen.add(int(tid))
                        arrivals.append(idx / fps)
            if progress and idx % (stride * 300) == 0:
                print(f"    {lane.id}: {idx}/{total} frames, {len(arrivals)} vehicles")
        idx += 1
    cap.release()
    return arrivals, idx / fps


def poisson_arrivals(rate_per_min: float, duration: float, seed: int) -> list[float]:
    rng, t, out = random.Random(seed), 0.0, []
    if rate_per_min <= 0:
        return out
    while True:
        t += rng.expovariate(rate_per_min / 60.0)
        if t >= duration:
            return out
        out.append(t)


@dataclass
class SimResult:
    mode: str
    vehicles: int
    served: int
    avg_wait: float
    max_wait: float
    idle_green: float          # seconds of green shown to an empty lane
    green_total: float
    cycles: int
    stops: int
    per_lane_wait: dict[str, float]


def simulate(signal: SignalConfig, arrivals: dict[str, list[float]], mode: str, duration: float,
             dt: float = 0.1, headway: float = 2.0, startup_lost: float = 2.0) -> SimResult:
    lanes = list(arrivals)
    ctl = SignalController(signal, lanes, mode)
    pending = {l: deque(sorted(a)) for l, a in arrivals.items()}
    queue: dict[str, deque] = {l: deque() for l in lanes}
    waits: dict[str, list[float]] = {l: [] for l in lanes}
    last_discharge = {l: -1e9 for l in lanes}
    idle = green_total = 0.0
    cycles = stops = 0
    t = 0.0
    while t < duration:
        for l in lanes:
            while pending[l] and pending[l][0] <= t:
                queue[l].append(pending[l].popleft())
        inputs = {l: LaneInput(monitored=True, healthy=True, occupied=bool(queue[l]), count=len(queue[l]))
                  for l in lanes}
        for ev in ctl.tick(t, inputs):
            if ev.kind == "stop":
                stops += 1
            if ev.kind == "phase_start" and ev.data["color"] == "green" and ev.lane == lanes[0]:
                cycles += 1
        if ctl.phase == Phase.GREEN:
            l = ctl.current_lane
            green_total += dt
            if not queue[l]:
                idle += dt
            elif t - ctl.phase_start >= startup_lost and t - last_discharge[l] >= headway:
                waits[l].append(t - queue[l].popleft())
                last_discharge[l] = t
        t += dt
    # vehicles still queued at the end count with the wait accumulated so far
    unserved = [duration - a for l in lanes for a in queue[l]]
    served = [w for ws in waits.values() for w in ws]
    allw = served + unserved
    return SimResult(
        mode=mode, vehicles=len(allw), served=len(served),
        avg_wait=sum(allw) / len(allw) if allw else 0.0, max_wait=max(allw, default=0.0),
        idle_green=idle, green_total=green_total, cycles=cycles, stops=stops,
        per_lane_wait={l: (sum(waits[l]) + sum(duration - a for a in queue[l])) / max(1, len(waits[l]) + len(queue[l]))
                       for l in lanes},
    )
