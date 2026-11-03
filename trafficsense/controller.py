"""Adaptive traffic signal state machine (flowchart in Section 3.12.5 of the paper).

The controller is pure logic: it receives the current time and per-lane detection
inputs, and returns events. It does no I/O, so the live system, the fixed-vs-adaptive
simulation, and the unit tests all run exactly the same code.

Cycle:  GREEN(lane) -> YELLOW(lane) -> ALL_RED -> GREEN(next lane) ...
A green phase ends when
  * the lane has been empty for `empty_timeout` seconds (after `min_green`)  -> "stop_empty"
  * the planned green (max_green, or count-based) has elapsed               -> "max_green"
  * the lane is camera-less, unhealthy (failsafe) or the mode is fixed      -> "fixed" / "failsafe"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import SignalConfig


class Phase(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ALL_RED = "all_red"


class Mode(str, Enum):
    ADAPTIVE = "adaptive"
    FIXED = "fixed"
    MANUAL = "manual"      # lights held by an operator override
    OFF = "off"            # signal switched off from the control panel


@dataclass
class LaneInput:
    monitored: bool = False   # lane has a camera
    healthy: bool = True      # camera + model OK (watchdog)
    occupied: bool = False    # at least one vehicle inside the lane zone
    count: int = 0            # vehicles inside the lane zone


@dataclass
class Event:
    kind: str                 # phase_start | phase_end | stop | skip
    lane: str
    time: float
    data: dict = field(default_factory=dict)


class SignalController:
    def __init__(self, cfg: SignalConfig, lane_ids: list[str], mode: str | None = None):
        if not lane_ids:
            raise ValueError("at least one lane is required")
        self.cfg = cfg
        self.lanes = list(lane_ids)
        self.mode = Mode(mode or cfg.mode)
        self.index = 0
        self.phase = Phase.GREEN
        self.phase_start: float | None = None
        self.planned_green = 0.0
        self.green_reason = ""
        self.empty_since: float | None = None
        self.manual_lights: dict[str, str] = {}

    # ----------------------------------------------------------------- state
    @property
    def current_lane(self) -> str:
        return self.lanes[self.index]

    def lights(self) -> dict[str, str]:
        """Color of every lane's signal head: green | yellow | red | off."""
        if self.mode == Mode.OFF:
            return {l: "off" for l in self.lanes}
        if self.mode == Mode.MANUAL:
            return {l: self.manual_lights.get(l, "red") for l in self.lanes}
        colors = {l: "red" for l in self.lanes}
        if self.phase == Phase.GREEN:
            colors[self.current_lane] = "green"
        elif self.phase == Phase.YELLOW:
            colors[self.current_lane] = "yellow"
        return colors

    def snapshot(self, now: float) -> dict:
        return {
            "mode": self.mode.value,
            "lane": self.current_lane,
            "phase": self.phase.value,
            "elapsed": round(now - self.phase_start, 1) if self.phase_start is not None else 0.0,
            "planned_green": self.planned_green,
            "empty_for": round(now - self.empty_since, 1) if self.empty_since is not None else 0.0,
            "lights": self.lights(),
        }

    # -------------------------------------------------------------- commands
    def set_manual(self, lane: str, color: str) -> None:
        """Operator override: hold `lane` at `color`; every other lane is forced red."""
        if lane not in self.lanes or color not in ("green", "yellow", "red"):
            raise ValueError(f"invalid manual command {lane}={color}")
        self.mode = Mode.MANUAL
        self.manual_lights = {l: "red" for l in self.lanes}
        self.manual_lights[lane] = color

    def emergency_stop(self) -> None:
        self.mode = Mode.MANUAL
        self.manual_lights = {l: "red" for l in self.lanes}

    def switch_off(self) -> None:
        self.mode = Mode.OFF

    def resume(self, now: float, mode: str = "adaptive") -> list[Event]:
        """Leave manual/off and restart automatic control at the next lane."""
        self.mode = Mode(mode)
        self.manual_lights = {}
        self.index = (self.index + 1) % len(self.lanes)
        return self._start_green(now, None)

    # ------------------------------------------------------------------ tick
    def tick(self, now: float, inputs: dict[str, LaneInput]) -> list[Event]:
        if self.mode in (Mode.MANUAL, Mode.OFF):
            return []
        if self.phase_start is None:  # first tick
            return self._start_green(now, inputs)

        elapsed = now - self.phase_start
        lane = self.current_lane

        if self.phase == Phase.GREEN:
            inp = inputs.get(lane, LaneInput())
            adaptive = self.mode == Mode.ADAPTIVE and inp.monitored and inp.healthy
            if not adaptive:
                if elapsed >= self.cfg.fixed_green:
                    # A lane that failed mid-green ends on failsafe timing.
                    reason = "failsafe" if self.green_reason == "adaptive" else self.green_reason
                    return self._end_green(now, reason, elapsed)
                return []
            if inp.occupied:
                self.empty_since = None
            elif self.empty_since is None:
                self.empty_since = now
            if (self.empty_since is not None and elapsed >= self.cfg.min_green
                    and now - self.empty_since >= self.cfg.empty_timeout):
                events = [Event("stop", lane, now, {
                    "reason": "no vehicle detected",
                    "empty_since": self.empty_since,
                    "green_elapsed": elapsed,
                    "saved_green": max(0.0, self.planned_green - elapsed),
                })]
                return events + self._end_green(now, "stop_empty", elapsed)
            if elapsed >= self.planned_green:
                return self._end_green(now, "max_green", elapsed)
            return []

        if self.phase == Phase.YELLOW and elapsed >= self.cfg.yellow:
            events = [Event("phase_end", lane, now, {"color": "yellow", "duration": elapsed})]
            if self.cfg.all_red > 0:
                self.phase, self.phase_start = Phase.ALL_RED, now
                return events
            self.index = (self.index + 1) % len(self.lanes)
            return events + self._start_green(now, inputs)

        if self.phase == Phase.ALL_RED and elapsed >= self.cfg.all_red:
            self.index = (self.index + 1) % len(self.lanes)
            return self._start_green(now, inputs)
        return []

    # -------------------------------------------------------------- internal
    def _plan(self, inp: LaneInput) -> tuple[float, str]:
        if self.mode == Mode.FIXED:
            return self.cfg.fixed_green, "fixed"
        if not inp.monitored:
            return self.cfg.fixed_green, "fixed"
        if not inp.healthy:
            return self.cfg.fixed_green, "failsafe"
        if self.cfg.count_based:
            g = self.cfg.base_green + self.cfg.per_vehicle * inp.count
            return float(min(self.cfg.max_green, max(self.cfg.min_green, g))), "adaptive"
        return self.cfg.max_green, "adaptive"

    def _start_green(self, now: float, inputs: dict[str, LaneInput] | None) -> list[Event]:
        inputs = inputs or {}
        events: list[Event] = []
        # Optionally skip lanes that are already empty (at most one full pass, so an
        # entirely empty intersection still cycles instead of spinning forever).
        if self.mode == Mode.ADAPTIVE and self.cfg.skip_empty_lanes:
            for _ in range(len(self.lanes) - 1):
                inp = inputs.get(self.current_lane, LaneInput())
                if inp.monitored and inp.healthy and not inp.occupied:
                    events.append(Event("skip", self.current_lane, now, {"reason": "lane empty at green start"}))
                    self.index = (self.index + 1) % len(self.lanes)
                else:
                    break
        inp = inputs.get(self.current_lane, LaneInput())
        self.planned_green, self.green_reason = self._plan(inp)
        self.phase, self.phase_start = Phase.GREEN, now
        self.empty_since = None if inp.occupied else now  # empty timer starts at green onset
        events.append(Event("phase_start", self.current_lane, now, {
            "color": "green", "planned": self.planned_green, "basis": self.green_reason,
            "count": inp.count, "mode": self.mode.value,
        }))
        return events

    def _end_green(self, now: float, reason: str, elapsed: float) -> list[Event]:
        lane = self.current_lane
        self.phase, self.phase_start = Phase.YELLOW, now
        return [
            Event("phase_end", lane, now, {"color": "green", "duration": elapsed, "reason": reason,
                                           "planned": self.planned_green}),
            Event("phase_start", lane, now, {"color": "yellow", "planned": self.cfg.yellow}),
        ]
