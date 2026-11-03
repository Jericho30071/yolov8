"""Signal controller behavior vs. the flowchart in Section 3.12.5 of the paper."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trafficsense.config import SignalConfig
from trafficsense.controller import LaneInput, Mode, Phase, SignalController
from trafficsense.simulation import simulate

LANES = ["S1", "S2", "S3", "S4"]


def cfg(**kw):
    base = dict(fixed_green=30, min_green=5, max_green=30, empty_timeout=5, yellow=3, all_red=1)
    base.update(kw)
    return SignalConfig(**base)


def run(ctl, inputs_fn, until, dt=0.1):
    events = []
    for i in range(int(round(until / dt)) + 1):
        t = round(i * dt, 3)
        events += ctl.tick(t, inputs_fn(t))
    return events


def ends(events):
    return [(e.lane, e.data["reason"], round(e.data["duration"], 1))
            for e in events if e.kind == "phase_end" and e.data["color"] == "green"]


def monitored(occupied=False, healthy=True, count=0):
    return {l: LaneInput(monitored=True, healthy=healthy, occupied=occupied, count=count) for l in LANES}


def test_empty_lane_stops_after_timeout():
    ctl = SignalController(cfg(), LANES)
    ev = run(ctl, lambda t: monitored(occupied=False), until=6)
    assert ends(ev) == [("S1", "stop_empty", 5.0)]
    assert any(e.kind == "stop" and e.lane == "S1" for e in ev)
    assert ctl.phase == Phase.YELLOW and ctl.lights()["S1"] == "yellow"


def test_vehicle_present_resets_empty_timer_and_keeps_normal_timing():
    ctl = SignalController(cfg(), LANES)
    # vehicle in S1 for the first 20 s, then gone: stop at 20 + 5
    ev = run(ctl, lambda t: monitored(occupied=t < 20), until=26)
    assert ends(ev) == [("S1", "stop_empty", 25.0)]


def test_busy_lane_runs_to_max_green():
    ctl = SignalController(cfg(), LANES)
    ev = run(ctl, lambda t: monitored(occupied=True), until=31)
    assert ends(ev) == [("S1", "max_green", 30.0)]


def test_yellow_three_seconds_then_all_red_then_next_lane_in_sequence():
    ctl = SignalController(cfg(), LANES)
    ev = run(ctl, lambda t: monitored(occupied=False), until=40)
    greens = [(e.lane, round(e.time, 1)) for e in ev if e.kind == "phase_start" and e.data["color"] == "green"]
    # green 5 s + yellow 3 s + all-red 1 s = 9 s per empty lane
    assert greens == [("S1", 0.0), ("S2", 9.0), ("S3", 18.0), ("S4", 27.0), ("S1", 36.0)]


def test_unhealthy_lane_falls_back_to_fixed_time():
    ctl = SignalController(cfg(fixed_green=20), LANES)
    ev = run(ctl, lambda t: monitored(occupied=False, healthy=False), until=21)
    assert ends(ev) == [("S1", "failsafe", 20.0)]


def test_camera_less_lane_uses_fixed_time():
    ctl = SignalController(cfg(fixed_green=12), LANES)
    ev = run(ctl, lambda t: {l: LaneInput(monitored=False) for l in LANES}, until=13)
    assert ends(ev) == [("S1", "fixed", 12.0)]


def test_fixed_mode_ignores_detection():
    ctl = SignalController(cfg(fixed_green=15), LANES, mode="fixed")
    ev = run(ctl, lambda t: monitored(occupied=False), until=16)
    assert ends(ev) == [("S1", "fixed", 15.0)]


def test_count_based_green():
    ctl = SignalController(cfg(count_based=True, base_green=10, per_vehicle=2, max_green=40), LANES)
    ctl.tick(0, monitored(occupied=True, count=6))
    assert ctl.planned_green == 22


def test_skip_empty_lanes():
    ctl = SignalController(cfg(skip_empty_lanes=True), LANES)
    inputs = monitored(occupied=False)
    inputs["S3"] = LaneInput(monitored=True, occupied=True, count=2)
    ev = ctl.tick(0, inputs)
    assert [e.lane for e in ev if e.kind == "skip"] == ["S1", "S2"]
    assert ctl.current_lane == "S3" and ctl.lights()["S3"] == "green"


def test_manual_override_and_resume():
    ctl = SignalController(cfg(), LANES)
    ctl.tick(0, monitored(occupied=True))
    ctl.set_manual("S2", "green")
    assert ctl.mode == Mode.MANUAL
    assert ctl.lights() == {"S1": "red", "S2": "green", "S3": "red", "S4": "red"}
    assert ctl.tick(100, monitored()) == []          # adaptive loop suspended
    ctl.emergency_stop()
    assert set(ctl.lights().values()) == {"red"}
    ctl.resume(200)
    assert ctl.mode == Mode.ADAPTIVE and ctl.current_lane == "S2" and ctl.lights()["S2"] == "green"


def test_adaptive_beats_fixed_on_light_traffic():
    arrivals = {"S1": [float(t) for t in range(0, 600, 20)], "S2": [], "S3": [5.0, 300.0], "S4": []}
    fixed = simulate(cfg(), arrivals, "fixed", 600)
    adaptive = simulate(cfg(), arrivals, "adaptive", 600)
    assert adaptive.idle_green < fixed.idle_green
    assert adaptive.avg_wait < fixed.avg_wait
    assert adaptive.vehicles == fixed.vehicles
