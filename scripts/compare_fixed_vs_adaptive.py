"""Fixed-time vs. adaptive signal comparison on recorded footage (Sections 3.10.6 / 3.11.2).

  # one video per approach you recorded; lanes without video get random arrivals
  python scripts/compare_fixed_vs_adaptive.py --video S1=data/raw_videos/lane1.mp4 --rate 4

  # re-run the simulation with new timing without re-detecting
  python scripts/compare_fixed_vs_adaptive.py --arrivals reports/arrivals.json

Outputs reports/comparison.csv, reports/comparison.png, reports/arrivals.json and an
evaluation_results row per mode.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trafficsense.config import load_config, resolve_path
from trafficsense.db import Database, EvaluationResult
from trafficsense.simulation import poisson_arrivals, simulate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", default=[], help="LANE=path, repeatable")
    ap.add_argument("--arrivals", help="reuse a saved arrivals.json")
    ap.add_argument("--rate", type=float, default=4.0, help="veh/min for lanes without video")
    ap.add_argument("--duration", type=float, help="seconds to simulate (default: longest video)")
    ap.add_argument("--headway", type=float, default=2.0, help="saturation headway, s/vehicle")
    ap.add_argument("--stride", type=int, default=2, help="process every Nth video frame")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    if args.arrivals:
        saved = json.loads(Path(args.arrivals).read_text())
        arrivals, duration, origin = saved["arrivals"], saved["duration"], saved["origin"]
    else:
        from trafficsense.simulation import extract_arrivals

        arrivals, origin, durations = {}, {}, []
        for item in args.video:
            lane_id, path = item.split("=", 1)
            print(f"detecting vehicles in {path} for lane {lane_id} ...")
            arr, dur = extract_arrivals(cfg, cfg.lane(lane_id), str(resolve_path(path)), stride=args.stride)
            arrivals[lane_id], origin[lane_id] = arr, f"video {Path(path).name}"
            durations.append(dur)
            print(f"  {len(arr)} vehicles in {dur:.0f}s ({len(arr) / dur * 60:.1f} veh/min)")
        duration = args.duration or (max(durations) if durations else 900.0)
        for i, lane in enumerate(cfg.lanes):
            if lane.id not in arrivals:
                arrivals[lane.id] = poisson_arrivals(args.rate, duration, args.seed + i)
                origin[lane.id] = f"random {args.rate} veh/min"
        arrivals = {l.id: arrivals[l.id] for l in cfg.lanes}  # signal order
        (reports / "arrivals.json").write_text(json.dumps(
            {"duration": duration, "origin": origin, "arrivals": arrivals}, indent=1))
    if args.duration:
        duration = args.duration

    results = [simulate(cfg.signal, arrivals, mode, duration, headway=args.headway)
               for mode in ("fixed", "adaptive")]

    print(f"\nsimulated {duration:.0f}s; arrivals: " + ", ".join(f"{k}={len(v)} ({origin[k]})"
                                                         for k, v in arrivals.items()))
    print(f"{'metric':28s} {'fixed':>10s} {'adaptive':>10s}")
    fixed, adaptive = results
    rows = [
        ("vehicles", fixed.vehicles, adaptive.vehicles),
        ("served", fixed.served, adaptive.served),
        ("average wait (s)", fixed.avg_wait, adaptive.avg_wait),
        ("max wait (s)", fixed.max_wait, adaptive.max_wait),
        ("idle green time (s)", fixed.idle_green, adaptive.idle_green),
        ("idle green share (%)", 100 * fixed.idle_green / max(1, fixed.green_total),
         100 * adaptive.idle_green / max(1, adaptive.green_total)),
        ("signal cycles", fixed.cycles, adaptive.cycles),
        ("green stops (empty lane)", fixed.stops, adaptive.stops),
    ]
    for name, f, a in rows:
        print(f"{name:28s} {f:10.1f} {a:10.1f}")
    for lane in arrivals:
        print(f"{'avg wait ' + lane + ' (s)':28s} {fixed.per_lane_wait[lane]:10.1f} {adaptive.per_lane_wait[lane]:10.1f}")
    if fixed.avg_wait:
        print(f"\nadaptive changes average waiting time by {100 * (adaptive.avg_wait - fixed.avg_wait) / fixed.avg_wait:+.1f}%")

    with open(reports / "comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "fixed", "adaptive"])
        w.writerows((n, round(f, 2), round(a, 2)) for n, f, a in rows)
        for lane in arrivals:
            w.writerow([f"avg wait {lane} (s)", round(fixed.per_lane_wait[lane], 2),
                        round(adaptive.per_lane_wait[lane], 2)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (title, key) in zip(axes, [("Average waiting time (s)", "avg_wait"),
                                       ("Max waiting time (s)", "max_wait"),
                                       ("Idle green time (s)", "idle_green")]):
        vals = [getattr(r, key) for r in results]
        bars = ax.bar(["Fixed-time", "Adaptive"], vals, color=["#8d99ae", "#2a9d55"])
        ax.bar_label(bars, fmt="%.1f")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Fixed-time vs. adaptive signal ({duration / 60:.0f} min of traffic)")
    fig.tight_layout()
    fig.savefig(reports / "comparison.png", dpi=150)
    print(f"\nreports -> {reports / 'comparison.csv'}, {reports / 'comparison.png'}")

    db = Database(cfg.db_path)
    sig = asdict(cfg.signal)
    for r in results:
        db.add(EvaluationResult(test_name="fixed vs adaptive simulation", mode=r.mode, total_runs=1,
                                avg_wait_time=r.avg_wait, idle_green_time=r.idle_green,
                                notes=json.dumps({"duration": duration, "vehicles": r.vehicles, "served": r.served,
                                                  "max_wait": r.max_wait, "cycles": r.cycles, "origin": origin,
                                                  "signal": sig})))


if __name__ == "__main__":
    main()
