"""Export a Chapter 4 summary of live test sessions from the SQLite database.

  python scripts/export_report.py              # latest session
  python scripts/export_report.py --session 3
  python scripts/export_report.py --all        # every session, one row each

Writes reports/session_<id>/ with every table as CSV, summary.csv, and green_phases.png.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from trafficsense.config import load_config

TABLES = ["detection_records", "signal_timing_logs", "stop_skip_events", "fault_records"]


def summarize(con, sid: int) -> dict:
    q = lambda sql: pd.read_sql_query(sql, con, params=(sid,))  # noqa: E731
    sess = q("SELECT * FROM run_sessions WHERE id = ?").iloc[0]
    green = q("SELECT * FROM signal_timing_logs WHERE session_id = ? AND signal_color = 'green' "
              "AND actual_duration IS NOT NULL")
    stops = q("SELECT * FROM stop_skip_events WHERE session_id = ?")
    vehicles = q("SELECT vehicle_type, COUNT(*) n FROM detection_records WHERE session_id = ? "
                 "AND detection_status = 'vehicle' GROUP BY vehicle_type")
    faults = q("SELECT * FROM fault_records WHERE session_id = ?")
    waited = green["vehicles_waited"].sum()
    return {
        "session": sid, "mode": sess["mode"], "started": sess["started_at"], "ended": sess["ended_at"],
        "green_phases": len(green),
        "avg_green_s": round(green["actual_duration"].mean(), 2) if len(green) else 0,
        "stopped_early": int((stops["event_type"] == "stop").sum()),
        "skipped": int((stops["event_type"] == "skip").sum()),
        "idle_green_saved_s": round(stops["saved_green"].fillna(0).sum(), 1),
        "avg_response_s": round(stops["response_time"].dropna().mean(), 3) if stops["response_time"].notna().any() else None,
        "avg_wait_s": round(green["total_wait"].sum() / waited, 2) if waited else None,
        "unique_vehicles": int(vehicles["n"].sum()),
        **{f"vehicles_{r.vehicle_type}": int(r.n) for r in vehicles.itertuples()},
        "faults": len(faults),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(load_config().db_path)
    ids = [r[0] for r in con.execute("SELECT id FROM run_sessions ORDER BY id")]
    if not ids:
        raise SystemExit("no sessions recorded yet — run `python run.py` first")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)

    if args.all:
        df = pd.DataFrame([summarize(con, i) for i in ids])
        df.to_csv(out / "all_sessions.csv", index=False)
        print(df.to_string(index=False))
        print(f"\n-> {out / 'all_sessions.csv'}")
        return

    sid = args.session or ids[-1]
    folder = out / f"session_{sid}"
    folder.mkdir(exist_ok=True)
    for t in TABLES:
        pd.read_sql_query(f"SELECT * FROM {t} WHERE session_id = ?", con, params=(sid,)).to_csv(
            folder / f"{t}.csv", index=False)
    summary = summarize(con, sid)
    pd.DataFrame([summary]).to_csv(folder / "summary.csv", index=False)
    for k, v in summary.items():
        print(f"{k:22s} {v}")

    green = pd.read_sql_query("SELECT lane_id, actual_duration, action FROM signal_timing_logs WHERE session_id = ? "
                              "AND signal_color = 'green' AND actual_duration IS NOT NULL", con, params=(sid,))
    if len(green):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"stop_empty": "#2a9d55", "max_green": "#0077b6", "fixed": "#8d99ae", "failsafe": "#e63946"}
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.bar(range(len(green)), green["actual_duration"], color=[colors.get(a, "#adb5bd") for a in green["action"]])
        ax.set_xlabel("green phase #")
        ax.set_ylabel("duration (s)")
        ax.set_title(f"Session {sid}: green phase durations by end reason")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
        ax.legend(handles, ["stopped (lane empty)", "max green", "fixed timing", "failsafe"], frameon=False,
                  ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(folder / "green_phases.png", dpi=150)
    print(f"\n-> {folder}")


if __name__ == "__main__":
    main()
