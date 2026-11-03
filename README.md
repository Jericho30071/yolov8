# TrafficSense

Traffic Light System with Vehicle Detection for Real-Time Traffic Management
(Arat, Junaid, Olila, Juario, Gordo — USTP, BSIT).

YOLOv8 detects vehicles in the monitored lane. Ultralytics' ByteTrack/BoT-SORT gives each
vehicle an ID, and a Supervision polygon zone decides whether the lane is occupied. The
adaptive controller ends a green early when the lane stays empty for 5 s. Everything is
logged to SQLite and shown on a FastAPI dashboard with manual override.

## Just watch the detection on a video (no server)

```powershell
.venv\Scripts\python.exe detect.py                 # file dialog to pick a video
.venv\Scripts\python.exe detect.py myvideo.mp4     # specific video;  0 = webcam
.venv\Scripts\python.exe detect.py myvideo.mp4 --save   # also save reports/annotated_<name>.mp4
```

Space pauses; q, Esc or the window's X closes it.

## Traffic-light simulation (no server)

```powershell
.venv\Scripts\python.exe simulate.py data\raw_videos\SAMPLE1.mp4
```

The video (lane S1, YOLOv8 detection) drives a 4-way signal panel run by the same
controller as the real system:

| Behavior | Paper |
|---|---|
| Green with no vehicle for 5 s → yellow 3 s → red → next lane | Objective 2, Flowchart 3.12.5 |
| Vehicles keep coming → normal green up to 30 s | Objective 2 |
| Sequence S1 → S2 → S3 → S4 (1 s all-red between lanes) | Flowchart 3.12.5 |
| Camera/detection fault → that lane runs fixed 30 s | Sec. 3.9, Table 2 |
| Fixed-time mode for comparison | Sec. 3.10.6 |
| Emergency stop (all red) / resume | Sec. 3.9 manual override |

S2–S4 have no camera yet, so they're simulated. Keys: `2`/`3`/`4` = vehicles waiting on
S2–S4 on/off, `f` = simulate S1 camera failure, `m` = adaptive/fixed, `e` = emergency
stop/resume, `o` = open video, `z` = draw zone, Space = pause, `q` = quit. Each signal event
is saved to `reports/simulation_<date>_<time>.csv`.

## Quick start: full system with dashboard (Windows PC)

```powershell
cd D:\YOLOV8
.venv\Scripts\activate
python run.py                                    # uses config.yaml (sample video on lane S1)
```

Open http://localhost:8000 and log in as `admin` / `admin123` (change it under Settings).

```powershell
python run.py --source data\raw_videos\lane1.mp4   # your own video on lane S1
python run.py --source 0                           # webcam
python run.py --mode fixed                         # conventional fixed-time baseline
python -m pytest tests                             # controller logic tests
```

`data/raw_videos/sample.mp4` is a synthetic 16 s clip (a bus, then an empty road) used only
to test the system. Replace it with real footage.

## When you have your videos: training workflow

Copy the videos into `data\raw_videos\`, then:

| Step | Command | Result |
|---|---|---|
| 1. Extract frames | `python scripts\extract_frames.py --every 1.0` | `data/frames/` (blurry and duplicate frames removed) |
| 2. Auto-label | `python scripts\autolabel.py` | `data/autolabel/` pre-labeled by YOLOv8-L, plus `review.csv` |
| 3. Correct labels | `python scripts\label_tool.py --review` then without `--review` | every box set to one of the 8 local classes |
| 4. Split 70/20/10 | `python scripts\split_dataset.py` (add `--group-by-video` with 3+ videos) | `data/dataset/data.yaml` |
| 5. Train | `python scripts\train.py --model yolov8n.pt --epochs 100` | `models/trafficsense_best.pt` |
| 6. Evaluate | `python scripts\evaluate.py` | P / R / mAP per class, `reports/model_metrics.csv`, confusion matrix in `runs/eval_test/` |
| 7. Use it | set `model.weights: models/trafficsense_best.pt` in `config.yaml` | live system uses your classes |

Notes:
- The stock COCO model only knows car, motorcycle, bus and truck. Auto-labeling turns those
  into best guesses (truck → jeepney) and `review.csv` lists what to check: motorcycle →
  motorela?, car → suv_van or multicab?, jeepney → multicab or truck?
- Label tool keys: drag = new box, `1`–`8` = class (legend on screen), right-click = delete,
  `d`/`a` = next/previous, `q` = quit. Progress is remembered.
- GTX 960 (4 GB): if CUDA runs out of memory, use `--batch 4`. `yolov8s.pt` is more
  accurate but about 3x slower on the Pi.
- Optional: merge downloaded public datasets (e.g. Kaggle) with
  `python scripts\merge_datasets.py --sources <folders>`, which remaps their class names.
- After the first model is trained, re-run `autolabel.py --model models\trafficsense_best.pt`
  on new frames. It then predicts all 8 local classes, so each labeling round gets faster.

## Vehicle classes

The model is built around 8 localized Cagayan de Oro classes (`classes:` in `config.yaml`;
the order is the YOLO class id). Use these rules so every annotator labels the same way.

| Key | Class | Includes | Labeling rule |
|---|---|---|---|
| 1 | `motorcycle` | Motorcycles, scooters, underbones, big bikes | Two wheels. Box the bike and rider together. A motorcycle with a sidecar is a `motorela`. |
| 2 | `motorela` | CDO motorela, tricycles, e-trikes | Three-wheeled public transport with a passenger cab. Box the whole vehicle, cab included. |
| 3 | `multicab` | Suzuki multicab / mini-cab PUVs and private multicabs | Small kei-size body, clearly shorter and narrower than a jeepney. |
| 4 | `jeepney` | Traditional jeepneys and modern PUV jeepneys (e-jeepneys, minibus-style PUVs) | Long passenger body with side entrance or route signboard. Modern jeepneys go here, not in `bus`. |
| 5 | `car` | Sedans, hatchbacks, taxis | Low passenger car with a separate trunk or hatch. |
| 6 | `suv_van` | SUVs, MPVs, pickups, UV Express, private vans | Tall passenger vehicles and pickups. UV Express vans go here, not in `jeepney`. |
| 7 | `bus` | City, provincial and tourist buses | Full-size buses only. Modern jeepneys are `jeepney`. |
| 8 | `truck` | Bongo, closed vans (aluminum/cargo), heavy and dump trucks, trailers | Built for cargo. Small cargo multicabs are `multicab`. |

General rules:
- Box every vehicle that is at least about 50% visible, including in other lanes. Training
  uses the whole frame even though live detection only looks at the zone.
- Skip vehicles so far away or blurred that you can't tell their class.
- For a vehicle partly hidden by another, box only its visible part.
- Aim for at least 150–300 boxes per class. Motorela, multicab and jeepney matter most, since
  the stock model has never seen them.

## Set up the lane zone

```powershell
python scripts\draw_zone.py --source data\raw_videos\lane1.mp4
```

Click the area in front of the stop line, press Enter, click the two ends of the count line,
press Enter, and paste the printed `zone:` / `count_line:` into the lane in `config.yaml`.

## Evaluation for Chapter 4

| Thesis section | Command / source |
|---|---|
| 3.11.1 Detection accuracy | `scripts\evaluate.py` → `reports/model_metrics.csv` and the `evaluation_results` table |
| 3.10.6 / 3.11.2 Fixed vs. adaptive | `python scripts\compare_fixed_vs_adaptive.py --video S1=data\raw_videos\lane1.mp4` → `reports/comparison.csv`, `comparison.png` |
| Live test sessions | `python scripts\export_report.py` (or `--all`) → `reports/session_<id>/` |
| Raw logs | dashboard → Logs → Export CSV |

The comparison script takes real arrival times from your footage (YOLOv8 + tracking) and
replays them through a queue model, once with fixed timing and once with adaptive control.
Both runs use the same controller code as the live system. Recorded vehicles don't respond to
our signal, so waiting time has to be simulated from the footage. Describe it this way in the
methodology. Lanes without video get random (Poisson) arrivals at `--rate` vehicles/min. Give
each lane its own video where possible: `--video S1=a.mp4 --video S2=b.mp4 ...`.

## How it maps to the paper

| Paper | Code |
|---|---|
| Flowchart 3.12.5 (S1→S4, 5 s empty, 3 s yellow) | `trafficsense/controller.py` (tested in `tests/test_controller.py`) |
| Detection / counting / classification | `trafficsense/detector.py` |
| Failsafe Table 2 (camera, detection, communication) | `trafficsense/system.py` watchdog; `web/static/app.js` queues commands while offline |
| ERD 3.12.6 | `trafficsense/db.py` (users, lanes, devices, run_sessions, detection_records, signal_timing_logs, stop_skip_events, manual_overrides, fault_records, evaluation_results) |
| Wireframes 3.7.2 | `trafficsense/web/` (login, dashboard, camera, controls, logs, settings) |
| Relay module 3.7.1 | `trafficsense/signals.py` (`hardware.driver: gpio`) |
| YOLO workflow 3.8 | `scripts/` |

Signal behavior is set in the `signal:` section of `config.yaml`. Timing can also be changed
live under Settings, and those changes survive a restart (`data/runtime_settings.json`).
- `count_based: true` sets green length from the queue size (Section 1.5 Scope). By default,
  green follows the objectives: normal timing while vehicles are present, stopped when empty.
- `skip_empty_lanes: true` skips a lane whose zone is already empty when its turn comes.

## Raspberry Pi 4 deployment

1. On the PC, export the model: `python scripts\export_model.py --format ncnn --imgsz 416`.
2. Copy the project (without `.venv`) to the Pi, then:
   `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-pi.txt`
3. In `config.yaml`, set:
   - `model.weights: models/trafficsense_best_ncnn_model`
   - `model.imgsz: 416`
   - `hardware.driver: gpio`
   - lane S1 `source: 0` (the C920 webcam)
   - the BCM pins for your relay channels
4. `python run.py`. To start it on boot, add a systemd service with `Restart=always` and
   `WatchdogSec`. Shutting down leaves the lamps red, and the hardware fixed-time fallback
   (Table 2) has to be wired separately at the relay.

Expect a few FPS (roughly 4–8) for YOLOv8n with NCNN at 416 px on a Pi 4. Measure the real
figure on the dashboard's Camera page and report it under 3.10.4 Performance Testing. A few FPS
is enough for presence detection, because the 5 s empty timer only needs a decision every few
hundred milliseconds.
