"""Loads config.yaml into typed dataclasses."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    weights: str = "yolov8n.pt"
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.5
    device: str = ""
    tracker: str = "bytetrack.yaml"
    coco_class_map: dict[str, str] = field(default_factory=dict)


@dataclass
class LaneConfig:
    id: str
    name: str = ""
    direction: str = ""
    source: Any = None
    loop_video: bool = True
    zone: list[list[float]] | None = None
    count_line: list[list[float]] | None = None

    @property
    def monitored(self) -> bool:
        return self.source is not None


@dataclass
class SignalConfig:
    mode: str = "adaptive"
    fixed_green: float = 30
    min_green: float = 5
    max_green: float = 30
    empty_timeout: float = 5
    yellow: float = 3
    all_red: float = 1
    skip_empty_lanes: bool = False
    count_based: bool = False
    base_green: float = 10
    per_vehicle: float = 2


@dataclass
class FailsafeConfig:
    frame_timeout: float = 3.0
    error_limit: int = 3
    recover_after: float = 5.0


@dataclass
class HardwareConfig:
    driver: str = "mock"
    relay_active_low: bool = True
    pins: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    secret_key: str = "change-me"
    default_admin: dict[str, str] = field(default_factory=lambda: {"username": "admin", "password": "admin123"})
    stream_fps: int = 10


@dataclass
class Config:
    classes: list[str]
    model: ModelConfig
    lanes: list[LaneConfig]
    signal: SignalConfig
    failsafe: FailsafeConfig
    hardware: HardwareConfig
    db_path: Path
    web: WebConfig

    def lane(self, lane_id: str) -> LaneConfig:
        return next(l for l in self.lanes if l.id == lane_id)


def _build(cls, data: dict | None):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def zone_file(source) -> Path | None:
    if isinstance(source, str) and "://" not in source:
        p = Path(source)
        return p.with_name(p.stem + ".zone.json")
    return None


def geometry_for(source, lane: LaneConfig):
    """Zone + count line for a source: <video>.zone.json if present, else the lane's config."""
    f = zone_file(source)
    if f is not None and f.exists():
        data = json.loads(f.read_text())
        return data.get("zone"), data.get("count_line")
    return lane.zone, lane.count_line


def save_geometry(source, zone, count_line) -> Path | None:
    f = zone_file(source)
    if f is not None:
        f.write_text(json.dumps({"zone": zone, "count_line": count_line}, indent=1))
    return f


def load_config(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    lanes = [_build(LaneConfig, l) for l in raw.get("lanes", [])]
    for lane in lanes:
        # Resolve relative video paths; keep webcam indexes (ints) and URLs as-is.
        if isinstance(lane.source, str) and "://" not in lane.source and not lane.source.isdigit():
            lane.source = str(resolve_path(lane.source))
        elif isinstance(lane.source, str) and lane.source.isdigit():
            lane.source = int(lane.source)
    return Config(
        classes=raw.get("classes", []),
        model=_build(ModelConfig, raw.get("model")),
        lanes=lanes,
        signal=_build(SignalConfig, raw.get("signal")),
        failsafe=_build(FailsafeConfig, raw.get("failsafe")),
        hardware=_build(HardwareConfig, raw.get("hardware")),
        db_path=resolve_path(raw.get("database", {}).get("path", "data/trafficsense.db")),
        web=_build(WebConfig, raw.get("web")),
    )
