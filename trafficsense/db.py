"""SQLite database (ERD in Section 3.12.6) via SQLAlchemy.

Tables: users, lanes, devices, run_sessions, detection_records, signal_timing_logs,
stop_skip_events, manual_overrides, fault_records, evaluation_results.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import threading
from datetime import datetime
from pathlib import Path

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine, select)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def now() -> datetime:
    return datetime.now()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(32), default="personnel")   # admin | personnel
    created_at = Column(DateTime, default=now)


class Lane(Base):
    __tablename__ = "lanes"
    id = Column(String(8), primary_key=True)          # S1..S4
    name = Column(String(64))
    direction = Column(String(32))
    camera_source = Column(String(256))
    status = Column(String(16), default="red")        # current signal color


class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True)
    kind = Column(String(32))                         # raspberry_pi | camera | relay
    name = Column(String(64))
    lane_id = Column(String(8), ForeignKey("lanes.id"), nullable=True)
    details = Column(Text)
    status = Column(String(16), default="unknown")
    updated_at = Column(DateTime, default=now, onupdate=now)


class RunSession(Base):
    __tablename__ = "run_sessions"
    id = Column(Integer, primary_key=True)
    mode = Column(String(16))
    source = Column(Text)
    model = Column(String(256))
    started_at = Column(DateTime, default=now)
    ended_at = Column(DateTime, nullable=True)


class DetectionRecord(Base):
    __tablename__ = "detection_records"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("run_sessions.id"))
    lane_id = Column(String(8), ForeignKey("lanes.id"))
    track_id = Column(Integer, nullable=True)
    vehicle_type = Column(String(32), nullable=True)  # null for lane status rows
    confidence = Column(Float, nullable=True)
    vehicle_count = Column(Integer)                   # vehicles in the lane zone at that moment
    detection_status = Column(String(16))             # vehicle | occupied | empty
    detected_at = Column(DateTime, default=now)


class SignalTimingLog(Base):
    __tablename__ = "signal_timing_logs"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("run_sessions.id"))
    lane_id = Column(String(8), ForeignKey("lanes.id"))
    signal_color = Column(String(8))                  # green | yellow
    mode = Column(String(16))                         # adaptive | fixed | manual
    basis = Column(String(16))                        # adaptive | fixed | failsafe
    planned_duration = Column(Float)
    actual_duration = Column(Float, nullable=True)
    action = Column(String(24), nullable=True)        # stop_empty | max_green | fixed | failsafe | manual
    vehicles_at_start = Column(Integer, default=0)
    vehicles_waited = Column(Integer, default=0)      # vehicles that queued during red
    total_wait = Column(Float, default=0.0)           # sum of their waiting seconds
    started_at = Column(DateTime, default=now)
    ended_at = Column(DateTime, nullable=True)


class StopSkipEvent(Base):
    __tablename__ = "stop_skip_events"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("run_sessions.id"))
    lane_id = Column(String(8), ForeignKey("lanes.id"))
    signal_log_id = Column(Integer, ForeignKey("signal_timing_logs.id"), nullable=True)
    detection_record_id = Column(Integer, ForeignKey("detection_records.id"), nullable=True)
    event_type = Column(String(8))                    # stop | skip
    reason = Column(String(128))
    green_elapsed = Column(Float, nullable=True)
    saved_green = Column(Float, nullable=True)        # idle green seconds avoided
    response_time = Column(Float, nullable=True)      # delay after the empty timeout expired
    occurred_at = Column(DateTime, default=now)


class ManualOverride(Base):
    __tablename__ = "manual_overrides"
    id = Column(Integer, primary_key=True)
    username = Column(String(64))
    action = Column(String(32))                       # manual | emergency_stop | off | resume | settings
    lane_id = Column(String(8), nullable=True)
    details = Column(Text)
    occurred_at = Column(DateTime, default=now)


class FaultRecord(Base):
    __tablename__ = "fault_records"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("run_sessions.id"))
    lane_id = Column(String(8), nullable=True)
    fault_type = Column(String(32))                   # camera | detection | communication
    message = Column(Text)
    response = Column(String(64))
    occurred_at = Column(DateTime, default=now)
    resolved_at = Column(DateTime, nullable=True)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id = Column(Integer, primary_key=True)
    test_name = Column(String(128))
    mode = Column(String(16))
    total_runs = Column(Integer, default=1)
    correct_detections = Column(Integer, nullable=True)
    incorrect_detections = Column(Integer, nullable=True)
    detection_accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    map50 = Column(Float, nullable=True)
    map50_95 = Column(Float, nullable=True)
    avg_wait_time = Column(Float, nullable=True)
    idle_green_time = Column(Float, nullable=True)
    signal_response_time = Column(Float, nullable=True)
    notes = Column(Text)
    created_at = Column(DateTime, default=now)


# --------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt, digest = stored.split("$")
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return hmac.compare_digest(check.hex(), digest)


# ---------------------------------------------------------------- database
class Database:
    """Thread-safe helper; every write opens a short session under a lock."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        self._lock = threading.Lock()

    def add(self, obj):
        with self._lock, self.Session() as s:
            s.add(obj)
            s.commit()
            return obj

    def update(self, model, obj_id, **values) -> None:
        if obj_id is None:
            return
        with self._lock, self.Session() as s:
            obj = s.get(model, obj_id)
            if obj is not None:
                for k, v in values.items():
                    setattr(obj, k, v)
                s.commit()

    def query(self, stmt):
        with self.Session() as s:
            return list(s.scalars(stmt))

    def sync_lanes_and_devices(self, cfg) -> None:
        with self._lock, self.Session() as s:
            for lane in cfg.lanes:
                row = s.get(Lane, lane.id) or Lane(id=lane.id)
                row.name, row.direction = lane.name, lane.direction
                row.camera_source = None if lane.source is None else str(lane.source)
                s.merge(row)
            s.query(Device).delete()
            s.add(Device(kind="raspberry_pi", name="Controller", details=f"driver={cfg.hardware.driver}",
                         status="online"))
            for lane in cfg.lanes:
                if lane.monitored:
                    s.add(Device(kind="camera", name=f"Camera {lane.id}", lane_id=lane.id,
                                 details=str(lane.source)))
            for lane_id, pins in cfg.hardware.pins.items():
                s.add(Device(kind="relay", name=f"Relay {lane_id}", lane_id=lane_id, details=str(pins),
                             status="online"))
            s.commit()

    def ensure_admin(self, username: str, password: str) -> None:
        with self._lock, self.Session() as s:
            if not s.scalars(select(User).limit(1)).first():
                s.add(User(username=username, password_hash=hash_password(password), role="admin"))
                s.commit()

    def authenticate(self, username: str, password: str) -> User | None:
        with self.Session() as s:
            user = s.scalars(select(User).where(User.username == username)).first()
            return user if user and verify_password(password, user.password_hash) else None
