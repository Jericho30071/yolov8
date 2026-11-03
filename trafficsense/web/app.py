"""FastAPI monitoring dashboard (wireframes in Section 3.7.2 of the paper).

Pages: login, dashboard, camera monitoring, system controls, logs, settings.
Live status and override commands travel over a WebSocket (/ws); the annotated
camera feed is an MJPEG stream (/video/{lane}).
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, select
from starlette.middleware.sessions import SessionMiddleware

from .. import db as m
from ..config import ROOT
from ..db import FaultRecord, User, hash_password
from ..system import TrafficSystem

log = logging.getLogger(__name__)
HERE = Path(__file__).parent
VIDEO_DIR = ROOT / "data" / "raw_videos"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
TABLES = {
    "detections": m.DetectionRecord, "signals": m.SignalTimingLog, "stop_skip": m.StopSkipEvent,
    "overrides": m.ManualOverride, "faults": m.FaultRecord, "evaluations": m.EvaluationResult,
}


def create_app(system: TrafficSystem) -> FastAPI:
    app = FastAPI(title="TrafficSense")
    app.add_middleware(SessionMiddleware, secret_key=system.cfg.web.secret_key)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")

    def user(request: Request) -> dict | None:
        return request.session.get("user")

    def page(request: Request, name: str, **ctx):
        if not user(request):
            return RedirectResponse("/login", status_code=303)
        return templates.TemplateResponse(request, f"{name}.html", {"user": user(request), "page": name,
                                                                   "lanes": system.cfg.lanes, **ctx})

    # --------------------------------------------------------------- auth
    @app.get("/login", response_class=HTMLResponse)
    def login_form(request: Request):
        return templates.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login")
    def login(request: Request, username: str = Form(...), password: str = Form(...)):
        u = system.db.authenticate(username, password)
        if not u:
            return templates.TemplateResponse(request, "login.html", {"error": "Invalid username or password"},
                                              status_code=401)
        request.session["user"] = {"username": u.username, "role": u.role}
        return RedirectResponse("/", status_code=303)

    @app.get("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    # -------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return page(request, "dashboard")

    @app.get("/camera", response_class=HTMLResponse)
    def camera(request: Request, msg: str = ""):
        videos = sorted(p.name for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXT) \
            if VIDEO_DIR.exists() else []
        return page(request, "camera", monitored=[l for l in system.cfg.lanes if l.monitored], videos=videos,
                    msg=msg, current={l.id: Path(str(l.source)).name for l in system.cfg.lanes})

    @app.post("/upload/{lane_id}")
    async def upload(request: Request, lane_id: str, file: UploadFile = File(...)):
        """Save an uploaded video to data/raw_videos and play it on the lane immediately."""
        u = user(request)
        if not u:
            return RedirectResponse("/login", status_code=303)
        name = Path(file.filename or "").name
        if Path(name).suffix.lower() not in VIDEO_EXT:
            return RedirectResponse(f"/camera?msg=Unsupported file type: {name}", status_code=303)
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        target = VIDEO_DIR / name
        with open(target, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
        system.set_source(u["username"], lane_id, str(target))
        return RedirectResponse(f"/camera?msg=Now playing {name} on {lane_id}", status_code=303)

    @app.post("/source/{lane_id}")
    def choose_source(request: Request, lane_id: str, source: str = Form(...)):
        """Play a video that is already in data/raw_videos, or a webcam index."""
        u = user(request)
        if not u:
            return RedirectResponse("/login", status_code=303)
        if source.isdigit():
            src, label = int(source), f"webcam {source}"
        else:
            path = VIDEO_DIR / Path(source).name
            if not path.exists():
                return RedirectResponse(f"/camera?msg=Not found: {source}", status_code=303)
            src, label = str(path), path.name
        system.set_source(u["username"], lane_id, src)
        return RedirectResponse(f"/camera?msg=Now playing {label} on {lane_id}", status_code=303)

    @app.get("/controls", response_class=HTMLResponse)
    def controls(request: Request):
        return page(request, "controls")

    @app.get("/logs", response_class=HTMLResponse)
    def logs(request: Request, table: str = "signals"):
        model = TABLES.get(table, m.SignalTimingLog)
        rows = system.db.query(select(model).order_by(desc(model.id)).limit(200))
        cols = [c.name for c in model.__table__.columns]
        return page(request, "logs", table=table, tables=list(TABLES), cols=cols,
                    rows=[[getattr(r, c) for c in cols] for r in rows])

    @app.get("/settings", response_class=HTMLResponse)
    def settings(request: Request, saved: int = 0):
        devices = system.db.query(select(m.Device))
        users = system.db.query(select(User))
        return page(request, "settings", signal=system.cfg.signal, devices=devices, users=users, saved=saved)

    @app.post("/settings")
    async def save_settings(request: Request):
        u = user(request)
        if not u or u["role"] != "admin":
            return RedirectResponse("/login", status_code=303)
        form = dict(await request.form())
        # unchecked checkboxes are not submitted
        for flag in ("skip_empty_lanes", "count_based"):
            form.setdefault(flag, "false")
        system.command(u["username"], "settings", settings=form)
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.post("/users")
    def add_user(request: Request, username: str = Form(...), password: str = Form(...),
                 role: str = Form("personnel")):
        u = user(request)
        if not u or u["role"] != "admin":
            return RedirectResponse("/login", status_code=303)
        if not system.db.query(select(User).where(User.username == username)):
            system.db.add(User(username=username, password_hash=hash_password(password), role=role))
        return RedirectResponse("/settings", status_code=303)

    # ---------------------------------------------------------------- api
    @app.get("/api/status")
    def api_status(request: Request):
        if not user(request):
            return Response(status_code=401)
        return system.status()

    @app.get("/export/{table}.csv")
    def export(request: Request, table: str):
        if not user(request) or table not in TABLES:
            return Response(status_code=404)
        model = TABLES[table]
        cols = [c.name for c in model.__table__.columns]
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(cols)
        for r in system.db.query(select(model).order_by(model.id)):
            w.writerow([getattr(r, c) for c in cols])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={table}.csv"})

    @app.get("/video/{lane_id}")
    async def video(request: Request, lane_id: str):
        mon = system.monitors.get(lane_id)
        if not user(request) or mon is None:
            return Response(status_code=404)
        delay = 1.0 / max(1, system.cfg.web.stream_fps)

        async def frames():
            while not await request.is_disconnected():
                jpg = mon.jpeg()
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(delay)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        u = websocket.session.get("user")
        if not u:
            await websocket.close(code=4401)
            return
        await websocket.accept()

        async def push():
            while True:
                await websocket.send_text(json.dumps({"type": "status", "data": system.status()}, default=str))
                await asyncio.sleep(0.5)

        pusher = asyncio.create_task(push())
        try:
            while True:
                msg = json.loads(await websocket.receive_text())
                try:
                    system.command(u["username"], msg["action"], msg.get("lane"), msg.get("color"))
                    reply = {"type": "ack", "id": msg.get("id"), "action": msg["action"]}
                except (ValueError, KeyError) as exc:
                    reply = {"type": "error", "id": msg.get("id"), "message": str(exc)}
                await websocket.send_text(json.dumps(reply))
        except WebSocketDisconnect as exc:
            if exc.code not in (1000, 1001):  # not a normal page close -> communication fault
                system.db.add(FaultRecord(session_id=system.session.id, fault_type="communication",
                                          message=f"dashboard link lost (code {exc.code}) user={u['username']}",
                                          response="controller continues on last valid configuration"))
        finally:
            pusher.cancel()

    return app
