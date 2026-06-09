"""
Deployment endpoints.

POST /api/deployment/{session_id}/start   — launch inference server + demo client
GET  /api/deployment/{session_id}/status  — poll live dashboard data
POST /api/deployment/callback             — internal: demo client posts metrics here
POST /api/deployment/{session_id}/stop    — kill both processes
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["deployment"])

# ── In-memory state ───────────────────────────────────────────────────────────

_status: dict = {
    "status": "stopped",   # stopped | starting | running | restarting | done | error
    "error": None,
    "episode": 0,
    "last": None,          # most recent callback payload
    "log": deque(maxlen=50),
}
_procs: dict[str, subprocess.Popen] = {}   # "inference" | "demo"

# ── Path helpers (mirror eval logic) ─────────────────────────────────────────

_BACKEND_DIR = Path(__file__).parent.parent
_MODEL_DIR   = _BACKEND_DIR / "model"      # /app/model inside Docker
_DATA_DIR    = _BACKEND_DIR / "data"


def _resolve(raw: str) -> str:
    if Path(raw).exists():
        return raw
    posix = raw.replace("\\", "/")
    idx = posix.find("/data/")
    if idx != -1:
        return str(_DATA_DIR / posix[idx + len("/data/"):])
    return raw


def _production_checkpoint() -> Path:
    ml_root = Path(os.environ.get("ML_PROJECT_ROOT", str(_BACKEND_DIR.parent)))
    session_ckpt = None   # no session-specific ckpt for deployment
    prod = ml_root / "model" / "checkpoints" / "production" / "final.pt"
    if prod.exists():
        return prod
    raise FileNotFoundError(f"Production checkpoint not found at {prod}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/deployment/{session_id}/start")
async def start_deployment(session_id: str) -> dict:
    global _status, _procs

    if _status["status"] in ("starting", "running"):
        return {"started": False, "reason": "already running"}

    # ── Resolve network files from training config ────────────────────────
    import yaml

    config_dir = _DATA_DIR / "uploads" / session_id
    config_path = next(config_dir.glob("*_training_config.yaml"), None)
    if not config_path:
        raise HTTPException(status_code=404, detail="Training config not found for session")

    cfg = yaml.safe_load(config_path.read_text())
    net_file   = _resolve(cfg["network"]["net"])
    route_file = _resolve(cfg["network"]["rou"])
    max_steps  = cfg["network"].get("max_steps", 720)
    begin_time = cfg["network"].get("begin_time", 0)

    try:
        checkpoint = str(_production_checkpoint())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ── Kill any lingering processes ──────────────────────────────────────
    _kill_all()

    _status.update({"status": "starting", "error": None, "episode": 0, "last": None})
    _status["log"].clear()

    ml_root = Path(os.environ.get("ML_PROJECT_ROOT", str(_BACKEND_DIR.parent)))
    python  = sys.executable

    inference_srv = ml_root / "model" / "serve" / "inference_server.py"
    demo_client   = ml_root / "model" / "serve" / "demo_deployment.py"

    env_copy = {**os.environ, "DISPLAY": ":99"}

    # Start inference server on port 8001
    _procs["inference"] = subprocess.Popen(
        [
            python, str(inference_srv),
            "--checkpoint", checkpoint,
            "--net-file",   net_file,
            "--route-file", route_file,
            "--port", "8001",
        ],
        env=env_copy,
    )

    # Start demo client only once the inference server is actually ready
    import threading, time, urllib.request

    def _start_demo_after_ready():
        # Poll /health until 200 OK or 30 s timeout
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen("http://localhost:8001/health", timeout=1)
                break  # got a 200
            except Exception:
                time.sleep(0.5)
        if _status["status"] == "stopped":
            return
        _procs["demo"] = subprocess.Popen(
            [
                python, str(demo_client),
                "--net-file",       net_file,
                "--route-file",     route_file,
                "--inference-url",  "http://localhost:8001",
                "--callback-url",   "http://localhost:7860/api/deployment/callback",
                "--display",        ":99",
                "--max-steps",      str(max_steps),
                "--begin-time",     str(begin_time),
            ],
            env=env_copy,
        )

    threading.Thread(target=_start_demo_after_ready, daemon=True).start()
    _status["status"] = "starting"
    return {"started": True}


@router.get("/api/deployment/{session_id}/status")
async def deployment_status(session_id: str) -> dict:
    return {
        "status":  _status["status"],
        "error":   _status["error"],
        "episode": _status["episode"],
        "last":    _status["last"],
        "log":     list(_status["log"]),
    }


@router.post("/api/deployment/callback")
async def deployment_callback(request: Request) -> dict:
    """Receives step metrics from demo_deployment.py."""
    body = await request.json()
    event = body.get("event")

    if event == "started":
        _status["status"] = "running"
        _status["episode"] = 1

    elif event == "step":
        _status["status"] = "running"
        _status["last"] = body
        _status["log"].append({
            "step":          body.get("step"),
            "sim_time":      body.get("sim_time"),
            "latency_ms":    body.get("latency_ms"),
            "fallback_tier": body.get("fallback_tier"),
            "waiting_time":  body.get("waiting_time"),
            "vehicles":      body.get("vehicles"),
        })

    elif event == "episode_end":
        _status["status"] = "restarting"
        _status["episode"] = body.get("episode", _status["episode"]) + 1

    elif event == "done":
        _status["status"] = "done"

    return {"ok": True}


@router.post("/api/deployment/{session_id}/stop")
async def stop_deployment(session_id: str) -> dict:
    _kill_all()
    _status["status"] = "stopped"
    return {"stopped": True}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _kill_all() -> None:
    for name, proc in list(_procs.items()):
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        _procs.pop(name, None)
