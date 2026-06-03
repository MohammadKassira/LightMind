from __future__ import annotations

import datetime
import json
import os
import sys
import threading
from pathlib import Path

# Add both backend/ and backend/traffic_rl/ to path
# backend/ allows: import traffic_rl.benchmark...
# backend/traffic_rl/ allows: import benchmark... (used internally by the package)
_backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_backend_dir))
sys.path.insert(0, str(_backend_dir / "traffic_rl"))

from traffic_rl.benchmark.controllers.independent_dqn_v2_web import run_web_job, WebJobConfig

WEB_JOBS_ROOT = Path(__file__).parent.parent / "data" / "web_jobs"

_CORRECT_SUMO_HOME = "/Users/hasanhaidar/Library/Python/3.9/lib/python/site-packages/sumo"
_SUMO_BIN_DIR = "/Users/hasanhaidar/Library/Python/3.9/bin"

# Track in-memory running sessions so duplicate /start calls are ignored
_running_sessions: set[str] = set()
_running_lock = threading.Lock()


def _enforce_sumo_env() -> None:
    """Force SUMO_HOME and PATH so all SUMO tools (netconvert, duarouter, sumo) are reachable.

    randomTrips.py --validate spawns duarouter as a subprocess, so duarouter must be on PATH
    inside the training thread — not just importable via shutil.which from the uvicorn process.
    """
    current = os.environ.get("SUMO_HOME", "")
    typemap = Path(current) / "data" / "typemap" / "osmNetconvert.typ.xml"
    if not typemap.exists():
        os.environ["SUMO_HOME"] = _CORRECT_SUMO_HOME

    current_path = os.environ.get("PATH", "")
    if _SUMO_BIN_DIR not in current_path:
        os.environ["PATH"] = _SUMO_BIN_DIR + os.pathsep + current_path


def append_training_log(session_id: str, message: str) -> None:
    log_path = WEB_JOBS_ROOT / session_id / "reports" / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logs: list[dict] = []
    if log_path.exists():
        try:
            logs = json.loads(log_path.read_text())
        except Exception:
            logs = []
    logs.append({"time": datetime.datetime.utcnow().isoformat(), "message": message})
    log_path.write_text(json.dumps(logs[-100:]))


def get_training_logs(session_id: str) -> list[dict]:
    log_path = WEB_JOBS_ROOT / session_id / "reports" / "training_log.json"
    if not log_path.exists():
        return []
    try:
        return json.loads(log_path.read_text())
    except Exception:
        return []


def get_training_progress(session_id: str) -> dict:
    job_dir = WEB_JOBS_ROOT / session_id
    result: dict = {
        "available": False,
        "current_episode": 0,
        "total_episodes": 500,
        "phase": "waiting",
        "wall_clock_seconds": 0,
    }

    smoke_summary = job_dir / "training" / "smoke" / "training_summary.json"
    if smoke_summary.exists():
        result["phase"] = "smoke_complete"

    rollouts_dir = job_dir / "training" / "full" / "training_rollouts"
    if rollouts_dir.exists():
        try:
            episodes = [d for d in rollouts_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
            result["current_episode"] = len(episodes)
            result["phase"] = "training"
            result["available"] = True
        except Exception:
            pass

    full_summary = job_dir / "training" / "full" / "training_summary.json"
    if full_summary.exists():
        try:
            data = json.loads(full_summary.read_text())
            result["total_episodes"] = data.get("total_episodes", 500)
            result["wall_clock_seconds"] = data.get("wall_clock_seconds", 0)
        except Exception:
            pass

    return result


def build_config(session_id: str, osm_path: str) -> WebJobConfig:
    return WebJobConfig(
        job_id=session_id,
        osm_path=Path(osm_path),
        output_root=WEB_JOBS_ROOT,
        network_name="uploaded_map",
        train_episodes=500,
        train_max_steps_per_episode=720,
        wall_clock_cap_minutes=240.0,
    )


def start_real_training(session_id: str, osm_path: str) -> bool:
    """Start training in a background thread. Returns False if already running."""
    with _running_lock:
        if session_id in _running_sessions:
            return False

    # Also check if a status file exists from a previous (non-failed) run
    status_path = WEB_JOBS_ROOT / session_id / "reports" / "job_status.json"
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text())
            if existing.get("status") not in ("failed",):
                return False
        except Exception:
            pass

    with _running_lock:
        _running_sessions.add(session_id)

    append_training_log(session_id, f"Real DQN training started — session {session_id[:12]}")
    config = build_config(session_id, osm_path)

    def run() -> None:
        _enforce_sumo_env()
        try:
            run_web_job(config)
            append_training_log(session_id, "Training pipeline completed successfully")
        except Exception as exc:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps({
                "job_id": session_id,
                "stage": "unknown",
                "status": "failed",
                "error": str(exc),
                "updated_at_utc": datetime.datetime.utcnow().isoformat(),
            }))
            append_training_log(session_id, f"Training failed: {exc}")
        finally:
            with _running_lock:
                _running_sessions.discard(session_id)

    threading.Thread(target=run, daemon=True).start()
    return True


def get_job_status(session_id: str) -> dict:
    status_path = WEB_JOBS_ROOT / session_id / "reports" / "job_status.json"
    if not status_path.exists():
        return {"status": "pending", "stage": "initializing"}
    return json.loads(status_path.read_text())


def get_job_result(session_id: str) -> dict | None:
    result_path = WEB_JOBS_ROOT / session_id / "reports" / "job_result.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text())


def get_kpi_summary(session_id: str) -> dict | None:
    kpi_path = WEB_JOBS_ROOT / session_id / "evaluation" / "results" / "kpi_summary_by_level.json"
    if not kpi_path.exists():
        return None
    return json.loads(kpi_path.read_text())


def get_model_checkpoint_path(session_id: str) -> Path | None:
    checkpoint = WEB_JOBS_ROOT / session_id / "training" / "full" / "controller.pt"
    return checkpoint if checkpoint.exists() else None


def get_package_zip_path(session_id: str) -> Path | None:
    path = WEB_JOBS_ROOT / "packages" / f"{session_id}_independent_dqn_v2_web_bundle.zip"
    return path if path.exists() else None


def is_sumo_available() -> bool:
    import shutil
    return bool(shutil.which("sumo") and shutil.which("netconvert"))
