from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

# Add both backend/ and backend/traffic_rl/ to path
_backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_backend_dir))
sys.path.insert(0, str(_backend_dir / "traffic_rl"))

from services.route_generator import generate_routes_for_network
from services.config_generator import create_training_config

WEB_JOBS_ROOT = Path(__file__).parent.parent / "data" / "web_jobs"

# ML_PROJECT_ROOT env var is set in Docker (ML_PROJECT_ROOT=/app).
# Falls back to 4 parent levels for local dev (ml_project/website/backend/services/).
_ml_project_root = Path(os.environ.get(
    "ML_PROJECT_ROOT",
    str(Path(__file__).parent.parent.parent.parent),
))

_running_sessions: set[str] = set()
_running_processes: dict[str, subprocess.Popen] = {}
_running_lock = threading.Lock()


def _enforce_sumo_env() -> None:
    """Verify SUMO tools are reachable. Raises early with a clear message if not found.

    In Docker, SUMO is pre-installed via apt-get and SUMO_HOME is set — this is a no-op.
    """
    import shutil
    missing = [t for t in ("sumo", "netconvert") if not shutil.which(t)]
    if missing:
        raise EnvironmentError(
            f"SUMO tools not found on PATH: {missing}. "
            "Ensure SUMO is installed in the deployment environment and SUMO_HOME is set."
        )


def _write_status(session_id: str, stage: str, status: str, details: dict | None = None) -> None:
    status_path = WEB_JOBS_ROOT / session_id / "reports" / "job_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({
        "job_id": session_id,
        "stage": stage,
        "status": status,
        "details": details or {},
        "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }))


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

    # Live progress from streaming subprocess reader
    progress_file = job_dir / "reports" / "training_progress.json"
    if progress_file.exists():
        try:
            data = json.loads(progress_file.read_text())
            if data.get("available"):
                return data
        except Exception:
            pass

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


def build_config(session_id: str, net_path: str):
    from traffic_rl.benchmark.controllers.independent_dqn_v2_web import WebJobConfig
    return WebJobConfig(
        job_id=session_id,
        osm_path=Path(net_path),
        output_root=WEB_JOBS_ROOT,
        network_name="uploaded_map",
        train_episodes=500,
        train_max_steps_per_episode=720,
        wall_clock_cap_minutes=240.0,
    )


def start_real_training(session_id: str, net_path: str, pass_threshold_pct: float = 25.0) -> bool:
    """Start GAT model training with capacity-based routes. Returns False if already running."""
    with _running_lock:
        if session_id in _running_sessions:
            return False

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

    net_path_obj = Path(net_path)
    ml_project_root = _ml_project_root

    def run() -> None:
        _enforce_sumo_env()
        current_stage = "initialization"
        try:
            _write_status(session_id, "initialization", "running", {"net_path": str(net_path_obj)})
            append_training_log(session_id, f"GAT model training started — session {session_id[:12]}")

            # Step 1: Generate capacity-based routes
            current_stage = "random_route_generation"
            _write_status(session_id, "random_route_generation", "running")
            append_training_log(session_id, "Generating capacity-based route files...")
            routes_result = generate_routes_for_network(
                net_path=net_path_obj,
                source_rou_path=None,
                output_dir=Path(__file__).parent.parent / "data" / "uploads" / session_id,
                network_name=session_id,
                begin_time=0,
                time_window=3600,
            )

            if routes_result["status"] != "success":
                raise Exception(f"Route generation failed: {routes_result.get('error', 'unknown error')}")

            append_training_log(
                session_id,
                f"Routes generated: {routes_result['metadata']['num_od_pairs']} OD pairs, "
                f"capacity={routes_result['jam_capacity']:.0f} vehicles"
            )
            _write_status(session_id, "random_route_generation", "passed", {
                "num_od_pairs": routes_result["metadata"]["num_od_pairs"],
                "jam_capacity": routes_result["jam_capacity"],
            })

            # Step 2: Create training config — episodes scale with TL node count
            current_stage = "scenario_manifest_creation"
            _write_status(session_id, "scenario_manifest_creation", "running")
            append_training_log(session_id, "Creating training configuration...")
            config_dir = Path(__file__).parent.parent / "data" / "uploads" / session_id
            try:
                from services.osm_converter import extract_network_data
                tl_count = extract_network_data(str(net_path_obj)).get("tl_count", 1) or 1
            except Exception:
                tl_count = 1
            num_episodes = max(200, min(2000, tl_count * 100))
            append_training_log(session_id, f"Network: {tl_count} TL nodes → {num_episodes} episodes")
            stop_file_path = WEB_JOBS_ROOT / session_id / "reports" / "stop_requested"
            config_path = create_training_config(
                session_id=session_id,
                net_path=net_path_obj,
                route_files=routes_result["route_files"],
                output_dir=config_dir,
                episodes=num_episodes,
                begin_time=0,
                stop_file=stop_file_path.as_posix(),
                pass_threshold_pct=pass_threshold_pct,
            )
            append_training_log(session_id, f"Training config created: {config_path}")
            _write_status(session_id, "scenario_manifest_creation", "passed")

            # Step 3: Run model/train.py and stream output for live progress
            current_stage = "independent_dqn_training"
            _write_status(session_id, "independent_dqn_training", "running")
            append_training_log(session_id, "Starting GAT model training with route randomization...")

            # training_metrics.jsonl — live per-episode KPI stream (read by /latest-episode-kpis)
            # training_metrics.json  — final full summary written by model/train.py at end (read below for job_result)
            jsonl_path = WEB_JOBS_ROOT / session_id / "reports" / "training_metrics.jsonl"

            cmd = [
                sys.executable,
                "model/train.py",
                "--config",        str(config_path),
                "--metrics-file",  str(jsonl_path),
                "--eval-episodes", "0",   # web app runs its own GUI eval separately
            ]

            _env = {**os.environ, "DISPLAY": ":99",
                    "SUMO_HOME": os.environ.get("SUMO_HOME", "/usr/share/sumo"),
                    "ML_PROJECT_ROOT": str(ml_project_root)}
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=ml_project_root,
                env=_env,
            )
            with _running_lock:
                _running_processes[session_id] = process

            ep_pattern = re.compile(r"\[Ep\s+(\d+)/(\d+)\]")
            progress_path = WEB_JOBS_ROOT / session_id / "reports" / "training_progress.json"

            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                m = ep_pattern.search(line)
                if m:
                    cur, total = int(m.group(1)), int(m.group(2))
                    progress_path.write_text(json.dumps({
                        "current_episode": cur,
                        "total_episodes": total,
                        "phase": "training",
                        "available": True,
                    }))
                    append_training_log(session_id, f"[Ep {cur}/{total}]")

            process.wait()
            if process.returncode != 0:
                raise Exception(f"Training subprocess failed (exit code {process.returncode})")

            append_training_log(session_id, "GAT model training completed")

            # Step 4: Copy checkpoint to web_jobs
            trained_checkpoint = ml_project_root / "checkpoints" / session_id / "final.pt"
            if trained_checkpoint.exists():
                results_checkpoint = WEB_JOBS_ROOT / session_id / "training" / "full"
                results_checkpoint.mkdir(parents=True, exist_ok=True)
                import shutil as _shutil
                _shutil.copy(trained_checkpoint, results_checkpoint / "controller.pt")
                append_training_log(session_id, "Model checkpoint saved")

            # Step 5: Write job_result.json — evaluation happens on page 4
            metrics_path = ml_project_root / "checkpoints" / session_id / "training_metrics.json"
            metrics_data = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

            result_path = WEB_JOBS_ROOT / session_id / "reports" / "job_result.json"
            result_path.write_text(json.dumps({
                "available": True,
                "source": "real_training",
                "session_id": session_id,
                "eval_metrics": {},
                "baseline_metrics": {"available": False},
                "training_metrics": {
                    "episode_returns":  metrics_data.get("episode_returns", []),
                    "losses":           metrics_data.get("losses", []),
                    "avg_waiting_time": metrics_data.get("avg_waiting_time", []),
                    "avg_queue_length": metrics_data.get("avg_queue_length", []),
                    "throughput":       metrics_data.get("throughput", []),
                    "stop_reason":      metrics_data.get("stop_reason", "completed"),
                },
            }))

            _write_status(session_id, "complete", "passed")
            append_training_log(session_id, "Training complete — proceed to Results to run evaluation")

        except Exception as exc:
            _write_status(session_id, current_stage, "failed", {"error": str(exc)})
            append_training_log(session_id, f"Training failed: {exc}")
        finally:
            with _running_lock:
                _running_sessions.discard(session_id)
                _running_processes.pop(session_id, None)

    threading.Thread(target=run, daemon=True).start()
    return True


def stop_training(session_id: str) -> bool:
    """Request early stop. Writes the stop-signal file; the trainer detects it after the
    current episode, saves the checkpoint, and exits cleanly."""
    stop_file = WEB_JOBS_ROOT / session_id / "reports" / "stop_requested"
    if not (WEB_JOBS_ROOT / session_id / "reports").exists():
        return False
    stop_file.touch()
    return True


def resume_training(session_id: str) -> bool:
    """Resume a stopped training run from the saved checkpoint."""
    with _running_lock:
        if session_id in _running_sessions:
            return False

    reports_dir = WEB_JOBS_ROOT / session_id / "reports"
    checkpoint  = _ml_project_root / "checkpoints" / session_id / "final.pt"
    config_dir  = Path(__file__).parent.parent / "data" / "uploads" / session_id
    config_path = next(config_dir.glob("*_training_config.yaml"), None)

    if not checkpoint.exists() or not config_path:
        return False

    # Count completed episodes from jsonl
    jsonl_path = reports_dir / "training_metrics.jsonl"
    completed  = len(jsonl_path.read_text().strip().splitlines()) if jsonl_path.exists() else 0

    # Read total_episodes from progress file
    progress_path = reports_dir / "training_progress.json"
    total = 1000
    if progress_path.exists():
        try:
            total = json.loads(progress_path.read_text()).get("total_episodes", 1000)
        except Exception:
            pass

    remaining = max(1, total - completed)

    # Clear stop flag so trainer doesn't immediately stop again
    stop_file = reports_dir / "stop_requested"
    stop_file.unlink(missing_ok=True)

    with _running_lock:
        _running_sessions.add(session_id)

    ml_project_root = _ml_project_root

    def run() -> None:
        current_stage = "independent_dqn_training"
        try:
            _write_status(session_id, "independent_dqn_training", "running")
            append_training_log(session_id, f"Resuming training — {completed} episodes done, {remaining} remaining")

            cmd = [
                sys.executable,
                "model/train.py",
                "--config",        str(config_path),
                "--checkpoint",    str(checkpoint),
                "--episodes",      str(remaining),
                "--metrics-file",  str(jsonl_path),
                "--eval-episodes", "0",
            ]

            _env = {**os.environ, "DISPLAY": ":99",
                    "SUMO_HOME": os.environ.get("SUMO_HOME", "/usr/share/sumo"),
                    "ML_PROJECT_ROOT": str(ml_project_root)}
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=ml_project_root,
                env=_env,
            )
            with _running_lock:
                _running_processes[session_id] = process

            ep_pattern    = re.compile(r"\[Ep\s+(\d+)/(\d+)\]")
            progress_path2 = reports_dir / "training_progress.json"

            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                m = ep_pattern.search(line)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
                    # Show overall progress (completed + new)
                    progress_path2.write_text(json.dumps({
                        "current_episode": completed + cur,
                        "total_episodes":  total,
                        "phase":           "training",
                        "available":       True,
                    }))
                append_training_log(session_id, line)

            process.wait()
            if process.returncode != 0:
                raise Exception(f"Resumed training subprocess failed (exit code {process.returncode})")

            append_training_log(session_id, "Resumed training completed")

            # Copy updated checkpoint
            if checkpoint.exists():
                dest = WEB_JOBS_ROOT / session_id / "training" / "full"
                dest.mkdir(parents=True, exist_ok=True)
                import shutil as _shutil
                _shutil.copy(checkpoint, dest / "controller.pt")

            # Evaluation + result (same as initial run)
            current_stage = "evaluation"
            _write_status(session_id, "evaluation", "running")
            append_training_log(session_id, "Starting fixed-time baseline evaluation (5 episodes, built-in TL program)...")
            baseline_path = reports_dir / "baseline_summary.json"
            try:
                import sys as _sys
                _model_path = str(ml_project_root / "model")
                if _model_path not in _sys.path:
                    _sys.path.insert(0, _model_path)
                import yaml as _yaml
                _cfg = _yaml.safe_load(config_path.read_text())
                _net_path = Path(_cfg["network"]["net"])
                _rou_path = Path(_cfg["network"]["rou"])
                from evaluation.eval_runner import evaluate_fixed_time
                bl = evaluate_fixed_time(
                    net_file=str(_net_path),
                    route_file=str(_rou_path),
                    num_episodes=5,
                    seeds=[0, 100, 200, 300, 400],
                    max_steps=_cfg["network"].get("max_steps", 720),
                    begin_time=_cfg["network"].get("begin_time", 0),
                )
                bl_out = {k: v for k, v in bl.items() if k != "episode_records"}
                baseline_path.write_text(json.dumps({
                    "available": True,
                    "source": "fixed_time_builtin",
                    "metrics": bl_out,
                }))
                append_training_log(session_id, "Fixed-time baseline complete")
            except Exception as bl_exc:
                baseline_path.write_text(json.dumps({"available": False, "error": str(bl_exc)}))
                append_training_log(session_id, f"Fixed-time baseline failed: {bl_exc}")

            eval_path    = ml_project_root / "checkpoints" / session_id / "eval_metrics.json"
            metrics_path = ml_project_root / "checkpoints" / session_id / "training_metrics.json"
            eval_data     = json.loads(eval_path.read_text()) if eval_path.exists() else {}
            metrics_data  = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            baseline_data = json.loads(baseline_path.read_text()) if baseline_path.exists() else {"available": False}

            result_path = reports_dir / "job_result.json"
            result_path.write_text(json.dumps({
                "available":        True,
                "source":           "resumed_training",
                "session_id":       session_id,
                "eval_metrics":     eval_data,
                "baseline_metrics": baseline_data,
                "training_metrics": {
                    "episode_returns":  metrics_data.get("episode_returns", []),
                    "losses":           metrics_data.get("losses", []),
                    "avg_waiting_time": metrics_data.get("avg_waiting_time", []),
                    "avg_queue_length": metrics_data.get("avg_queue_length", []),
                    "throughput":       metrics_data.get("throughput", []),
                    "stop_reason":      metrics_data.get("stop_reason", "completed"),
                },
            }))
            _write_status(session_id, "complete", "passed")
            append_training_log(session_id, "Resumed training pipeline complete")

        except Exception as exc:
            _write_status(session_id, current_stage, "failed", {"error": str(exc)})
            append_training_log(session_id, f"Resume failed: {exc}")
        finally:
            with _running_lock:
                _running_sessions.discard(session_id)
                _running_processes.pop(session_id, None)

    threading.Thread(target=run, daemon=True).start()
    return True


def retrain(session_id: str, mode: str) -> bool:
    """Retraining pipeline triggered after a failed eval verdict.

    mode="continue" — load checkpoint, preserve epsilon schedule (same grad_steps).
    mode="explore"  — load checkpoint, reset epsilon to 0.5 and grad_steps to 0 so
                      the model re-explores. The existing replay buffer is kept; old
                      transitions provide diversity while new exploration fills it.
    Both modes run num_episodes from the original training config.
    """
    with _running_lock:
        if session_id in _running_sessions:
            return False

    reports_dir  = WEB_JOBS_ROOT / session_id / "reports"
    ckpt_dir     = _ml_project_root / "checkpoints" / session_id
    checkpoint   = ckpt_dir / "final.pt"
    if not checkpoint.exists():
        # Training may have been terminated before final.pt was written; fall back to
        # the latest per-episode checkpoint so retraining can still proceed.
        candidates = sorted(ckpt_dir.glob("checkpoint_ep*.pt"))
        if candidates:
            checkpoint = candidates[-1]
    config_dir  = Path(__file__).parent.parent / "data" / "uploads" / session_id
    config_path = next(config_dir.glob("*_training_config.yaml"), None)

    if not checkpoint.exists() or not config_path:
        return False

    import yaml as _yaml
    cfg_data     = _yaml.safe_load(config_path.read_text())
    num_episodes = cfg_data.get("trainer", {}).get("num_episodes", 200)

    # Clear previous eval comparison so the new run starts fresh
    (reports_dir / "eval_comparison.json").unlink(missing_ok=True)

    # Start fresh jsonl so progress bar resets for this run
    jsonl_path = reports_dir / "training_metrics.jsonl"
    jsonl_path.unlink(missing_ok=True)

    # Remove any lingering stop flag
    (reports_dir / "stop_requested").unlink(missing_ok=True)

    with _running_lock:
        _running_sessions.add(session_id)

    ml_project_root = _ml_project_root
    label = "Re-explore (ε=0.5)" if mode == "explore" else "Continue"

    def run() -> None:
        current_stage = "independent_dqn_training"
        try:
            _write_status(session_id, "independent_dqn_training", "running")
            append_training_log(session_id, f"Retraining [{label}] — {num_episodes} episodes from checkpoint")

            cmd = [
                sys.executable,
                "model/train.py",
                "--config",        str(config_path),
                "--checkpoint",    str(checkpoint),
                "--episodes",      str(num_episodes),
                "--metrics-file",  str(jsonl_path),
                "--eval-episodes", "0",
            ]
            if mode == "explore":
                cmd += ["--epsilon-start", "0.5"]

            _env = {**os.environ, "DISPLAY": ":99",
                    "SUMO_HOME": os.environ.get("SUMO_HOME", "/usr/share/sumo"),
                    "ML_PROJECT_ROOT": str(ml_project_root)}
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=ml_project_root,
                env=_env,
            )
            with _running_lock:
                _running_processes[session_id] = process

            ep_pattern    = re.compile(r"\[Ep\s+(\d+)/(\d+)\]")
            progress_path = reports_dir / "training_progress.json"

            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                m = ep_pattern.search(line)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
                    progress_path.write_text(json.dumps({
                        "current_episode": cur,
                        "total_episodes":  tot,
                        "phase":           "training",
                        "available":       True,
                    }))
                append_training_log(session_id, line)

            process.wait()
            if process.returncode != 0:
                raise Exception(f"Retrain subprocess failed (exit code {process.returncode})")

            append_training_log(session_id, f"Retraining [{label}] complete")

            # Copy updated checkpoint
            if checkpoint.exists():
                dest = WEB_JOBS_ROOT / session_id / "training" / "full"
                dest.mkdir(parents=True, exist_ok=True)
                import shutil as _shutil
                _shutil.copy(checkpoint, dest / "controller.pt")

            # Fixed-time baseline
            current_stage = "evaluation"
            _write_status(session_id, "evaluation", "running")
            append_training_log(session_id, "Starting fixed-time baseline evaluation...")
            baseline_path = reports_dir / "baseline_summary.json"
            try:
                import sys as _sys
                _model_path = str(ml_project_root / "model")
                if _model_path not in _sys.path:
                    _sys.path.insert(0, _model_path)
                _cfg = _yaml.safe_load(config_path.read_text())
                from evaluation.eval_runner import evaluate_fixed_time
                bl = evaluate_fixed_time(
                    net_file=str(Path(_cfg["network"]["net"])),
                    route_file=str(Path(_cfg["network"]["rou"])),
                    num_episodes=5,
                    seeds=[0, 100, 200, 300, 400],
                    max_steps=_cfg["network"].get("max_steps", 720),
                    begin_time=_cfg["network"].get("begin_time", 0),
                )
                baseline_path.write_text(json.dumps({
                    "available": True,
                    "source":   "fixed_time_builtin",
                    "metrics":  {k: v for k, v in bl.items() if k != "episode_records"},
                }))
                append_training_log(session_id, "Fixed-time baseline complete")
            except Exception as bl_exc:
                baseline_path.write_text(json.dumps({"available": False, "error": str(bl_exc)}))
                append_training_log(session_id, f"Fixed-time baseline failed: {bl_exc}")

            eval_path     = ml_project_root / "checkpoints" / session_id / "eval_metrics.json"
            metrics_path  = ml_project_root / "checkpoints" / session_id / "training_metrics.json"
            eval_data     = json.loads(eval_path.read_text()) if eval_path.exists() else {}
            metrics_data  = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            baseline_data = json.loads(baseline_path.read_text()) if baseline_path.exists() else {"available": False}

            result_path = reports_dir / "job_result.json"
            result_path.write_text(json.dumps({
                "available":        True,
                "source":           f"retrain_{mode}",
                "session_id":       session_id,
                "eval_metrics":     eval_data,
                "baseline_metrics": baseline_data,
                "training_metrics": {
                    "episode_returns":  metrics_data.get("episode_returns", []),
                    "losses":           metrics_data.get("losses", []),
                    "avg_waiting_time": metrics_data.get("avg_waiting_time", []),
                    "avg_queue_length": metrics_data.get("avg_queue_length", []),
                    "throughput":       metrics_data.get("throughput", []),
                    "stop_reason":      metrics_data.get("stop_reason", "completed"),
                },
            }))
            _write_status(session_id, "complete", "passed")
            append_training_log(session_id, f"Retraining [{label}] pipeline complete")

        except Exception as exc:
            _write_status(session_id, current_stage, "failed", {"error": str(exc)})
            append_training_log(session_id, f"Retrain failed: {exc}")
        finally:
            with _running_lock:
                _running_sessions.discard(session_id)
                _running_processes.pop(session_id, None)

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
