from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from services.real_trainer import (
    WEB_JOBS_ROOT,
    get_job_result,
    get_job_status,
    get_kpi_summary,
    get_model_checkpoint_path,
    get_package_zip_path,
    get_training_logs,
    get_training_progress,
    resume_training,
    retrain,
    start_real_training,
    stop_training,
)

router = APIRouter()

# ── Eval session state (in-memory, keyed by session_id) ──────────────────────
_eval_status: dict[str, dict] = {}
# {session_id: {phase: str, episode1_running: bool, error: str|None}}

# Must stay in sync with frontend STAGE_ORDER in TrainingScreen.jsx
STAGE_LABELS = {
    "initialization":             "Initializing job…",
    "random_route_generation":    "Generating traffic demand scenarios…",
    "scenario_manifest_creation": "Building training configuration…",
    "independent_dqn_training":   "Training GAT model…",
    "evaluation":                 "Running fixed-time baseline…",
    "complete":                   "Training complete!",
}

STAGE_PROGRESS = {
    "initialization":             10,
    "random_route_generation":    22,
    "scenario_manifest_creation": 32,
    "independent_dqn_training":   35,  # 35→75% filled by episode progress on frontend
    "evaluation":                 80,
    "complete":                   100,
}

STAGE_ORDER = list(STAGE_PROGRESS.keys())


class RealTrainRequest(BaseModel):
    session_id: str
    net_path: str
    pass_threshold_pct: float = 25.0


@router.post("/api/real-train/start")
async def start_training(req: RealTrainRequest) -> dict:
    started = start_real_training(req.session_id, req.net_path,
                                  pass_threshold_pct=req.pass_threshold_pct)
    return {"started": started, "session_id": req.session_id}


@router.get("/api/real-train/{session_id}/status")
async def training_status(session_id: str) -> dict:
    raw = get_job_status(session_id)
    stage = raw.get("stage", "initializing")
    completed_stages = STAGE_ORDER[: STAGE_ORDER.index(stage)] if stage in STAGE_ORDER else []
    return {
        **raw,
        "label": STAGE_LABELS.get(stage, stage),
        "progress_pct": STAGE_PROGRESS.get(stage, 0),
        "completed_stages": [
            {"stage": s, "label": STAGE_LABELS.get(s, s)} for s in completed_stages
        ],
    }


@router.post("/api/real-train/{session_id}/stop")
async def stop_training_endpoint(session_id: str) -> dict:
    """Request early stop. Trainer saves checkpoint after current episode and exits cleanly."""
    ok = stop_training(session_id)
    return {"requested": ok}


@router.post("/api/real-train/{session_id}/resume")
async def resume_training_endpoint(session_id: str) -> dict:
    """Resume a stopped training run from the saved checkpoint for the remaining episodes."""
    ok = resume_training(session_id)
    return {"started": ok}


class RetrainRequest(BaseModel):
    mode: str = "continue"  # "continue" | "explore"


@router.post("/api/real-train/{session_id}/retrain")
async def retrain_endpoint(session_id: str, req: RetrainRequest) -> dict:
    """Trigger retraining after a failed eval verdict.

    mode='continue' — preserve epsilon schedule, run num_episodes from checkpoint.
    mode='explore'  — reset epsilon to 0.5 and grad-steps to 0, run num_episodes.
    """
    ok = retrain(session_id, req.mode)
    return {"started": ok}


@router.get("/api/real-train/{session_id}/training-progress")
async def training_progress_detail(session_id: str) -> dict:
    return get_training_progress(session_id)


@router.get("/api/real-train/{session_id}/logs")
async def training_logs(session_id: str) -> dict:
    return {"logs": get_training_logs(session_id)}


@router.get("/api/real-train/{session_id}/latest-episode-kpis")
async def latest_episode_kpis(session_id: str) -> dict:
    """Return the most recent per-episode KPIs from training_metrics.jsonl (live stream file)."""
    path = WEB_JOBS_ROOT / session_id / "reports" / "training_metrics.jsonl"
    if not path.exists():
        return {"available": False}
    try:
        lines = path.read_text().strip().splitlines()
        if not lines:
            return {"available": False}
        return {"available": True, **json.loads(lines[-1])}
    except Exception:
        return {"available": False}


@router.get("/api/real-train/{session_id}/result")
async def training_result(session_id: str) -> dict:
    result = get_job_result(session_id)
    if not result:
        return {"available": False}
    return result  # already has "available": True; eval_metrics/baseline_metrics at top level


@router.get("/api/real-train/{session_id}/debug-metrics")
async def debug_metrics(session_id: str) -> dict:
    """Temporary debug endpoint — shows the raw structure of training_metrics.json."""
    metrics_file = WEB_JOBS_ROOT / session_id / "training" / "full" / "training_metrics.json"
    if not metrics_file.exists():
        return {"exists": False, "checked_path": str(metrics_file)}
    try:
        data = json.loads(metrics_file.read_text())
    except Exception as exc:
        return {"exists": True, "parse_error": str(exc)}

    if isinstance(data, list):
        first = data[0] if data else {}
        last = data[-1] if data else {}
        return {
            "exists": True,
            "type": "list",
            "length": len(data),
            "first_record_keys": list(first.keys()) if isinstance(first, dict) else str(first)[:200],
            "sample_first_record": first,
            "sample_last_record": last,
        }
    if isinstance(data, dict):
        first_key = next(iter(data), None)
        first_val = data.get(first_key, [])
        first_item = first_val[0] if isinstance(first_val, list) and first_val else None
        last_item = first_val[-1] if isinstance(first_val, list) and first_val else None
        return {
            "exists": True,
            "type": "dict",
            "top_level_keys": list(data.keys()),
            "first_record_keys": list(first_item.keys()) if isinstance(first_item, dict) else "not a list",
            "sample_first_record": first_item,
            "sample_last_record": last_item,
        }
    return {"exists": True, "type": type(data).__name__, "raw_preview": str(data)[:500]}


@router.get("/api/real-train/{session_id}/episode-metrics")
async def episode_metrics(session_id: str) -> dict:
    import re

    # training_metrics.csv/json only appear AFTER training completes and contain
    # gradient-update records (loss, epsilon) — no KPI data.
    # The only live source of per-episode KPIs is run.log in each rollout dir.
    result: dict = {
        "available": False,
        "current_episode": 0,
        "convergence_pct": None,
        "reward_history": [],
        "latest_kpis": {
            "waiting_time": None,
            "queue_length": None,
            "throughput": None,
            "phase_change_rate": None,
        },
    }

    rollouts_dir = None
    for phase in ("full", "smoke"):
        candidate = WEB_JOBS_ROOT / session_id / "training" / phase / "training_rollouts"
        if candidate.exists():
            rollouts_dir = candidate
            break

    if not rollouts_dir:
        return result

    try:
        episodes = sorted(
            [d for d in rollouts_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
        )
    except Exception:
        return result

    if not episodes:
        return result

    result["available"] = True
    result["current_episode"] = len(episodes)

    def parse_run_log(log_path: Path):
        """Extract KPIs from SUMO run.log.
        Returns (waiting_time_s, queue_length_m, completed_trips).
        run.log fields used:
          - 'WaitingTime: N.NN'  → mean wait per completed trip (seconds)
          - 'Statistics (avg of N)' → N completed trips (throughput)
          - ' Waiting: N'  → vehicles still waiting at sim end (queue proxy: N × 8m)
        """
        try:
            text = log_path.read_text(errors="replace")

            m = re.search(r"WaitingTime:\s*([\d.]+)", text)
            waiting_time = float(m.group(1)) if m else None

            # Note: leading space distinguishes " Waiting:" from "WaitingTime:"
            m = re.search(r" Waiting:\s*(\d+)", text)
            waiting_count = int(m.group(1)) if m else None
            # 0 waiting vehicles is valid (good!) — don't turn it into None
            queue_length = round(waiting_count * 8.0, 1) if waiting_count is not None else None

            # "avg of N" = N vehicles that completed their trips this episode
            m = re.search(r"Statistics \(avg of (\d+)\)", text)
            completed = int(m.group(1)) if m else None

            return waiting_time, queue_length, completed
        except Exception:
            return None, None, None

    # KPIs from the most recently completed episode.
    # The last episode dir may be in-progress (incomplete run.log) — scan backwards
    # until we find one with a complete WaitingTime.
    for ep_dir in reversed(episodes):
        log = ep_dir / "run.log"
        if log.exists():
            wt, ql, tp = parse_run_log(log)
            if wt is not None:
                result["latest_kpis"]["waiting_time"] = wt
                result["latest_kpis"]["queue_length"] = ql
                result["latest_kpis"]["throughput"] = tp
                # phase_change_rate is not emitted by SUMO default output — stays None
                break

    # Reward history + convergence from all episode run.logs
    wt_series: list[float] = []
    for ep_dir in episodes:
        log = ep_dir / "run.log"
        if log.exists():
            wt, _, _ = parse_run_log(log)
            if wt is not None:
                wt_series.append(wt)

    if wt_series:
        max_wt = max(wt_series) or 1.0
        result["reward_history"] = [round((max_wt - wt) / max_wt * 100, 1) for wt in wt_series]

        # Convergence: coefficient of variation over last 20 episodes
        # Low CV (stable performance) → high convergence_pct
        if len(wt_series) >= 10:
            recent = wt_series[-min(20, len(wt_series)):]
            mean_wt = sum(recent) / len(recent)
            if mean_wt > 0:
                std = (sum((x - mean_wt) ** 2 for x in recent) / len(recent)) ** 0.5
                cv = std / mean_wt  # 0 = perfectly stable, >0.5 = highly variable
                # Scale: cv=0 → 100%, cv=0.5 → 0%
                result["convergence_pct"] = max(0, min(100, int((1 - min(cv, 0.5) / 0.5) * 100)))

    return result


@router.get("/api/real-train/{session_id}/download-model")
async def download_model(session_id: str) -> FileResponse:
    path = get_model_checkpoint_path(session_id)
    if not path:
        raise HTTPException(status_code=404, detail="Model checkpoint not ready yet")
    return FileResponse(
        str(path),
        filename=f"lightmind_dqn_{session_id}.pt",
        media_type="application/octet-stream",
    )


@router.get("/api/real-train/{session_id}/download-bundle")
async def download_bundle(session_id: str) -> FileResponse:
    path = get_package_zip_path(session_id)
    if not path:
        raise HTTPException(status_code=404, detail="Bundle not ready yet")
    return FileResponse(
        str(path),
        filename=f"lightmind_bundle_{session_id}.zip",
        media_type="application/zip",
    )


# ── Fixed-time baseline routes ────────────────────────────────────────────────

class BaselineRequest(BaseModel):
    green_duration_s: int = 60


@router.post("/api/real-train/{session_id}/run-baseline")
async def run_baseline(session_id: str, req: BaselineRequest = BaselineRequest()) -> dict:
    import threading  # noqa: PLC0415
    from services.fake_trainer import load_status  # noqa: PLC0415

    try:
        status = load_status(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    net_path_str = status.get("net_absolute_path", "")
    if not net_path_str:
        raise HTTPException(status_code=400, detail="No network file found for this session")

    green_duration = max(10, min(300, req.green_duration_s))

    baseline_status_path = WEB_JOBS_ROOT / session_id / "reports" / "baseline_status.json"
    if baseline_status_path.exists():
        try:
            existing = json.loads(baseline_status_path.read_text())
            if existing.get("status") in ("running", "complete"):
                return {"started": False, "message": "Baseline already running or complete"}
        except Exception:
            pass

    baseline_status_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_status_path.write_text(json.dumps({
        "status": "running",
        "runs_complete": 0,
        "total_runs": 9,
        "green_duration_s": green_duration,
    }))

    def run_bg() -> None:
        try:
            from services.fixed_time_runner import run_fixed_time_baseline  # noqa: PLC0415
            summary = run_fixed_time_baseline(
                net_file=Path(net_path_str),
                session_id=session_id,
                output_root=WEB_JOBS_ROOT,
                green_duration_s=green_duration,
            )
            baseline_status_path.write_text(json.dumps({
                "status": "complete",
                "runs_complete": 9,
                "total_runs": 9,
                "green_duration_s": green_duration,
                "summary": summary,
            }))
        except Exception as exc:
            baseline_status_path.write_text(json.dumps({"status": "failed", "error": str(exc)}))

    threading.Thread(target=run_bg, daemon=True).start()
    return {"started": True, "green_duration_s": green_duration}


@router.get("/api/real-train/{session_id}/baseline-status")
async def baseline_status(session_id: str) -> dict:
    path = WEB_JOBS_ROOT / session_id / "reports" / "baseline_status.json"
    if not path.exists():
        return {"status": "not_started"}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"status": "unknown"}


@router.get("/api/real-train/{session_id}/baseline-results")
async def baseline_results(session_id: str) -> dict:
    path = WEB_JOBS_ROOT / session_id / "reports" / "baseline_summary.json"
    if not path.exists():
        return {"available": False}
    try:
        return {"available": True, "summary": json.loads(path.read_text())}
    except Exception:
        return {"available": False}


# ── SUMO-GUI evaluation endpoints ─────────────────────────────────────────────

@router.post("/api/real-train/{session_id}/start-eval")
async def start_eval(session_id: str, background_tasks: BackgroundTasks) -> dict:
    """Kick off the two-phase SUMO-GUI eval: fixed-time first, then GAT."""
    if _eval_status.get(session_id, {}).get("phase") in ("fixed_time", "gat"):
        return {"started": False, "reason": "already running"}
    _eval_status[session_id] = {"phase": "fixed_time", "episode1_running": False, "error": None}
    background_tasks.add_task(_run_eval, session_id)
    return {"started": True}


@router.get("/api/real-train/{session_id}/eval-status")
async def eval_status_endpoint(session_id: str) -> dict:
    return _eval_status.get(session_id, {"phase": "not_started"})


@router.get("/api/real-train/{session_id}/eval-comparison")
async def eval_comparison(session_id: str) -> dict:
    path = WEB_JOBS_ROOT / session_id / "reports" / "eval_comparison.json"
    if not path.exists():
        return {"available": False}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"available": False}


def _eval_verdict(gat: dict, ft: dict, threshold_pct: float) -> dict:
    def _mean(lst):
        return sum(lst) / len(lst) if lst else None

    gat_wait = _mean(gat.get("avg_waiting_time", [])) or gat.get("mean_waiting_time")
    ft_wait  = _mean(ft.get("avg_waiting_time",  [])) or ft.get("mean_waiting_time")

    if not gat_wait or not ft_wait or ft_wait == 0:
        return {"passed": None, "reason": "insufficient data"}

    improvement_pct = (ft_wait - gat_wait) / ft_wait * 100
    return {
        "passed":               improvement_pct >= threshold_pct,
        "wait_improvement_pct": round(improvement_pct, 2),
        "threshold_pct":        threshold_pct,
        "gat_avg_wait":         round(gat_wait, 2),
        "baseline_avg_wait":    round(ft_wait, 2),
    }


async def _run_eval(session_id: str) -> None:
    """Background task: run fixed-time then GAT eval, each with GUI on episode 1."""
    import yaml

    # Resolve ml_project root (env var set in Docker; fall back for local dev)
    ml_project_root = Path(os.environ.get(
        "ML_PROJECT_ROOT",
        str(Path(__file__).parent.parent.parent.parent),
    ))
    ml_model_dir = ml_project_root / "model"
    if str(ml_model_dir) not in sys.path:
        sys.path.insert(0, str(ml_model_dir))

    # Import the phase runner after sys.path is set
    from services.eval_runner_service import run_phase_with_gui  # noqa: PLC0415

    # Locate training config written by real_trainer.py
    config_dir = Path(__file__).parent.parent / "data" / "uploads" / session_id
    config_path = next(config_dir.glob("*_training_config.yaml"), None)
    if not config_path:
        _eval_status[session_id] = {"phase": "error", "episode1_running": False,
                                    "error": "Training config not found"}
        return

    cfg = yaml.safe_load(config_path.read_text())
    _data_dir = Path(__file__).parent.parent / "data"

    def _resolve_cfg_path(raw: str) -> str:
        if Path(raw).exists():
            return raw
        posix = raw.replace("\\", "/")
        idx = posix.find("/data/")
        if idx != -1:
            return str(_data_dir / posix[idx + len("/data/"):])
        return raw

    net_file  = _resolve_cfg_path(cfg["network"]["net"])
    rou_file  = _resolve_cfg_path(cfg["network"]["rou"])
    max_steps = cfg["network"].get("max_steps", 720)
    begin_time = cfg["network"].get("begin_time", 0)
    seeds     = [0, 100, 200, 300, 400]

    reports_dir = WEB_JOBS_ROOT / session_id / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()

    try:
        # ── Phase 1: fixed-time baseline ──────────────────────────────────────
        _eval_status[session_id]["phase"] = "fixed_time"
        _eval_status[session_id]["episode1_running"] = True

        ft_metrics = await loop.run_in_executor(None, lambda: run_phase_with_gui(
            mode="fixed_time",
            display=":99",
            net_file=net_file,
            route_file=rou_file,
            max_steps=max_steps,
            begin_time=begin_time,
            seeds=seeds,
        ))

        _eval_status[session_id]["episode1_running"] = False
        (reports_dir / "fixed_time_eval.json").write_text(
            json.dumps({"available": True, "metrics": ft_metrics})
        )

        # ── Phase 2: GAT model ────────────────────────────────────────────────
        _eval_status[session_id]["phase"] = "gat"
        _eval_status[session_id]["episode1_running"] = True

        ckpt_dir  = ml_project_root / "checkpoints" / session_id
        ckpt_path = ckpt_dir / "final.pt"
        if not ckpt_path.exists():
            candidates = sorted(ckpt_dir.glob("*.pt"))
            if candidates:
                ckpt_path = candidates[-1]
            else:
                # Fall back to the baked-in production model (single-municipality deployment)
                production = ml_project_root / "model" / "checkpoints" / "production" / "final.pt"
                if not production.exists():
                    raise FileNotFoundError(
                        f"No session checkpoint in {ckpt_dir} and no production model at {production}"
                    )
                ckpt_path = production

        gat_metrics = await loop.run_in_executor(None, lambda: run_phase_with_gui(
            mode="gat",
            display=":99",
            net_file=net_file,
            route_file=rou_file,
            max_steps=max_steps,
            begin_time=begin_time,
            seeds=seeds,
            checkpoint=str(ckpt_path),
            cfg=cfg,
        ))

        _eval_status[session_id]["episode1_running"] = False
        (reports_dir / "gat_eval.json").write_text(
            json.dumps({"available": True, "metrics": gat_metrics})
        )

        # ── Write comparison and signal done ──────────────────────────────────
        threshold = float(cfg.get("eval_pass_threshold_pct", 25.0))
        verdict   = _eval_verdict(gat_metrics, ft_metrics, threshold)
        (reports_dir / "eval_comparison.json").write_text(json.dumps({
            "available":  True,
            "gat":        gat_metrics,
            "fixed_time": ft_metrics,
            "verdict":    verdict,
        }))
        _eval_status[session_id]["phase"] = "done"
        _eval_status[session_id]["episode1_running"] = False

    except Exception as exc:
        _eval_status[session_id]["phase"] = "error"
        _eval_status[session_id]["episode1_running"] = False
        _eval_status[session_id]["error"] = str(exc)
