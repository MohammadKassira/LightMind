from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
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
    start_real_training,
)

router = APIRouter()

STAGE_LABELS = {
    "initialization": "Initializing job...",
    "traffic_light_detection": "Detecting traffic signals in network...",
    "random_route_generation": "Generating traffic demand scenarios...",
    "scenario_manifest_creation": "Building training scenarios...",
    "independent_dqn_training": "Training Independent DQN model...",
    "evaluation": "Evaluating trained model on held-out scenarios...",
    "kpi_extraction": "Extracting KPI results...",
    "output_packaging": "Packaging outputs for download...",
    "complete": "Training complete!",
}

STAGE_PROGRESS = {
    "initialization": 2,
    "traffic_light_detection": 15,
    "random_route_generation": 22,
    "scenario_manifest_creation": 28,
    "independent_dqn_training": 35,  # episodes fill 35→75%; evaluation picks up at 80
    "evaluation": 80,
    "kpi_extraction": 90,
    "output_packaging": 96,
    "complete": 100,
}

STAGE_ORDER = list(STAGE_PROGRESS.keys())


class RealTrainRequest(BaseModel):
    session_id: str
    net_path: str


@router.post("/api/real-train/start")
async def start_training(req: RealTrainRequest) -> dict:
    started = start_real_training(req.session_id, req.net_path)
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


@router.get("/api/real-train/{session_id}/training-progress")
async def training_progress_detail(session_id: str) -> dict:
    return get_training_progress(session_id)


@router.get("/api/real-train/{session_id}/logs")
async def training_logs(session_id: str) -> dict:
    return {"logs": get_training_logs(session_id)}


@router.get("/api/real-train/{session_id}/result")
async def training_result(session_id: str) -> dict:
    result = get_job_result(session_id)
    if not result:
        return {"available": False}
    kpis = get_kpi_summary(session_id)
    return {"available": True, "result": result, "kpis": kpis}


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
