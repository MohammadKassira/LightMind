from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from services.real_trainer import (
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
    "osm_validation": "Validating OSM map...",
    "osm_to_sumo_conversion": "Converting OSM to SUMO network...",
    "traffic_light_detection": "Detecting real traffic signals...",
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
    "osm_validation": 5,
    "osm_to_sumo_conversion": 15,
    "traffic_light_detection": 20,
    "random_route_generation": 30,
    "scenario_manifest_creation": 35,
    "independent_dqn_training": 80,
    "evaluation": 92,
    "kpi_extraction": 96,
    "output_packaging": 99,
    "complete": 100,
}

STAGE_ORDER = list(STAGE_PROGRESS.keys())


class RealTrainRequest(BaseModel):
    session_id: str
    osm_path: str


@router.post("/api/real-train/start")
async def start_training(req: RealTrainRequest) -> dict:
    started = start_real_training(req.session_id, req.osm_path)
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
