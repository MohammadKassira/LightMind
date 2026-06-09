from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.fake_trainer import load_status, start_training


router = APIRouter(prefix="/api", tags=["train"])


class TrainRequest(BaseModel):
    session_id: str
    baseline: str
    demand_level: str
    custom_demand: Optional[dict[str, Any]] = None
    demand_schedule: Optional[list[dict[str, Any]]] = None


@router.post("/train")
def train_model(payload: TrainRequest) -> dict[str, str]:
    try:
        load_status(payload.session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc

    start_training(
        payload.session_id,
        payload.baseline,
        payload.demand_level,
        payload.custom_demand,
        payload.demand_schedule,
    )
    return {"status": "started"}
