from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

_TEMPLATE_PATH = Path(__file__).parent.parent / "static" / "demand_template.xlsx"


@router.get("/api/demand-template")
async def download_demand_template() -> FileResponse:
    if not _TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404, detail="Template file not found")
    return FileResponse(
        str(_TEMPLATE_PATH),
        filename="lightmind_demand_template.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
