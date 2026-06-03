from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.fake_trainer import (
    create_status,
    ensure_data_dirs,
    ensure_session_dirs,
    load_status,
    save_status,
    session_upload_dir,
)


router = APIRouter(prefix="/api/upload", tags=["upload"])
network_router = APIRouter(prefix="/api/sessions", tags=["network"])


@router.post("/osm")
async def upload_osm(file: UploadFile = File(...)) -> dict:
    extension = Path(file.filename or "").suffix.lower()
    if extension != ".osm":
        raise HTTPException(status_code=400, detail="Only .osm files are accepted")

    ensure_data_dirs()
    session_id = uuid4().hex
    upload_dir, _ = ensure_session_dirs(session_id)
    target_path = upload_dir / "map.osm"

    with target_path.open("wb") as handle:
        handle.write(await file.read())

    create_status(session_id, osm_filename=file.filename or "map.osm")
    # Persist absolute OSM path so real_train can find it later
    from services.fake_trainer import load_status
    _status = load_status(session_id)
    _status["osm_absolute_path"] = str(target_path.resolve())
    save_status(session_id, _status)

    network_summary = None
    sumo_error = None

    import main as app_main  # noqa: PLC0415 — imported here to avoid circular import at module load
    if app_main.SUMO_AVAILABLE:
        try:
            from services.osm_converter import convert_osm_to_sumo, get_network_summary
            result = convert_osm_to_sumo(str(target_path), session_id)
            network_summary = get_network_summary(result["network_data"], result["net_file"])
        except Exception as exc:
            sumo_error = str(exc)

    return {
        "session_id": session_id,
        "osm_absolute_path": str(target_path.resolve()),
        "message": "OSM uploaded successfully",
        "osm_uploaded": True,
        "sumo_converted": network_summary is not None,
        "sumo_error": sumo_error,
        "network_summary": network_summary,
    }


@router.post("/demand")
async def upload_demand(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, str]:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in {".xlsx", ".csv"}:
        raise HTTPException(status_code=400, detail="Only .xlsx or .csv files are accepted")

    try:
        status = load_status(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc

    upload_dir = session_upload_dir(session_id)
    filename = f"demand{extension}"
    target_path = upload_dir / filename

    with target_path.open("wb") as handle:
        handle.write(await file.read())

    status["demand_filename"] = file.filename or filename
    save_status(session_id, status)
    return {"status": "success", "message": "Demand file uploaded successfully"}


@network_router.get("/{session_id}/network")
async def get_network(session_id: str) -> dict:
    net_file = f"data/sessions/{session_id}/sumo/network.net.xml"

    if not os.path.exists(net_file):
        return {"available": False}

    try:
        from services.osm_converter import extract_network_data, get_network_summary
        network_data = extract_network_data(net_file)
        summary = get_network_summary(network_data, net_file)
        return {"available": True, **summary}
    except Exception as exc:
        return {"available": False, "error": str(exc)}
