from __future__ import annotations

import os
import xml.etree.ElementTree as ET
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


@router.post("/net")
async def upload_net(file: UploadFile = File(...)) -> dict:
    filename = file.filename or "uploaded_map.net.xml"
    lower = filename.lower()
    if not (lower.endswith(".net.xml") or lower.endswith(".xml")):
        raise HTTPException(status_code=400, detail="Only .net.xml or .xml files are accepted")

    content = await file.read()

    # Validate it is a SUMO net file
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid XML: {exc}") from exc
    if root.tag != "net":
        raise HTTPException(status_code=400, detail=f"Root XML tag must be <net>, got <{root.tag}>. Upload a SUMO .net.xml file.")

    ensure_data_dirs()
    session_id = uuid4().hex
    upload_dir, _ = ensure_session_dirs(session_id)
    target_path = upload_dir / "uploaded_map.net.xml"
    target_path.write_bytes(content)

    create_status(session_id, net_filename=filename)
    _status = load_status(session_id)
    _status["net_absolute_path"] = str(target_path.resolve())
    save_status(session_id, _status)

    network_summary = None
    try:
        from services.osm_converter import extract_network_data, get_network_summary  # noqa: PLC0415
        network_data = extract_network_data(str(target_path))
        network_summary = get_network_summary(network_data, str(target_path))
    except Exception:
        pass  # Non-fatal — summary only used for UI display

    return {
        "session_id": session_id,
        "net_absolute_path": str(target_path.resolve()),
        "message": "SUMO network file uploaded successfully",
        "net_uploaded": True,
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
