from fastapi import APIRouter, HTTPException

from services.fake_trainer import generate_results, load_status


router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("/{session_id}")
def get_results(session_id: str) -> dict:
    try:
        load_status(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc

    return generate_results(session_id)
