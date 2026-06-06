from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from services.fake_trainer import SESSIONS_DIR

router = APIRouter(prefix="/api/sessions", tags=["model"])


@router.get("/{session_id}/model")
def download_model(session_id: str) -> FileResponse:
    model_path = SESSIONS_DIR / session_id / "model.json"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model file not found. Training may not have completed yet.",
        )
    filename = f"lightmind_model_{session_id}.json"
    return FileResponse(
        path=str(model_path),
        media_type="application/json",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
