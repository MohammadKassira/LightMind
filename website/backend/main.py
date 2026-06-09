from dotenv import load_dotenv
load_dotenv(override=True)  # override=True ensures .env values beat any pre-set shell vars

import asyncio
import os
# Force the virtual Xvfb display so any X11 subprocess always uses it,
# overriding anything injected by WSLg or Docker Desktop on Windows.
os.environ.setdefault("DISPLAY", ":99")
import shutil
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from routers import demand_template, deployment, model_download, parse_schedule, real_train, results, train, upload, ws
from services.real_trainer import is_sumo_available


def check_sumo_installed():
    if not shutil.which("netconvert"):
        print("WARNING: SUMO/netconvert not found. OSM conversion unavailable.")
        return False
    return True

SUMO_AVAILABLE = check_sumo_installed()

app = FastAPI(title="LightMind API", version="0.1.0")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://frontend:5173",
    "http://0.0.0.0:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(upload.network_router)
app.include_router(deployment.router)
app.include_router(real_train.router)
app.include_router(train.router)
app.include_router(ws.router)
app.include_router(results.router)
app.include_router(parse_schedule.router)
app.include_router(model_download.router)
app.include_router(demand_template.router)


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/system/status")
def system_status() -> dict:
    sumo = is_sumo_available()
    return {
        "sumo_available": sumo,
        "real_training_available": sumo,
    }


# ── noVNC reverse proxy (HF Spaces only exposes port 7860) ───────────────────

_NOVNC_BASE = "http://localhost:6080"
_NOVNC_WS_URL = "ws://localhost:6080"
_SKIP_RESPONSE_HEADERS = {"connection", "transfer-encoding", "content-encoding"}


@app.api_route("/novnc/{path:path}", methods=["GET", "HEAD"])
async def novnc_proxy(path: str, request: Request) -> Response:
    """Proxy noVNC static files through FastAPI so port 7860 is the only entry point."""
    try:
        import httpx
        url = f"{_NOVNC_BASE}/{path}"
        if request.query_params:
            url += f"?{request.query_params}"
        async with httpx.AsyncClient() as client:
            resp = await client.request(request.method, url, timeout=10.0)
        headers = {k: v for k, v in resp.headers.items() if k.lower() not in _SKIP_RESPONSE_HEADERS}
        return Response(content=resp.content, status_code=resp.status_code, headers=headers)
    except Exception:
        return Response(content=b"noVNC not ready", status_code=503)


@app.websocket("/novnc/websockify")
async def novnc_ws_proxy(websocket: WebSocket) -> None:
    """Proxy WebSocket VNC traffic so the browser can reach x11vnc via port 7860."""
    client_protocols = websocket.headers.get("sec-websocket-protocol", "")
    # Only accept/request a subprotocol if the browser offered one.
    # Accepting with a subprotocol the client never offered causes an immediate RFC 6455 close.
    subprotocol = client_protocols.split(",")[0].strip() if client_protocols else None
    await websocket.accept(subprotocol=subprotocol)
    try:
        import websockets
        ws_kwargs: dict = {}
        if subprotocol:
            ws_kwargs["subprotocols"] = [subprotocol]
        async with websockets.connect(f"{_NOVNC_WS_URL}/", **ws_kwargs) as vnc_ws:

            async def browser_to_vnc():
                try:
                    async for msg in websocket.iter_bytes():
                        await vnc_ws.send(msg)
                except Exception:
                    pass

            async def vnc_to_browser():
                try:
                    async for msg in vnc_ws:
                        if isinstance(msg, bytes):
                            await websocket.send_bytes(msg)
                        else:
                            await websocket.send_text(msg)
                except Exception:
                    pass

            await asyncio.gather(browser_to_vnc(), vnc_to_browser())
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
