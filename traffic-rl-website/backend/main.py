from dotenv import load_dotenv
load_dotenv(override=True)  # override=True ensures .env values beat any pre-set shell vars

import shutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import demand_template, model_download, parse_schedule, real_train, results, train, upload, ws
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
