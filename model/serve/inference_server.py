"""
Inference server for LightMind traffic RL model.

Loads the trained checkpoint once at startup.
POST /decide  — runs encoder → GAT → PhaseHead, returns phase decisions.
GET  /health  — reports latency, fallback tier, cycle count.

Watchdog:
  - inference > 500 ms → fall back to MaxPressure
  - MaxPressure error  → fall back to fixed-time
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_MODEL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_MODEL_DIR))

from models.node_encoder import pad_obs_dict
from models.phase_head import pad_phase_features

app = FastAPI(title="LightMind Inference Server")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Global state (populated at startup) ──────────────────────────────────────

_trainer = None
_graph: dict | None = None
_padded_pf: list | None = None
_cycle_count: int = 0
_last_latency_ms: float = 0.0
_fallback_tier: int = 0   # 0 = model, 1 = MaxPressure, 2 = fixed-time
_fixed_time_state: dict = {}


# ── Request / response schemas ────────────────────────────────────────────────

class NodeObsIn(BaseModel):
    obs: list[float]
    validity: list[float]

class DecideRequest(BaseModel):
    obs: dict[str, NodeObsIn]
    # raw incoming-lane queue counts (halting vehicles, un-normalised) for MaxPressure fallback
    raw_queues: dict[str, list[float]] = {}

class DecideResponse(BaseModel):
    actions: dict[str, int]
    fallback_tier: int
    latency_ms: float
    cycle: int


# ── Fallback helpers ──────────────────────────────────────────────────────────

def _max_pressure(req: DecideRequest) -> dict[str, int]:
    """Pick the phase with the highest sum of green-lane queue lengths."""
    actions: dict[str, int] = {}
    for i, nid in enumerate(_graph["node_ids"]):
        if nid not in req.obs:
            continue
        queues = req.raw_queues.get(nid, [])
        phase_feats = _graph["phase_features"][i]
        best_phase, best_p = 0, -1.0
        for ph_idx, pf in enumerate(phase_feats):
            pressure = sum(
                float(pf[j]) * (queues[j] if j < len(queues) else 0.0)
                for j in range(len(pf))
            )
            if pressure > best_p:
                best_p = pressure
                best_phase = ph_idx
        actions[nid] = best_phase
    return actions


def _fixed_time() -> dict[str, int]:
    """Cycle through phases on a fixed schedule (~30 s per phase at 5 s steps)."""
    DURATION = 6
    actions: dict[str, int] = {}
    for i, nid in enumerate(_graph["node_ids"]):
        st = _fixed_time_state.setdefault(nid, {"phase": 0, "ticks": 0})
        st["ticks"] += 1
        if st["ticks"] >= DURATION:
            st["phase"] = (st["phase"] + 1) % _graph["node_meta"][i]["num_phases"]
            st["ticks"] = 0
        actions[nid] = st["phase"]
    return actions


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/decide", response_model=DecideResponse)
async def decide(req: DecideRequest) -> DecideResponse:
    global _cycle_count, _last_latency_ms, _fallback_tier

    _cycle_count += 1
    t0 = time.perf_counter()

    try:
        obs_dict = {
            nid: (
                torch.tensor(v.obs, dtype=torch.float32),
                torch.tensor(v.validity, dtype=torch.float32),
            )
            for nid, v in req.obs.items()
        }
        _, padded_obs = pad_obs_dict(obs_dict)
        actions = _trainer._select_actions(padded_obs, _graph, _padded_pf, epsilon=0.0)
        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed > 500:
            raise TimeoutError(f"{elapsed:.0f} ms exceeds 500 ms budget")
        _fallback_tier = 0
        _last_latency_ms = elapsed

    except Exception:
        try:
            actions = _max_pressure(req)
            _fallback_tier = 1
        except Exception:
            actions = _fixed_time()
            _fallback_tier = 2
        _last_latency_ms = (time.perf_counter() - t0) * 1000

    return DecideResponse(
        actions={k: int(v) for k, v in actions.items()},
        fallback_tier=_fallback_tier,
        latency_ms=round(_last_latency_ms, 2),
        cycle=_cycle_count,
    )


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "cycle_count": _cycle_count,
        "last_latency_ms": round(_last_latency_ms, 2),
        "fallback_tier": _fallback_tier,
        "fallback_name": ["model", "MaxPressure", "fixed-time"][min(_fallback_tier, 2)],
    }


# ── Startup ───────────────────────────────────────────────────────────────────

def _load(checkpoint: str, net_file: str, route_file: str) -> None:
    global _trainer, _graph, _padded_pf, _fixed_time_state

    from data.graph_builder import build_graph
    from env.traffic_env import TrafficEnv
    from training.trainer import DQNTrainer

    _graph = build_graph(net_file)

    # Probe env: headless SUMO started briefly to establish obs dims for DQNTrainer
    probe = TrafficEnv(net_file=net_file, route_file=route_file, use_gui=False, max_steps=1)
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    _trainer = DQNTrainer.load_checkpoint(checkpoint, cfg, probe)
    probe.close()

    _, _padded_pf = pad_phase_features(_graph)
    _fixed_time_state.clear()
    print(f"[inference_server] ready — {len(_graph['node_ids'])} nodes", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--net-file", required=True)
    ap.add_argument("--route-file", required=True)
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    _load(args.checkpoint, args.net_file, args.route_file)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
