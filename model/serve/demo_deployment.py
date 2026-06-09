"""
Demo deployment client.

Runs SUMO via TraCI (with GUI on the configured DISPLAY) and drives it using
the inference server.  Every decision_interval seconds it:
  1. Reads the current obs_dict from the environment.
  2. Serialises it and POSTs to the inference server's /decide endpoint.
  3. Applies the returned phase decisions back to SUMO.
  4. POSTs a metrics snapshot to the backend callback URL.

In a real deployment, replace step 1 with camera / sensor output and remove
the TrafficEnv wrapper — the inference server interface stays identical.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

_MODEL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_MODEL_DIR))


def _serialize_obs(obs_dict: dict, graph: dict) -> dict:
    """Convert obs_dict tensors → JSON-serialisable format for /decide."""
    obs_payload: dict = {}
    raw_queues: dict = {}
    num_phases_map = {
        nid: graph["node_meta"][i]["num_phases"]
        for i, nid in enumerate(graph["node_ids"])
    }
    for nid, (obs_t, val_t) in obs_dict.items():
        obs_payload[nid] = {
            "obs": obs_t.tolist(),
            "validity": val_t.tolist(),
        }
        # Extract un-normalised incoming queue for MaxPressure fallback.
        # obs layout: [phase_onehot(P), time_in_phase(1), queue_in(L_in), ...]
        np_ = num_phases_map.get(nid, 1)
        queue_start = np_ + 1
        obs_list = obs_t.tolist()
        # queue_in values are halting/30 — multiply back to vehicle count
        raw_queues[nid] = [v * 30.0 for v in obs_list[queue_start:]]
    return {"obs": obs_payload, "raw_queues": raw_queues}


def _per_node_stats(obs_dict: dict, graph: dict, actions: dict) -> dict:
    """Extract per-intersection stats for the dashboard."""
    stats: dict = {}
    for i, nid in enumerate(graph["node_ids"]):
        if nid not in obs_dict:
            continue
        obs_t, _ = obs_dict[nid]
        np_ = graph["node_meta"][i]["num_phases"]
        queue_start = np_ + 1
        obs_list = obs_t.tolist()
        queue_vals = obs_list[queue_start:]
        total_queue = sum(v * 30.0 for v in queue_vals)
        stats[nid] = {
            "phase": actions.get(nid, 0),
            "queue": round(total_queue, 1),
        }
    return stats


def run(
    net_file: str,
    route_file: str,
    inference_url: str,
    callback_url: str,
    display: str = ":99",
    max_steps: int = 720,
    begin_time: int = 0,
    seed: int = 42,
) -> None:
    import os
    os.environ["DISPLAY"] = display

    from env.traffic_env import TrafficEnv

    env = TrafficEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=True,
        max_steps=max_steps,
        begin_time=begin_time,
        override_tl_program=True,
    )

    obs_dict, graph = env.reset(seed=seed)
    done = False
    step = 0

    # Notify backend: deployment started
    try:
        requests.post(callback_url, json={"event": "started"}, timeout=2)
    except Exception:
        pass

    while not done:
        payload = _serialize_obs(obs_dict, graph)

        # ── Call inference server ─────────────────────────────────────────
        try:
            resp = requests.post(f"{inference_url}/decide", json=payload, timeout=2.0)
            result = resp.json()
            actions = {k: int(v) for k, v in result["actions"].items()}
            fallback_tier = result.get("fallback_tier", 0)
            latency_ms = result.get("latency_ms", 0.0)
            cycle = result.get("cycle", step)
        except Exception as exc:
            # Inference server unreachable — use hold-current (empty actions)
            actions = {}
            fallback_tier = 2
            latency_ms = 0.0
            cycle = step
            print(f"[demo] inference error: {exc}", flush=True)

        # ── Step environment ──────────────────────────────────────────────
        obs_dict, graph, _, done, info = env.step(actions)
        step += 1

        # ── Report to backend dashboard ───────────────────────────────────
        try:
            requests.post(
                callback_url,
                json={
                    "event": "step",
                    "step": step,
                    "cycle": cycle,
                    "sim_time": info.get("sim_time", 0.0),
                    "fallback_tier": fallback_tier,
                    "latency_ms": latency_ms,
                    "waiting_time": round(info.get("step_mean_waiting_time", 0.0), 2),
                    "throughput": info.get("step_throughput", 0),
                    "vehicles": info.get("step_num_vehicles", 0),
                    "queue_length": round(info.get("step_queue_length", 0.0), 1),
                    "per_node": _per_node_stats(obs_dict, graph, actions),
                    "raw_obs": {
                        nid: obs_t.tolist()[:20]   # first 20 dims — enough for the panel
                        for nid, (obs_t, _) in obs_dict.items()
                    },
                },
                timeout=1.0,
            )
        except Exception:
            pass  # dashboard update is best-effort

    env.close()

    # Notify backend: deployment finished
    try:
        requests.post(callback_url, json={"event": "done"}, timeout=2)
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-file", required=True)
    ap.add_argument("--route-file", required=True)
    ap.add_argument("--inference-url", default="http://localhost:8001")
    ap.add_argument("--callback-url", default="http://localhost:7860/api/deployment/callback")
    ap.add_argument("--display", default=":99")
    ap.add_argument("--max-steps", type=int, default=720)
    ap.add_argument("--begin-time", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(
        net_file=args.net_file,
        route_file=args.route_file,
        inference_url=args.inference_url,
        callback_url=args.callback_url,
        display=args.display,
        max_steps=args.max_steps,
        begin_time=args.begin_time,
        seed=args.seed,
    )
