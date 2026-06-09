from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import ScenarioSpec
from .utils import json_dump, now_iso


def create_generated_scenario_manifest(
    *,
    output_path: Path,
    job_id: str,
    network_name: str,
    net_file: Path,
    train_scenario: ScenarioSpec,
    eval_scenarios: list[ScenarioSpec],
    level_period_s: dict[str, float],
) -> dict[str, Any]:
    payload = {
        "created_at_utc": now_iso(),
        "job_id": job_id,
        "network": network_name,
        "network_file": str(net_file),
        "manifest_type": "uploaded_map_generated_scenarios",
        "training_scenario": train_scenario.to_json(),
        "evaluation_scenarios": [s.to_json() for s in eval_scenarios],
        "evaluation_count": len(eval_scenarios),
        "demand_level_period_s": {k: float(v) for k, v in level_period_s.items()},
    }
    json_dump(output_path, payload)
    return payload
