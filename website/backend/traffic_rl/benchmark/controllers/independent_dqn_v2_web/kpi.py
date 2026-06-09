from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import json_dump, now_iso


def extract_kpis_from_evaluation_outputs(
    *,
    evaluation_payload: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    payload = {
        "created_at_utc": now_iso(),
        "kpi_definitions": [
            "mean_waiting_time_completed_s",
            "throughput_completed_trips",
            "mean_total_queue_length_m",
            "phase_change_rate_per_tls_per_min",
        ],
        "per_run_csv": evaluation_payload.get("per_run_csv"),
        "per_run_json": evaluation_payload.get("per_run_json"),
        "summary_csv": evaluation_payload.get("summary_csv"),
        "summary_json": evaluation_payload.get("summary_json"),
        "checks": evaluation_payload.get("checks", {}),
    }
    json_dump(output_path, payload)
    return payload
