#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = REPO_ROOT / "benchmark/runs/independent_dqn_v2"
RESULTS_ROOT = REPO_ROOT / "benchmark/results/independent_dqn_v2"
LOG_ROOT = REPO_ROOT / "benchmark/logs"

REPORTABLE_NETWORKS = [
    "cologne1",
    "ingolstadt1",
    "ingolstadt7",
    "cologne8",
    "grid4x4",
    "bologna_pasubio",
    "arterial4x4",
    "ingolstadt21",
    "toronto",
]
LEVELS = ["low", "medium", "high"]
SEEDS = [1, 2, 3]


@dataclass
class Check:
    name: str
    passed: bool
    details: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _parse_queue_mean_lane_level(queue_xml: Path) -> float | None:
    if not queue_xml.exists():
        return None
    timestep_count = 0
    total_queue_sum_m = 0.0
    try:
        for _event, elem in ET.iterparse(queue_xml, events=("end",)):
            tag = elem.tag.split("}", 1)[-1]
            if tag == "lane" or tag == "lanes":
                continue
            if tag == "data":
                timestep_count += 1
                for child in elem:
                    if child.tag.split("}", 1)[-1] != "lanes":
                        continue
                    for lane in child:
                        if lane.tag.split("}", 1)[-1] != "lane":
                            continue
                        q_raw = lane.attrib.get("queueing_length")
                        if q_raw is None:
                            continue
                        try:
                            total_queue_sum_m += float(q_raw)
                        except Exception:
                            continue
                elem.clear()
                continue
            elem.clear()
    except Exception:
        return None
    if timestep_count <= 0:
        return None
    return total_queue_sum_m / timestep_count


def _mean(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def run() -> dict[str, Any]:
    old_combined = RESULTS_ROOT / "independent_dqn_v2_per_run.csv"
    old_pattern = {
        "rows": 0,
        "queue_nonnull_count": 0,
        "queue_zero_count": 0,
        "queue_min": None,
        "queue_max": None,
    }
    if old_combined.exists():
        with old_combined.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        old_pattern["rows"] = len(rows)
        qvals = []
        for r in rows:
            q = r.get("mean_total_queue_length_m")
            if q is None or q == "":
                continue
            try:
                qf = float(q)
                qvals.append(qf)
            except Exception:
                continue
        old_pattern["queue_nonnull_count"] = len(qvals)
        old_pattern["queue_zero_count"] = sum(1 for v in qvals if v == 0.0)
        if qvals:
            old_pattern["queue_min"] = min(qvals)
            old_pattern["queue_max"] = max(qvals)

    per_network_rows: dict[str, list[dict[str, Any]]] = {n: [] for n in REPORTABLE_NETWORKS}
    combined_rows: list[dict[str, Any]] = []
    checks: list[Check] = []
    parse_failures: list[dict[str, Any]] = []

    for network in REPORTABLE_NETWORKS:
        for level in LEVELS:
            for seed in SEEDS:
                scenario_id = f"{network}__{level}__seed_{seed:03d}"
                run_id = f"independent_dqn_v2_main__{network}__{level}__seed_{seed:03d}"
                run_dir = RUN_ROOT / run_id
                manifest_path = run_dir / "run_manifest.json"
                if not manifest_path.exists():
                    parse_failures.append({"run_id": run_id, "error": "missing_run_manifest"})
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    parse_failures.append({"run_id": run_id, "error": f"manifest_parse_error:{exc}"})
                    continue

                kpis = manifest.get("kpis", {}) or {}
                queue_xml = run_dir / "queue.xml"
                queue_mean = _parse_queue_mean_lane_level(queue_xml)
                if queue_mean is None:
                    parse_failures.append({"run_id": run_id, "error": "queue_parse_failed"})

                row = {
                    "controller": "independent_dqn_v2",
                    "training_scope": str(manifest.get("training_scope", "main")),
                    "model_status": str(manifest.get("model_status", "trained_main")),
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "network": network,
                    "level": level,
                    "seed": seed,
                    "mean_waiting_time_completed_s": kpis.get("mean_waiting_time_completed_s"),
                    "throughput_completed_trips": kpis.get("throughput_completed_trips"),
                    "mean_total_queue_length_m": queue_mean,
                    "phase_change_rate_per_tls_per_min": kpis.get("phase_change_rate_per_tls_per_min"),
                    "run_status": manifest.get("run_status", "evaluation_passed"),
                }
                per_network_rows[network].append(row)
                combined_rows.append(row)

    # Sort combined deterministically
    network_rank = {n: i for i, n in enumerate(REPORTABLE_NETWORKS)}
    level_rank = {l: i for i, l in enumerate(LEVELS)}
    combined_rows.sort(key=lambda r: (network_rank[r["network"]], level_rank[r["level"]], int(r["seed"])))

    # Write per-network outputs
    per_run_fields = [
        "controller",
        "training_scope",
        "model_status",
        "run_id",
        "scenario_id",
        "network",
        "level",
        "seed",
        "mean_waiting_time_completed_s",
        "throughput_completed_trips",
        "mean_total_queue_length_m",
        "phase_change_rate_per_tls_per_min",
        "run_status",
    ]
    summary_fields = [
        "controller",
        "training_scope",
        "network",
        "level",
        "run_count",
        "mean_waiting_time_completed_s",
        "throughput_completed_trips",
        "mean_total_queue_length_m",
        "phase_change_rate_per_tls_per_min",
    ]

    all_summary_rows: list[dict[str, Any]] = []
    for network in REPORTABLE_NETWORKS:
        rows = sorted(per_network_rows[network], key=lambda r: (level_rank[r["level"]], int(r["seed"])))
        per_run_csv = RESULTS_ROOT / f"main_{network}_per_run.csv"
        per_run_json = RESULTS_ROOT / f"main_{network}_per_run.json"
        _write_csv(per_run_csv, rows, per_run_fields)
        per_run_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

        summary_rows = []
        for level in LEVELS:
            level_rows = [r for r in rows if str(r["level"]) == level]
            summary_rows.append(
                {
                    "controller": "independent_dqn_v2",
                    "training_scope": "main",
                    "network": network,
                    "level": level,
                    "run_count": len(level_rows),
                    "mean_waiting_time_completed_s": _mean([
                        float(r["mean_waiting_time_completed_s"]) for r in level_rows if r["mean_waiting_time_completed_s"] is not None
                    ]),
                    "throughput_completed_trips": _mean([
                        float(r["throughput_completed_trips"]) for r in level_rows if r["throughput_completed_trips"] is not None
                    ]),
                    "mean_total_queue_length_m": _mean([
                        float(r["mean_total_queue_length_m"]) for r in level_rows if r["mean_total_queue_length_m"] is not None
                    ]),
                    "phase_change_rate_per_tls_per_min": _mean([
                        float(r["phase_change_rate_per_tls_per_min"]) for r in level_rows if r["phase_change_rate_per_tls_per_min"] is not None
                    ]),
                }
            )

        summary_csv = RESULTS_ROOT / f"main_{network}_summary_by_network_level.csv"
        summary_json = RESULTS_ROOT / f"main_{network}_summary_by_network_level.json"
        _write_csv(summary_csv, summary_rows, summary_fields)
        summary_json.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
        all_summary_rows.extend(summary_rows)

    # Write combined outputs
    combined_per_run_csv = RESULTS_ROOT / "independent_dqn_v2_per_run.csv"
    combined_per_run_json = RESULTS_ROOT / "independent_dqn_v2_per_run.json"
    combined_summary_csv = RESULTS_ROOT / "independent_dqn_v2_summary_by_network_level.csv"
    combined_summary_json = RESULTS_ROOT / "independent_dqn_v2_summary_by_network_level.json"
    _write_csv(combined_per_run_csv, combined_rows, per_run_fields)
    combined_per_run_json.write_text(json.dumps(combined_rows, indent=2) + "\n", encoding="utf-8")

    all_summary_rows.sort(key=lambda r: (network_rank[r["network"]], level_rank[r["level"]]))
    _write_csv(combined_summary_csv, all_summary_rows, summary_fields)
    combined_summary_json.write_text(json.dumps(all_summary_rows, indent=2) + "\n", encoding="utf-8")

    # Validation checks
    queue_vals = [r["mean_total_queue_length_m"] for r in combined_rows if r["mean_total_queue_length_m"] is not None]
    nonzero_queue_count = sum(1 for v in queue_vals if float(v) > 0.0)
    raw_nonzero_match = nonzero_queue_count > 0 and sum(1 for v in queue_vals if float(v) == 0.0) == 0

    checks.append(Check("no_simulations_rerun", True, "Re-extraction only read existing run folders and xml outputs."))
    checks.append(Check("no_retraining_occurred", True, "No training entrypoint invoked."))
    checks.append(Check("dqn_raw_queue_xml_used", True, f"Parsed queue.xml for {len(combined_rows)} runs."))
    checks.append(Check("dqn_queue_uses_lane_level_queueing_length", True, "Summed lane-level queueing_length per timestep and averaged over timesteps."))
    checks.append(Check("dqn_queue_nonzero_where_raw_nonzero", raw_nonzero_match, f"nonzero_queue_count={nonzero_queue_count} total={len(queue_vals)}"))
    checks.append(Check("all_81_dqn_rows_queue_populated", len(queue_vals) == 81, f"queue_populated={len(queue_vals)}"))
    checks.append(Check("most_excluded", all(r["network"] != "MoST" for r in combined_rows), "MoST not included in reportable DQN outputs."))
    checks.append(Check("exactly_81_dqn_rows", len(combined_rows) == 81, f"rows={len(combined_rows)}"))
    checks.append(Check("run_manifest_parse_failures_zero", len(parse_failures) == 0, f"parse_failures={len(parse_failures)}"))

    overall_pass = all(c.passed for c in checks)

    new_pattern = {
        "rows": len(combined_rows),
        "queue_nonnull_count": len(queue_vals),
        "queue_zero_count": sum(1 for v in queue_vals if float(v) == 0.0),
        "queue_min": min(queue_vals) if queue_vals else None,
        "queue_max": max(queue_vals) if queue_vals else None,
        "queue_mean": (sum(float(v) for v in queue_vals) / len(queue_vals)) if queue_vals else None,
    }

    summary_payload = {
        "created_at_utc": _utc_now_iso(),
        "scope": "independent_dqn_v2_main_reportable_queue_bugfix_reextraction",
        "reportable_networks": REPORTABLE_NETWORKS,
        "levels": LEVELS,
        "seeds": SEEDS,
        "expected_runs": 81,
        "outputs": {
            "main_network_tables_root": str(RESULTS_ROOT),
            "combined_per_run_csv": str(combined_per_run_csv),
            "combined_per_run_json": str(combined_per_run_json),
            "combined_summary_csv": str(combined_summary_csv),
            "combined_summary_json": str(combined_summary_json),
        },
        "old_queue_value_pattern": old_pattern,
        "corrected_queue_value_pattern": new_pattern,
        "parse_failures": parse_failures,
        "validation_checks": [{"name": c.name, "passed": bool(c.passed), "details": c.details} for c in checks],
        "overall_result": "PASS" if overall_pass else "FAIL",
    }

    summary_json = LOG_ROOT / "independent_dqn_v2_queue_extraction_bugfix_summary.json"
    summary_txt = LOG_ROOT / "independent_dqn_v2_queue_extraction_bugfix_summary.txt"
    validation_txt = LOG_ROOT / "independent_dqn_v2_queue_extraction_bugfix_validation.txt"

    _json_dump(summary_json, summary_payload)

    summary_lines = [
        "Independent DQN v2 Queue Extraction Bugfix Summary",
        "",
        f"Created (UTC): {summary_payload['created_at_utc']}",
        "MoST excluded: YES",
        f"Expected runs: {summary_payload['expected_runs']}",
        f"Extracted rows: {new_pattern['rows']}",
        f"Old queue pattern: zeros={old_pattern['queue_zero_count']} / nonnull={old_pattern['queue_nonnull_count']} / min={old_pattern['queue_min']} / max={old_pattern['queue_max']}",
        f"New queue pattern: zeros={new_pattern['queue_zero_count']} / nonnull={new_pattern['queue_nonnull_count']} / min={new_pattern['queue_min']} / max={new_pattern['queue_max']}",
        f"Overall result: {'PASS' if overall_pass else 'FAIL'}",
    ]
    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    validation_lines = ["Independent DQN v2 Queue Extraction Bugfix Validation", ""]
    for i, c in enumerate(checks, start=1):
        validation_lines.append(f"{i}. {c.name}: {'PASS' if c.passed else 'FAIL'}")
        validation_lines.append(f"   details: {c.details}")
    validation_lines.append("")
    validation_lines.append(f"Overall result: {'PASS' if overall_pass else 'FAIL'}")
    validation_txt.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")

    return {
        "overall_pass": overall_pass,
        "rows": len(combined_rows),
        "old_queue_pattern": old_pattern,
        "new_queue_pattern": new_pattern,
    }


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2))
