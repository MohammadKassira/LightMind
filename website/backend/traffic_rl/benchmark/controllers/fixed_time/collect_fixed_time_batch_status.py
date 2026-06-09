from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SHARD_RESULTS_DIR = REPO_ROOT / "benchmark/logs/fixed_time_shard_results"
DEFAULT_EXEC_PLAN = REPO_ROOT / "benchmark/logs/fixed_time_full_matrix_execution_plan.json"
DEFAULT_OUT_JSON = REPO_ROOT / "benchmark/logs/fixed_time_full_matrix_batch_status.json"
DEFAULT_OUT_TXT = REPO_ROOT / "benchmark/logs/fixed_time_full_matrix_batch_status.txt"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def collect_status(
    *,
    shard_results_dir: Path,
    execution_plan_path: Path,
    out_json: Path,
    out_txt: Path,
) -> dict[str, Any]:
    execution_plan = _load_json(execution_plan_path)
    expected_runs = execution_plan.get("runs", [])
    expected_run_ids = {r["run_id"] for r in expected_runs}
    run_to_network = {r["run_id"]: r["network"] for r in expected_runs}
    run_to_validation_class = {
        r["run_id"]: (
            "staged_probe_valid_full_context_deferred"
            if r["expected_execution_mode"] == "full_most_expensive"
            else "full_candidate_validation_passed"
        )
        for r in expected_runs
    }

    shard_summaries = []
    observed_runs: dict[str, dict[str, Any]] = {}
    for p in sorted(shard_results_dir.glob("*_summary.json")):
        data = _load_json(p)
        shard_summaries.append({"path": str(p), "summary": data})
        for row in data.get("per_run", []):
            run_id = row.get("run_id")
            if run_id:
                observed_runs[run_id] = row

    completed_passed = [rid for rid, row in observed_runs.items() if row.get("status") == "passed"]
    failed_runs = [
        rid
        for rid, row in observed_runs.items()
        if row.get("status") in {"failed", "failed_exception"}
    ]
    skipped = [
        rid
        for rid, row in observed_runs.items()
        if str(row.get("status", "")).startswith("skipped_")
    ]
    missing_runs = sorted(expected_run_ids - set(observed_runs.keys()))

    by_shard = {}
    for s in shard_summaries:
        summary = s["summary"]
        by_shard[summary["shard_id"]] = {
            "runs_planned": summary.get("runs_planned", 0),
            "runs_attempted": summary.get("runs_attempted", 0),
            "runs_passed": summary.get("runs_passed", 0),
            "runs_failed": summary.get("runs_failed", 0),
            "runs_skipped_already_completed": summary.get(
                "runs_skipped_already_completed", 0
            ),
            "summary_json_path": s["path"],
        }

    by_network = defaultdict(lambda: {"passed": 0, "failed": 0, "missing": 0, "skipped": 0})
    for rid in expected_run_ids:
        net = run_to_network[rid]
        row = observed_runs.get(rid)
        if row is None:
            by_network[net]["missing"] += 1
        else:
            st = row.get("status")
            if st == "passed":
                by_network[net]["passed"] += 1
            elif st in {"failed", "failed_exception"}:
                by_network[net]["failed"] += 1
            elif str(st).startswith("skipped_"):
                by_network[net]["skipped"] += 1

    by_validation_class = defaultdict(lambda: {"passed": 0, "failed": 0, "missing": 0, "skipped": 0})
    for rid in expected_run_ids:
        cls = run_to_validation_class[rid]
        row = observed_runs.get(rid)
        if row is None:
            by_validation_class[cls]["missing"] += 1
        else:
            st = row.get("status")
            if st == "passed":
                by_validation_class[cls]["passed"] += 1
            elif st in {"failed", "failed_exception"}:
                by_validation_class[cls]["failed"] += 1
            elif str(st).startswith("skipped_"):
                by_validation_class[cls]["skipped"] += 1

    payload = {
        "created_at_utc": _now_iso(),
        "total_expected_runs": len(expected_runs),
        "completed_passed_runs": len(completed_passed),
        "failed_runs": len(failed_runs),
        "missing_runs": len(missing_runs),
        "skipped_runs": len(skipped),
        "completion_by_shard": by_shard,
        "completion_by_network": dict(by_network),
        "completion_by_validation_class": dict(by_validation_class),
        "failed_run_ids": sorted(failed_runs),
        "missing_run_ids": missing_runs,
        "skipped_run_ids": sorted(skipped),
    }
    _json_dump(out_json, payload)

    lines = [
        "Fixed-Time Full Matrix Batch Status",
        "",
        f"Created (UTC): {payload['created_at_utc']}",
        f"Total expected runs: {payload['total_expected_runs']}",
        f"Completed passed runs: {payload['completed_passed_runs']}",
        f"Failed runs: {payload['failed_runs']}",
        f"Missing runs: {payload['missing_runs']}",
        f"Skipped runs: {payload['skipped_runs']}",
        "",
        "Completion by shard:",
    ]
    for shard_id in sorted(payload["completion_by_shard"]):
        lines.append(f"- {shard_id}: {payload['completion_by_shard'][shard_id]}")
    lines.extend(["", "Completion by network:"])
    for network in sorted(payload["completion_by_network"]):
        lines.append(f"- {network}: {payload['completion_by_network'][network]}")
    lines.extend(["", "Completion by validation class:"])
    for cls in sorted(payload["completion_by_validation_class"]):
        lines.append(f"- {cls}: {payload['completion_by_validation_class'][cls]}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect status across fixed-time shard summaries."
    )
    parser.add_argument(
        "--shard-results-dir",
        type=Path,
        default=DEFAULT_SHARD_RESULTS_DIR,
        help="Directory containing <shard_id>_summary.json files.",
    )
    parser.add_argument(
        "--execution-plan",
        type=Path,
        default=DEFAULT_EXEC_PLAN,
        help="Path to fixed-time full matrix execution plan.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-txt",
        type=Path,
        default=DEFAULT_OUT_TXT,
        help="Output TXT path.",
    )
    args = parser.parse_args()

    payload = collect_status(
        shard_results_dir=args.shard_results_dir,
        execution_plan_path=args.execution_plan,
        out_json=args.out_json,
        out_txt=args.out_txt,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

