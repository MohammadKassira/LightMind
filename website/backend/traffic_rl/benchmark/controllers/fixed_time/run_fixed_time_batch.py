from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_fixed_time import (
    DEFAULT_BANK_MANIFEST,
    DEFAULT_RUN_ROOT,
    REPO_ROOT,
    run_fixed_time_scenario,
)


DEFAULT_EXEC_PLAN = REPO_ROOT / "benchmark/logs/fixed_time_full_matrix_execution_plan.json"
DEFAULT_SHARD_PLAN = REPO_ROOT / "benchmark/logs/fixed_time_full_matrix_shard_plan.json"
DEFAULT_SHARD_RESULTS_DIR = REPO_ROOT / "benchmark/logs/fixed_time_shard_results"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _required_outputs_present(run_manifest: dict[str, Any], execution_mode: str) -> bool:
    output_files = run_manifest.get("output_files", {})
    exists_map = output_files.get("exists", {})
    core_required = ("tripinfo_xml", "summary_xml", "queue_xml", "run_log")
    if not all(bool(exists_map.get(k)) for k in core_required):
        return False

    # MoST full runs may intentionally use switch-state-only TLS output.
    if bool(exists_map.get("tls_output_xml")):
        return True
    if bool(exists_map.get("tls_state_xml")):
        return True
    return bool(exists_map.get("tls_switch_states_xml"))


def _is_successful_completed_run(run_manifest_path: Path, execution_mode: str) -> bool:
    if not run_manifest_path.exists():
        return False
    try:
        data = _load_json(run_manifest_path)
    except Exception:
        return False
    status = data.get("run_status")
    if execution_mode == "full_most_expensive":
        ok_status = status == "smoke_run_passed"
    else:
        ok_status = status == "smoke_run_passed"
    if not ok_status:
        return False
    return _required_outputs_present(data, execution_mode)


def _fatal_error_lines_from_run_manifest(run_manifest_path: Path) -> list[str]:
    if not run_manifest_path.exists():
        return []
    try:
        data = _load_json(run_manifest_path)
    except Exception:
        return []
    warnings = (
        data.get("basic_metrics_qc", {})
        .get("warning_summary", {})
    )
    quit_count = warnings.get("quitting_on_error_lines", 0)
    error_count = warnings.get("error_lines", 0)
    lines: list[str] = []
    if quit_count:
        lines.append(f"quitting_on_error_lines={quit_count}")
    if error_count:
        lines.append(f"error_lines={error_count}")
    return lines


def _resolve_runs_for_shard(
    *,
    shard_id: str | None,
    scenario_list: Path | None,
    execution_plan: dict[str, Any],
    shard_plan: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    entries = execution_plan.get("runs", [])
    by_run_id = {e["run_id"]: e for e in entries}
    by_scenario_id = {e["scenario_id"]: e for e in entries}

    if scenario_list is not None:
        data = _load_json(scenario_list)
        sid = data.get("shard_id") or scenario_list.stem
        runs: list[dict[str, Any]] = []
        for scenario_id in data.get("scenario_ids", []):
            if scenario_id not in by_scenario_id:
                raise ValueError(f"Scenario ID {scenario_id} from {scenario_list} not found in execution plan.")
            runs.append(by_scenario_id[scenario_id])
        return sid, runs

    if shard_id is None:
        raise ValueError("Provide either --shard-id or --scenario-list.")

    shard = next((s for s in shard_plan.get("shards", []) if s.get("shard_id") == shard_id), None)
    if shard is None:
        raise ValueError(f"Shard id not found: {shard_id}")
    runs: list[dict[str, Any]] = []
    for run_id in shard.get("run_ids", []):
        if run_id not in by_run_id:
            raise ValueError(f"Run id {run_id} listed in shard {shard_id} not found in execution plan.")
        runs.append(by_run_id[run_id])
    return shard_id, runs


def run_batch(
    *,
    bank_manifest_path: Path,
    execution_plan_path: Path,
    shard_plan_path: Path,
    shard_id: str | None,
    scenario_list: Path | None,
    run_root: Path,
    rerun_failed: bool,
    shard_results_dir: Path,
) -> dict[str, Any]:
    bank_manifest = _load_json(bank_manifest_path)
    execution_plan = _load_json(execution_plan_path)
    shard_plan = _load_json(shard_plan_path)

    resolved_shard_id, runs = _resolve_runs_for_shard(
        shard_id=shard_id,
        scenario_list=scenario_list,
        execution_plan=execution_plan,
        shard_plan=shard_plan,
    )

    per_run: list[dict[str, Any]] = []
    passed = failed = skipped = attempted = 0

    for run in runs:
        run_id = run["run_id"]
        scenario_id = run["scenario_id"]
        execution_mode = run["expected_execution_mode"]
        run_dir = run_root / run_id
        run_manifest_path = run_dir / "run_manifest.json"

        if _is_successful_completed_run(run_manifest_path, execution_mode):
            skipped += 1
            per_run.append(
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "status": "skipped_already_completed",
                    "run_output_path": str(run_dir),
                    "sumo_return_code": _load_json(run_manifest_path).get("sumo_return_code"),
                    "fatal_error_lines": _fatal_error_lines_from_run_manifest(run_manifest_path),
                }
            )
            continue

        if run_dir.exists() and rerun_failed:
            # Preserve successful runs; for incomplete/failed runs we reset directory and rerun deterministically.
            import shutil

            shutil.rmtree(run_dir)

        if run_dir.exists() and not rerun_failed:
            skipped += 1
            per_run.append(
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "status": "skipped_incomplete_or_failed_no_rerun_flag",
                    "run_output_path": str(run_dir),
                    "sumo_return_code": (
                        _load_json(run_manifest_path).get("sumo_return_code")
                        if run_manifest_path.exists()
                        else None
                    ),
                    "fatal_error_lines": _fatal_error_lines_from_run_manifest(run_manifest_path),
                }
            )
            continue

        attempted += 1
        most_probe = False
        # Full matrix path intentionally keeps MoST as expensive full run mode.
        if execution_mode == "full_most_expensive":
            most_probe = False
        try:
            result = run_fixed_time_scenario(
                bank_manifest=bank_manifest,
                scenario_id=scenario_id,
                run_root=run_root,
                most_probe=most_probe,
                run_id_override=run_id,
            )
            run_status = result.get("run_status")
            run_ok = run_status == "smoke_run_passed"
            if run_ok:
                passed += 1
                status = "passed"
            else:
                failed += 1
                status = "failed"
            per_run.append(
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "status": status,
                    "run_output_path": result.get("run_dir"),
                    "sumo_return_code": result.get("sumo_return_code"),
                    "run_status": run_status,
                    "fatal_error_lines": _fatal_error_lines_from_run_manifest(
                        Path(result["run_manifest_path"])
                    ),
                }
            )
        except Exception as exc:  # pragma: no cover
            failed += 1
            per_run.append(
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "status": "failed_exception",
                    "run_output_path": str(run_dir),
                    "sumo_return_code": None,
                    "run_status": None,
                    "exception": str(exc),
                    "fatal_error_lines": _fatal_error_lines_from_run_manifest(run_manifest_path),
                }
            )

    summary = {
        "created_at_utc": _now_iso(),
        "shard_id": resolved_shard_id,
        "runs_planned": len(runs),
        "runs_attempted": attempted,
        "runs_passed": passed,
        "runs_failed": failed,
        "runs_skipped_already_completed": skipped,
        "per_run": per_run,
    }
    shard_results_dir.mkdir(parents=True, exist_ok=True)
    json_path = shard_results_dir / f"{resolved_shard_id}_summary.json"
    txt_path = shard_results_dir / f"{resolved_shard_id}_summary.txt"
    _json_dump(json_path, summary)

    lines = [
        f"Fixed-Time Shard Summary: {resolved_shard_id}",
        "",
        f"Created (UTC): {summary['created_at_utc']}",
        f"Planned: {summary['runs_planned']}",
        f"Attempted: {summary['runs_attempted']}",
        f"Passed: {summary['runs_passed']}",
        f"Failed: {summary['runs_failed']}",
        f"Skipped already complete: {summary['runs_skipped_already_completed']}",
        "",
        "Per-run:",
    ]
    for row in per_run:
        lines.append(
            f"- {row['run_id']} | {row['scenario_id']} | status={row['status']} | rc={row.get('sumo_return_code')} | out={row['run_output_path']} | fatal={row.get('fatal_error_lines')}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["summary_json_path"] = str(json_path)
    summary["summary_txt_path"] = str(txt_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-safe parallel batch runner for full frozen Fixed-Time matrix."
    )
    parser.add_argument("--shard-id", default=None, help="Shard id from shard plan.")
    parser.add_argument(
        "--scenario-list",
        type=Path,
        default=None,
        help="Optional shard scenario list JSON path (alternative to --shard-id).",
    )
    parser.add_argument(
        "--bank-manifest",
        type=Path,
        default=DEFAULT_BANK_MANIFEST,
        help="Path to frozen bank manifest JSON.",
    )
    parser.add_argument(
        "--execution-plan",
        type=Path,
        default=DEFAULT_EXEC_PLAN,
        help="Path to full matrix execution plan JSON.",
    )
    parser.add_argument(
        "--shard-plan",
        type=Path,
        default=DEFAULT_SHARD_PLAN,
        help="Path to shard plan JSON.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--shard-results-dir",
        type=Path,
        default=DEFAULT_SHARD_RESULTS_DIR,
        help="Directory for per-shard summary outputs.",
    )
    parser.add_argument(
        "--no-rerun-failed",
        action="store_true",
        help="Do not rerun incomplete/failed runs; just report/skip.",
    )
    args = parser.parse_args()

    summary = run_batch(
        bank_manifest_path=args.bank_manifest,
        execution_plan_path=args.execution_plan,
        shard_plan_path=args.shard_plan,
        shard_id=args.shard_id,
        scenario_list=args.scenario_list,
        run_root=args.run_root,
        rerun_failed=not args.no_rerun_failed,
        shard_results_dir=args.shard_results_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
