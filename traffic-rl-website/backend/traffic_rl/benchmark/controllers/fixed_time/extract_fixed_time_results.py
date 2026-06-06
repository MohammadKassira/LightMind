from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEGACY_REPO_ROOT = Path("/home/mohammad-kassira/traffic_rl")


def _discover_repo_root() -> Path:
    module_path = Path(__file__).resolve()
    for candidate in [module_path.parent, *module_path.parents]:
        if (candidate / "benchmark/scenarios").is_dir() and (candidate / "benchmark/controllers").is_dir():
            return candidate
    return module_path.parents[3]


REPO_ROOT = _discover_repo_root()
DEFAULT_BANK_MANIFEST = REPO_ROOT / "benchmark/scenarios/frozen_reportable_scenario_bank_manifest.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "benchmark/runs/fixed_time"
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmark/results/fixed_time"
DEFAULT_LOG_DIR = REPO_ROOT / "benchmark/logs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        return REPO_ROOT / p
    if p == LEGACY_REPO_ROOT or str(p).startswith(f"{LEGACY_REPO_ROOT}/"):
        return REPO_ROOT / p.relative_to(LEGACY_REPO_ROOT)
    return p


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        if isinstance(v, str) and not v.strip():
            return None
        return float(v)
    except Exception:
        return None


def _as_int(v: Any) -> int | None:
    fv = _as_float(v)
    if fv is None:
        return None
    try:
        return int(fv)
    except Exception:
        return None


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


@dataclass
class ParseResult:
    success: bool
    error: str | None


def _parse_tripinfo(path: Path) -> tuple[ParseResult, int | None, float | None]:
    completed = 0
    waiting_sum = 0.0
    waiting_count = 0
    try:
        for _event, elem in ET.iterparse(path, events=("end",)):
            if _strip_ns(elem.tag) != "tripinfo":
                elem.clear()
                continue
            completed += 1
            wt = _as_float(elem.attrib.get("waitingTime"))
            if wt is not None:
                waiting_sum += wt
                waiting_count += 1
            elem.clear()
    except Exception as exc:
        return ParseResult(False, str(exc)), None, None
    mean_wait = (waiting_sum / waiting_count) if waiting_count else None
    return ParseResult(True, None), completed, mean_wait


def _parse_summary(path: Path) -> tuple[ParseResult, dict[str, int | float | None]]:
    last: dict[str, str] | None = None
    try:
        for _event, elem in ET.iterparse(path, events=("end",)):
            if _strip_ns(elem.tag) == "step":
                last = dict(elem.attrib)
            elem.clear()
    except Exception as exc:
        return ParseResult(False, str(exc)), {}
    if last is None:
        return ParseResult(True, None), {}
    out: dict[str, int | float | None] = {
        "final_loaded": _as_int(last.get("loaded")),
        "final_inserted": _as_int(last.get("inserted")),
        "final_running": _as_int(last.get("running")),
        "final_waiting": _as_int(last.get("waiting")),
        "final_ended": _as_int(last.get("ended")),
        "final_arrived": _as_int(last.get("arrived")),
        "final_teleports": _as_int(last.get("teleports")),
        "final_time": _as_float(last.get("time")),
    }
    return ParseResult(True, None), out


def _parse_queue(path: Path) -> tuple[ParseResult, float | None]:
    timestep_count = 0
    total_queue_sum_m = 0.0
    try:
        for _event, elem in ET.iterparse(path, events=("end",)):
            tag = _strip_ns(elem.tag)
            if tag == "lane" or tag == "lanes":
                # Keep lane attributes available for the parent <data> event.
                continue
            if tag == "data":
                timestep_count += 1
                for child in elem:
                    if _strip_ns(child.tag) != "lanes":
                        continue
                    for lane in child:
                        if _strip_ns(lane.tag) != "lane":
                            continue
                        q = _as_float(lane.attrib.get("queueing_length"))
                        if q is not None:
                            total_queue_sum_m += q
                elem.clear()
                continue
            elem.clear()
    except Exception as exc:
        return ParseResult(False, str(exc)), None
    mean_total_queue_m = (total_queue_sum_m / timestep_count) if timestep_count else None
    return ParseResult(True, None), mean_total_queue_m


def _parse_tls_for_phase_changes(
    path: Path,
    simulation_minutes: float | None,
) -> tuple[ParseResult, float | None, str, dict[str, int]]:
    prev_by_tls: dict[str, tuple[str | None, str | None, str | None]] = {}
    unique_tls: set[str] = set()
    tls_state_events = 0
    phase_change_events = 0
    timestep_events = 0
    link_events = 0
    try:
        for _event, elem in ET.iterparse(path, events=("end",)):
            tag = _strip_ns(elem.tag)
            if tag == "tlsState":
                tls_state_events += 1
                tls_id = elem.attrib.get("id")
                if tls_id:
                    unique_tls.add(tls_id)
                    state_tuple = (
                        elem.attrib.get("programID"),
                        elem.attrib.get("phase"),
                        elem.attrib.get("state"),
                    )
                    prev = prev_by_tls.get(tls_id)
                    if prev is not None and prev != state_tuple:
                        phase_change_events += 1
                    prev_by_tls[tls_id] = state_tuple
            elif tag == "timestep":
                timestep_events += 1
            elif tag == "link":
                link_events += 1
            elem.clear()
    except Exception as exc:
        return ParseResult(False, str(exc)), None, "parse_error", {}

    detail = {
        "tls_state_events": tls_state_events,
        "tls_unique_count": len(unique_tls),
        "phase_change_events": phase_change_events,
        "timestep_events": timestep_events,
        "link_events": link_events,
    }

    if tls_state_events == 0:
        if timestep_events > 0:
            return ParseResult(True, None), None, "unavailable_link_output_no_tls_states", detail
        return ParseResult(True, None), None, "unavailable_no_tls_state_events", detail

    if not unique_tls:
        return ParseResult(True, None), None, "unavailable_missing_tls_ids", detail
    if simulation_minutes is None or simulation_minutes <= 0:
        return ParseResult(True, None), None, "unavailable_invalid_simulation_duration", detail

    rate = phase_change_events / (len(unique_tls) * simulation_minutes)
    return ParseResult(True, None), rate, "ok", detail


def _warning_stats_from_logs(run_dir: Path) -> dict[str, int]:
    warning_lines = 0
    teleport_warning_lines = 0
    quitting_on_error_lines = 0
    error_lines = 0
    for name in ("sumo_stderr.txt", "sumo.error.log", "run.log"):
        path = run_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("Warning:"):
                    warning_lines += 1
                if "Teleporting vehicle" in line:
                    teleport_warning_lines += 1
                if "Quitting (on error)." in line:
                    quitting_on_error_lines += 1
                if line.startswith("Error:"):
                    error_lines += 1
    return {
        "warning_line_count": warning_lines,
        "teleport_warning_count": teleport_warning_lines,
        "quitting_on_error_lines": quitting_on_error_lines,
        "error_line_count": error_lines,
    }


def _metric_stats(values: list[float | None]) -> dict[str, float | None]:
    xs = [x for x in values if x is not None and not math.isnan(x)]
    if not xs:
        return {"mean": None, "std": None, "min": None, "max": None}
    if len(xs) == 1:
        return {"mean": xs[0], "std": 0.0, "min": xs[0], "max": xs[0]}
    return {
        "mean": statistics.mean(xs),
        "std": statistics.stdev(xs),
        "min": min(xs),
        "max": max(xs),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_extraction(
    *,
    bank_manifest_path: Path,
    run_root: Path,
    results_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
    bank = _load_json(bank_manifest_path)
    entries = bank.get("entries", [])
    expected_scenarios = [e["scenario_id"] for e in entries]
    expected_networks = sorted({e["network"] for e in entries})
    expected_levels = sorted({e["level"] for e in entries})

    per_run_rows: list[dict[str, Any]] = []
    parse_failures: dict[str, int] = {
        "run_manifest_parse_failures": 0,
        "tripinfo_parse_failures": 0,
        "summary_parse_failures": 0,
        "queue_parse_failures": 0,
        "tls_parse_failures": 0,
    }
    metric_failures: dict[str, int] = {
        "phase_change_rate_unavailable_rows": 0,
    }
    runs_missing = 0

    for scenario_id in expected_scenarios:
        run_id = f"fixed_time__{scenario_id}"
        run_dir = run_root / run_id
        run_manifest_path = run_dir / "run_manifest.json"
        if not run_dir.exists() or not run_manifest_path.exists():
            runs_missing += 1
            continue

        try:
            manifest = _load_json(run_manifest_path)
        except Exception:
            parse_failures["run_manifest_parse_failures"] += 1
            continue

        network = manifest.get("network")
        level = manifest.get("level")
        seed = manifest.get("seed")
        controller = manifest.get("controller", "fixed_time")
        sumo_return_code = manifest.get("sumo_return_code")
        run_status = manifest.get("run_status")

        tripinfo_path = run_dir / "tripinfo.xml"
        summary_path = run_dir / "summary.xml"
        queue_path = run_dir / "queue.xml"
        output_files = manifest.get("output_files", {}) or {}
        tls_output_mode = manifest.get("tls_output_mode") or output_files.get("tls_output_mode")
        tls_output_file_raw = manifest.get("tls_output_file") or output_files.get("tls_output_file")
        if tls_output_file_raw:
            tls_output_file = _resolve_path(str(tls_output_file_raw))
            if not tls_output_file.exists():
                tls_output_file = run_dir / Path(str(tls_output_file_raw)).name
        else:
            if network == "MoST":
                tls_output_mode = tls_output_mode or "switch_states_only"
                tls_output_file = run_dir / "tls_switch_states.xml"
            else:
                tls_output_mode = tls_output_mode or "per_step_tls_states"
                tls_output_file = run_dir / "tls_state.xml"

        tripinfo_result, throughput_completed_trips, mean_waiting = _parse_tripinfo(tripinfo_path)
        if not tripinfo_result.success:
            parse_failures["tripinfo_parse_failures"] += 1

        summary_result, summary_stats = _parse_summary(summary_path)
        if not summary_result.success:
            parse_failures["summary_parse_failures"] += 1

        queue_result, mean_queue = _parse_queue(queue_path)
        if not queue_result.success:
            parse_failures["queue_parse_failures"] += 1

        sim_begin = _as_float(manifest.get("simulation_begin_s"))
        sim_end = _as_float(manifest.get("simulation_end_s"))
        sim_minutes = None
        if sim_begin is not None and sim_end is not None and sim_end > sim_begin:
            sim_minutes = (sim_end - sim_begin) / 60.0
        tls_result, phase_rate, phase_status, tls_detail = _parse_tls_for_phase_changes(
            tls_output_file, sim_minutes
        )
        if not tls_result.success:
            parse_failures["tls_parse_failures"] += 1
        if phase_rate is None:
            metric_failures["phase_change_rate_unavailable_rows"] += 1

        final_running = _as_int(summary_stats.get("final_running"))
        final_waiting = _as_int(summary_stats.get("final_waiting"))
        unfinished_trips_estimated = (
            final_running + final_waiting
            if final_running is not None and final_waiting is not None
            else None
        )
        completion_ratio = None
        if throughput_completed_trips is not None and unfinished_trips_estimated is not None:
            denom = throughput_completed_trips + unfinished_trips_estimated
            if denom > 0:
                completion_ratio = throughput_completed_trips / denom

        warning_stats = _warning_stats_from_logs(run_dir)
        fatal_error_present = (
            warning_stats["quitting_on_error_lines"] > 0 or warning_stats["error_line_count"] > 0
        )

        run_valid = (
            sumo_return_code == 0
            and run_status == "smoke_run_passed"
            and tripinfo_result.success
            and summary_result.success
            and queue_result.success
            and tls_result.success
        )

        row = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "network": network,
            "level": level,
            "seed": seed,
            "controller": controller,
            "run_output_path": str(run_dir),
            "sumo_return_code": sumo_return_code,
            "run_status": run_status,
            "tripinfo_parse_success": tripinfo_result.success,
            "summary_parse_success": summary_result.success,
            "queue_parse_success": queue_result.success,
            "tls_parse_success": tls_result.success,
            "tls_output_mode": tls_output_mode,
            "tls_output_file": str(tls_output_file),
            "fatal_error_present": fatal_error_present,
            "warning_line_count": warning_stats["warning_line_count"],
            "teleport_warning_count": warning_stats["teleport_warning_count"],
            "unfinished_trips_estimated": unfinished_trips_estimated,
            "completion_ratio": completion_ratio,
            "mean_waiting_time_completed_s": mean_waiting,
            "throughput_completed_trips": throughput_completed_trips,
            "mean_total_queue_length_m": mean_queue,
            "phase_change_rate_per_tls_per_min": phase_rate,
            "phase_change_rate_status": phase_status,
            "tls_state_event_count": tls_detail.get("tls_state_events"),
            "tls_unique_count": tls_detail.get("tls_unique_count"),
            "tls_phase_change_event_count": tls_detail.get("phase_change_events"),
            "tls_timestep_event_count": tls_detail.get("timestep_events"),
            "tls_link_event_count": tls_detail.get("link_events"),
            "final_loaded": summary_stats.get("final_loaded"),
            "final_inserted": summary_stats.get("final_inserted"),
            "final_running": final_running,
            "final_waiting": final_waiting,
            "final_ended": summary_stats.get("final_ended"),
            "final_arrived": summary_stats.get("final_arrived"),
            "final_teleports": summary_stats.get("final_teleports"),
            "is_valid_run": run_valid,
        }
        per_run_rows.append(row)

    per_run_rows.sort(key=lambda r: (str(r["network"]), str(r["level"]), int(r["seed"])))

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_run_rows:
        grouped.setdefault((str(row["network"]), str(row["level"])), []).append(row)

    for network in expected_networks:
        for level in expected_levels:
            rows = grouped.get((network, level), [])
            seeds_present = sorted({int(r["seed"]) for r in rows})
            valid_rows = [r for r in rows if r["is_valid_run"]]

            wait_stats = _metric_stats([r["mean_waiting_time_completed_s"] for r in valid_rows])
            throughput_stats = _metric_stats([_as_float(r["throughput_completed_trips"]) for r in valid_rows])
            queue_stats = _metric_stats([r["mean_total_queue_length_m"] for r in valid_rows])
            phase_stats = _metric_stats([r["phase_change_rate_per_tls_per_min"] for r in valid_rows])

            summary_rows.append(
                {
                    "network": network,
                    "level": level,
                    "seeds_expected": 3,
                    "seeds_present_count": len(seeds_present),
                    "seeds_present": ",".join(str(s) for s in seeds_present),
                    "all_3_seeds_present": len(seeds_present) == 3,
                    "valid_runs": len(valid_rows),
                    "invalid_runs": max(3 - len(valid_rows), 0),
                    "mean_waiting_time_completed_s_mean": wait_stats["mean"],
                    "mean_waiting_time_completed_s_std": wait_stats["std"],
                    "mean_waiting_time_completed_s_min": wait_stats["min"],
                    "mean_waiting_time_completed_s_max": wait_stats["max"],
                    "throughput_completed_trips_mean": throughput_stats["mean"],
                    "throughput_completed_trips_std": throughput_stats["std"],
                    "throughput_completed_trips_min": throughput_stats["min"],
                    "throughput_completed_trips_max": throughput_stats["max"],
                    "mean_total_queue_length_m_mean": queue_stats["mean"],
                    "mean_total_queue_length_m_std": queue_stats["std"],
                    "mean_total_queue_length_m_min": queue_stats["min"],
                    "mean_total_queue_length_m_max": queue_stats["max"],
                    "phase_change_rate_per_tls_per_min_mean": phase_stats["mean"],
                    "phase_change_rate_per_tls_per_min_std": phase_stats["std"],
                    "phase_change_rate_per_tls_per_min_min": phase_stats["min"],
                    "phase_change_rate_per_tls_per_min_max": phase_stats["max"],
                    "phase_change_rate_available_runs": sum(
                        1 for r in valid_rows if r["phase_change_rate_per_tls_per_min"] is not None
                    ),
                }
            )

    summary_rows.sort(key=lambda r: (str(r["network"]), str(r["level"])))

    per_run_csv = results_dir / "fixed_time_per_run.csv"
    per_run_json = results_dir / "fixed_time_per_run.json"
    summary_csv = results_dir / "fixed_time_summary_by_network_level.csv"
    summary_json = results_dir / "fixed_time_summary_by_network_level.json"

    per_run_fieldnames = list(per_run_rows[0].keys()) if per_run_rows else []
    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    _write_csv(per_run_csv, per_run_rows, per_run_fieldnames)
    _json_dump(per_run_json, per_run_rows)
    _write_csv(summary_csv, summary_rows, summary_fieldnames)
    _json_dump(summary_json, summary_rows)

    discovered_runs = len(per_run_rows)
    valid_rows_count = sum(1 for r in per_run_rows if r["is_valid_run"])

    networks_present = sorted({str(r["network"]) for r in per_run_rows})
    levels_present = sorted({str(r["level"]) for r in per_run_rows})
    invalid_status_rows = [
        r
        for r in per_run_rows
        if r["sumo_return_code"] != 0 or r["run_status"] != "smoke_run_passed"
    ]

    group_seed_coverage_ok = True
    for row in summary_rows:
        if not row["all_3_seeds_present"]:
            group_seed_coverage_ok = False
            break

    validation_checks = {
        "cli_runs_successfully": True,
        "discovered_completed_runs_eq_90": discovered_runs == 90 and runs_missing == 0,
        "per_run_rows_eq_90": len(per_run_rows) == 90,
        "summary_rows_eq_30": len(summary_rows) == 30,
        "all_10_networks_represented": len(networks_present) == 10,
        "all_3_levels_represented": sorted(levels_present) == ["high", "low", "medium"],
        "all_3_seeds_per_network_level": group_seed_coverage_ok,
        "no_debug_only_network_present": sorted(networks_present) == sorted(expected_networks),
        "run_manifest_parse_all_90": parse_failures["run_manifest_parse_failures"] == 0,
        "tripinfo_parse_all_90": parse_failures["tripinfo_parse_failures"] == 0,
        "summary_parse_all_90": parse_failures["summary_parse_failures"] == 0,
        "queue_parse_all_90": parse_failures["queue_parse_failures"] == 0,
        "tls_parse_all_90": parse_failures["tls_parse_failures"] == 0,
        "all_sumo_return_code_zero": len([r for r in per_run_rows if r["sumo_return_code"] != 0]) == 0,
        "all_run_status_passed": len([r for r in per_run_rows if r["run_status"] != "smoke_run_passed"]) == 0,
        "kpi_fields_populated_or_flagged": all(
            r["mean_waiting_time_completed_s"] is not None
            and r["throughput_completed_trips"] is not None
            and r["mean_total_queue_length_m"] is not None
            and (
                r["phase_change_rate_per_tls_per_min"] is not None
                or str(r["phase_change_rate_status"]).startswith("unavailable_")
            )
            for r in per_run_rows
        ),
    }
    overall_pass = all(validation_checks.values())

    extraction_summary = {
        "created_at_utc": _now_iso(),
        "source_run_root": str(run_root),
        "expected_runs": 90,
        "runs_discovered": discovered_runs,
        "runs_missing": runs_missing,
        "runs_extracted": len(per_run_rows),
        "valid_rows": valid_rows_count,
        "invalid_rows": len(per_run_rows) - valid_rows_count,
        "parse_failures": parse_failures,
        "metric_extraction_failures": metric_failures,
        "invalid_status_rows": len(invalid_status_rows),
        "most_tls_switch_state_handling": {
            "rows_using_switch_states_only": sum(
                1 for r in per_run_rows if r.get("tls_output_mode") == "switch_states_only"
            ),
            "rows_using_per_step_tls_states": sum(
                1 for r in per_run_rows if r.get("tls_output_mode") == "per_step_tls_states"
            ),
        },
        "output_tables": {
            "per_run_csv": str(per_run_csv),
            "per_run_json": str(per_run_json),
            "per_run_row_count": len(per_run_rows),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "summary_row_count": len(summary_rows),
        },
        "validation_checks": validation_checks,
        "overall_result": "PASS" if overall_pass else "FAIL",
    }

    summary_json_path = log_dir / "fixed_time_result_extraction_summary.json"
    summary_txt_path = log_dir / "fixed_time_result_extraction_summary.txt"
    validation_txt_path = log_dir / "fixed_time_result_extraction_validation.txt"

    _json_dump(summary_json_path, extraction_summary)

    summary_lines = [
        "Fixed-Time Result Extraction Summary",
        "",
        f"Created (UTC): {extraction_summary['created_at_utc']}",
        f"Source run root: {run_root}",
        f"Expected runs: {extraction_summary['expected_runs']}",
        f"Discovered runs: {extraction_summary['runs_discovered']}",
        f"Extracted rows: {extraction_summary['runs_extracted']}",
        f"Valid rows: {extraction_summary['valid_rows']}",
        f"Invalid rows: {extraction_summary['invalid_rows']}",
        f"Run missing count: {extraction_summary['runs_missing']}",
        "",
        "Parse failures:",
        f"- run_manifest: {parse_failures['run_manifest_parse_failures']}",
        f"- tripinfo: {parse_failures['tripinfo_parse_failures']}",
        f"- summary: {parse_failures['summary_parse_failures']}",
        f"- queue: {parse_failures['queue_parse_failures']}",
        f"- tls: {parse_failures['tls_parse_failures']}",
        "",
        "MoST TLS handling:",
        f"- switch_states_only rows: {extraction_summary['most_tls_switch_state_handling']['rows_using_switch_states_only']}",
        f"- per_step_tls_states rows: {extraction_summary['most_tls_switch_state_handling']['rows_using_per_step_tls_states']}",
        "",
        "Output tables:",
        f"- per-run CSV rows: {extraction_summary['output_tables']['per_run_row_count']}",
        f"- summary CSV rows: {extraction_summary['output_tables']['summary_row_count']}",
        "",
        f"Overall result: {extraction_summary['overall_result']}",
    ]
    summary_txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    validation_lines = [
        "Fixed-Time Result Extraction Validation",
        "",
        "Checks:",
        f"1. extractor CLI imports/runs successfully: {'PASS' if validation_checks['cli_runs_successfully'] else 'FAIL'}",
        f"2. exactly 90 completed Fixed-Time runs discovered: {'PASS' if validation_checks['discovered_completed_runs_eq_90'] else 'FAIL'}",
        f"3. exactly 90 per-run result rows written: {'PASS' if validation_checks['per_run_rows_eq_90'] else 'FAIL'}",
        f"4. exactly 30 network-level summary rows written: {'PASS' if validation_checks['summary_rows_eq_30'] else 'FAIL'}",
        f"5. all 10 reportable networks represented: {'PASS' if validation_checks['all_10_networks_represented'] else 'FAIL'}",
        f"6. all 3 levels represented for every network: {'PASS' if validation_checks['all_3_levels_represented'] else 'FAIL'}",
        f"7. all 3 seeds represented for every network-level group: {'PASS' if validation_checks['all_3_seeds_per_network_level'] else 'FAIL'}",
        f"8. no debug-only network appears: {'PASS' if validation_checks['no_debug_only_network_present'] else 'FAIL'}",
        f"9. all 90 run manifests parse: {'PASS' if validation_checks['run_manifest_parse_all_90'] else 'FAIL'}",
        f"10. all 90 tripinfo files parse: {'PASS' if validation_checks['tripinfo_parse_all_90'] else 'FAIL'}",
        f"11. all 90 summary files parse: {'PASS' if validation_checks['summary_parse_all_90'] else 'FAIL'}",
        f"12. all 90 queue files parse: {'PASS' if validation_checks['queue_parse_all_90'] else 'FAIL'}",
        f"13. TLS parsing succeeds for normal tls_state and MoST tls_switch_states: {'PASS' if validation_checks['tls_parse_all_90'] else 'FAIL'}",
        f"14. no run with sumo_return_code != 0: {'PASS' if validation_checks['all_sumo_return_code_zero'] else 'FAIL'}",
        f"15. no run with failed run status: {'PASS' if validation_checks['all_run_status_passed'] else 'FAIL'}",
        f"16. KPI fields populated or clearly flagged unavailable: {'PASS' if validation_checks['kpi_fields_populated_or_flagged'] else 'FAIL'}",
        "",
        f"17. overall result = {'PASS' if overall_pass else 'FAIL'}",
        "",
        f"Per-run rows: {len(per_run_rows)}",
        f"Summary rows: {len(summary_rows)}",
    ]
    validation_txt_path.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")

    extraction_summary["summary_paths"] = {
        "summary_json": str(summary_json_path),
        "summary_txt": str(summary_txt_path),
        "validation_txt": str(validation_txt_path),
    }
    return extraction_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and aggregate fixed-time benchmark results from completed run outputs."
    )
    parser.add_argument("--bank-manifest", type=Path, default=DEFAULT_BANK_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    summary = run_extraction(
        bank_manifest_path=args.bank_manifest,
        run_root=args.run_root,
        results_dir=args.results_dir,
        log_dir=args.log_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
