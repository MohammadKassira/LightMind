from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEGACY_REPO_ROOTS = (
    Path("/home/mohammad-kassira/traffic_rl"),
    Path("/content/traffic_rl"),
)


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

NON_MOST_NETWORKS = [
    "cologne1",
    "ingolstadt1",
    "cologne8",
    "ingolstadt7",
    "ingolstadt21",
    "grid4x4",
    "bologna_pasubio",
    "arterial4x4",
    "toronto",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        return REPO_ROOT / p
    for legacy in LEGACY_REPO_ROOTS:
        if p == legacy or str(p).startswith(f"{legacy}/"):
            return REPO_ROOT / p.relative_to(legacy)
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


@dataclass
class TLSProgram:
    tls_id: str
    program_id: str
    tl_type: str
    offset_raw: str
    durations: list[float]


@dataclass
class NetworkTLSAudit:
    network: str
    net_file: str
    net_file_exists: bool
    tls_controller_count: int
    tl_logic_count: int
    tl_type_values: list[str]
    has_non_static_logic: bool
    max_programs_per_tls: int
    min_phase_count: int | None
    max_phase_count: int | None
    explicit_positive_phase_durations: bool
    analytically_derivable: bool
    non_derivable_reasons: list[str]
    notes: str


def _parse_tls_programs_from_net(net_file: Path) -> tuple[dict[str, list[TLSProgram]], dict[str, Any]]:
    by_tls: dict[str, list[TLSProgram]] = defaultdict(list)
    tl_type_values: set[str] = set()
    phase_counts: list[int] = []
    explicit_positive_phase_durations = True
    tl_logic_count = 0
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
    except Exception as exc:
        return {}, {
            "error": str(exc),
            "tl_logic_count": 0,
            "tl_type_values": [],
            "phase_counts": [],
            "explicit_positive_phase_durations": False,
        }

    for tl in root.findall("tlLogic"):
        tl_logic_count += 1
        tls_id = tl.attrib.get("id", "")
        program_id = tl.attrib.get("programID", "")
        tl_type = tl.attrib.get("type", "")
        offset_raw = tl.attrib.get("offset", "0")
        tl_type_values.add(tl_type)
        durations: list[float] = []
        for phase in tl.findall("phase"):
            d = _as_float(phase.attrib.get("duration"))
            if d is None or d <= 0:
                explicit_positive_phase_durations = False
                continue
            durations.append(d)
        phase_counts.append(len(durations))
        by_tls[tls_id].append(
            TLSProgram(
                tls_id=tls_id,
                program_id=program_id,
                tl_type=tl_type,
                offset_raw=offset_raw,
                durations=durations,
            )
        )

    return by_tls, {
        "error": None,
        "tl_logic_count": tl_logic_count,
        "tl_type_values": sorted(tl_type_values),
        "phase_counts": phase_counts,
        "explicit_positive_phase_durations": explicit_positive_phase_durations,
    }


def _audit_network_derivability(network: str, net_file: Path) -> tuple[NetworkTLSAudit, dict[str, list[TLSProgram]]]:
    net_exists = net_file.exists()
    if not net_exists:
        audit = NetworkTLSAudit(
            network=network,
            net_file=str(net_file),
            net_file_exists=False,
            tls_controller_count=0,
            tl_logic_count=0,
            tl_type_values=[],
            has_non_static_logic=True,
            max_programs_per_tls=0,
            min_phase_count=None,
            max_phase_count=None,
            explicit_positive_phase_durations=False,
            analytically_derivable=False,
            non_derivable_reasons=["net_file_missing"],
            notes="Network net.xml not found.",
        )
        return audit, {}

    by_tls, meta = _parse_tls_programs_from_net(net_file)
    reasons: list[str] = []
    if meta["error"] is not None:
        reasons.append("net_xml_parse_error")
    tl_type_values: list[str] = list(meta["tl_type_values"])
    has_non_static = any(t != "static" for t in tl_type_values)
    if has_non_static:
        reasons.append("contains_non_static_tlLogic")

    max_programs_per_tls = max((len(v) for v in by_tls.values()), default=0)
    if max_programs_per_tls > 1:
        reasons.append("multiple_programs_per_tls")

    phase_counts = list(meta["phase_counts"])
    if any(c <= 0 for c in phase_counts):
        reasons.append("missing_or_invalid_phase_durations")
    if not meta["explicit_positive_phase_durations"]:
        reasons.append("non_positive_phase_durations")

    derivable = len(reasons) == 0 and len(by_tls) > 0
    note = (
        "All TLS are static with single deterministic program and explicit phase durations."
        if derivable
        else "At least one derivability precondition failed."
    )
    audit = NetworkTLSAudit(
        network=network,
        net_file=str(net_file),
        net_file_exists=True,
        tls_controller_count=len(by_tls),
        tl_logic_count=int(meta["tl_logic_count"]),
        tl_type_values=sorted(tl_type_values),
        has_non_static_logic=has_non_static,
        max_programs_per_tls=max_programs_per_tls,
        min_phase_count=min(phase_counts) if phase_counts else None,
        max_phase_count=max(phase_counts) if phase_counts else None,
        explicit_positive_phase_durations=bool(meta["explicit_positive_phase_durations"]),
        analytically_derivable=derivable,
        non_derivable_reasons=sorted(set(reasons)),
        notes=note,
    )
    return audit, by_tls


def _count_transitions_for_program(
    *,
    begin_s: float,
    end_s: float,
    offset_raw: str,
    durations: list[float],
) -> int:
    if not durations or end_s <= begin_s:
        return 0
    cycle = sum(durations)
    if cycle <= 0:
        return 0
    if offset_raw == "begin":
        offset = begin_s
    else:
        offset = _as_float(offset_raw) or 0.0

    boundaries: list[float] = []
    cumulative = 0.0
    for d in durations:
        cumulative += d
        boundaries.append(cumulative)

    transitions = 0
    for boundary in boundaries:
        # Count integers n with begin < offset + boundary + n*cycle < end
        n_min = math.floor((begin_s - offset - boundary) / cycle) + 1
        n_max = math.ceil((end_s - offset - boundary) / cycle) - 1
        if n_max >= n_min:
            transitions += int(n_max - n_min + 1)
    return transitions


def _derive_phase_change_rate(
    *,
    begin_s: float,
    end_s: float,
    programs_by_tls: dict[str, list[TLSProgram]],
) -> float | None:
    if end_s <= begin_s:
        return None
    if not programs_by_tls:
        return None
    sim_minutes = (end_s - begin_s) / 60.0
    if sim_minutes <= 0:
        return None

    total_transitions = 0
    for tls_id, programs in programs_by_tls.items():
        _ = tls_id
        program = programs[0]
        total_transitions += _count_transitions_for_program(
            begin_s=begin_s,
            end_s=end_s,
            offset_raw=program.offset_raw,
            durations=program.durations,
        )
    tls_count = len(programs_by_tls)
    if tls_count == 0:
        return None
    return total_transitions / (tls_count * sim_minutes)


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


def run_backfill(
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

    per_run_json_path = results_dir / "fixed_time_per_run.json"
    per_run_csv_path = results_dir / "fixed_time_per_run.csv"
    summary_json_path = results_dir / "fixed_time_summary_by_network_level.json"
    summary_csv_path = results_dir / "fixed_time_summary_by_network_level.csv"
    if not per_run_json_path.exists():
        raise FileNotFoundError(f"Missing per-run table: {per_run_json_path}")
    per_run_rows: list[dict[str, Any]] = _load_json(per_run_json_path)
    if len(per_run_rows) != 90:
        raise ValueError(f"Expected 90 per-run rows before backfill, got {len(per_run_rows)}")

    row_by_run_id = {str(r["run_id"]): r for r in per_run_rows}

    # Resolve network -> net file path from canonical run manifests.
    network_net_paths: dict[str, Path] = {}
    for scenario_id in expected_scenarios:
        run_id = f"fixed_time__{scenario_id}"
        run_manifest = run_root / run_id / "run_manifest.json"
        if not run_manifest.exists():
            continue
        manifest = _load_json(run_manifest)
        network = manifest.get("network")
        path = manifest.get("network_file_path_used")
        if not network or not path:
            continue
        if network == "MoST":
            continue
        network_net_paths.setdefault(str(network), _resolve_path(str(path)))

    audit_payload: dict[str, Any] = {
        "created_at_utc": _now_iso(),
        "networks": {},
        "overall_derivable": True,
    }
    programs_by_network: dict[str, dict[str, list[TLSProgram]]] = {}
    non_derivable_networks: list[str] = []
    audit_lines = [
        "Fixed-Time Non-MoST Stability Derivability Audit",
        "",
    ]
    for network in NON_MOST_NETWORKS:
        net_path = network_net_paths.get(network, Path(""))
        audit, programs_by_tls = _audit_network_derivability(network, net_path)
        programs_by_network[network] = programs_by_tls
        if not audit.analytically_derivable:
            non_derivable_networks.append(network)
        audit_payload["networks"][network] = audit.__dict__
        audit_lines.extend(
            [
                f"- {network}",
                f"  net_file: {audit.net_file}",
                f"  tls_controller_count: {audit.tls_controller_count}",
                f"  tlLogic_count: {audit.tl_logic_count}",
                f"  tlLogic_types: {audit.tl_type_values}",
                f"  static_deterministic: {audit.analytically_derivable}",
                f"  explicit_phase_durations: {audit.explicit_positive_phase_durations}",
                f"  max_programs_per_tls: {audit.max_programs_per_tls}",
                f"  notes: {audit.notes}",
                (
                    f"  non_derivable_reasons: {audit.non_derivable_reasons}"
                    if audit.non_derivable_reasons
                    else "  non_derivable_reasons: []"
                ),
            ]
        )

    if non_derivable_networks:
        audit_payload["overall_derivable"] = False
        audit_payload["non_derivable_networks"] = non_derivable_networks
    else:
        audit_payload["overall_derivable"] = True
        audit_payload["non_derivable_networks"] = []

    audit_json_path = log_dir / "fixed_time_non_most_stability_derivability_audit.json"
    audit_txt_path = log_dir / "fixed_time_non_most_stability_derivability_audit.txt"
    _json_dump(audit_json_path, audit_payload)
    audit_txt_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    if non_derivable_networks:
        summary_payload = {
            "created_at_utc": _now_iso(),
            "overall_result": "FAIL",
            "message": "At least one non-MoST network is not analytically derivable. No backfill applied.",
            "non_derivable_networks": non_derivable_networks,
            "audit_json_path": str(audit_json_path),
            "audit_txt_path": str(audit_txt_path),
        }
        _json_dump(log_dir / "fixed_time_stability_backfill_summary.json", summary_payload)
        (log_dir / "fixed_time_stability_backfill_summary.txt").write_text(
            "Fixed-Time Stability Backfill Summary\n\nFAIL: non-derivable networks found.\n"
            + f"Networks: {', '.join(non_derivable_networks)}\n",
            encoding="utf-8",
        )
        (log_dir / "fixed_time_stability_backfill_validation.txt").write_text(
            "Fixed-Time Stability Backfill Validation\n\nOverall result = FAIL\n",
            encoding="utf-8",
        )
        return summary_payload

    most_phase_values_before = {
        str(r["run_id"]): r.get("phase_change_rate_per_tls_per_min")
        for r in per_run_rows
        if r.get("network") == "MoST"
    }

    backfilled_rows = 0
    unchanged_rows = 0
    for scenario_id in expected_scenarios:
        run_id = f"fixed_time__{scenario_id}"
        row = row_by_run_id.get(run_id)
        if row is None:
            continue
        network = str(row.get("network"))
        if network == "MoST":
            unchanged_rows += 1
            continue
        begin = _as_float(row.get("final_loaded"))  # placeholder; overwritten from manifest below
        _ = begin
        manifest = _load_json(run_root / run_id / "run_manifest.json")
        begin_s = _as_float(manifest.get("simulation_begin_s"))
        end_s = _as_float(manifest.get("simulation_end_s"))
        if begin_s is None or end_s is None:
            continue
        rate = _derive_phase_change_rate(
            begin_s=begin_s,
            end_s=end_s,
            programs_by_tls=programs_by_network[network],
        )
        row["phase_change_rate_per_tls_per_min"] = rate
        row["phase_change_rate_status"] = "derived_from_static_tls_program"
        row["tls_state_event_count"] = None
        row["tls_unique_count"] = len(programs_by_network[network])
        row["tls_phase_change_event_count"] = None
        row["tls_timestep_event_count"] = None
        row["tls_link_event_count"] = None
        backfilled_rows += 1

    per_run_rows.sort(key=lambda r: (str(r["network"]), str(r["level"]), int(r["seed"])))

    # Rebuild 30-row summary table.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_run_rows:
        grouped[(str(row["network"]), str(row["level"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for network in expected_networks:
        for level in expected_levels:
            rows = grouped.get((network, level), [])
            seeds_present = sorted({int(r["seed"]) for r in rows})
            valid_rows = [r for r in rows if bool(r.get("is_valid_run"))]
            wait_stats = _metric_stats([_as_float(r.get("mean_waiting_time_completed_s")) for r in valid_rows])
            throughput_stats = _metric_stats([_as_float(r.get("throughput_completed_trips")) for r in valid_rows])
            queue_stats = _metric_stats([_as_float(r.get("mean_total_queue_length_m")) for r in valid_rows])
            phase_stats = _metric_stats([_as_float(r.get("phase_change_rate_per_tls_per_min")) for r in valid_rows])
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
                        1 for r in valid_rows if r.get("phase_change_rate_per_tls_per_min") is not None
                    ),
                }
            )
    summary_rows.sort(key=lambda r: (str(r["network"]), str(r["level"])))

    # Persist updated tables.
    per_run_fields = list(per_run_rows[0].keys())
    summary_fields = list(summary_rows[0].keys())
    _write_csv(per_run_csv_path, per_run_rows, per_run_fields)
    _json_dump(per_run_json_path, per_run_rows)
    _write_csv(summary_csv_path, summary_rows, summary_fields)
    _json_dump(summary_json_path, summary_rows)

    most_phase_unchanged = all(
        row.get("phase_change_rate_per_tls_per_min") == most_phase_values_before.get(str(row.get("run_id")))
        for row in per_run_rows
        if row.get("network") == "MoST"
    )

    total_phase_populated = sum(
        1 for r in per_run_rows if _as_float(r.get("phase_change_rate_per_tls_per_min")) is not None
    )
    non_most_derived_status_count = sum(
        1
        for r in per_run_rows
        if r.get("network") != "MoST"
        and r.get("phase_change_rate_status") == "derived_from_static_tls_program"
    )

    validation_checks = {
        "all_9_non_most_networks_derivable": len(non_derivable_networks) == 0,
        "per_run_rows_still_90": len(per_run_rows) == 90,
        "summary_rows_still_30": len(summary_rows) == 30,
        "phase_change_rate_populated_all_90": total_phase_populated == 90,
        "most_values_unchanged": most_phase_unchanged,
        "non_most_status_is_derived": non_most_derived_status_count == 81,
        "no_simulation_reruns_occurred": True,
    }
    overall_pass = all(validation_checks.values())

    validation_lines = [
        "Fixed-Time Stability Backfill Validation",
        "",
        f"1. all 9 non-MoST networks derivable: {'PASS' if validation_checks['all_9_non_most_networks_derivable'] else 'FAIL'}",
        f"2. per-run result table still has 90 rows: {'PASS' if validation_checks['per_run_rows_still_90'] else 'FAIL'}",
        f"3. summary table still has 30 rows: {'PASS' if validation_checks['summary_rows_still_30'] else 'FAIL'}",
        f"4. phase_change_rate populated for all 90 runs: {'PASS' if validation_checks['phase_change_rate_populated_all_90'] else 'FAIL'}",
        f"5. MoST values remain unchanged: {'PASS' if validation_checks['most_values_unchanged'] else 'FAIL'}",
        f"6. non-MoST status indicates analytical derivation: {'PASS' if validation_checks['non_most_status_is_derived'] else 'FAIL'}",
        f"7. no simulation reruns occurred: {'PASS' if validation_checks['no_simulation_reruns_occurred'] else 'FAIL'}",
        "",
        f"overall result = {'PASS' if overall_pass else 'FAIL'}",
    ]
    validation_path = log_dir / "fixed_time_stability_backfill_validation.txt"
    validation_path.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")

    summary_payload = {
        "created_at_utc": _now_iso(),
        "overall_result": "PASS" if overall_pass else "FAIL",
        "all_four_headline_kpis_available_for_all_90_runs": total_phase_populated == 90,
        "non_most_analytical_derivation_applied_rows": backfilled_rows,
        "most_rows_unchanged": unchanged_rows,
        "phase_change_rate_populated_rows": total_phase_populated,
        "phase_change_rate_status_counts": {
            "derived_from_static_tls_program": non_most_derived_status_count,
            "ok": sum(1 for r in per_run_rows if r.get("phase_change_rate_status") == "ok"),
        },
        "future_output_policy_recommendation": (
            "For MaxPressure/RL, emit SaveTLSSwitchStates for all networks to avoid relying on analytical backfill."
        ),
        "files_updated": {
            "per_run_csv": str(per_run_csv_path),
            "per_run_json": str(per_run_json_path),
            "summary_csv": str(summary_csv_path),
            "summary_json": str(summary_json_path),
            "audit_json": str(audit_json_path),
            "audit_txt": str(audit_txt_path),
            "validation_txt": str(validation_path),
        },
    }
    summary_json_out = log_dir / "fixed_time_stability_backfill_summary.json"
    summary_txt_out = log_dir / "fixed_time_stability_backfill_summary.txt"
    _json_dump(summary_json_out, summary_payload)
    summary_txt_out.write_text(
        "\n".join(
            [
                "Fixed-Time Stability Backfill Summary",
                "",
                f"Created (UTC): {summary_payload['created_at_utc']}",
                f"Overall result: {summary_payload['overall_result']}",
                f"All 4 headline KPIs available for all 90 runs: {summary_payload['all_four_headline_kpis_available_for_all_90_runs']}",
                f"Non-MoST analytical backfill rows: {summary_payload['non_most_analytical_derivation_applied_rows']}",
                f"MoST rows unchanged: {summary_payload['most_rows_unchanged']}",
                "Recommendation:",
                summary_payload["future_output_policy_recommendation"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "audit": audit_payload,
        "validation_checks": validation_checks,
        "summary": summary_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill non-MoST Fixed-Time stability KPI analytically from static TLS programs."
    )
    parser.add_argument("--bank-manifest", type=Path, default=DEFAULT_BANK_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    result = run_backfill(
        bank_manifest_path=args.bank_manifest,
        run_root=args.run_root,
        results_dir=args.results_dir,
        log_dir=args.log_dir,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
