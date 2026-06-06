from __future__ import annotations

import argparse
import json
import re
import subprocess
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
    # Fallback to historical layout expectation.
    return module_path.parents[3]


REPO_ROOT = _discover_repo_root()
DEFAULT_BANK_MANIFEST = (
    REPO_ROOT / "benchmark/scenarios/frozen_reportable_scenario_bank_manifest.json"
)
DEFAULT_RUN_ROOT = REPO_ROOT / "benchmark/runs/fixed_time"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        return REPO_ROOT / p
    if p == LEGACY_REPO_ROOT or str(p).startswith(f"{LEGACY_REPO_ROOT}/"):
        return REPO_ROOT / p.relative_to(LEGACY_REPO_ROOT)
    return p


def _parse_xml_ok(path: Path) -> tuple[bool, str | None]:
    try:
        # Stream-parse to validate well-formedness without constructing full trees for
        # large outputs (for example, multi-GB tls_state.xml in MoST full runs).
        for _event, elem in ET.iterparse(path, events=("end",)):
            elem.clear()
        return True, None
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def _tripinfo_metrics(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tripinfo_parse_success": False,
        "tripinfo_parse_error": None,
        "completed_trip_count": 0,
        "mean_waiting_time_s": None,
    }
    try:
        completed_trip_count = 0
        waiting_sum_s = 0.0
        waiting_count = 0
        for _event, trip in ET.iterparse(path, events=("end",)):
            if trip.tag != "tripinfo":
                trip.clear()
                continue
            completed_trip_count += 1
            waiting_attr = trip.attrib.get("waitingTime")
            if waiting_attr is not None:
                try:
                    waiting_sum_s += float(waiting_attr)
                    waiting_count += 1
                except ValueError:
                    pass
            trip.clear()
        result["tripinfo_parse_success"] = True
        result["completed_trip_count"] = completed_trip_count
        if waiting_count:
            result["mean_waiting_time_s"] = waiting_sum_s / waiting_count
    except Exception as exc:  # pragma: no cover
        result["tripinfo_parse_error"] = str(exc)
    return result


def _run_sumo_with_streamed_logs(cmd: list[str], stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess:
    # Explicitly stream SUMO output to disk. Do not use capture_output=True for
    # long-running/full simulations because warning-heavy workloads can be large.
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        return subprocess.run(cmd, stdout=out, stderr=err, check=False)


def _iter_log_lines(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line


def _parse_terminal_stats_from_run_log(run_log_path: Path) -> dict[str, int | None]:
    patterns = {
        "inserted": re.compile(r"Inserted:\s*(\d+)"),
        "running": re.compile(r"Running:\s*(\d+)"),
        "waiting": re.compile(r"Waiting:\s*(\d+)"),
    }
    values: dict[str, int | None] = {k: None for k in patterns}
    for line in _iter_log_lines(run_log_path):
        for key, pattern in patterns.items():
            if values[key] is not None:
                continue
            match = pattern.search(line)
            if match:
                values[key] = int(match.group(1))
    completed = None
    inserted = values["inserted"]
    running = values["running"]
    waiting = values["waiting"]
    if inserted is not None and running is not None and waiting is not None:
        completed = max(inserted - running - waiting, 0)
    values["completed_estimated"] = completed
    return values


def _summarize_warning_logs(stderr_path: Path, sumo_error_log: Path) -> dict[str, int]:
    summary = {
        "warning_lines": 0,
        "error_lines": 0,
        "quitting_on_error_lines": 0,
        "no_connection_stop_lines": 0,
        "person_disconnect_lines": 0,
    }
    person_disconnect_pattern = re.compile(r"No connection found between edge .* for person")
    for path in (stderr_path, sumo_error_log):
        for line in _iter_log_lines(path):
            if line.startswith("Warning:"):
                summary["warning_lines"] += 1
            if line.startswith("Error:"):
                summary["error_lines"] += 1
            if "Quitting (on error)." in line:
                summary["quitting_on_error_lines"] += 1
            if "No connection between stop" in line:
                summary["no_connection_stop_lines"] += 1
            if person_disconnect_pattern.search(line):
                summary["person_disconnect_lines"] += 1
    return summary


@dataclass
class ScenarioContext:
    scenario_id: str
    bank_entry: dict[str, Any]
    frozen_dir: Path
    frozen_manifest_path: Path
    frozen_manifest: dict[str, Any]
    source_manifest_path: Path
    source_manifest: dict[str, Any]
    source_validation_path: Path
    source_validation: dict[str, Any]


def _resolve_scenario_context(bank_manifest: dict[str, Any], scenario_id: str) -> ScenarioContext:
    entries = bank_manifest.get("entries", [])
    entry = next((x for x in entries if x.get("scenario_id") == scenario_id), None)
    if entry is None:
        raise ValueError(f"Scenario ID not found in bank manifest: {scenario_id}")
    frozen_dir = _resolve_path(entry["frozen_target_path"])
    frozen_manifest_path = _resolve_path(entry["scenario_manifest_path"])
    frozen_manifest = _load_json(frozen_manifest_path)
    source_manifest_path = frozen_dir / "source_candidate_manifest.json"
    source_validation_path = frozen_dir / "source_candidate_validation_report.json"
    source_manifest = _load_json(source_manifest_path)
    source_validation = _load_json(source_validation_path)
    return ScenarioContext(
        scenario_id=scenario_id,
        bank_entry=entry,
        frozen_dir=frozen_dir,
        frozen_manifest_path=frozen_manifest_path,
        frozen_manifest=frozen_manifest,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        source_validation_path=source_validation_path,
        source_validation=source_validation,
    )


def _resolve_demand_files_from_frozen_manifest(frozen_manifest: dict[str, Any]) -> list[Path]:
    files: list[Path] = []
    for item in frozen_manifest.get("copied_demand_files", []):
        files.append(_resolve_path(item["frozen_target_path"]))
    if not files:
        raise ValueError("Frozen manifest has no copied demand files.")
    return files


def _resolve_validation_end_time(ctx: ScenarioContext) -> int:
    timing = ctx.source_validation.get("validation_timing", {})
    v = timing.get("selected_validation_end_time_s")
    if v is None:
        v = (
            ctx.source_validation.get("clearance_calibration", {}) or {}
        ).get("selected_validation_end_time_s")
    if v is None:
        v = (ctx.source_manifest.get("timing_window_used", {}) or {}).get("generated_end_s")
    if v is None:
        raise ValueError(
            f"Unable to resolve validated end time for scenario {ctx.scenario_id}"
        )
    return int(v)


def _ensure_cfg_elem(parent: ET.Element, tag: str) -> ET.Element:
    elem = parent.find(tag)
    if elem is None:
        elem = ET.SubElement(parent, tag)
    return elem


def _resolve_canonical_additional_files_from_sumocfg(canonical_sumocfg: Path) -> list[Path]:
    root = ET.parse(canonical_sumocfg).getroot()
    add_elem = root.find("./input/additional-files")
    if add_elem is None:
        return []
    raw = add_elem.attrib.get("value", "")
    if not raw:
        return []
    cfg_base = canonical_sumocfg.resolve().parent
    items = [s.strip() for s in raw.split(",") if s.strip()]
    resolved: list[Path] = []
    for item in items:
        p = Path(item)
        resolved.append((cfg_base / p).resolve() if not p.is_absolute() else p)
    return resolved


def _write_tls_switch_additional_file(path: Path, dest: Path) -> None:
    root = ET.Element("additional")
    ET.SubElement(
        root,
        "timedEvent",
        {
            "type": "SaveTLSSwitchStates",
            "dest": str(dest.resolve()),
        },
    )
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def _build_most_parity_sumocfg(
    *,
    canonical_sumocfg: Path,
    generated_sumocfg: Path,
    net_file: Path,
    route_files: list[Path],
    additional_files: list[Path],
    begin_s: int,
    end_s: int,
    outputs: dict[str, Path],
    tls_output_mode: str,
) -> Path:
    cfg_tree = ET.parse(canonical_sumocfg)
    cfg_root = cfg_tree.getroot()

    input_elem = _ensure_cfg_elem(cfg_root, "input")
    time_elem = _ensure_cfg_elem(cfg_root, "time")
    report_elem = _ensure_cfg_elem(cfg_root, "report")
    output_elem = _ensure_cfg_elem(cfg_root, "output")
    processing_elem = _ensure_cfg_elem(cfg_root, "processing")

    _ensure_cfg_elem(input_elem, "net-file").attrib["value"] = str(net_file.resolve())
    _ensure_cfg_elem(input_elem, "route-files").attrib["value"] = ",".join(
        str(p.resolve()) for p in route_files
    )
    resolved_additional_files = [p.resolve() for p in additional_files]
    if tls_output_mode == "switch_states_only":
        switch_additional = outputs["tls_switch_additional_xml"]
        _write_tls_switch_additional_file(
            path=switch_additional,
            dest=outputs["tls_switch_states_xml"],
        )
        resolved_additional_files.append(switch_additional.resolve())
    if resolved_additional_files:
        _ensure_cfg_elem(input_elem, "additional-files").attrib["value"] = ",".join(
            str(p) for p in resolved_additional_files
        )
    _ensure_cfg_elem(time_elem, "begin").attrib["value"] = str(begin_s)
    _ensure_cfg_elem(time_elem, "end").attrib["value"] = str(end_s)

    # Keep output names deterministic and local to this run directory.
    _ensure_cfg_elem(output_elem, "output-prefix").attrib["value"] = ""
    _ensure_cfg_elem(output_elem, "tripinfo-output").attrib["value"] = str(
        outputs["tripinfo_xml"].resolve()
    )
    _ensure_cfg_elem(output_elem, "summary-output").attrib["value"] = str(
        outputs["summary_xml"].resolve()
    )
    _ensure_cfg_elem(output_elem, "queue-output").attrib["value"] = str(
        outputs["queue_xml"].resolve()
    )
    if tls_output_mode == "switch_states_only":
        link_elem = output_elem.find("link-output")
        if link_elem is not None:
            output_elem.remove(link_elem)
    else:
        _ensure_cfg_elem(output_elem, "link-output").attrib["value"] = str(
            outputs["tls_state_xml"].resolve()
        )
    _ensure_cfg_elem(report_elem, "log").attrib["value"] = str(outputs["run_log"].resolve())
    _ensure_cfg_elem(report_elem, "duration-log.statistics").attrib["value"] = "true"

    # Explicit parity guardrail for MoST canonical behavior.
    _ensure_cfg_elem(processing_elem, "ignore-route-errors").attrib["value"] = "true"

    generated_sumocfg.parent.mkdir(parents=True, exist_ok=True)
    cfg_tree.write(generated_sumocfg, encoding="UTF-8", xml_declaration=True)
    return generated_sumocfg


def _build_full_fixed_time_command(
    net_file: Path,
    route_files: list[Path],
    additional_files: list[Path],
    begin_s: int,
    end_s: int,
    run_dir: Path,
) -> tuple[list[str], dict[str, Path]]:
    outputs = {
        "tripinfo_xml": run_dir / "tripinfo.xml",
        "summary_xml": run_dir / "summary.xml",
        "queue_xml": run_dir / "queue.xml",
        "tls_state_xml": run_dir / "tls_state.xml",
        "run_log": run_dir / "run.log",
        "sumo_stdout": run_dir / "sumo_stdout.txt",
        "sumo_stderr": run_dir / "sumo_stderr.txt",
        "sumo_error_log": run_dir / "sumo.error.log",
    }
    cmd = [
        "sumo",
        "--net-file",
        str(net_file),
        "--route-files",
        ",".join(str(p) for p in route_files),
        "--begin",
        str(begin_s),
        "--end",
        str(end_s),
        "--no-step-log",
        "true",
        "--duration-log.statistics",
        "true",
        "--tripinfo-output",
        str(outputs["tripinfo_xml"]),
        "--summary-output",
        str(outputs["summary_xml"]),
        "--queue-output",
        str(outputs["queue_xml"]),
        "--link-output",
        str(outputs["tls_state_xml"]),
        "--log",
        str(outputs["run_log"]),
        "--error-log",
        str(outputs["sumo_error_log"]),
    ]
    if additional_files:
        cmd.extend(["--additional-files", ",".join(str(p) for p in additional_files)])
    return cmd, outputs


def _build_most_probe_command(
    canonical_sumocfg: Path,
    net_file: Path,
    route_files: list[Path],
    additional_files: list[Path],
    begin_s: int,
    end_s: int,
    run_dir: Path,
    tls_output_mode: str,
) -> tuple[list[str], dict[str, Path]]:
    outputs = {
        "tripinfo_xml": run_dir / "tripinfo.xml",
        "summary_xml": run_dir / "summary.xml",
        "queue_xml": run_dir / "queue.xml",
        "tls_state_xml": run_dir / "tls_state.xml",
        "run_log": run_dir / "run.log",
        "sumo_stdout": run_dir / "sumo_stdout.txt",
        "sumo_stderr": run_dir / "sumo_stderr.txt",
        "sumo_error_log": run_dir / "sumo.error.log",
        "tls_switch_states_xml": run_dir / "tls_switch_states.xml",
        "tls_switch_additional_xml": run_dir / "tls_switch_states.additional.xml",
        "generated_sumocfg": run_dir / "most_parity_generated.sumocfg",
    }
    generated_sumocfg = _build_most_parity_sumocfg(
        canonical_sumocfg=canonical_sumocfg,
        generated_sumocfg=outputs["generated_sumocfg"],
        net_file=net_file,
        route_files=route_files,
        additional_files=additional_files,
        begin_s=begin_s,
        end_s=end_s,
        outputs=outputs,
        tls_output_mode=tls_output_mode,
    )
    cmd = [
        "sumo",
        "-c",
        str(generated_sumocfg.resolve()),
        "--no-step-log",
        "true",
        "--error-log",
        str(outputs["sumo_error_log"].resolve()),
    ]
    return cmd, outputs


def run_fixed_time_scenario(
    *,
    bank_manifest: dict[str, Any],
    scenario_id: str,
    run_root: Path,
    most_probe: bool = False,
    most_probe_begin: int = 23950,
    most_probe_end: int = 24100,
    run_id_override: str | None = None,
) -> dict[str, Any]:
    ctx = _resolve_scenario_context(bank_manifest, scenario_id)
    network = ctx.frozen_manifest["network"]
    level = ctx.frozen_manifest["level"]
    seed = int(ctx.frozen_manifest["seed"])
    run_id = run_id_override or (
        f"fixed_time__{scenario_id}__{_now_iso().replace(':', '').replace('-', '')}"
    )
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    route_files = _resolve_demand_files_from_frozen_manifest(ctx.frozen_manifest)
    for rf in route_files:
        rf_resolved = rf.resolve()
        if str(REPO_ROOT / "benchmark/scenarios") not in str(rf_resolved):
            raise ValueError(
                f"Frozen smoke run must use demand under benchmark/scenarios only: {rf}"
            )

    net_file = _resolve_path(ctx.source_manifest["source_network_artifact_path"])
    candidate_validation = ctx.source_manifest.get("candidate_validation", {}) or {}
    canonical_sumocfg: Path | None = None
    if network == "MoST":
        canonical = candidate_validation.get("canonical_source_sumocfg_path") or ctx.source_validation.get(
            "canonical_source_sumocfg_path"
        )
        if not canonical:
            raise ValueError(f"MoST run requires canonical source sumocfg path: {scenario_id}")
        canonical_sumocfg = _resolve_path(canonical)

    additional_files = []
    for p in candidate_validation.get("validation_additional_files", []) or []:
        if p:
            additional_files.append(_resolve_path(p))
    if not additional_files and canonical_sumocfg is not None:
        additional_files = _resolve_canonical_additional_files_from_sumocfg(canonical_sumocfg)

    if most_probe:
        if canonical_sumocfg is None:
            raise ValueError(f"MoST probe requested but canonical source sumocfg path missing: {scenario_id}")
        cmd, outputs = _build_most_probe_command(
            canonical_sumocfg=canonical_sumocfg,
            net_file=net_file,
            route_files=route_files,
            additional_files=additional_files,
            begin_s=most_probe_begin,
            end_s=most_probe_end,
            run_dir=run_dir,
            tls_output_mode="per_step_tls_states",
        )
        begin_s = most_probe_begin
        end_s = most_probe_end
    else:
        begin_s = int(
            (ctx.source_manifest.get("timing_window_used", {}) or {}).get(
                "generated_begin_s", 0
            )
            or 0
        )
        end_s = _resolve_validation_end_time(ctx)
        if network == "MoST":
            if canonical_sumocfg is None:
                raise ValueError(
                    f"MoST full run requested but canonical source sumocfg path missing: {scenario_id}"
                )
            cmd, outputs = _build_most_probe_command(
                canonical_sumocfg=canonical_sumocfg,
                net_file=net_file,
                route_files=route_files,
                additional_files=additional_files,
                begin_s=begin_s,
                end_s=end_s,
                run_dir=run_dir,
                tls_output_mode="switch_states_only",
            )
        else:
            cmd, outputs = _build_full_fixed_time_command(
                net_file=net_file,
                route_files=route_files,
                additional_files=additional_files,
                begin_s=begin_s,
                end_s=end_s,
                run_dir=run_dir,
            )

    proc = _run_sumo_with_streamed_logs(
        cmd=cmd,
        stdout_path=outputs["sumo_stdout"],
        stderr_path=outputs["sumo_stderr"],
    )

    tls_output_mode = (
        "switch_states_only" if (network == "MoST" and not most_probe) else "per_step_tls_states"
    )
    tls_output_file = (
        outputs["tls_switch_states_xml"]
        if tls_output_mode == "switch_states_only"
        else outputs["tls_state_xml"]
    )

    required_files = {
        "tripinfo_xml": outputs["tripinfo_xml"],
        "summary_xml": outputs["summary_xml"],
        "queue_xml": outputs["queue_xml"],
        "tls_output_xml": tls_output_file,
        "run_log": outputs["run_log"],
        "run_manifest": run_dir / "run_manifest.json",
    }
    required_output_exists = {
        k: v.exists() for k, v in required_files.items() if k != "run_manifest"
    }
    output_exists = {
        "tripinfo_xml": outputs["tripinfo_xml"].exists(),
        "summary_xml": outputs["summary_xml"].exists(),
        "queue_xml": outputs["queue_xml"].exists(),
        "tls_state_xml": outputs["tls_state_xml"].exists(),
        "tls_switch_states_xml": (
            outputs["tls_switch_states_xml"].exists()
            if outputs.get("tls_switch_states_xml")
            else False
        ),
        "tls_output_xml": tls_output_file.exists(),
        "run_log": outputs["run_log"].exists(),
        "sumo_stdout": outputs["sumo_stdout"].exists(),
        "sumo_stderr": outputs["sumo_stderr"].exists(),
        "sumo_error_log": outputs["sumo_error_log"].exists(),
        "generated_sumocfg": (
            outputs["generated_sumocfg"].exists()
            if outputs.get("generated_sumocfg")
            else False
        ),
        "tls_switch_additional_xml": (
            outputs["tls_switch_additional_xml"].exists()
            if outputs.get("tls_switch_additional_xml")
            else False
        ),
    }

    tripinfo_parse_ok, tripinfo_parse_err = _parse_xml_ok(outputs["tripinfo_xml"])
    summary_parse_ok, summary_parse_err = _parse_xml_ok(outputs["summary_xml"])
    queue_parse_ok, queue_parse_err = _parse_xml_ok(outputs["queue_xml"])
    tls_parse_ok, tls_parse_err = _parse_xml_ok(tls_output_file)
    tripinfo_metrics = _tripinfo_metrics(outputs["tripinfo_xml"])

    terminal_stats = _parse_terminal_stats_from_run_log(outputs["run_log"])
    warning_summary = _summarize_warning_logs(outputs["sumo_stderr"], outputs["sumo_error_log"])
    fatal_quit_present = warning_summary.get("quitting_on_error_lines", 0) > 0

    if most_probe:
        run_status = (
            "fixed_time_frozen_most_probe_valid"
            if proc.returncode == 0 and not fatal_quit_present
            else "fixed_time_frozen_most_probe_failed"
        )
    else:
        run_status = (
            "smoke_run_passed"
            if (
                proc.returncode == 0
                and all(required_output_exists.values())
                and tripinfo_parse_ok
                and summary_parse_ok
                and queue_parse_ok
                and tls_parse_ok
            )
            else "smoke_run_failed"
        )

    manifest = {
        "created_at_utc": _now_iso(),
        "run_id": run_id,
        "controller": "fixed_time",
        "scenario_id": scenario_id,
        "frozen_scenario_path": str(ctx.frozen_dir),
        "frozen_scenario_manifest_path": str(ctx.frozen_manifest_path),
        "network": network,
        "level": level,
        "seed": seed,
        "demand_file_paths_used": [str(p) for p in route_files],
        "network_file_path_used": str(net_file),
        "additional_file_paths_used": [str(p) for p in additional_files],
        "simulation_begin_s": begin_s,
        "simulation_end_s": end_s,
        "sumo_command": cmd,
        "sumo_return_code": proc.returncode,
        "output_files": {
            "tripinfo_xml": str(outputs["tripinfo_xml"]),
            "summary_xml": str(outputs["summary_xml"]),
            "queue_xml": str(outputs["queue_xml"]),
            "tls_state_xml": str(outputs["tls_state_xml"]),
            "tls_switch_states_xml": (
                str(outputs["tls_switch_states_xml"])
                if outputs.get("tls_switch_states_xml")
                else None
            ),
            "run_log": str(outputs["run_log"]),
            "sumo_stdout": str(outputs["sumo_stdout"]),
            "sumo_stderr": str(outputs["sumo_stderr"]),
            "sumo_error_log": str(outputs["sumo_error_log"]),
            "generated_sumocfg": str(outputs.get("generated_sumocfg")) if outputs.get("generated_sumocfg") else None,
            "tls_output_mode": tls_output_mode,
            "tls_output_file": str(tls_output_file),
            "exists": output_exists,
        },
        "xml_parse_status": {
            "tripinfo_parse_success": tripinfo_parse_ok,
            "tripinfo_parse_error": tripinfo_parse_err,
            "summary_parse_success": summary_parse_ok,
            "summary_parse_error": summary_parse_err,
            "queue_parse_success": queue_parse_ok,
            "queue_parse_error": queue_parse_err,
            "tls_state_parse_success": tls_parse_ok,
            "tls_state_parse_error": tls_parse_err,
            "tls_output_mode": tls_output_mode,
            "tls_output_file": str(tls_output_file),
            "tls_output_parse_success": tls_parse_ok,
            "tls_output_parse_error": tls_parse_err,
        },
        "basic_metrics_qc": {
            "completed_trip_count": tripinfo_metrics["completed_trip_count"],
            "mean_waiting_time_s": tripinfo_metrics["mean_waiting_time_s"],
            "terminal_inserted": terminal_stats.get("inserted"),
            "terminal_running": terminal_stats.get("running"),
            "terminal_waiting": terminal_stats.get("waiting"),
            "terminal_completed_estimated": terminal_stats.get("completed_estimated"),
            "warning_summary": warning_summary,
            "fatal_quitting_on_error_present": fatal_quit_present,
        },
        "source_frozen_scenario_status": ctx.frozen_manifest.get("frozen_bank_status"),
        "source_candidate_validation_status_snapshot": ctx.frozen_manifest.get(
            "source_candidate_validation_status_actual"
        ),
        "source_candidate_validation_mode_snapshot": ctx.frozen_manifest.get(
            "source_candidate_validation_mode"
        ),
        "source_manifest_sha256": ctx.frozen_manifest.get(
            "source_candidate_manifest_sha256"
        ),
        "source_validation_report_sha256": ctx.frozen_manifest.get(
            "source_candidate_validation_report_sha256"
        ),
        "run_status": run_status,
        "mode": "most_probe" if most_probe else "full_smoke",
        "tls_output_mode": tls_output_mode,
        "tls_output_file": str(tls_output_file),
    }
    _json_dump(required_files["run_manifest"], manifest)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "scenario_id": scenario_id,
        "network": network,
        "level": level,
        "seed": seed,
        "sumo_return_code": proc.returncode,
        "run_status": run_status,
        "output_exists": output_exists,
        "xml_parse_status": manifest["xml_parse_status"],
        "basic_metrics_qc": manifest["basic_metrics_qc"],
        "run_manifest_path": str(required_files["run_manifest"]),
        "mode": manifest["mode"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Fixed-Time baseline smoke/probe on frozen scenarios from bank manifest."
    )
    parser.add_argument("--scenario-id", required=True, help="Scenario ID from bank manifest.")
    parser.add_argument(
        "--bank-manifest",
        type=Path,
        default=DEFAULT_BANK_MANIFEST,
        help="Path to frozen bank manifest JSON.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Directory root for Fixed-Time run outputs.",
    )
    parser.add_argument(
        "--most-probe",
        action="store_true",
        help="Run MoST short canonical-style probe mode (no long full-context run).",
    )
    parser.add_argument("--most-probe-begin", type=int, default=23950)
    parser.add_argument("--most-probe-end", type=int, default=24100)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional deterministic run id. Useful for batch orchestration/resume.",
    )
    args = parser.parse_args()

    bank_manifest = _load_json(args.bank_manifest)
    result = run_fixed_time_scenario(
        bank_manifest=bank_manifest,
        scenario_id=args.scenario_id,
        run_root=args.run_root,
        most_probe=args.most_probe,
        most_probe_begin=args.most_probe_begin,
        most_probe_end=args.most_probe_end,
        run_id_override=args.run_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
