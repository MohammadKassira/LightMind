from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.controllers.fixed_time.run_fixed_time import _parse_xml_ok, _tripinfo_metrics, _write_tls_switch_additional_file
from benchmark.controllers.independent_dqn_v2.env_adapter import (
    DEFAULT_CONTRACT_PATH,
    ForbiddenCallTracker,
    IndependentDQNV2EnvAdapter,
    ScenarioInputs,
    _allocate_traci_port,
    _ensure_traci_import,
    parse_queue_mean_from_xml,
)
from models.independent_dqn_v2 import AgentSpec, DQNAgentConfig, IndependentDQNController

from .exceptions import WebIntegrationError
from .schemas import ScenarioSpec
from .utils import csv_dump, json_dump, now_iso


def _as_adapter_scenario(spec: ScenarioSpec) -> ScenarioInputs:
    return ScenarioInputs(
        scenario_id=spec.scenario_id,
        network="uploaded_map",
        level=spec.level,
        seed=int(spec.seed),
        begin_s=int(spec.begin_s),
        end_s=int(spec.end_s),
        net_file=spec.net_file,
        route_files=[spec.route_file],
        additional_files=[Path(p) for p in spec.additional_files],
        canonical_sumocfg_file=spec.sumocfg_file,
    )


def _build_eval_command(scenario: ScenarioInputs, run_dir: Path) -> tuple[list[str], dict[str, Path]]:
    outputs = {
        "tripinfo_xml": run_dir / "tripinfo.xml",
        "summary_xml": run_dir / "summary.xml",
        "queue_xml": run_dir / "queue.xml",
        "tls_switch_states_xml": run_dir / "tls_switch_states.xml",
        "tls_switch_additional_xml": run_dir / "tls_switch_states.additional.xml",
        "run_log": run_dir / "run.log",
        "sumo_stdout": run_dir / "sumo_stdout.txt",
        "sumo_error_log": run_dir / "sumo.error.log",
        "controller_diagnostics_json": run_dir / "controller_diagnostics.json",
        "run_manifest_json": run_dir / "run_manifest.json",
    }
    _write_tls_switch_additional_file(outputs["tls_switch_additional_xml"], outputs["tls_switch_states_xml"])

    cmd = [
        "sumo",
        "--net-file",
        str(scenario.net_file.resolve()),
        "--route-files",
        ",".join(str(p.resolve()) for p in scenario.route_files),
        "--begin",
        str(int(scenario.begin_s)),
        "--end",
        str(int(scenario.end_s)),
        "--no-step-log",
        "true",
        "--duration-log.statistics",
        "true",
        "--no-warnings",
        "true",
        "--tripinfo-output",
        str(outputs["tripinfo_xml"].resolve()),
        "--summary-output",
        str(outputs["summary_xml"].resolve()),
        "--queue-output",
        str(outputs["queue_xml"].resolve()),
        "--log",
        str(outputs["run_log"].resolve()),
        "--error-log",
        str(outputs["sumo_error_log"].resolve()),
        "--additional-files",
        ",".join([str(p.resolve()) for p in scenario.additional_files + [outputs["tls_switch_additional_xml"]]]),
    ]
    return cmd, outputs


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def evaluate_on_heldout_scenarios(
    *,
    checkpoint_path: Path,
    scenarios: list[ScenarioSpec],
    output_root: Path,
    run_root: Path,
    seed: int,
) -> dict[str, Any]:
    stage = "evaluation"
    if not checkpoint_path.exists():
        raise WebIntegrationError(stage, f"Checkpoint missing: {checkpoint_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    adapter = IndependentDQNV2EnvAdapter(contract_path=DEFAULT_CONTRACT_PATH)
    traci = _ensure_traci_import()

    eval_results: list[dict[str, Any]] = []
    per_run_rows: list[dict[str, Any]] = []

    for spec in scenarios:
        scenario = _as_adapter_scenario(spec)
        run_id = f"independent_dqn_v2_web_eval__{scenario.scenario_id}"
        run_dir = run_root / run_id
        if run_dir.exists():
            import shutil

            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)

        cmd, outputs = _build_eval_command(scenario, run_dir)
        label = f"idqn_v2_web_eval_{int(time.time() * 1_000_000)}"
        port = _allocate_traci_port()

        decision_ticks = 0
        phase_change_count = 0
        eval_invalid_actions = 0
        eval_yellow_selected = 0
        eval_forbidden_calls = 0
        eval_errors: list[str] = []
        process = None

        try:
            with outputs["sumo_stdout"].open("w", encoding="utf-8") as sumo_stdout:
                traci.start(cmd, port=port, label=label, numRetries=40, stdout=sumo_stdout, doSwitch=True)
                conn = traci.getConnection(label)
                process = conn._process

                adapter._build_static_cache(conn)
                specs: dict[str, AgentSpec] = {}
                for i, (tls_id, tls_spec) in enumerate(sorted(adapter.static_cache.tls_specs.items())):
                    specs[tls_id] = AgentSpec(
                        obs_dim=int(tls_spec.obs_dim),
                        num_actions=int(len(tls_spec.action_phase_indices)),
                        config_overrides={"seed": int(seed + i), "device": "cpu"},
                    )
                eval_controller = IndependentDQNController(agent_specs=specs, default_config=DQNAgentConfig(seed=seed))
                eval_controller.load(checkpoint_path, map_location="cpu")

                next_decision_time_s = float(scenario.begin_s) + float(adapter.normalization.decision_interval_s)
                with ForbiddenCallTracker(conn) as forbidden_tracker:
                    while float(conn.simulation.getTime()) < float(scenario.end_s):
                        conn.simulationStep()
                        sim_time_s = float(conn.simulation.getTime())
                        adapter._update_runtime_phases(conn, sim_time_s)

                        if sim_time_s + 1e-9 < next_decision_time_s:
                            continue
                        next_decision_time_s = sim_time_s + float(adapter.normalization.decision_interval_s)

                        observations: dict[str, np.ndarray] = {}
                        action_masks: dict[str, np.ndarray] = {}
                        for tls_id in adapter.static_cache.tls_specs:
                            obs, _ = adapter._build_observation_and_reward(conn, tls_id, sim_time_s)
                            observations[tls_id] = obs
                            action_masks[tls_id] = adapter._build_action_mask(tls_id)

                        actions = eval_controller.select_actions(
                            observations=observations,
                            action_masks=action_masks,
                            explore=False,
                        )

                        for tls_id, action_idx in actions.items():
                            mask = action_masks[tls_id]
                            if not bool(mask[int(action_idx)]):
                                eval_invalid_actions += 1
                                raise WebIntegrationError(stage, f"Invalid eval action for {tls_id}: {action_idx}")

                            tls_spec = adapter.static_cache.tls_specs[tls_id]
                            runtime = adapter.static_cache.tls_runtime[tls_id]
                            current_phase = int(runtime.current_phase_index)
                            target_phase = int(tls_spec.action_phase_indices[int(action_idx)])

                            if target_phase in set(tls_spec.yellow_all_red_phase_indices):
                                eval_yellow_selected += 1
                                raise WebIntegrationError(
                                    stage,
                                    f"Eval selected yellow/all-red phase for {tls_id}: {target_phase}",
                                )
                            if target_phase != current_phase:
                                phase_change_count += 1

                            conn.trafficlight.setPhase(tls_id, target_phase)
                            conn.trafficlight.setPhaseDuration(tls_id, float(adapter.normalization.decision_interval_s))
                            runtime.current_phase_index = target_phase
                            runtime.phase_enter_time_s = sim_time_s

                        decision_ticks += 1

                    eval_forbidden_calls = int(sum(int(v) for v in forbidden_tracker.counts.values()))

                try:
                    traci.close(wait=True)
                except Exception:
                    pass

        except Exception as exc:
            eval_errors.append(f"{exc}\n{traceback.format_exc()}")
            try:
                traci.close(wait=True)
            except Exception:
                pass

        return_code = int(process.returncode if process is not None and process.returncode is not None else 1)

        trip_ok, trip_err = _parse_xml_ok(outputs["tripinfo_xml"])
        sum_ok, sum_err = _parse_xml_ok(outputs["summary_xml"])
        queue_ok, queue_err = _parse_xml_ok(outputs["queue_xml"])
        tls_ok, tls_err = _parse_xml_ok(outputs["tls_switch_states_xml"])

        trip_metrics = _tripinfo_metrics(outputs["tripinfo_xml"]) if outputs["tripinfo_xml"].exists() else {
            "completed_trip_count": None,
            "mean_waiting_time_s": None,
        }
        queue_mean = parse_queue_mean_from_xml(outputs["queue_xml"])

        tls_count = max(len(adapter.static_cache.tls_specs), 1)
        sim_minutes = max((scenario.end_s - scenario.begin_s) / 60.0, 1e-6)
        phase_change_rate = float(phase_change_count / (tls_count * sim_minutes))

        run_status = (
            "evaluation_passed"
            if (
                return_code == 0
                and trip_ok
                and sum_ok
                and queue_ok
                and tls_ok
                and eval_invalid_actions == 0
                and eval_forbidden_calls == 0
                and len(eval_errors) == 0
            )
            else "evaluation_failed"
        )

        manifest = {
            "created_at_utc": now_iso(),
            "controller": "independent_dqn_v2",
            "training_scope": "web",
            "model_status": "trained_from_scratch_uploaded_map",
            "checkpoint_source": str(checkpoint_path),
            "scenario_id": scenario.scenario_id,
            "network": scenario.network,
            "level": scenario.level,
            "seed": int(scenario.seed),
            "simulation_begin_s": int(scenario.begin_s),
            "simulation_end_s": int(scenario.end_s),
            "decision_ticks": int(decision_ticks),
            "phase_change_count": int(phase_change_count),
            "invalid_action_count": int(eval_invalid_actions),
            "yellow_all_red_selected_count": int(eval_yellow_selected),
            "forbidden_traci_call_count": int(eval_forbidden_calls),
            "sumo_return_code": int(return_code),
            "run_status": run_status,
            "output_files": {
                "tripinfo_xml": str(outputs["tripinfo_xml"]),
                "summary_xml": str(outputs["summary_xml"]),
                "queue_xml": str(outputs["queue_xml"]),
                "tls_switch_states_xml": str(outputs["tls_switch_states_xml"]),
                "run_log": str(outputs["run_log"]),
                "controller_diagnostics_json": str(outputs["controller_diagnostics_json"]),
            },
            "xml_parse_status": {
                "tripinfo_parse_success": trip_ok,
                "tripinfo_parse_error": trip_err,
                "summary_parse_success": sum_ok,
                "summary_parse_error": sum_err,
                "queue_parse_success": queue_ok,
                "queue_parse_error": queue_err,
                "tls_parse_success": tls_ok,
                "tls_parse_error": tls_err,
            },
            "kpis": {
                "mean_waiting_time_completed_s": trip_metrics.get("mean_waiting_time_s"),
                "throughput_completed_trips": trip_metrics.get("completed_trip_count"),
                "mean_total_queue_length_m": queue_mean,
                "phase_change_rate_per_tls_per_min": phase_change_rate,
            },
        }

        diagnostics = {
            "controller": "independent_dqn_v2",
            "training_scope": "web",
            "model_status": "trained_from_scratch_uploaded_map",
            "decision_ticks": int(decision_ticks),
            "phase_change_count": int(phase_change_count),
            "invalid_action_count": int(eval_invalid_actions),
            "yellow_all_red_selected_count": int(eval_yellow_selected),
            "forbidden_traci_call_count": int(eval_forbidden_calls),
            "errors": eval_errors,
        }

        json_dump(outputs["run_manifest_json"], manifest)
        json_dump(outputs["controller_diagnostics_json"], diagnostics)

        per_run_row = {
            "controller": "independent_dqn_v2",
            "training_scope": "web",
            "model_status": "trained_from_scratch_uploaded_map",
            "run_id": run_id,
            "scenario_id": scenario.scenario_id,
            "network": scenario.network,
            "level": scenario.level,
            "seed": int(scenario.seed),
            "mean_waiting_time_completed_s": trip_metrics.get("mean_waiting_time_s"),
            "throughput_completed_trips": trip_metrics.get("completed_trip_count"),
            "mean_total_queue_length_m": queue_mean,
            "phase_change_rate_per_tls_per_min": phase_change_rate,
            "run_status": run_status,
        }
        per_run_rows.append(per_run_row)

        eval_results.append(
            {
                "scenario_id": scenario.scenario_id,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "run_status": run_status,
                "sumo_return_code": int(return_code),
                "invalid_action_count": int(eval_invalid_actions),
                "forbidden_traci_call_count": int(eval_forbidden_calls),
                "manifest_path": str(outputs["run_manifest_json"]),
            }
        )

    per_run_csv = output_root / "kpi_per_run.csv"
    per_run_json = output_root / "kpi_per_run.json"
    csv_dump(per_run_csv, per_run_rows)
    per_run_json.write_text(__import__("json").dumps(per_run_rows, indent=2) + "\n", encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    for level in ["low", "medium", "high"]:
        level_rows = [r for r in per_run_rows if str(r["level"]) == level]
        summary_rows.append(
            {
                "controller": "independent_dqn_v2",
                "training_scope": "web",
                "network": "uploaded_map",
                "level": level,
                "run_count": len(level_rows),
                "mean_waiting_time_completed_s": _mean(
                    [float(r["mean_waiting_time_completed_s"]) for r in level_rows if r["mean_waiting_time_completed_s"] is not None]
                ),
                "throughput_completed_trips": _mean(
                    [float(r["throughput_completed_trips"]) for r in level_rows if r["throughput_completed_trips"] is not None]
                ),
                "mean_total_queue_length_m": _mean(
                    [float(r["mean_total_queue_length_m"]) for r in level_rows if r["mean_total_queue_length_m"] is not None]
                ),
                "phase_change_rate_per_tls_per_min": _mean(
                    [float(r["phase_change_rate_per_tls_per_min"]) for r in level_rows if r["phase_change_rate_per_tls_per_min"] is not None]
                ),
            }
        )

    summary_csv = output_root / "kpi_summary_by_level.csv"
    summary_json = output_root / "kpi_summary_by_level.json"
    csv_dump(summary_csv, summary_rows)
    summary_json.write_text(__import__("json").dumps(summary_rows, indent=2) + "\n", encoding="utf-8")

    checks = {
        "all_runs_passed": all(r["run_status"] == "evaluation_passed" for r in eval_results),
        "kpi_csv_written": per_run_csv.exists(),
        "kpi_json_written": per_run_json.exists(),
        "summary_csv_written": summary_csv.exists(),
        "summary_json_written": summary_json.exists(),
        "all_4_kpis_populated": all(
            row["mean_waiting_time_completed_s"] is not None
            and row["throughput_completed_trips"] is not None
            and row["mean_total_queue_length_m"] is not None
            and row["phase_change_rate_per_tls_per_min"] is not None
            for row in per_run_rows
        ),
    }
    checks["overall_pass"] = all(bool(v) for v in checks.values())

    payload = {
        "created_at_utc": now_iso(),
        "controller": "independent_dqn_v2",
        "evaluation_scope": "generated_heldout",
        "runs": eval_results,
        "per_run_csv": str(per_run_csv),
        "per_run_json": str(per_run_json),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "checks": checks,
    }
    summary_path = output_root / "evaluation_summary.json"
    json_dump(summary_path, payload)

    return {
        "summary_path": str(summary_path),
        "payload": payload,
        "checks": checks,
    }
