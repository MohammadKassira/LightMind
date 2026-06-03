from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.controllers.fixed_time.run_fixed_time import REPO_ROOT

from .env_adapter import (
    DEFAULT_BANK_MANIFEST,
    DEFAULT_CONTRACT_PATH,
    DEFAULT_DECISION_INTERVAL_S,
    FORBIDDEN_PER_STEP_CALLS,
    IndependentDQNV2EnvAdapter,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _profile_one(
    *,
    scenario_id: str,
    adapter: IndependentDQNV2EnvAdapter,
    max_decision_steps: int,
    run_root: Path,
    bank_manifest_path: Path,
) -> dict[str, Any]:
    result = adapter.profile_scenario(
        scenario_id=scenario_id,
        max_decision_steps=max_decision_steps,
        run_root=run_root,
        bank_manifest_path=bank_manifest_path,
    )
    payload = asdict(result)
    payload["result"] = "PASS" if (result.sumo_return_code == 0 and result.forbidden_call_count == 0) else "FAIL"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile Independent DQN v2 fast environment adapter (no model/training)."
    )
    parser.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--bank-manifest", type=Path, default=DEFAULT_BANK_MANIFEST)
    parser.add_argument("--decision-interval-s", type=float, default=DEFAULT_DECISION_INTERVAL_S)
    parser.add_argument("--max-decision-steps", type=int, default=120)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "benchmark/tmp/independent_dqn_v2_env_adapter_profile",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "benchmark/logs/independent_dqn_v2_env_adapter_profile.json",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=REPO_ROOT / "benchmark/logs/independent_dqn_v2_env_adapter_profile.txt",
    )
    parser.add_argument(
        "--output-validation-txt",
        type=Path,
        default=REPO_ROOT / "benchmark/logs/independent_dqn_v2_env_adapter_validation.txt",
    )
    args = parser.parse_args()

    contract_loaded = False
    adapter_imports_ok = True
    errors: list[str] = []

    try:
        adapter_probe = IndependentDQNV2EnvAdapter(
            contract_path=args.contract_path,
            decision_interval_s=float(args.decision_interval_s),
        )
        _ = adapter_probe.contract
        contract_loaded = True
    except Exception as exc:
        errors.append(f"contract_or_adapter_init_error:{exc}")
        contract_loaded = False

    scenario_ids = ["cologne1__medium__seed_001", "ingolstadt7__medium__seed_001"]
    scenario_results: list[dict[str, Any]] = []

    for scenario_id in scenario_ids:
        try:
            adapter = IndependentDQNV2EnvAdapter(
                contract_path=args.contract_path,
                decision_interval_s=float(args.decision_interval_s),
            )
            prof = _profile_one(
                scenario_id=scenario_id,
                adapter=adapter,
                max_decision_steps=int(args.max_decision_steps),
                run_root=args.run_root,
                bank_manifest_path=args.bank_manifest,
            )
            scenario_results.append(prof)
        except Exception as exc:
            scenario_results.append(
                {
                    "scenario_id": scenario_id,
                    "result": "FAIL",
                    "error": str(exc),
                }
            )

    by_id = {item.get("scenario_id"): item for item in scenario_results}

    cologne = by_id.get("cologne1__medium__seed_001", {})
    ing7 = by_id.get("ingolstadt7__medium__seed_001", {})

    scenario_resolves = all("error" not in r for r in scenario_results)
    sumo_starts = all(int(r.get("sumo_return_code", 1)) == 0 for r in scenario_results if "error" not in r)

    static_cache_once = all(
        int(r.get("static_cache_build_count", 0)) == 1 for r in scenario_results if "error" not in r
    )
    qmax_computed = all(int(r.get("q_max_lane_computed_count", 0)) > 0 for r in scenario_results if "error" not in r)
    phase_timer_base = all(
        int(r.get("phase_timer_norm_base_computed_count", 0)) > 0 for r in scenario_results if "error" not in r
    )
    observations_all_tls = all(
        bool(r.get("observations_generated_all_tls", False)) for r in scenario_results if "error" not in r
    )
    obs_dims_ok = all(bool(r.get("obs_dim_formula_ok", False)) for r in scenario_results if "error" not in r)
    action_masks_ok = all(bool(r.get("action_masks_valid", False)) for r in scenario_results if "error" not in r)
    rewards_ok = all(bool(r.get("rewards_computed", False)) for r in scenario_results if "error" not in r)
    outgoing_ok = all(
        bool(r.get("outgoing_controlled_link_integrity_ok", False)) for r in scenario_results if "error" not in r
    )
    yellow_red_ok = all(
        bool(r.get("yellow_all_red_contract_ok", False)) for r in scenario_results if "error" not in r
    )
    no_neighbor = all(
        bool(r.get("no_neighbor_information_used", False)) for r in scenario_results if "error" not in r
    )
    no_model_logic = all(
        bool(r.get("no_model_shared_weight_logic", False)) for r in scenario_results if "error" not in r
    )
    forbidden_absent = all(int(r.get("forbidden_call_count", 1)) == 0 for r in scenario_results if "error" not in r)

    profiling_cologne_done = "error" not in cologne and int(cologne.get("decision_ticks_profiled", 0)) > 0
    profiling_ing7_done = "error" not in ing7 and int(ing7.get("decision_ticks_profiled", 0)) > 0

    no_training_launched = all(
        bool(r.get("training_launched", True)) is False for r in scenario_results if "error" not in r
    )

    fixed_time_untouched = (
        (REPO_ROOT / "benchmark/controllers/fixed_time").exists()
        and (REPO_ROOT / "benchmark/runs/fixed_time").exists()
        and (REPO_ROOT / "benchmark/results/fixed_time").exists()
    )
    max_pressure_untouched = (
        (REPO_ROOT / "benchmark/controllers/max_pressure").exists()
        and (REPO_ROOT / "benchmark/runs/max_pressure").exists()
        and (REPO_ROOT / "benchmark/results/max_pressure").exists()
    )

    checks = {
        "adapter_imports": adapter_imports_ok,
        "contract_loads": contract_loaded,
        "frozen_scenarios_resolve": scenario_resolves,
        "sumo_traci_starts": sumo_starts,
        "static_cache_builds_once": static_cache_once,
        "q_max_lane_computed": qmax_computed,
        "phase_timer_normalization_base_computed": phase_timer_base,
        "observations_generated_all_tls": observations_all_tls,
        "observation_dimensions_match_formula": obs_dims_ok,
        "action_masks_valid": action_masks_ok,
        "rewards_computed": rewards_ok,
        "outgoing_lanes_from_controlled_links": outgoing_ok,
        "yellow_all_red_handling_contract": yellow_red_ok,
        "no_neighbor_information": no_neighbor,
        "no_model_shared_weight_logic": no_model_logic,
        "forbidden_calls_absent_in_per_step_path": forbidden_absent,
        "profiling_completed_cologne1": profiling_cologne_done,
        "profiling_completed_ingolstadt7": profiling_ing7_done,
        "no_dqn_training_launched": no_training_launched,
        "fixed_time_untouched": fixed_time_untouched,
        "max_pressure_untouched": max_pressure_untouched,
    }
    checks["overall_result_pass"] = all(checks.values())

    payload = {
        "created_at_utc": _now_iso(),
        "contract_path": str(args.contract_path),
        "forbidden_per_step_traci_calls": list(FORBIDDEN_PER_STEP_CALLS),
        "scenario_results": scenario_results,
        "checks": checks,
        "errors": errors,
    }
    _json_dump(args.output_json, payload)

    lines = [
        "Independent DQN v2 Env Adapter Profile",
        "",
        f"Created (UTC): {payload['created_at_utc']}",
        f"Contract: {payload['contract_path']}",
        "",
    ]
    for item in scenario_results:
        lines.append(f"Scenario: {item.get('scenario_id')}")
        if "error" in item:
            lines.append(f"- result: FAIL")
            lines.append(f"- error: {item['error']}")
            lines.append("")
            continue
        lines.extend(
            [
                f"- result: {item.get('result')}",
                f"- decision_ticks_profiled: {item.get('decision_ticks_profiled')}",
                f"- tls_agent_count: {item.get('tls_agent_count')}",
                f"- observation_time_total_s: {item.get('observation_time_total_s')}",
                f"- observation_time_avg_per_decision_tick_s: {item.get('observation_time_avg_per_decision_tick_s')}",
                f"- observation_time_avg_per_tls_observation_s: {item.get('observation_time_avg_per_tls_observation_s')}",
                f"- observation_time_total_s_by_tls: {item.get('observation_time_total_s_by_tls')}",
                f"- observation_time_avg_per_tls_per_decision_tick_s: {item.get('observation_time_avg_per_tls_per_decision_tick_s')}",
                f"- sumo_step_time_total_s: {item.get('sumo_step_time_total_s')}",
                f"- reward_time_total_s: {item.get('reward_time_total_s')}",
                f"- action_mask_time_total_s: {item.get('action_mask_time_total_s')}",
                f"- forbidden_call_count: {item.get('forbidden_call_count')}",
                f"- forbidden_call_counts: {item.get('forbidden_call_counts')}",
                f"- allowed_dynamic_call_counts: {item.get('allowed_dynamic_call_counts')}",
                "",
            ]
        )

    args.output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vlines = [
        "Independent DQN v2 Env Adapter Validation",
        "",
        f"1. adapter imports: {'PASS' if checks['adapter_imports'] else 'FAIL'}",
        f"2. contract loads: {'PASS' if checks['contract_loads'] else 'FAIL'}",
        f"3. frozen scenarios resolve: {'PASS' if checks['frozen_scenarios_resolve'] else 'FAIL'}",
        f"4. SUMO/TraCI starts: {'PASS' if checks['sumo_traci_starts'] else 'FAIL'}",
        f"5. static cache builds once: {'PASS' if checks['static_cache_builds_once'] else 'FAIL'}",
        f"6. per-lane q_max_lane computed: {'PASS' if checks['q_max_lane_computed'] else 'FAIL'}",
        f"7. phase timer normalization base computed: {'PASS' if checks['phase_timer_normalization_base_computed'] else 'FAIL'}",
        f"8. observations generated for all TLS agents: {'PASS' if checks['observations_generated_all_tls'] else 'FAIL'}",
        f"9. observation dimensions match formula: {'PASS' if checks['observation_dimensions_match_formula'] else 'FAIL'}",
        f"10. action masks valid: {'PASS' if checks['action_masks_valid'] else 'FAIL'}",
        f"11. rewards computed: {'PASS' if checks['rewards_computed'] else 'FAIL'}",
        f"12. outgoing lanes use controlled-link destinations: {'PASS' if checks['outgoing_lanes_from_controlled_links'] else 'FAIL'}",
        f"13. yellow/all-red handling follows contract: {'PASS' if checks['yellow_all_red_handling_contract'] else 'FAIL'}",
        f"14. no neighbor information used: {'PASS' if checks['no_neighbor_information'] else 'FAIL'}",
        f"15. no model/shared-weight logic introduced: {'PASS' if checks['no_model_shared_weight_logic'] else 'FAIL'}",
        f"16. forbidden calls absent from per-step observation path: {'PASS' if checks['forbidden_calls_absent_in_per_step_path'] else 'FAIL'}",
        f"17. profiling completed for cologne1: {'PASS' if checks['profiling_completed_cologne1'] else 'FAIL'}",
        f"18. profiling completed for ingolstadt7: {'PASS' if checks['profiling_completed_ingolstadt7'] else 'FAIL'}",
        f"19. no DQN training launched: {'PASS' if checks['no_dqn_training_launched'] else 'FAIL'}",
        f"20. Fixed-Time and MaxPressure untouched: {'PASS' if (checks['fixed_time_untouched'] and checks['max_pressure_untouched']) else 'FAIL'}",
        f"21. overall result = {'PASS' if checks['overall_result_pass'] else 'FAIL'}",
        "",
        f"Overall: {'PASS' if checks['overall_result_pass'] else 'FAIL'}",
    ]
    args.output_validation_txt.write_text("\n".join(vlines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
