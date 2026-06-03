from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.controllers.fixed_time.run_fixed_time import (
    REPO_ROOT,
    _build_most_parity_sumocfg,
    _load_json,
    _parse_xml_ok,
    _resolve_scenario_context,
    _tripinfo_metrics,
    _write_tls_switch_additional_file,
)
from benchmark.controllers.independent_dqn_v2.env_adapter import (
    DEFAULT_BANK_MANIFEST,
    DEFAULT_CONTRACT_PATH,
    ForbiddenCallTracker,
    IndependentDQNV2EnvAdapter,
    parse_queue_mean_from_xml,
)
from models.independent_dqn_v2 import (
    AgentSpec,
    DQNAgentConfig,
    IndependentDQNController,
    safe_agent_filename,
)


DEFAULT_NETWORK = "cologne1"
DEFAULT_EPISODES = 50
DEFAULT_MAX_STEPS_PER_EPISODE = 720
DEFAULT_WALL_CLOCK_CAP_MINUTES = 45.0
DEFAULT_PILOT_MODEL_ROOT = REPO_ROOT / "benchmark/models/independent_dqn_v2_pilot"
DEFAULT_PILOT_RUN_ROOT = REPO_ROOT / "benchmark/runs/independent_dqn_v2_pilot"
DEFAULT_PILOT_RESULTS_ROOT = REPO_ROOT / "benchmark/results/independent_dqn_v2_pilot"
DEFAULT_MAIN_MODEL_ROOT = REPO_ROOT / "benchmark/models/independent_dqn_v2"
DEFAULT_MAIN_RUN_ROOT = REPO_ROOT / "benchmark/runs/independent_dqn_v2"
ALLOWED_PILOT_NETWORKS = {
    "MoST",
    "cologne1",
    "ingolstadt7",
    "ingolstadt21",
    "cologne8",
    "bologna_pasubio",
    "grid4x4",
    "arterial4x4",
    "toronto",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _torch_readiness() -> dict[str, Any]:
    out = {
        "torch_import_ok": False,
        "torch_version": None,
        "cuda_available": None,
    }
    try:
        import torch

        out["torch_import_ok"] = True
        out["torch_version"] = str(torch.__version__)
        out["cuda_available"] = bool(torch.cuda.is_available())
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out


def _build_eval_command(adapter: IndependentDQNV2EnvAdapter, scenario: Any, run_dir: Path) -> tuple[list[str], dict[str, Path]]:
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
    if str(getattr(scenario, "network", "")) == "MoST" and getattr(scenario, "canonical_sumocfg_file", None):
        outputs["generated_sumocfg"] = run_dir / "most_parity_generated.sumocfg"
        generated_sumocfg = _build_most_parity_sumocfg(
            canonical_sumocfg=scenario.canonical_sumocfg_file,
            generated_sumocfg=outputs["generated_sumocfg"],
            net_file=scenario.net_file,
            route_files=scenario.route_files,
            additional_files=scenario.additional_files,
            begin_s=int(scenario.begin_s),
            end_s=int(scenario.end_s),
            outputs=outputs,
            tls_output_mode="switch_states_only",
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


def _trend_label(first_mean: float | None, last_mean: float | None, flat_pct_threshold: float = 5.0) -> tuple[str, float | None, float | None]:
    if first_mean is None or last_mean is None:
        return "unknown", None, None
    delta = float(last_mean - first_mean)
    base = max(abs(first_mean), 1e-6)
    delta_pct = float((delta / base) * 100.0)
    if abs(delta_pct) <= float(flat_pct_threshold):
        return "flat", delta, delta_pct
    return ("improving" if delta > 0 else "degrading"), delta, delta_pct


def run_pilot(
    *,
    network: str,
    episodes: int,
    max_steps_per_episode: int,
    wall_clock_cap_minutes: float,
    bank_manifest_path: Path,
    contract_path: Path,
    pilot_model_root: Path,
    pilot_run_root: Path,
    pilot_results_root: Path,
    seed: int,
) -> dict[str, Any]:
    np.random.seed(seed)
    if str(network) not in ALLOWED_PILOT_NETWORKS:
        raise RuntimeError(
            f"Pilot v2 runner is locked to {sorted(ALLOWED_PILOT_NETWORKS)} only, got: {network}"
        )

    adapter = IndependentDQNV2EnvAdapter(contract_path=contract_path)
    contract = adapter.contract
    main_episode_budget = int(
        ((contract.get("fixed_compute_budget_policy", {}) or {}).get("per_network_episode_budgets", {}) or {}).get(
            str(network),
            500,
        )
    )

    train_every_steps = int(contract.get("train_every_steps", 8))
    learning_starts_steps = int(contract.get("learning_starts_steps", 1000))
    target_update_interval = 1000

    torch_status = _torch_readiness()
    if not bool(torch_status["torch_import_ok"]):
        raise RuntimeError(f"Torch import failed: {torch_status.get('error')}")

    bank = _load_json(bank_manifest_path)
    entries = [e for e in (bank.get("entries", []) or []) if str(e.get("network")) == str(network)]
    scenario_ids = sorted([str(e["scenario_id"]) for e in entries])
    if len(scenario_ids) != 9:
        raise RuntimeError(f"Expected 9 frozen scenarios for {network}, found {len(scenario_ids)}")

    train_scenario_id = f"{network}__medium__seed_001"
    if train_scenario_id not in scenario_ids:
        raise RuntimeError(f"Training scenario not found in frozen set: {train_scenario_id}")

    train_scenario = adapter.resolve_scenario(train_scenario_id, bank_manifest_path=bank_manifest_path)

    network_model_dir = pilot_model_root / network
    checkpoint_source_rel = f"benchmark/models/independent_dqn_v2_pilot/{network}/"
    preexisting_pilot_run_dirs = (
        {p.name for p in pilot_run_root.iterdir() if p.is_dir() and p.name.startswith("independent_dqn_v2_pilot__")}
        if pilot_run_root.exists()
        else set()
    )
    preexisting_main_model_dirs = (
        {p.name for p in DEFAULT_MAIN_MODEL_ROOT.iterdir() if p.is_dir()} if DEFAULT_MAIN_MODEL_ROOT.exists() else set()
    )
    preexisting_main_run_dirs = (
        {p.name for p in DEFAULT_MAIN_RUN_ROOT.iterdir() if p.is_dir() and p.name.startswith("independent_dqn_v2_main__")}
        if DEFAULT_MAIN_RUN_ROOT.exists()
        else set()
    )
    if network_model_dir.exists():
        import shutil

        shutil.rmtree(network_model_dir)
    network_model_dir.mkdir(parents=True, exist_ok=False)
    rollout_root = network_model_dir / "training_rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)

    from benchmark.controllers.independent_dqn_v2.env_adapter import _ensure_traci_import, _allocate_traci_port

    traci = _ensure_traci_import()

    controller: IndependentDQNController | None = None
    agent_ids: list[str] = []

    metrics_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []

    global_transition_steps = 0
    total_gradient_updates = 0
    gradient_updates_by_agent: dict[str, int] = {}
    total_target_sync_updates = 0
    invalid_action_count = 0
    yellow_all_red_selected_count = 0
    total_forbidden_calls = 0
    total_phase_change_count = 0
    replay_size_max = 0
    losses_finite = True
    training_started = False

    train_wall_start = time.perf_counter()
    wall_clock_cap_exceeded = False
    wall_clock_cap_exceeded_at_episode: int | None = None

    for ep in range(1, int(episodes) + 1):
        ep_wall_start = time.perf_counter()
        ep_dir = rollout_root / f"episode_{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        cmd, outputs = adapter._build_traci_command(
            train_scenario,
            ep_dir,
            include_xml_outputs=False,
            suppress_warnings=True,
        )
        label = f"idqn_v2_pilot_train_{int(time.time() * 1_000_000)}_{ep}"
        port = _allocate_traci_port()

        ep_decision_steps = 0
        ep_reward_sum = 0.0
        ep_updates = 0
        ep_losses: list[float] = []
        ep_arrived = 0
        ep_queue_proxy_sum = 0.0
        ep_phase_change_count = 0
        ep_forbidden_calls = 0

        prev_obs: dict[str, np.ndarray] | None = None
        prev_actions: dict[str, int] | None = None
        prev_masks: dict[str, np.ndarray] | None = None
        last_rewards: dict[str, float] | None = None

        process = None
        try:
            with outputs["sumo_stdout"].open("w", encoding="utf-8") as sumo_stdout:
                traci.start(cmd, port=port, label=label, numRetries=40, stdout=sumo_stdout, doSwitch=True)
                conn = traci.getConnection(label)
                process = conn._process
                training_started = True

                adapter._build_static_cache(conn)
                if controller is None:
                    specs: dict[str, AgentSpec] = {}
                    for i, (tls_id, tls_spec) in enumerate(sorted(adapter.static_cache.tls_specs.items())):
                        specs[tls_id] = AgentSpec(
                            obs_dim=int(tls_spec.obs_dim),
                            num_actions=int(len(tls_spec.action_phase_indices)),
                            config_overrides={
                                "seed": int(seed + i),
                                "device": "cpu",
                                "batch_size": 64,
                                "replay_buffer_size": 50_000,
                                "min_replay_size": 1_000,
                                "target_update_interval": target_update_interval,
                                "epsilon_start": 1.0,
                                "epsilon_end": 0.05,
                                "epsilon_decay_steps": 10_000,
                                "train_every_steps": train_every_steps,
                                "learning_starts_steps": learning_starts_steps,
                            },
                        )
                    controller = IndependentDQNController(agent_specs=specs, default_config=DQNAgentConfig(seed=seed))
                    agent_ids = sorted(specs.keys())
                    gradient_updates_by_agent = {aid: 0 for aid in agent_ids}

                if controller is None:
                    raise RuntimeError("Controller initialization failed.")

                next_decision_time_s = float(train_scenario.begin_s) + float(adapter.normalization.decision_interval_s)

                with ForbiddenCallTracker(conn) as forbidden_tracker:
                    while float(conn.simulation.getTime()) < float(train_scenario.end_s) and ep_decision_steps < int(
                        max_steps_per_episode
                    ):
                        conn.simulationStep()
                        sim_time_s = float(conn.simulation.getTime())
                        ep_arrived += int(conn.simulation.getArrivedNumber())

                        adapter._update_runtime_phases(conn, sim_time_s)
                        if sim_time_s + 1e-9 < next_decision_time_s:
                            continue
                        next_decision_time_s = sim_time_s + float(adapter.normalization.decision_interval_s)

                        observations: dict[str, np.ndarray] = {}
                        action_masks: dict[str, np.ndarray] = {}
                        rewards: dict[str, float] = {}

                        for tls_id, spec in adapter.static_cache.tls_specs.items():
                            obs, rew = adapter._build_observation_and_reward(conn, tls_id, sim_time_s)
                            mask = adapter._build_action_mask(tls_id)

                            if obs.shape != (int(spec.obs_dim),):
                                raise RuntimeError(
                                    f"Observation dimension mismatch for {tls_id}: {obs.shape} vs {(int(spec.obs_dim),)}"
                                )
                            if mask.shape != (len(spec.action_phase_indices),):
                                raise RuntimeError(
                                    f"Mask shape mismatch for {tls_id}: {mask.shape} vs {(len(spec.action_phase_indices),)}"
                                )

                            observations[tls_id] = obs
                            action_masks[tls_id] = mask
                            rewards[tls_id] = float(rew)
                            ep_queue_proxy_sum += float(np.sum(obs[len(spec.phases) + 1 :]))

                        if prev_obs is not None and prev_actions is not None and prev_masks is not None:
                            controller.store_transitions(
                                observations=prev_obs,
                                actions=prev_actions,
                                rewards=rewards,
                                next_observations=observations,
                                dones={aid: False for aid in agent_ids},
                                next_action_masks=action_masks,
                            )
                            global_transition_steps += 1

                            if (
                                global_transition_steps >= learning_starts_steps
                                and global_transition_steps % train_every_steps == 0
                            ):
                                step_metrics = controller.train_step()
                                for aid, m in step_metrics.items():
                                    if m is None:
                                        continue
                                    loss = float(m["loss"])
                                    finite = bool(math.isfinite(loss))
                                    losses_finite = losses_finite and finite
                                    target_updated = int(m.get("target_updated", 0.0))
                                    total_gradient_updates += 1
                                    gradient_updates_by_agent[aid] = int(gradient_updates_by_agent.get(aid, 0) + 1)
                                    ep_updates += 1
                                    total_target_sync_updates += target_updated
                                    ep_losses.append(loss)
                                    metrics_rows.append(
                                        {
                                            "episode": ep,
                                            "agent_id": aid,
                                            "global_transition_step": int(global_transition_steps),
                                            "loss": loss,
                                            "loss_is_finite": finite,
                                            "epsilon": float(m.get("epsilon", 0.0)),
                                            "target_updated": target_updated,
                                            "target_sync_count": int(m.get("target_sync_count", 0.0)),
                                            "replay_size": int(m.get("replay_size", 0.0)),
                                        }
                                    )

                        actions = controller.select_actions(
                            observations=observations,
                            action_masks=action_masks,
                            explore=True,
                        )

                        for tls_id, action_idx in actions.items():
                            mask = action_masks[tls_id]
                            if not bool(mask[int(action_idx)]):
                                invalid_action_count += 1
                                raise RuntimeError(f"Invalid masked action for {tls_id}: {action_idx}")

                            spec = adapter.static_cache.tls_specs[tls_id]
                            runtime = adapter.static_cache.tls_runtime[tls_id]
                            current_phase = int(runtime.current_phase_index)
                            target_phase = int(spec.action_phase_indices[int(action_idx)])

                            if target_phase in set(spec.yellow_all_red_phase_indices):
                                yellow_all_red_selected_count += 1
                                raise RuntimeError(f"Selected yellow/all-red phase for {tls_id}: {target_phase}")

                            if target_phase != current_phase:
                                ep_phase_change_count += 1

                            conn.trafficlight.setPhase(tls_id, target_phase)
                            conn.trafficlight.setPhaseDuration(tls_id, float(adapter.normalization.decision_interval_s))
                            runtime.current_phase_index = target_phase
                            runtime.phase_enter_time_s = sim_time_s

                        ep_reward_sum += float(sum(rewards.values()))
                        prev_obs = observations
                        prev_actions = actions
                        prev_masks = action_masks
                        last_rewards = rewards
                        ep_decision_steps += 1

                    if prev_obs is not None and prev_actions is not None and prev_masks is not None and last_rewards is not None:
                        controller.store_transitions(
                            observations=prev_obs,
                            actions=prev_actions,
                            rewards=last_rewards,
                            next_observations=prev_obs,
                            dones={aid: True for aid in agent_ids},
                            next_action_masks=prev_masks,
                        )
                        global_transition_steps += 1

                        if (
                            global_transition_steps >= learning_starts_steps
                            and global_transition_steps % train_every_steps == 0
                        ):
                            step_metrics = controller.train_step()
                            for aid, m in step_metrics.items():
                                if m is None:
                                    continue
                                loss = float(m["loss"])
                                finite = bool(math.isfinite(loss))
                                losses_finite = losses_finite and finite
                                target_updated = int(m.get("target_updated", 0.0))
                                total_gradient_updates += 1
                                gradient_updates_by_agent[aid] = int(gradient_updates_by_agent.get(aid, 0) + 1)
                                ep_updates += 1
                                total_target_sync_updates += target_updated
                                ep_losses.append(loss)
                                metrics_rows.append(
                                    {
                                        "episode": ep,
                                        "agent_id": aid,
                                        "global_transition_step": int(global_transition_steps),
                                        "loss": loss,
                                        "loss_is_finite": finite,
                                        "epsilon": float(m.get("epsilon", 0.0)),
                                        "target_updated": target_updated,
                                        "target_sync_count": int(m.get("target_sync_count", 0.0)),
                                        "replay_size": int(m.get("replay_size", 0.0)),
                                    }
                                )

                    ep_forbidden_calls = int(sum(int(v) for v in forbidden_tracker.counts.values()))
                    total_forbidden_calls += ep_forbidden_calls

                try:
                    traci.close(wait=True)
                except Exception:
                    pass

            ret = int(process.returncode if process is not None and process.returncode is not None else 1)
            replay_size_end = 0
            if controller is not None and agent_ids:
                replay_size_end = max(len(controller.agents[aid].replay) for aid in agent_ids)
                replay_size_max = max(replay_size_max, replay_size_end)

            ep_wall = max(time.perf_counter() - ep_wall_start, 0.0)
            total_phase_change_count += ep_phase_change_count
            episode_summaries.append(
                {
                    "episode": ep,
                    "return_code": ret,
                    "episode_steps": int(ep_decision_steps),
                    "episode_reward_sum": float(ep_reward_sum),
                    "episode_loss_mean": _mean(ep_losses),
                    "episode_loss_update_count": int(ep_updates),
                    "episode_wall_time_s": float(ep_wall),
                    "episode_arrived_vehicles": int(ep_arrived),
                    "episode_throughput_proxy": int(ep_arrived),
                    "episode_mean_queue_count_or_length_proxy": (
                        float(ep_queue_proxy_sum / max(ep_decision_steps, 1))
                        if ep_decision_steps > 0
                        else None
                    ),
                    "episode_phase_change_count": int(ep_phase_change_count),
                    "episode_phase_change_rate": (
                        float(ep_phase_change_count / max(ep_decision_steps, 1))
                        if ep_decision_steps > 0
                        else None
                    ),
                    "episode_waiting_proxy": None,
                    "replay_size_end": int(replay_size_end),
                    "forbidden_call_count": int(ep_forbidden_calls),
                }
            )

        except Exception as exc:
            episode_summaries.append(
                {
                    "episode": ep,
                    "return_code": 1,
                    "episode_steps": int(ep_decision_steps),
                    "episode_reward_sum": float(ep_reward_sum),
                    "episode_loss_mean": None,
                    "episode_loss_update_count": int(ep_updates),
                    "episode_wall_time_s": float(max(time.perf_counter() - ep_wall_start, 0.0)),
                    "episode_arrived_vehicles": int(ep_arrived),
                    "episode_throughput_proxy": int(ep_arrived),
                    "episode_mean_queue_count_or_length_proxy": None,
                    "episode_phase_change_count": int(ep_phase_change_count),
                    "episode_phase_change_rate": None,
                    "episode_waiting_proxy": None,
                    "replay_size_end": 0,
                    "forbidden_call_count": int(ep_forbidden_calls),
                    "error": f"{exc}\n{traceback.format_exc()}",
                }
            )
            try:
                traci.close(wait=True)
            except Exception:
                pass

        elapsed_min = (time.perf_counter() - train_wall_start) / 60.0
        if (not wall_clock_cap_exceeded) and elapsed_min > float(wall_clock_cap_minutes):
            wall_clock_cap_exceeded = True
            wall_clock_cap_exceeded_at_episode = ep

    if controller is None:
        raise RuntimeError("Pilot training did not initialize controller.")

    total_wall_s = max(time.perf_counter() - train_wall_start, 0.0)
    avg_ep_s = total_wall_s / max(len(episode_summaries), 1)
    projected_500_s = avg_ep_s * 500.0
    projected_main_s = avg_ep_s * float(main_episode_budget)

    reward_values = [float(ep.get("episode_reward_sum", 0.0)) for ep in episode_summaries if "error" not in ep]
    reward_first10_mean = _mean(reward_values[:10])
    reward_last10_mean = _mean(reward_values[-10:])
    reward_trend_label, reward_trend_delta, reward_trend_delta_pct = _trend_label(
        reward_first10_mean,
        reward_last10_mean,
        flat_pct_threshold=5.0,
    )

    update_losses = [float(r["loss"]) for r in metrics_rows if bool(r.get("loss_is_finite", False))]
    update_window = min(50, len(update_losses))
    loss_first_window_mean = _mean(update_losses[:update_window]) if update_window > 0 else None
    loss_last_window_mean = _mean(update_losses[-update_window:]) if update_window > 0 else None

    target_sync_by_agent = {aid: int(controller.agents[aid].target_sync_count) for aid in agent_ids}

    num_actions = {aid: int(controller.agents[aid].num_actions) for aid in agent_ids}
    degenerate_warning = None
    if all(v == 1 for v in num_actions.values()):
        degenerate_warning = "single_action_space_per_tls"
    elif total_phase_change_count == 0:
        degenerate_warning = "no_phase_changes_detected"

    controller_ckpt = network_model_dir / "controller.pt"
    controller.save(controller_ckpt)
    per_agent_ckpts: list[str] = []
    for aid in agent_ids:
        p = network_model_dir / safe_agent_filename(aid)
        controller.agents[aid].save(p)
        per_agent_ckpts.append(str(p))

    training_metrics_csv = network_model_dir / "training_metrics.csv"
    training_metrics_json = network_model_dir / "training_metrics.json"
    _write_csv(training_metrics_csv, metrics_rows)
    training_metrics_json.write_text(json.dumps(metrics_rows, indent=2) + "\n", encoding="utf-8")

    training_summary = {
        "created_at_utc": _now_iso(),
        "controller": "independent_dqn_v2",
        "training_scope": "pilot",
        "network": network,
        "scenario_id": train_scenario_id,
        "episodes": int(episodes),
        "max_steps_per_episode": int(max_steps_per_episode),
        "train_every_steps": int(train_every_steps),
        "learning_starts_steps": int(learning_starts_steps),
        "target_update_interval": int(target_update_interval),
        "tls_agent_count": int(len(agent_ids)),
        "agent_ids": agent_ids,
        "total_wall_clock_s": float(total_wall_s),
        "average_seconds_per_episode": float(avg_ep_s),
        "projected_500_episode_runtime_s": float(projected_500_s),
        "projected_500_episode_runtime_h": float(projected_500_s / 3600.0),
        "projected_main_episode_count": int(main_episode_budget),
        "projected_main_episode_runtime_s": float(projected_main_s),
        "projected_main_episode_runtime_h": float(projected_main_s / 3600.0),
        "wall_clock_cap_minutes": float(wall_clock_cap_minutes),
        "wall_clock_cap_exceeded": bool(wall_clock_cap_exceeded),
        "wall_clock_cap_exceeded_at_episode": wall_clock_cap_exceeded_at_episode,
        "total_transitions": int(global_transition_steps),
        "replay_buffer_size_max": int(replay_size_max),
        "total_gradient_updates": int(total_gradient_updates),
        "gradient_updates_by_agent": {aid: int(gradient_updates_by_agent.get(aid, 0)) for aid in agent_ids},
        "losses_finite": bool(losses_finite),
        "target_sync_count_by_agent": target_sync_by_agent,
        "target_sync_total": int(total_target_sync_updates),
        "invalid_action_count": int(invalid_action_count),
        "forbidden_traci_call_count": int(total_forbidden_calls),
        "reward_first_10_mean": reward_first10_mean,
        "reward_last_10_mean": reward_last10_mean,
        "reward_trend_delta": reward_trend_delta,
        "reward_trend_delta_pct": reward_trend_delta_pct,
        "reward_trend_direction": reward_trend_label,
        "reward_trend_flat_threshold_pct": 5.0,
        "loss_first_updating_window_mean": loss_first_window_mean,
        "loss_last_updating_window_mean": loss_last_window_mean,
        "loss_window_size_updates": int(update_window),
        "phase_change_count_total": int(total_phase_change_count),
        "degenerate_action_warning": degenerate_warning,
        "episode_summaries": episode_summaries,
        "checkpoints": {
            "controller": str(controller_ckpt),
            "per_agent": per_agent_ckpts,
        },
        "metrics_files": {
            "csv": str(training_metrics_csv),
            "json": str(training_metrics_json),
        },
    }

    training_manifest = {
        "created_at_utc": _now_iso(),
        "controller": "independent_dqn_v2",
        "training_scope": "pilot",
        "network": network,
        "checkpoint_source": "none",
        "episodes": int(episodes),
        "max_steps_per_episode": int(max_steps_per_episode),
        "train_every_steps": int(train_every_steps),
        "learning_starts_steps": int(learning_starts_steps),
        "target_update_interval": int(target_update_interval),
        "full_main_training_launched": False,
        "full_90_run_evaluation_launched": False,
        "output_files": {
            "training_pilot_manifest_json": str(network_model_dir / "training_pilot_manifest.json"),
            "training_pilot_summary_json": str(network_model_dir / "training_pilot_summary.json"),
            "training_metrics_csv": str(training_metrics_csv),
            "training_metrics_json": str(training_metrics_json),
            "controller_checkpoint": str(controller_ckpt),
            "per_agent_checkpoints": per_agent_ckpts,
        },
    }

    _json_dump(network_model_dir / "training_pilot_summary.json", training_summary)
    _json_dump(network_model_dir / "training_pilot_manifest.json", training_manifest)

    # Evaluation across 9 frozen scenarios.
    eval_results: list[dict[str, Any]] = []
    per_run_rows: list[dict[str, Any]] = []

    from benchmark.controllers.independent_dqn_v2.env_adapter import _ensure_traci_import, _allocate_traci_port

    traci_eval = _ensure_traci_import()

    for scenario_id in scenario_ids:
        scenario = adapter.resolve_scenario(scenario_id, bank_manifest_path=bank_manifest_path)
        run_id = f"independent_dqn_v2_pilot__{scenario_id}"
        run_dir = pilot_run_root / run_id
        if run_dir.exists():
            import shutil

            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)

        cmd, outputs = _build_eval_command(adapter, scenario, run_dir)
        label = f"idqn_v2_pilot_eval_{int(time.time() * 1_000_000)}"
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
                traci_eval.start(cmd, port=port, label=label, numRetries=40, stdout=sumo_stdout, doSwitch=True)
                conn = traci_eval.getConnection(label)
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
                eval_controller.load(controller_ckpt, map_location="cpu")

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
                        for tls_id, spec in adapter.static_cache.tls_specs.items():
                            obs, _rew = adapter._build_observation_and_reward(conn, tls_id, sim_time_s)
                            mask = adapter._build_action_mask(tls_id)
                            observations[tls_id] = obs
                            action_masks[tls_id] = mask

                        actions = eval_controller.select_actions(
                            observations=observations,
                            action_masks=action_masks,
                            explore=False,
                        )

                        for tls_id, action_idx in actions.items():
                            mask = action_masks[tls_id]
                            if not bool(mask[int(action_idx)]):
                                eval_invalid_actions += 1
                                raise RuntimeError(f"Invalid eval action for {tls_id}: {action_idx}")

                            spec = adapter.static_cache.tls_specs[tls_id]
                            runtime = adapter.static_cache.tls_runtime[tls_id]
                            current_phase = int(runtime.current_phase_index)
                            target_phase = int(spec.action_phase_indices[int(action_idx)])
                            if target_phase in set(spec.yellow_all_red_phase_indices):
                                eval_yellow_selected += 1
                                raise RuntimeError(f"Eval selected yellow/all-red phase for {tls_id}: {target_phase}")
                            if target_phase != current_phase:
                                phase_change_count += 1

                            conn.trafficlight.setPhase(tls_id, target_phase)
                            conn.trafficlight.setPhaseDuration(tls_id, float(adapter.normalization.decision_interval_s))
                            runtime.current_phase_index = target_phase
                            runtime.phase_enter_time_s = sim_time_s

                        decision_ticks += 1

                    eval_forbidden_calls = int(sum(int(v) for v in forbidden_tracker.counts.values()))

                try:
                    traci_eval.close(wait=True)
                except Exception:
                    pass

        except Exception as exc:
            eval_errors.append(f"{exc}\n{traceback.format_exc()}")
            try:
                traci_eval.close(wait=True)
            except Exception:
                pass

        return_code = int(process.returncode if process is not None and process.returncode is not None else 1)

        trip_ok, trip_err = _parse_xml_ok(outputs["tripinfo_xml"])
        sum_ok, sum_err = _parse_xml_ok(outputs["summary_xml"])
        queue_ok, queue_err = _parse_xml_ok(outputs["queue_xml"])
        tls_ok, tls_err = _parse_xml_ok(outputs["tls_switch_states_xml"])

        trip_metrics = _tripinfo_metrics(outputs["tripinfo_xml"]) if outputs["tripinfo_xml"].exists() else {
            "completed_trip_count": 0,
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
            "created_at_utc": _now_iso(),
            "controller": "independent_dqn_v2",
            "training_scope": "pilot",
            "model_status": "trained_pilot",
            "checkpoint_source": checkpoint_source_rel,
            "scenario_id": scenario_id,
            "network": scenario.network,
            "level": scenario.level,
            "seed": int(scenario.seed),
            "simulation_begin_s": int(scenario.begin_s),
            "simulation_end_s": int(scenario.end_s),
            "validated_horizon_end_s": int(scenario.end_s),
            "decision_step_cap": None,
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
            "training_scope": "pilot",
            "model_status": "trained_pilot",
            "decision_ticks": int(decision_ticks),
            "phase_change_count": int(phase_change_count),
            "invalid_action_count": int(eval_invalid_actions),
            "yellow_all_red_selected_count": int(eval_yellow_selected),
            "forbidden_traci_call_count": int(eval_forbidden_calls),
            "errors": eval_errors,
        }

        _json_dump(outputs["run_manifest_json"], manifest)
        _json_dump(outputs["controller_diagnostics_json"], diagnostics)

        per_run_row = {
            "controller": "independent_dqn_v2",
            "training_scope": "pilot",
            "model_status": "trained_pilot",
            "run_id": run_id,
            "scenario_id": scenario_id,
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
                "scenario_id": scenario_id,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "run_status": run_status,
                "sumo_return_code": int(return_code),
                "model_status": "trained_pilot",
                "invalid_action_count": int(eval_invalid_actions),
                "forbidden_traci_call_count": int(eval_forbidden_calls),
                "horizon_parity_ok": int(manifest["simulation_end_s"]) == int(manifest["validated_horizon_end_s"]),
                "manifest_path": str(outputs["run_manifest_json"]),
            }
        )

    # Extraction outputs
    pilot_results_root.mkdir(parents=True, exist_ok=True)
    per_run_csv = pilot_results_root / f"{network}_per_run.csv"
    per_run_json = pilot_results_root / f"{network}_per_run.json"
    _write_csv(per_run_csv, per_run_rows)
    per_run_json.write_text(json.dumps(per_run_rows, indent=2) + "\n", encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    for level in ["low", "medium", "high"]:
        level_rows = [r for r in per_run_rows if str(r["level"]) == level]
        summary_rows.append(
            {
                "controller": "independent_dqn_v2",
                "training_scope": "pilot",
                "network": network,
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

    summary_csv = pilot_results_root / f"{network}_summary_by_network_level.csv"
    summary_json = pilot_results_root / f"{network}_summary_by_network_level.json"
    _write_csv(summary_csv, summary_rows)
    summary_json.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")

    # Consolidated logs
    pilot_summary = {
        "created_at_utc": _now_iso(),
        "controller": "independent_dqn_v2",
        "training_scope": "pilot",
        "network": network,
        "training": training_summary,
        "evaluation": {
            "runs": eval_results,
            "all_9_passed": all(r["run_status"] == "evaluation_passed" for r in eval_results),
        },
        "extraction": {
            "per_run_rows": len(per_run_rows),
            "summary_rows": len(summary_rows),
            "per_run_csv": str(per_run_csv),
            "per_run_json": str(per_run_json),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
        },
    }

    # Validation checks
    expected_tls_agents = 7 if str(network) == "ingolstadt7" else int(len(adapter.static_cache.tls_specs))
    post_pilot_run_dirs = (
        {p.name for p in pilot_run_root.iterdir() if p.is_dir() and p.name.startswith("independent_dqn_v2_pilot__")}
        if pilot_run_root.exists()
        else set()
    )
    new_pilot_run_dirs = sorted(post_pilot_run_dirs - preexisting_pilot_run_dirs)
    post_main_model_dirs = (
        {p.name for p in DEFAULT_MAIN_MODEL_ROOT.iterdir() if p.is_dir()} if DEFAULT_MAIN_MODEL_ROOT.exists() else set()
    )
    post_main_run_dirs = (
        {p.name for p in DEFAULT_MAIN_RUN_ROOT.iterdir() if p.is_dir() and p.name.startswith("independent_dqn_v2_main__")}
        if DEFAULT_MAIN_RUN_ROOT.exists()
        else set()
    )
    new_main_model_dirs = sorted(post_main_model_dirs - preexisting_main_model_dirs)
    new_main_run_dirs = sorted(post_main_run_dirs - preexisting_main_run_dirs)

    checks = {
        "pilot_trainer_imports": True,
        "v2_model_imports": True,
        "v2_env_adapter_imports": True,
        "torch_imports": bool(torch_status["torch_import_ok"]),
        "frozen_scenarios_resolve": len(scenario_ids) == 9,
        "tls_agents_detected_expected": int(len(adapter.static_cache.tls_specs)) == int(expected_tls_agents),
        "one_agent_per_tls": int(len(agent_ids)) == int(len(adapter.static_cache.tls_specs)),
        "pilot_training_completes_50_episodes": len(episode_summaries) == int(episodes) and all(
            int(ep.get("return_code", 1)) == 0 for ep in episode_summaries
        ),
        "replay_buffer_receives_transitions": int(global_transition_steps) > 0,
        "gradient_updates_occur": int(total_gradient_updates) > 0,
        "losses_finite": bool(losses_finite),
        "target_sync_occurs": int(total_target_sync_updates) > 0,
        "invalid_actions_zero": int(invalid_action_count) == 0,
        "forbidden_traci_calls_zero": int(total_forbidden_calls) == 0,
        "training_metrics_include_wall_clock": all("episode_wall_time_s" in ep for ep in episode_summaries),
        "projected_main_runtime_reported": (
            int(training_summary.get("projected_main_episode_count", 0)) == int(main_episode_budget)
            and training_summary.get("projected_main_episode_runtime_s") is not None
        ),
        "learning_signal_summary_reported": training_summary.get("reward_trend_direction") is not None,
        "all_9_evaluations_pass": all(r["run_status"] == "evaluation_passed" for r in eval_results),
        "full_horizon_parity_passes": all(bool(r["horizon_parity_ok"]) for r in eval_results),
        "extraction_per_run_9_rows": len(per_run_rows) == 9,
        "extraction_summary_3_rows": len(summary_rows) == 3,
        "all_4_kpis_populated": all(
            row["mean_waiting_time_completed_s"] is not None
            and row["throughput_completed_trips"] is not None
            and row["mean_total_queue_length_m"] is not None
            and row["phase_change_rate_per_tls_per_min"] is not None
            for row in summary_rows
        ),
        "no_main_training_launched": len(new_main_model_dirs) == 0 and len(new_main_run_dirs) == 0,
        "no_full_90_run_evaluation_launched": (
            len(new_pilot_run_dirs) == 9 and all(f"__{network}__" in rid for rid in new_pilot_run_dirs)
        ),
    }
    checks["overall_result_pass"] = all(checks.values())

    log_summary_json = REPO_ROOT / f"benchmark/logs/independent_dqn_v2_{network}_pilot_summary.json"
    log_summary_txt = REPO_ROOT / f"benchmark/logs/independent_dqn_v2_{network}_pilot_summary.txt"
    log_validation_txt = REPO_ROOT / f"benchmark/logs/independent_dqn_v2_{network}_pilot_validation.txt"

    _json_dump(log_summary_json, {"pilot_summary": pilot_summary, "checks": checks, "torch": torch_status})

    summary_lines = [
        f"Independent DQN v2 {network} Pilot Summary",
        "",
        f"Created (UTC): {_now_iso()}",
        f"Network: {network}",
        f"Episodes: {episodes}",
        f"Max steps/episode: {max_steps_per_episode}",
        f"Total wall-clock (s): {training_summary['total_wall_clock_s']}",
        f"Avg seconds/episode: {training_summary['average_seconds_per_episode']}",
        f"Projected {training_summary['projected_main_episode_count']}-episode runtime (s): {training_summary['projected_main_episode_runtime_s']}",
        f"Projected {training_summary['projected_main_episode_count']}-episode runtime (h): {training_summary['projected_main_episode_runtime_h']}",
        f"Wall-clock cap minutes: {training_summary['wall_clock_cap_minutes']}",
        f"Wall-clock cap exceeded: {training_summary['wall_clock_cap_exceeded']}",
        f"Gradient updates: {training_summary['total_gradient_updates']}",
        f"Per-agent gradient updates: {training_summary['gradient_updates_by_agent']}",
        f"Target sync total: {training_summary['target_sync_total']}",
        f"Forbidden TraCI call count: {training_summary['forbidden_traci_call_count']}",
        f"Reward first 10 mean: {training_summary['reward_first_10_mean']}",
        f"Reward last 10 mean: {training_summary['reward_last_10_mean']}",
        f"Reward trend: {training_summary['reward_trend_direction']} (delta={training_summary['reward_trend_delta']}, delta_pct={training_summary['reward_trend_delta_pct']})",
        f"Loss first window mean: {training_summary['loss_first_updating_window_mean']}",
        f"Loss last window mean: {training_summary['loss_last_updating_window_mean']}",
        f"9-scenario evaluation all passed: {pilot_summary['evaluation']['all_9_passed']}",
        f"Overall validation: {'PASS' if checks['overall_result_pass'] else 'FAIL'}",
    ]
    log_summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    vlines = [
        f"Independent DQN v2 {network} Pilot Validation",
        "",
        f"1. pilot trainer imports: {'PASS' if checks['pilot_trainer_imports'] else 'FAIL'}",
        f"2. v2 model imports: {'PASS' if checks['v2_model_imports'] else 'FAIL'}",
        f"3. v2 env adapter imports: {'PASS' if checks['v2_env_adapter_imports'] else 'FAIL'}",
        f"4. Torch imports: {'PASS' if checks['torch_imports'] else 'FAIL'}",
        f"5. frozen scenarios resolve: {'PASS' if checks['frozen_scenarios_resolve'] else 'FAIL'}",
        f"6. {expected_tls_agents} TLS agents are detected: {'PASS' if checks['tls_agents_detected_expected'] else 'FAIL'}",
        f"7. one independent DQN agent is created per TLS: {'PASS' if checks['one_agent_per_tls'] else 'FAIL'}",
        f"8. pilot training completes 50 episodes: {'PASS' if checks['pilot_training_completes_50_episodes'] else 'FAIL'}",
        f"9. replay buffers receive transitions: {'PASS' if checks['replay_buffer_receives_transitions'] else 'FAIL'}",
        f"10. gradient updates occur: {'PASS' if checks['gradient_updates_occur'] else 'FAIL'}",
        f"11. losses are finite: {'PASS' if checks['losses_finite'] else 'FAIL'}",
        f"12. target sync occurs: {'PASS' if checks['target_sync_occurs'] else 'FAIL'}",
        f"13. invalid actions = 0: {'PASS' if checks['invalid_actions_zero'] else 'FAIL'}",
        f"14. forbidden TraCI calls = 0: {'PASS' if checks['forbidden_traci_calls_zero'] else 'FAIL'}",
        f"15. training metrics include wall-clock timing: {'PASS' if checks['training_metrics_include_wall_clock'] else 'FAIL'}",
        f"16. projected {main_episode_budget}-episode runtime is reported: {'PASS' if checks['projected_main_runtime_reported'] else 'FAIL'}",
        f"17. learning-signal summary is reported: {'PASS' if checks['learning_signal_summary_reported'] else 'FAIL'}",
        f"18. all 9 frozen {network} evaluations pass: {'PASS' if checks['all_9_evaluations_pass'] else 'FAIL'}",
        f"19. full horizon parity passes: {'PASS' if checks['full_horizon_parity_passes'] else 'FAIL'}",
        f"20. extraction per-run table has 9 rows: {'PASS' if checks['extraction_per_run_9_rows'] else 'FAIL'}",
        f"21. extraction summary table has 3 rows: {'PASS' if checks['extraction_summary_3_rows'] else 'FAIL'}",
        f"22. all 4 KPIs populated: {'PASS' if checks['all_4_kpis_populated'] else 'FAIL'}",
        f"23. no main {main_episode_budget}-episode training launched: {'PASS' if checks['no_main_training_launched'] else 'FAIL'}",
        f"24. no full 90-run evaluation launched: {'PASS' if checks['no_full_90_run_evaluation_launched'] else 'FAIL'}",
        f"25. overall result = {'PASS' if checks['overall_result_pass'] else 'FAIL'}",
        "",
        f"Overall: {'PASS' if checks['overall_result_pass'] else 'FAIL'}",
    ]
    log_validation_txt.write_text("\n".join(vlines) + "\n", encoding="utf-8")

    return {
        "training_model_dir": str(network_model_dir),
        "pilot_runs_root": str(pilot_run_root),
        "pilot_results_root": str(pilot_results_root),
        "log_summary_json": str(log_summary_json),
        "log_validation_txt": str(log_validation_txt),
        "checks": checks,
        "pilot_summary": pilot_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent DQN v2 single-network pilot runner.")
    parser.add_argument("--network", default=DEFAULT_NETWORK)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE)
    parser.add_argument("--train-every-steps", type=int, default=8)
    parser.add_argument("--learning-starts-steps", type=int, default=1000)
    parser.add_argument("--wall-clock-cap-minutes", type=float, default=DEFAULT_WALL_CLOCK_CAP_MINUTES)
    parser.add_argument("--bank-manifest", type=Path, default=DEFAULT_BANK_MANIFEST)
    parser.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--pilot-model-root", type=Path, default=DEFAULT_PILOT_MODEL_ROOT)
    parser.add_argument("--pilot-run-root", type=Path, default=DEFAULT_PILOT_RUN_ROOT)
    parser.add_argument("--pilot-results-root", type=Path, default=DEFAULT_PILOT_RESULTS_ROOT)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    # Enforce contract settings for this pilot.
    if int(args.train_every_steps) != 8 or int(args.learning_starts_steps) != 1000:
        raise RuntimeError("Pilot must use train_every_steps=8 and learning_starts_steps=1000")

    result = run_pilot(
        network=str(args.network),
        episodes=int(args.episodes),
        max_steps_per_episode=int(args.max_steps_per_episode),
        wall_clock_cap_minutes=float(args.wall_clock_cap_minutes),
        bank_manifest_path=args.bank_manifest,
        contract_path=args.contract_path,
        pilot_model_root=args.pilot_model_root,
        pilot_run_root=args.pilot_run_root,
        pilot_results_root=args.pilot_results_root,
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
