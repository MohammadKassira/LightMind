from __future__ import annotations

import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.controllers.independent_dqn_v2.env_adapter import (
    DEFAULT_CONTRACT_PATH,
    ForbiddenCallTracker,
    IndependentDQNV2EnvAdapter,
    ScenarioInputs,
    _allocate_traci_port,
    _ensure_traci_import,
)
from models.independent_dqn_v2 import AgentSpec, DQNAgentConfig, IndependentDQNController, safe_agent_filename

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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def train_independent_dqn_from_scratch(
    *,
    train_scenario: ScenarioSpec,
    output_dir: Path,
    episodes: int,
    max_steps_per_episode: int,
    seed: int,
    train_every_steps: int,
    learning_starts_steps: int,
    target_update_interval: int,
    wall_clock_cap_minutes: float,
    training_label: str,
) -> dict[str, Any]:
    stage = "independent_dqn_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_root = output_dir / "training_rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(seed))
    torch_status = _torch_readiness()
    if not bool(torch_status.get("torch_import_ok", False)):
        raise WebIntegrationError(stage, f"Torch import failed: {torch_status.get('error')}")

    adapter = IndependentDQNV2EnvAdapter(contract_path=DEFAULT_CONTRACT_PATH)
    reward_version = str(adapter.contract.get("reward_version", ""))
    if reward_version != "aggregate_local_efficient_pressure_v2":
        raise WebIntegrationError(
            stage,
            f"Contract reward mismatch: expected aggregate_local_efficient_pressure_v2, got {reward_version}",
        )

    traci = _ensure_traci_import()
    scenario = _as_adapter_scenario(train_scenario)

    controller: IndependentDQNController | None = None
    agent_ids: list[str] = []

    metrics_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []

    global_transition_steps = 0
    total_gradient_updates = 0
    total_target_sync_updates = 0
    invalid_action_count = 0
    yellow_all_red_selected_count = 0
    total_forbidden_calls = 0
    total_phase_change_count = 0
    replay_size_max = 0
    losses_finite = True

    train_wall_start = time.perf_counter()
    wall_clock_cap_exceeded = False
    wall_clock_cap_exceeded_at_episode: int | None = None

    for ep in range(1, int(episodes) + 1):
        if wall_clock_cap_exceeded:
            break

        ep_wall_start = time.perf_counter()
        ep_dir = rollout_root / f"episode_{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        cmd, outputs = adapter._build_traci_command(
            scenario,
            ep_dir,
            include_xml_outputs=False,
            suppress_warnings=True,
        )
        label = f"{training_label}_{int(time.time() * 1_000_000)}_{ep}"
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
                                "target_update_interval": int(target_update_interval),
                                "epsilon_start": 1.0,
                                "epsilon_end": 0.05,
                                "epsilon_decay_steps": 10_000,
                                "train_every_steps": int(train_every_steps),
                                "learning_starts_steps": int(learning_starts_steps),
                            },
                        )
                    if not specs:
                        raise WebIntegrationError(stage, "No controllable TLS found during training setup.")
                    controller = IndependentDQNController(agent_specs=specs, default_config=DQNAgentConfig(seed=seed))
                    agent_ids = sorted(specs.keys())

                if controller is None:
                    raise WebIntegrationError(stage, "Controller initialization failed.")

                next_decision_time_s = float(scenario.begin_s) + float(adapter.normalization.decision_interval_s)

                with ForbiddenCallTracker(conn) as forbidden_tracker:
                    while float(conn.simulation.getTime()) < float(scenario.end_s) and ep_decision_steps < int(
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

                        for tls_id, tls_spec in adapter.static_cache.tls_specs.items():
                            obs, rew = adapter._build_observation_and_reward(conn, tls_id, sim_time_s)
                            mask = adapter._build_action_mask(tls_id)

                            if obs.shape != (int(tls_spec.obs_dim),):
                                raise WebIntegrationError(
                                    stage,
                                    f"Observation shape mismatch for {tls_id}: {obs.shape} vs {(int(tls_spec.obs_dim),)}",
                                )
                            if mask.shape != (len(tls_spec.action_phase_indices),):
                                raise WebIntegrationError(
                                    stage,
                                    f"Action mask mismatch for {tls_id}: {mask.shape} vs {(len(tls_spec.action_phase_indices),)}",
                                )

                            observations[tls_id] = obs
                            action_masks[tls_id] = mask
                            rewards[tls_id] = float(rew)
                            ep_queue_proxy_sum += float(np.sum(obs[len(tls_spec.phases) + 1 :]))

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
                                global_transition_steps >= int(learning_starts_steps)
                                and global_transition_steps % int(train_every_steps) == 0
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
                                raise WebIntegrationError(stage, f"Invalid masked action selected for {tls_id}: {action_idx}")

                            tls_spec = adapter.static_cache.tls_specs[tls_id]
                            runtime = adapter.static_cache.tls_runtime[tls_id]
                            current_phase = int(runtime.current_phase_index)
                            target_phase = int(tls_spec.action_phase_indices[int(action_idx)])

                            if target_phase in set(tls_spec.yellow_all_red_phase_indices):
                                yellow_all_red_selected_count += 1
                                raise WebIntegrationError(stage, f"Selected yellow/all-red phase for {tls_id}: {target_phase}")

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
                            global_transition_steps >= int(learning_starts_steps)
                            and global_transition_steps % int(train_every_steps) == 0
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
                        float(ep_queue_proxy_sum / max(ep_decision_steps, 1)) if ep_decision_steps > 0 else None
                    ),
                    "episode_phase_change_count": int(ep_phase_change_count),
                    "episode_phase_change_rate": (
                        float(ep_phase_change_count / max(ep_decision_steps, 1)) if ep_decision_steps > 0 else None
                    ),
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
        raise WebIntegrationError(stage, "Training failed before controller initialization.")

    controller_ckpt = output_dir / "controller.pt"
    controller.save(controller_ckpt)
    per_agent_ckpts: list[str] = []
    for aid in agent_ids:
        p = output_dir / safe_agent_filename(aid)
        controller.agents[aid].save(p)
        per_agent_ckpts.append(str(p))

    training_metrics_csv = output_dir / "training_metrics.csv"
    training_metrics_json = output_dir / "training_metrics.json"
    csv_dump(training_metrics_csv, metrics_rows)
    training_metrics_json.write_text(__import__("json").dumps(metrics_rows, indent=2) + "\n", encoding="utf-8")

    total_wall_s = max(time.perf_counter() - train_wall_start, 0.0)
    summary = {
        "created_at_utc": now_iso(),
        "controller": "independent_dqn_v2",
        "training_scope": training_label,
        "reward_version": reward_version,
        "observation_contract": str(adapter.contract.get("observation_version", "")),
        "scenario_id": train_scenario.scenario_id,
        "episodes_requested": int(episodes),
        "episodes_completed": len(episode_summaries),
        "max_steps_per_episode": int(max_steps_per_episode),
        "train_every_steps": int(train_every_steps),
        "learning_starts_steps": int(learning_starts_steps),
        "target_update_interval": int(target_update_interval),
        "tls_agent_count": len(agent_ids),
        "agent_ids": agent_ids,
        "total_wall_clock_s": float(total_wall_s),
        "wall_clock_cap_minutes": float(wall_clock_cap_minutes),
        "wall_clock_cap_exceeded": bool(wall_clock_cap_exceeded),
        "wall_clock_cap_exceeded_at_episode": wall_clock_cap_exceeded_at_episode,
        "total_transitions": int(global_transition_steps),
        "replay_buffer_size_max": int(replay_size_max),
        "total_gradient_updates": int(total_gradient_updates),
        "target_sync_total": int(total_target_sync_updates),
        "losses_finite": bool(losses_finite),
        "invalid_action_count": int(invalid_action_count),
        "yellow_all_red_selected_count": int(yellow_all_red_selected_count),
        "forbidden_traci_call_count": int(total_forbidden_calls),
        "phase_change_count_total": int(total_phase_change_count),
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

    checks = {
        "reward_is_aggregate_local_efficient_pressure_v2": reward_version == "aggregate_local_efficient_pressure_v2",
        "training_started": len(episode_summaries) > 0,
        "controller_checkpoint_written": controller_ckpt.exists(),
        "per_agent_checkpoints_written": all(Path(p).exists() for p in per_agent_ckpts),
        "forbidden_traci_calls_zero": int(total_forbidden_calls) == 0,
        "invalid_actions_zero": int(invalid_action_count) == 0,
        "yellow_all_red_selected_zero": int(yellow_all_red_selected_count) == 0,
    }
    checks["overall_pass"] = all(bool(v) for v in checks.values())

    summary_path = output_dir / "training_summary.json"
    manifest_path = output_dir / "training_manifest.json"
    json_dump(summary_path, summary)
    json_dump(
        manifest_path,
        {
            "created_at_utc": now_iso(),
            "controller": "independent_dqn_v2",
            "training_scope": training_label,
            "scenario_id": train_scenario.scenario_id,
            "checkpoint_source": "none",
            "episodes_requested": int(episodes),
            "episodes_completed": len(episode_summaries),
            "max_steps_per_episode": int(max_steps_per_episode),
            "train_every_steps": int(train_every_steps),
            "learning_starts_steps": int(learning_starts_steps),
            "reward_version": reward_version,
            "output_files": {
                "summary_json": str(summary_path),
                "manifest_json": str(manifest_path),
                "metrics_csv": str(training_metrics_csv),
                "metrics_json": str(training_metrics_json),
                "controller_checkpoint": str(controller_ckpt),
                "per_agent_checkpoints": per_agent_ckpts,
            },
            "checks": checks,
        },
    )

    return {
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "summary": summary,
        "checks": checks,
        "controller_checkpoint": str(controller_ckpt),
    }
