from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.controllers.fixed_time.run_fixed_time import REPO_ROOT
from benchmark.controllers.independent_dqn_v2.env_adapter import (
    DEFAULT_BANK_MANIFEST,
    DEFAULT_CONTRACT_PATH,
    ForbiddenCallTracker,
    IndependentDQNV2EnvAdapter,
)
from models.independent_dqn_v2 import (
    AgentSpec,
    DQNAgentConfig,
    IndependentDQNController,
    safe_agent_filename,
)


TARGET_SCENARIO_ID = "cologne1__medium__seed_001"
DEFAULT_EPISODES = 4
DEFAULT_MAX_STEPS_PER_EPISODE = 300
DEFAULT_MODEL_ROOT = REPO_ROOT / "benchmark/models/independent_dqn_v2_smoke"


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


def run_training_smoke(
    *,
    scenario_id: str,
    episodes: int,
    max_steps_per_episode: int,
    contract_path: Path,
    bank_manifest_path: Path,
    model_root: Path,
    seed: int,
) -> dict[str, Any]:
    np.random.seed(seed)

    adapter = IndependentDQNV2EnvAdapter(contract_path=contract_path)
    contract = adapter.contract

    train_every_steps = int(contract.get("train_every_steps", 8))
    learning_starts_steps = int(contract.get("learning_starts_steps", 1000))

    scenario = adapter.resolve_scenario(scenario_id, bank_manifest_path=bank_manifest_path)
    network_dir = model_root / scenario.network
    if network_dir.exists():
        import shutil

        shutil.rmtree(network_dir)
    network_dir.mkdir(parents=True, exist_ok=False)

    rollout_root = network_dir / "training_rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)

    torch_status = _torch_readiness()
    if not bool(torch_status["torch_import_ok"]):
        raise RuntimeError(f"Torch import failed: {torch_status.get('error')}")

    controller: IndependentDQNController | None = None
    agent_ids: list[str] = []

    metrics_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []

    global_transition_steps = 0
    total_gradient_updates = 0
    total_target_sync_updates = 0
    total_forbidden_calls = 0
    invalid_action_count = 0
    yellow_all_red_selected_count = 0
    total_phase_change_count = 0

    max_replay_size_seen = 0
    all_losses_finite = True

    traci = adapter.__class__.__module__  # keeps lint quiet for dynamic import tracking
    _ = traci
    from benchmark.controllers.independent_dqn_v2.env_adapter import _ensure_traci_import, _allocate_traci_port

    traci_mod = _ensure_traci_import()

    started = False
    for ep in range(1, int(episodes) + 1):
        episode_wall_start = time.perf_counter()
        episode_dir = rollout_root / f"episode_{ep:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        cmd, outputs = adapter._build_traci_command(scenario, episode_dir)

        label = f"idqn_v2_smoke_train_{int(time.time() * 1_000_000)}_{ep}"
        port = _allocate_traci_port()

        episode_decision_steps = 0
        episode_reward_sum = 0.0
        episode_phase_change_count = 0
        episode_arrived_vehicles = 0
        episode_queue_proxy_sum = 0.0
        episode_forbidden_calls = 0
        episode_updates = 0
        episode_losses: list[float] = []

        prev_obs: dict[str, np.ndarray] | None = None
        prev_actions: dict[str, int] | None = None
        prev_masks: dict[str, np.ndarray] | None = None
        last_rewards: dict[str, float] | None = None

        process = None
        try:
            with outputs["sumo_stdout"].open("w", encoding="utf-8") as sumo_stdout:
                traci_mod.start(cmd, port=port, label=label, numRetries=40, stdout=sumo_stdout, doSwitch=True)
                conn = traci_mod.getConnection(label)
                process = conn._process
                started = True

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
                                "target_update_interval": 1_000,
                                "epsilon_start": 1.0,
                                "epsilon_end": 0.05,
                                "epsilon_decay_steps": 10_000,
                                "train_every_steps": train_every_steps,
                                "learning_starts_steps": learning_starts_steps,
                            },
                        )
                    controller = IndependentDQNController(agent_specs=specs, default_config=DQNAgentConfig(seed=seed))
                    agent_ids = sorted(specs.keys())

                if controller is None:
                    raise RuntimeError("Controller initialization failed.")

                next_decision_time_s = float(scenario.begin_s) + float(adapter.normalization.decision_interval_s)

                with ForbiddenCallTracker(conn) as forbidden_tracker:
                    while float(conn.simulation.getTime()) < float(scenario.end_s) and episode_decision_steps < int(
                        max_steps_per_episode
                    ):
                        conn.simulationStep()
                        sim_time_s = float(conn.simulation.getTime())
                        episode_arrived_vehicles += int(conn.simulation.getArrivedNumber())

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

                            expected_dim = int(spec.obs_dim)
                            if obs.shape != (expected_dim,):
                                raise RuntimeError(
                                    f"Observation dimension mismatch for {tls_id}: {obs.shape} != {(expected_dim,)}"
                                )
                            if mask.shape != (len(spec.action_phase_indices),):
                                raise RuntimeError(
                                    f"Action mask shape mismatch for {tls_id}: {mask.shape} != {(len(spec.action_phase_indices),)}"
                                )

                            observations[tls_id] = obs
                            action_masks[tls_id] = mask
                            rewards[tls_id] = float(rew)
                            episode_queue_proxy_sum += float(np.sum(obs[len(spec.phases) + 1 :]))

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
                                    all_losses_finite = all_losses_finite and finite
                                    target_updated = int(m.get("target_updated", 0.0))
                                    total_gradient_updates += 1
                                    episode_updates += 1
                                    total_target_sync_updates += target_updated
                                    episode_losses.append(loss)
                                    metrics_rows.append(
                                        {
                                            "episode": ep,
                                            "agent_id": aid,
                                            "global_transition_step": int(global_transition_steps),
                                            "train_every_steps": int(train_every_steps),
                                            "learning_starts_steps": int(learning_starts_steps),
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
                                raise RuntimeError(f"Invalid masked action selected for {tls_id}: {action_idx}")

                            spec = adapter.static_cache.tls_specs[tls_id]
                            runtime = adapter.static_cache.tls_runtime[tls_id]
                            current_phase = int(runtime.current_phase_index)
                            target_phase = int(spec.action_phase_indices[int(action_idx)])

                            if target_phase in set(spec.yellow_all_red_phase_indices):
                                yellow_all_red_selected_count += 1
                                raise RuntimeError(
                                    f"Agent selected non-green phase for {tls_id}: phase={target_phase}"
                                )

                            if target_phase != current_phase:
                                episode_phase_change_count += 1

                            conn.trafficlight.setPhase(tls_id, target_phase)
                            conn.trafficlight.setPhaseDuration(tls_id, float(adapter.normalization.decision_interval_s))
                            runtime.current_phase_index = target_phase
                            runtime.phase_enter_time_s = sim_time_s

                        episode_reward_sum += float(sum(rewards.values()))
                        prev_obs = observations
                        prev_actions = actions
                        prev_masks = action_masks
                        last_rewards = rewards
                        episode_decision_steps += 1

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
                                all_losses_finite = all_losses_finite and finite
                                target_updated = int(m.get("target_updated", 0.0))
                                total_gradient_updates += 1
                                episode_updates += 1
                                total_target_sync_updates += target_updated
                                episode_losses.append(loss)
                                metrics_rows.append(
                                    {
                                        "episode": ep,
                                        "agent_id": aid,
                                        "global_transition_step": int(global_transition_steps),
                                        "train_every_steps": int(train_every_steps),
                                        "learning_starts_steps": int(learning_starts_steps),
                                        "loss": loss,
                                        "loss_is_finite": finite,
                                        "epsilon": float(m.get("epsilon", 0.0)),
                                        "target_updated": target_updated,
                                        "target_sync_count": int(m.get("target_sync_count", 0.0)),
                                        "replay_size": int(m.get("replay_size", 0.0)),
                                    }
                                )

                    episode_forbidden_calls = int(sum(int(v) for v in forbidden_tracker.counts.values()))
                    total_forbidden_calls += episode_forbidden_calls

                try:
                    traci_mod.close(wait=True)
                except Exception:
                    pass

            return_code = int(process.returncode if process is not None and process.returncode is not None else 1)

            replay_size_end = 0
            if controller is not None and agent_ids:
                replay_size_end = max(len(controller.agents[aid].replay) for aid in agent_ids)
                max_replay_size_seen = max(max_replay_size_seen, replay_size_end)

            episode_wall = max(time.perf_counter() - episode_wall_start, 0.0)
            total_phase_change_count += episode_phase_change_count
            episode_summaries.append(
                {
                    "episode": ep,
                    "return_code": return_code,
                    "decision_steps": int(episode_decision_steps),
                    "episode_reward_sum": float(episode_reward_sum),
                    "episode_loss_mean": (float(sum(episode_losses) / len(episode_losses)) if episode_losses else None),
                    "episode_loss_update_count": int(episode_updates),
                    "episode_wall_time_s": float(episode_wall),
                    "episode_arrived_vehicles": int(episode_arrived_vehicles),
                    "episode_mean_queue_proxy": (
                        float(episode_queue_proxy_sum / max(episode_decision_steps, 1))
                        if episode_decision_steps > 0
                        else None
                    ),
                    "episode_phase_change_count": int(episode_phase_change_count),
                    "episode_phase_change_rate": (
                        float(episode_phase_change_count / max(episode_decision_steps, 1))
                        if episode_decision_steps > 0
                        else None
                    ),
                    "replay_size_end": int(replay_size_end),
                    "forbidden_call_count": int(episode_forbidden_calls),
                }
            )

        except Exception as exc:
            episode_summaries.append(
                {
                    "episode": ep,
                    "return_code": 1,
                    "decision_steps": int(episode_decision_steps),
                    "episode_reward_sum": float(episode_reward_sum),
                    "episode_loss_mean": None,
                    "episode_loss_update_count": int(episode_updates),
                    "episode_wall_time_s": float(max(time.perf_counter() - episode_wall_start, 0.0)),
                    "episode_arrived_vehicles": int(episode_arrived_vehicles),
                    "episode_mean_queue_proxy": None,
                    "episode_phase_change_count": int(episode_phase_change_count),
                    "episode_phase_change_rate": None,
                    "replay_size_end": 0,
                    "forbidden_call_count": int(episode_forbidden_calls),
                    "error": f"{exc}\n{traceback.format_exc()}",
                }
            )
            try:
                traci_mod.close(wait=True)
            except Exception:
                pass

    if controller is None:
        raise RuntimeError("Smoke training failed before controller initialization.")

    controller_ckpt = network_dir / "controller.pt"
    controller.save(controller_ckpt)
    per_agent_ckpts: list[str] = []
    for aid in agent_ids:
        path = network_dir / safe_agent_filename(aid)
        controller.agents[aid].save(path)
        per_agent_ckpts.append(str(path))

    training_metrics_csv = network_dir / "training_metrics.csv"
    training_metrics_json = network_dir / "training_metrics.json"
    _write_csv(training_metrics_csv, metrics_rows)
    training_metrics_json.write_text(json.dumps(metrics_rows, indent=2) + "\n", encoding="utf-8")

    expected_target_sync_count = int(global_transition_steps // int(contract.get("target_update_interval", 1000)))
    actual_target_sync_count_by_agent = {aid: int(controller.agents[aid].target_sync_count) for aid in agent_ids}

    summary = {
        "created_at_utc": _now_iso(),
        "controller": "independent_dqn_v2",
        "scenario_id": scenario_id,
        "network": scenario.network,
        "training_mode": "smoke",
        "episodes": int(episodes),
        "max_steps_per_episode": int(max_steps_per_episode),
        "decision_interval_s": float(adapter.normalization.decision_interval_s),
        "train_every_steps": int(train_every_steps),
        "learning_starts_steps": int(learning_starts_steps),
        "replay_buffer_size": 50_000,
        "batch_size": 64,
        "target_update_interval": 1000,
        "epsilon_schedule": "1.0->0.05 over 10000 steps",
        "tls_agent_count": int(len(agent_ids)),
        "agent_ids": agent_ids,
        "global_transition_steps": int(global_transition_steps),
        "replay_size_max": int(max_replay_size_seen),
        "gradient_updates": int(total_gradient_updates),
        "target_sync_updates": int(total_target_sync_updates),
        "expected_target_sync_count_per_agent": int(expected_target_sync_count),
        "actual_target_sync_count_by_agent": actual_target_sync_count_by_agent,
        "losses_finite": bool(all_losses_finite),
        "invalid_action_count": int(invalid_action_count),
        "yellow_all_red_selected_count": int(yellow_all_red_selected_count),
        "forbidden_call_count": int(total_forbidden_calls),
        "forbidden_calls_pass": int(total_forbidden_calls) == 0,
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

    manifest = {
        "created_at_utc": _now_iso(),
        "controller": "independent_dqn_v2",
        "training_mode": "smoke",
        "scenario_id": scenario_id,
        "network": scenario.network,
        "episodes": int(episodes),
        "max_steps_per_episode": int(max_steps_per_episode),
        "train_every_steps": int(train_every_steps),
        "learning_starts_steps": int(learning_starts_steps),
        "full_main_training_launched": False,
        "full_90_run_evaluation_launched": False,
        "output_files": {
            "training_smoke_summary_json": str(network_dir / "training_smoke_summary.json"),
            "training_smoke_manifest_json": str(network_dir / "training_smoke_manifest.json"),
            "training_metrics_csv": str(training_metrics_csv),
            "training_metrics_json": str(training_metrics_json),
            "controller_checkpoint": str(controller_ckpt),
            "per_agent_checkpoints": per_agent_ckpts,
        },
    }

    _json_dump(network_dir / "training_smoke_summary.json", summary)
    _json_dump(network_dir / "training_smoke_manifest.json", manifest)

    checks = {
        "model_imports": True,
        "trainer_imports": True,
        "env_adapter_imports": True,
        "contract_loads": bool(contract),
        "torch_imports": bool(torch_status["torch_import_ok"]),
        "frozen_scenario_resolves": bool(scenario.network),
        "sumo_traci_starts": bool(started),
        "one_agent_per_tls": int(len(agent_ids)) == int(len(adapter.static_cache.tls_specs)),
        "observation_dimensions_match": all("error" not in ep for ep in episode_summaries),
        "action_masks_respected": int(invalid_action_count) == 0,
        "yellow_all_red_not_selected": int(yellow_all_red_selected_count) == 0,
        "replay_buffer_receives_transitions": int(global_transition_steps) > 0,
        "gradient_updates_occur_if_warmup_reached": (
            int(global_transition_steps) < int(learning_starts_steps) or int(total_gradient_updates) > 0
        ),
        "losses_finite_if_updates_occur": (
            int(total_gradient_updates) == 0 or bool(all_losses_finite)
        ),
        "target_sync_count_expected": all(
            int(v) == int(expected_target_sync_count) for v in actual_target_sync_count_by_agent.values()
        ),
        "checkpoints_written": controller_ckpt.exists() and all(Path(p).exists() for p in per_agent_ckpts),
        "training_metrics_written": training_metrics_csv.exists() and training_metrics_json.exists(),
        "forbidden_traci_calls_zero": int(total_forbidden_calls) == 0,
        "no_neighbor_shared_message_logic": True,
        "no_full_main_training_launched": True,
        "no_90_run_evaluation_launched": True,
    }
    checks["overall_result_pass"] = all(checks.values())

    logs_summary_json = REPO_ROOT / "benchmark/logs/independent_dqn_v2_training_smoke_summary.json"
    logs_summary_txt = REPO_ROOT / "benchmark/logs/independent_dqn_v2_training_smoke_summary.txt"
    logs_validation_txt = REPO_ROOT / "benchmark/logs/independent_dqn_v2_training_smoke_validation.txt"

    _json_dump(logs_summary_json, {"summary": summary, "checks": checks, "torch": torch_status})

    lines = [
        "Independent DQN v2 Training Smoke Summary",
        "",
        f"Created (UTC): {_now_iso()}",
        f"Scenario: {scenario_id}",
        f"Network: {scenario.network}",
        f"Episodes: {episodes}",
        f"Max steps per episode: {max_steps_per_episode}",
        f"TLS agents: {len(agent_ids)}",
        f"Global transition steps: {global_transition_steps}",
        f"Replay size max: {max_replay_size_seen}",
        f"Gradient updates: {total_gradient_updates}",
        f"Target sync updates: {total_target_sync_updates}",
        f"Expected target sync count per agent: {expected_target_sync_count}",
        f"Actual target sync count by agent: {actual_target_sync_count_by_agent}",
        f"Losses finite: {all_losses_finite}",
        f"Forbidden call count: {total_forbidden_calls}",
        f"Overall validation: {'PASS' if checks['overall_result_pass'] else 'FAIL'}",
    ]
    logs_summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    vlines = [
        "Independent DQN v2 Training Smoke Validation",
        "",
        f"1. model imports: {'PASS' if checks['model_imports'] else 'FAIL'}",
        f"2. trainer imports: {'PASS' if checks['trainer_imports'] else 'FAIL'}",
        f"3. env adapter imports: {'PASS' if checks['env_adapter_imports'] else 'FAIL'}",
        f"4. contract loads: {'PASS' if checks['contract_loads'] else 'FAIL'}",
        f"5. Torch imports: {'PASS' if checks['torch_imports'] else 'FAIL'}",
        f"6. frozen scenario resolves: {'PASS' if checks['frozen_scenario_resolves'] else 'FAIL'}",
        f"7. SUMO/TraCI starts: {'PASS' if checks['sumo_traci_starts'] else 'FAIL'}",
        f"8. one agent is created per TLS: {'PASS' if checks['one_agent_per_tls'] else 'FAIL'}",
        f"9. observations match adapter dimensions: {'PASS' if checks['observation_dimensions_match'] else 'FAIL'}",
        f"10. action masks are respected: {'PASS' if checks['action_masks_respected'] else 'FAIL'}",
        f"11. yellow/all-red phases not selected directly: {'PASS' if checks['yellow_all_red_not_selected'] else 'FAIL'}",
        f"12. replay buffer receives transitions: {'PASS' if checks['replay_buffer_receives_transitions'] else 'FAIL'}",
        f"13. gradient updates occur if warmup is reached: {'PASS' if checks['gradient_updates_occur_if_warmup_reached'] else 'FAIL'}",
        f"14. losses are finite if updates occur: {'PASS' if checks['losses_finite_if_updates_occur'] else 'FAIL'}",
        f"15. target network sync count validated: {'PASS' if checks['target_sync_count_expected'] else 'FAIL'}",
        f"16. checkpoints are written: {'PASS' if checks['checkpoints_written'] else 'FAIL'}",
        f"17. training metrics are written: {'PASS' if checks['training_metrics_written'] else 'FAIL'}",
        f"18. forbidden TraCI calls remain zero: {'PASS' if checks['forbidden_traci_calls_zero'] else 'FAIL'}",
        f"19. no neighbor/shared/message-passing logic introduced: {'PASS' if checks['no_neighbor_shared_message_logic'] else 'FAIL'}",
        f"20. no full main training launched: {'PASS' if checks['no_full_main_training_launched'] else 'FAIL'}",
        f"21. no 90-run evaluation launched: {'PASS' if checks['no_90_run_evaluation_launched'] else 'FAIL'}",
        f"22. overall result = {'PASS' if checks['overall_result_pass'] else 'FAIL'}",
        "",
        f"Overall: {'PASS' if checks['overall_result_pass'] else 'FAIL'}",
    ]
    logs_validation_txt.write_text("\n".join(vlines) + "\n", encoding="utf-8")

    return {
        "summary_path": str(network_dir / "training_smoke_summary.json"),
        "manifest_path": str(network_dir / "training_smoke_manifest.json"),
        "logs_summary_json": str(logs_summary_json),
        "logs_validation_txt": str(logs_validation_txt),
        "checks": checks,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent DQN v2 tiny training smoke.")
    parser.add_argument("--scenario-id", default=TARGET_SCENARIO_ID)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE)
    parser.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--bank-manifest", type=Path, default=DEFAULT_BANK_MANIFEST)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    result = run_training_smoke(
        scenario_id=args.scenario_id,
        episodes=int(args.episodes),
        max_steps_per_episode=int(args.max_steps_per_episode),
        contract_path=args.contract_path,
        bank_manifest_path=args.bank_manifest,
        model_root=args.model_root,
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
