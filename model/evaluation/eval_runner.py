"""Greedy policy evaluation for DQNTrainer (R4+).

Works with both MockEnv and TrafficEnv.  When running against TrafficEnv,
step info contains per-step vehicle metrics (mean_waiting_time, throughput,
num_vehicles) that are accumulated into episode records for compute_metrics().
"""

import torch

from env.perception import apply_perception
from models.node_encoder import pad_obs_dict
from models.phase_head import pad_phase_features
from training.reward import ObservationImputer, PressureReward
from evaluation.metrics import compute_metrics


def evaluate(
    trainer,
    env,
    num_episodes: int = 10,
    perception_severity: float = 0.0,
    use_pressure: bool = True,
    sentinel: float = -1.0,
    seed: int | None = None,
) -> dict:
    """Run the trained policy greedily and return a metrics summary.

    Args:
        trainer:              DQNTrainer with trained encoder/gat/head
        env:                  MockEnv or TrafficEnv
        num_episodes:         Number of evaluation episodes
        perception_severity:  Sensor corruption rate (match training value)
        use_pressure:         Whether to compute pressure reward for return logging
        sentinel:             Sentinel value used during training
        seed:                 RNG seed for first episode (subsequent are random)

    Returns:
        metrics dict from compute_metrics() plus raw per-episode records
        under key "episode_records"
    """
    pressure_fn  = PressureReward()
    obs_imputer  = ObservationImputer()
    episode_records = []

    for ep in range(num_episodes):
        ep_seed = seed + ep if seed is not None else None
        obs_dict, graph = env.reset(seed=ep_seed)
        obs_dict = apply_perception(obs_dict, perception_severity, sentinel)
        pressure_fn.reset()
        obs_imputer.reset()
        obs_dict = obs_imputer.impute(obs_dict)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)

        done              = False
        total_return      = 0.0
        step_count        = 0
        waiting_times     = []
        throughput_steps  = []
        num_vehicles_steps = []
        prev_actions      = {nid: 0 for nid in graph["node_ids"]}
        phase_changes     = 0

        while not done:
            actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)

            # Count phase changes across all nodes
            phase_changes += sum(
                1 for nid in graph["node_ids"]
                if actions.get(nid, 0) != prev_actions.get(nid, 0)
            )
            prev_actions = actions

            next_obs_dict, _, reward_dict, done, info = env.step(actions)
            next_obs_dict = apply_perception(next_obs_dict, perception_severity, sentinel)

            if use_pressure:
                reward_dict = pressure_fn.compute(next_obs_dict, graph)
            total_return += sum(reward_dict.values())

            # Collect SUMO vehicle metrics if available (TrafficEnv populates these)
            if "step_mean_waiting_time" in info:
                waiting_times.append(info["step_mean_waiting_time"])
            if "step_throughput" in info:
                throughput_steps.append(info["step_throughput"])
            if "step_num_vehicles" in info:
                num_vehicles_steps.append(info["step_num_vehicles"])

            next_obs_dict = obs_imputer.impute(next_obs_dict)
            _, padded_obs = pad_obs_dict(next_obs_dict)
            step_count += 1

        episode_records.append({
            "total_return":  total_return,
            "length":        step_count,
            "waiting_times": waiting_times,
            "throughput":    throughput_steps,
            "num_vehicles":  num_vehicles_steps,
            "phase_changes": phase_changes,
        })

    summary = compute_metrics(episode_records)
    summary["episode_records"] = episode_records
    return summary


def print_summary(label: str, metrics: dict) -> None:
    """Print a compact metrics summary to stdout."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Episodes:            {metrics.get('n_episodes', '?')}")
    print(f"  Mean return:         {metrics.get('mean_return', float('nan')):+.3f}"
          f"  ± {metrics.get('std_return', float('nan')):.3f}")
    print(f"  Mean ep length:      {metrics.get('mean_ep_length', float('nan')):.1f} steps")

    mwt = metrics.get("mean_waiting_time", float("nan"))
    import math
    if not math.isnan(mwt):
        print(f"  Mean waiting time:   {mwt:.2f} s")
        print(f"  p95 waiting time:    {metrics.get('p95_waiting_time', float('nan')):.2f} s")
        print(f"  Max waiting time:    {metrics.get('max_waiting_time', float('nan')):.2f} s")
        print(f"  Mean throughput/step:{metrics.get('mean_throughput_per_step', float('nan')):.2f} veh")
    else:
        print("  Waiting time:        N/A (MockEnv)")

    print(f"  Phase change rate:   {metrics.get('mean_phase_change_rate', float('nan')):.3f} /step")
    print(f"{'='*60}")
