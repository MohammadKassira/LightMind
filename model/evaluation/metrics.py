"""Evaluation metrics for traffic signal control (§8.1).

Computes mean/p95/max waiting time, throughput, queue, and phase-change rate
from episode records collected by eval_runner.evaluate().
"""

import math
from typing import Any


def compute_metrics(episode_records: list[dict]) -> dict:
    """Summarise a list of per-episode records into a metrics dict.

    Each record is produced by eval_runner.evaluate() and contains:
        total_return:         float  — sum of pressure rewards (or 0 if not used)
        length:               int    — episode length in action steps
        waiting_times:        list[float]  — per-step mean vehicle waiting time
                                            (empty for MockEnv)
        throughput:           list[int]    — per-step arrived vehicle count
                                            (empty for MockEnv)
        num_vehicles:         list[int]    — per-step vehicle count in network
        phase_changes:        int          — total phase changes across all nodes

    Returns a dict with:
        n_episodes
        mean_return, std_return
        mean_ep_length
        -- waiting time (NaN when no SUMO data) --
        mean_waiting_time, p95_waiting_time, max_waiting_time
        -- throughput --
        mean_throughput_per_step
        -- queue proxy (vehicles in network) --
        mean_vehicles_in_net, p95_vehicles_in_net
        -- stability --
        mean_phase_change_rate  (changes per action step)
    """
    if not episode_records:
        return {}

    n = len(episode_records)

    returns = [r["total_return"] for r in episode_records]
    lengths = [r["length"] for r in episode_records]

    # Waiting time — flatten per-step values across all episodes
    all_wait = [w for r in episode_records for w in r.get("waiting_times", [])]
    if all_wait:
        all_wait_sorted = sorted(all_wait)
        mean_wait = sum(all_wait) / len(all_wait)
        p95_wait  = all_wait_sorted[int(0.95 * len(all_wait_sorted))]
        max_wait  = all_wait_sorted[-1]
    else:
        mean_wait = p95_wait = max_wait = float("nan")

    # Throughput — per-step arrivals averaged across all steps
    all_tp = [t for r in episode_records for t in r.get("throughput", [])]
    mean_tp_per_step = sum(all_tp) / len(all_tp) if all_tp else float("nan")

    # Vehicles in network
    all_veh = [v for r in episode_records for v in r.get("num_vehicles", [])]
    if all_veh:
        all_veh_sorted = sorted(all_veh)
        mean_veh = sum(all_veh) / len(all_veh)
        p95_veh  = all_veh_sorted[int(0.95 * len(all_veh_sorted))]
    else:
        mean_veh = p95_veh = float("nan")

    # Phase-change rate
    phase_changes = [r.get("phase_changes", 0) for r in episode_records]
    mean_pc_rate = (
        sum(phase_changes[i] / max(lengths[i], 1) for i in range(n)) / n
    )

    return {
        "n_episodes":              n,
        "mean_return":             sum(returns) / n,
        "std_return":              _std(returns),
        "mean_ep_length":          sum(lengths) / n,
        "mean_waiting_time":       mean_wait,
        "p95_waiting_time":        p95_wait,
        "max_waiting_time":        max_wait,
        "mean_throughput_per_step": mean_tp_per_step,
        "mean_vehicles_in_net":    mean_veh,
        "p95_vehicles_in_net":     p95_veh,
        "mean_phase_change_rate":  mean_pc_rate,
    }


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
