"""
SUMO-GUI evaluation phase runner.

Runs one evaluation phase (fixed_time or gat) across 5 episodes:
  - Episode 0: SUMO-GUI on the given display (user watches live)
  - Episodes 1-4: headless, all started simultaneously via ThreadPoolExecutor

All 5 finish before the function returns. Results are aggregated and returned
as a metrics dict compatible with eval_comparison.json.
"""
from __future__ import annotations

import concurrent.futures
import os


def run_phase_with_gui(
    mode: str,              # "fixed_time" or "gat"
    display: str,           # ":99"
    net_file: str,
    route_file: str,
    max_steps: int,
    begin_time: int,
    seeds: list[int],
    checkpoint: str | None = None,
    cfg: dict | None = None,
) -> dict:
    """Run 5 eval episodes: episode 0 with SUMO-GUI, episodes 1-4 headless in parallel.

    All 5 are submitted to a ThreadPoolExecutor simultaneously so headless runs
    complete while the GUI episode plays. Returns aggregated metrics dict.

    Caller must have added model/ to sys.path before calling this.
    """
    os.environ["DISPLAY"] = display

    results: list[dict | None] = [None] * 5

    def run_single_episode(ep_idx: int, use_gui: bool) -> dict:
        """One self-contained episode: creates its own env (and trainer for GAT)."""
        from env.traffic_env import TrafficEnv  # noqa: PLC0415
        from models.node_encoder import pad_obs_dict  # noqa: PLC0415
        from models.phase_head import pad_phase_features  # noqa: PLC0415
        from training.reward import ObservationImputer  # noqa: PLC0415

        env = TrafficEnv(
            net_file=net_file,
            route_file=route_file,
            max_steps=max_steps,
            begin_time=begin_time,
            use_gui=use_gui,
            override_tl_program=(mode == "gat"),
            passive=(mode == "fixed_time"),
        )

        trainer = None
        if mode == "gat" and checkpoint:
            from training.trainer import DQNTrainer  # noqa: PLC0415
            # DQNTrainer.__init__ does a probe env.reset() to derive model dims.
            # We pass our episode env; it resets with cfg seed internally, then
            # we re-reset below with the correct episode seed.
            trainer = DQNTrainer.load_checkpoint(checkpoint, cfg or {}, env)

        obs_dict, graph = env.reset(seed=seeds[ep_idx])

        imputer = ObservationImputer()
        imputer.reset()
        obs_dict = imputer.impute(obs_dict)

        done = False
        waiting_times: list[float] = []
        throughput_steps: list[int] = []

        if mode == "gat":
            _, padded_obs = pad_obs_dict(obs_dict)
            _, padded_pf  = pad_phase_features(graph)

        while not done:
            if mode == "gat":
                actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)
            else:
                actions = {}

            obs_dict, graph, _, done, info = env.step(actions)

            if "step_mean_waiting_time" in info:
                waiting_times.append(info["step_mean_waiting_time"])
            if "step_throughput" in info:
                throughput_steps.append(info["step_throughput"])

            if mode == "gat" and not done:
                obs_dict = imputer.impute(obs_dict)
                _, padded_obs = pad_obs_dict(obs_dict)
                _, padded_pf  = pad_phase_features(graph)

        env.close()
        return {
            "waiting_times": waiting_times,
            "throughput": throughput_steps,
        }

    # Submit all 5 simultaneously: ep 0 with GUI, eps 1-4 headless
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {
            executor.submit(run_single_episode, 0, True): 0,
            executor.submit(run_single_episode, 1, False): 1,
            executor.submit(run_single_episode, 2, False): 2,
            executor.submit(run_single_episode, 3, False): 3,
            executor.submit(run_single_episode, 4, False): 4,
        }
        for future, idx in future_map.items():
            results[idx] = future.result()  # propagates exceptions

    # Aggregate: mean waiting time per episode, total throughput per episode
    avg_waits = [
        round(sum(r["waiting_times"]) / max(len(r["waiting_times"]), 1), 2)
        for r in results
    ]
    total_throughputs = [sum(r["throughput"]) for r in results]

    return {
        "avg_waiting_time": avg_waits,
        "throughput": total_throughputs,
        "mean_waiting_time": round(sum(avg_waits) / len(avg_waits), 2),
        "mean_throughput": round(sum(total_throughputs) / len(total_throughputs), 2),
    }
