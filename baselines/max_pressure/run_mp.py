"""Run the Max Pressure baseline with explicit external traffic-light control."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import traci

CURRENT_DIR = Path(__file__).resolve().parent
BASELINES_DIR = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from max_pressure import LaneMapping, MaxPressureController, load_lane_mapping
from rl.config import Config
from rl.env import TrafficLightEnv

CSV_FIELDNAMES = [
    "run_id",
    "episode",
    "timestamp_utc",
    "total_reward",
    "avg_reward",
    "steps",
    "last_step_index",
    "avg_queue_length",
    "total_waiting_time",
]


@dataclass
class EpisodeSummary:
    """Aggregated metrics for one evaluation episode."""

    total_reward: float
    avg_reward: float
    steps: int
    last_step_index: int
    avg_queue_length: float
    total_waiting_time: float


class EnvironmentStepAdapter:
    """Wrap the existing environment with an explicit external-control mode."""

    def __init__(self, base_env: TrafficLightEnv, external_control: bool = True) -> None:
        self.base_env = base_env
        self.external_control = external_control

    def reset(self) -> dict[str, list[float]]:
        """Reset the base environment."""
        return self.base_env.reset()

    def step(
        self,
        action_dict: dict[str, int],
    ) -> tuple[dict[str, list[float]], dict[str, float], bool, dict[str, Any]]:
        """Advance the environment without overriding external TLS decisions."""
        if not self.external_control:
            raise RuntimeError("Max Pressure requires external_control=True.")

        _ = action_dict
        traci.simulationStep()

        tls_ids = self.base_env.get_all_tls_ids()
        next_state = {tls_id: self.base_env.get_state(tls_id) for tls_id in tls_ids}
        step_metrics = self.base_env.get_step_metrics(tls_ids)
        rewards = {
            tls_id: self.base_env.compute_reward(tls_id, step_metrics)
            for tls_id in tls_ids
        }
        self.base_env._update_episode_metrics(step_metrics)
        info: dict[str, Any] = {
            "step_metrics": step_metrics,
            "episode_metrics": dict(self.base_env.episode_metrics),
            "external_control": True,
        }

        return next_state, rewards, False, info

    def get_all_tls_ids(self) -> list[str]:
        """Return every traffic light ID in the current simulation."""
        return self.base_env.get_all_tls_ids()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped environment."""
        return getattr(self.base_env, name)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the Max Pressure baseline."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Max Pressure baseline. "
            "The lane mapping JSON must look like "
            "{tls_id: {phase_id: [[incoming_lane, outgoing_lane], ...]}}."
        )
    )
    parser.add_argument(
        "--lane-mapping",
        required=True,
        help="Path to a JSON file containing the lane mapping for every traffic light.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--min-green-time",
        type=int,
        default=None,
        help="Minimum green time in simulation steps before switching away from a phase.",
    )
    parser.add_argument(
        "--yellow-time",
        type=int,
        default=3,
        help="Number of simulation steps to hold a yellow transition.",
    )
    parser.add_argument(
        "--disable-yellow",
        action="store_true",
        help="Disable yellow-phase transitions and switch directly to the next green phase.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(CURRENT_DIR / "results"),
        help="Base directory where per-run results will be saved.",
    )
    return parser.parse_args()


def initialize_environment(
    config: Config,
    external_control: bool,
) -> EnvironmentStepAdapter:
    """Create the environment with the same SUMO config as the RL baseline."""
    sumo_cmd = [config.sumo_binary, "-c", config.sumo_config_path]
    base_env = TrafficLightEnv(sumo_cmd)
    return EnvironmentStepAdapter(base_env=base_env, external_control=external_control)


def create_run_directory(results_root: Path) -> tuple[str, Path]:
    """Create a unique per-run directory so experiments never overwrite each other."""
    results_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = results_root / run_id
    suffix = 1

    while run_dir.exists():
        run_dir = results_root / f"{run_id}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.name, run_dir


def run_episode(
    env: EnvironmentStepAdapter,
    controller: MaxPressureController,
    max_steps: int,
) -> EpisodeSummary:
    """Run one evaluation episode and collect aggregated metrics."""
    env.reset()
    controller.reset()

    total_reward = 0.0
    total_queue = 0.0
    total_waiting_time = 0.0
    executed_steps = 0
    last_step_index = -1
    last_info: dict[str, Any] = {}

    for step in range(max_steps):
        action_dict = controller.update_all(step)
        _, rewards, done, info = env.step(action_dict)
        last_info = info

        total_reward += sum(rewards.values())
        total_queue += get_network_queue_length(env)
        total_waiting_time += get_step_waiting_time(info)
        executed_steps += 1
        last_step_index = step

        if done:
            break

    avg_reward = total_reward / executed_steps if executed_steps else 0.0
    avg_queue_length = total_queue / executed_steps if executed_steps else 0.0
    episode_waiting_time = get_episode_waiting_time(last_info, env, total_waiting_time)

    return EpisodeSummary(
        total_reward=total_reward,
        avg_reward=avg_reward,
        steps=executed_steps,
        last_step_index=last_step_index,
        avg_queue_length=avg_queue_length,
        total_waiting_time=episode_waiting_time,
    )


def get_network_queue_length(env: EnvironmentStepAdapter) -> float:
    """Return the total halted queue length across all controlled intersections."""
    tls_ids = env.get_all_tls_ids()
    return float(sum(env.get_total_queue(tls_id) for tls_id in tls_ids))


def get_step_waiting_time(info: dict[str, Any]) -> float:
    """Read step waiting time from env info when available."""
    step_metrics = info.get("step_metrics", {})
    waiting_time = step_metrics.get("waiting_time")
    return float(waiting_time) if waiting_time is not None else 0.0


def get_episode_waiting_time(
    info: dict[str, Any],
    env: EnvironmentStepAdapter,
    fallback: float,
) -> float:
    """Read episode waiting time from env info or the env metrics container."""
    episode_metrics = info.get("episode_metrics", {})
    waiting_time = episode_metrics.get("waiting_time")
    if waiting_time is not None:
        return float(waiting_time)

    env_waiting_time = getattr(env, "episode_metrics", {}).get("waiting_time")
    if env_waiting_time is not None:
        return float(env_waiting_time)

    return float(fallback)


def build_episode_result(
    run_id: str,
    episode: int,
    summary: EpisodeSummary,
) -> dict[str, str | int | float]:
    """Build a serializable episode result row."""
    return {
        "run_id": run_id,
        "episode": episode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_reward": round(summary.total_reward, 6),
        "avg_reward": round(summary.avg_reward, 6),
        "steps": summary.steps,
        "last_step_index": summary.last_step_index,
        "avg_queue_length": round(summary.avg_queue_length, 6),
        "total_waiting_time": round(summary.total_waiting_time, 6),
    }


def save_episode_result(
    result: dict[str, str | int | float],
    run_dir: Path,
) -> None:
    """Save per-episode JSON plus an accumulating run-local CSV."""
    episode = int(result["episode"])
    json_path = run_dir / f"episode_{episode:04d}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    csv_path = run_dir / "episodes.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def build_controller(
    lane_mapping: LaneMapping,
    tls_ids: list[str],
    action_interval: int,
    min_green_time: int,
    enable_yellow: bool,
    yellow_time: int,
) -> MaxPressureController:
    """Create the Max Pressure controller with validated control settings."""
    return MaxPressureController(
        lane_mapping=lane_mapping,
        tls_ids=tls_ids,
        action_interval=action_interval,
        min_green_time=min_green_time,
        enable_yellow=enable_yellow,
        yellow_time=yellow_time,
    )


def main() -> None:
    """Run the Max Pressure evaluation loop."""
    args = parse_args()
    config = Config()

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")
    if config.action_interval <= 0:
        raise ValueError("Config.action_interval must be greater than zero.")

    min_green_time = (
        args.min_green_time
        if args.min_green_time is not None
        else config.action_interval
    )
    if min_green_time < 0:
        raise ValueError("--min-green-time must be zero or greater.")
    if args.yellow_time < 0:
        raise ValueError("--yellow-time must be zero or greater.")

    enable_yellow = not args.disable_yellow

    results_root = Path(args.results_dir)
    lane_mapping = load_lane_mapping(args.lane_mapping)
    env = initialize_environment(config, external_control=True)

    try:
        tls_ids = env.get_all_tls_ids()
        if not tls_ids:
            raise RuntimeError("No traffic lights found in the SUMO network.")

        controller = build_controller(
            lane_mapping=lane_mapping,
            tls_ids=tls_ids,
            action_interval=config.action_interval,
            min_green_time=min_green_time,
            enable_yellow=enable_yellow,
            yellow_time=args.yellow_time,
        )
        run_id, run_dir = create_run_directory(results_root)

        for episode_index in range(args.episodes):
            episode = episode_index + 1
            summary = run_episode(env=env, controller=controller, max_steps=config.max_steps)
            result = build_episode_result(run_id=run_id, episode=episode, summary=summary)
            save_episode_result(result, run_dir)

            print(
                "episode="
                f"{episode} run_id={run_id} "
                f"total_reward={result['total_reward']:.2f} "
                f"avg_reward={result['avg_reward']:.4f} "
                f"steps={result['steps']} "
                f"last_step_index={result['last_step_index']} "
                f"avg_queue_length={result['avg_queue_length']:.2f} "
                f"total_waiting_time={result['total_waiting_time']:.2f}"
            )
    finally:
        if traci.isLoaded():
            traci.close()


if __name__ == "__main__":
    main()
