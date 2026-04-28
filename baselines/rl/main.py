"""Entry point for training or evaluating the traffic-light RL system.

This module runs a basic inference-only interaction loop between a
shared RL model and the SUMO traffic light environment.
"""

import copy
import os
import random
from collections import deque
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import traci

from baselines.rl.config import Config
from baselines.rl.env import TrafficLightEnv
from baselines.rl.model import TrafficModel


def _build_sumo_cmd(
    config: Config,
    seed: Optional[int] = None,
    scale: Optional[float] = None,
    sumo_cfg_path: Optional[Union[os.PathLike[str], str]] = None,
) -> list[str]:
    """Build the SUMO command for one training-network run."""
    resolved_sumo_cfg_path = (
        str(Path(sumo_cfg_path))
        if sumo_cfg_path is not None
        else config.sumo_config_path
    )
    sumo_cmd = [config.sumo_binary, "-c", resolved_sumo_cfg_path]
    if seed is not None:
        sumo_cmd.extend(["--seed", str(seed)])
    if scale is not None:
        sumo_cmd.extend(["--scale", str(scale)])
    return sumo_cmd


def _resolve_model_path(
    output_dir: Path,
    model_path: Optional[Union[os.PathLike[str], str]] = None,
) -> Path:
    """Return the checkpoint to use for evaluation."""
    if model_path is not None:
        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {resolved_path}")
        return resolved_path

    for candidate_name in ("best_model.pth", "latest_model.pth"):
        candidate_path = output_dir / candidate_name
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        "No evaluation checkpoint found. Expected best_model.pth or latest_model.pth "
        f"in {output_dir}."
    )


def _get_checkpoint_state_dim(checkpoint_state_dict: dict[str, torch.Tensor]) -> int:
    """Infer the trained model input width from the first linear layer."""
    input_layer_weights = checkpoint_state_dict.get("network.0.weight")
    if input_layer_weights is None or input_layer_weights.ndim != 2:
        raise RuntimeError("Unable to infer checkpoint input width from network.0.weight.")
    return int(input_layer_weights.shape[1])


def _get_episode_avg_speed(episode_metrics: dict[str, float]) -> float:
    """Convert accumulated speed totals into one episode-average value."""
    metric_steps = episode_metrics["metric_steps"]
    if metric_steps == 0.0:
        return 0.0
    return episode_metrics["avg_speed_total"] / metric_steps


def _get_state_dim(state_dict: dict[str, list[float]]) -> int:
    """Return the maximum state width across the current network."""
    if not state_dict:
        raise RuntimeError("No traffic light states were returned by the environment.")
    return max(len(state_vector) for state_vector in state_dict.values())


def _pad_state_vector(state_vector: list[float], target_dim: int) -> list[float]:
    """Resize one traffic-light state so every TLS matches the model input width."""
    if len(state_vector) > target_dim:
        if target_dim <= 0:
            raise RuntimeError(f"Configured model input {target_dim} is invalid.")
        # Keep the leading lane features and preserve the phase value at the end.
        if target_dim == 1:
            return [state_vector[-1]]
        return state_vector[: target_dim - 1] + [state_vector[-1]]
    if len(state_vector) == target_dim:
        return state_vector
    return state_vector + [0.0] * (target_dim - len(state_vector))


def evaluate_model(
    model_path: Optional[Union[os.PathLike[str], str]] = None,
    network_name: str = "cologne3",
    sumo_cfg_name: Optional[str] = None,
    scale: Optional[float] = 0.5,
    results_filename: str = "cologne3_low_results.xlsx",
    demand_label: str = "LOW",
) -> pd.DataFrame:
    """Evaluate the trained model on one network for five fixed seeds."""
    config = Config()
    output_dir = Path(__file__).resolve().parent / "training_artifacts"
    network_dir = (
        Path(__file__).resolve().parent.parent
        / "traffic_networks"
        / "Main_training_real"
        / network_name
    )
    resolved_sumo_cfg_name = sumo_cfg_name or f"{network_name}.sumocfg"
    sumo_cfg_path = (
        network_dir
        / resolved_sumo_cfg_name
    )
    results_path = output_dir / results_filename
    evaluation_seeds = [0, 1, 2, 3, 4]
    epsilon = 0.0
    action_dim = 2
    metric_columns = [
        "cars_passed",
        "waiting_time",
        "avg_speed",
        "teleports",
        "stopped_cars",
    ]

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = _resolve_model_path(output_dir, model_path)
    if not sumo_cfg_path.exists():
        raise FileNotFoundError(f"SUMO config not found: {sumo_cfg_path}")

    if traci.isLoaded():
        traci.close()

    checkpoint_state_dict = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_state_dim = _get_checkpoint_state_dim(checkpoint_state_dict)

    print("USING SUMO CONFIG:", str(sumo_cfg_path))
    print("USING SUMO SCALE:", scale if scale is not None else "default")
    print("USING CHECKPOINT:", str(checkpoint_path))
    env = TrafficLightEnv(
        _build_sumo_cmd(
            config,
            seed=evaluation_seeds[0],
            scale=scale,
            sumo_cfg_path=sumo_cfg_path,
        )
    )

    try:
        state_dict = env.reset()
        tls_ids = env.get_all_tls_ids()
        if not tls_ids:
            raise RuntimeError("No traffic lights found in the SUMO network.")

        state_dim = checkpoint_state_dim
        model = TrafficModel(state_dim=state_dim, action_dim=action_dim)
        model.load_state_dict(checkpoint_state_dict)
        model.eval()

        results = []

        for episode_index, seed in enumerate(evaluation_seeds, start=1):
            random.seed(seed)
            torch.manual_seed(seed)
            env.sumo_cmd = _build_sumo_cmd(
                config,
                seed=seed,
                scale=scale,
                sumo_cfg_path=sumo_cfg_path,
            )
            state_dict = env.reset()
            tls_ids = env.get_all_tls_ids()

            action_dict = {tls_id: 0 for tls_id in tls_ids}
            last_info: dict[str, dict[str, float]] = {
                "episode_metrics": dict(env.episode_metrics),
            }

            for step in range(config.max_steps):
                if step % config.action_interval == 0:
                    for tls_id in tls_ids:
                        state_vector = _pad_state_vector(state_dict[tls_id], state_dim)

                        if random.random() < epsilon:
                            action = random.randrange(action_dim)
                        else:
                            with torch.no_grad():
                                q_values = model(state_vector)
                            action = int(torch.argmax(q_values.squeeze(0)).item())

                        action_dict[tls_id] = action

                next_state_dict, _, done, info = env.step(action_dict)
                if done:
                    raise RuntimeError(
                        "Evaluation episode ended early. Expected a fixed number of steps "
                        f"({config.max_steps}) for every run."
                    )

                state_dict = next_state_dict
                last_info = info

            episode_metrics = last_info["episode_metrics"]
            episode_result = {
                "episode": episode_index,
                "seed": seed,
                "cars_passed": episode_metrics["cars_passed"],
                "waiting_time": episode_metrics["waiting_time"],
                "avg_speed": _get_episode_avg_speed(episode_metrics),
                "teleports": episode_metrics["teleports"],
                "stopped_cars": episode_metrics["stopped_cars"],
            }
            results.append(episode_result)

            print(
                "evaluation_episode="
                f"{episode_index} seed={seed} cars_passed={episode_result['cars_passed']:.0f} "
                f"waiting_time={episode_result['waiting_time']:.1f} "
                f"avg_speed={episode_result['avg_speed']:.2f} "
                f"teleports={episode_result['teleports']:.0f} "
                f"stopped_cars={episode_result['stopped_cars']:.1f}"
            )

        results_df = pd.DataFrame(results)
        results_df.to_excel(results_path, index=False)

        average_metrics = results_df[metric_columns].mean()
        print(f"\n{network_name} - {demand_label} demand results")
        for metric_name in metric_columns:
            print(f"{metric_name}: {average_metrics[metric_name]:.4f}")
        print(f"\nSaved evaluation results to {results_path}")

        return results_df
    finally:
        if traci.isLoaded():
            traci.close()


def main() -> None:
    """Run a simple SUMO control loop with one shared Q-network model."""
    config = Config()
    sumo_cmd = _build_sumo_cmd(config)
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    start_episode = 0
    num_episodes = 1500
    batch_size = 32
    target_update_interval = 200
    checkpoint_interval = 50
    output_dir = Path(__file__).resolve().parent / "training_artifacts"

    env = TrafficLightEnv(sumo_cmd)

    try:
        os.makedirs(output_dir, exist_ok=True)
        state_dict = env.reset()
        tls_ids = env.get_all_tls_ids()

        if not tls_ids:
            raise RuntimeError("No traffic lights found in the SUMO network.")

        state_dim = len(state_dict[tls_ids[0]])
        action_dim = 2
        model = TrafficModel(state_dim=state_dim, action_dim=action_dim)
        target_model = copy.deepcopy(model)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        replay_buffer = deque(maxlen=10000)
        global_step = 0
        recent_rewards: deque[float] = deque(maxlen=50)
        best_reward = float("-inf")

        # Reuse the latest chosen action between control updates.
        action_dict = {tls_id: 0 for tls_id in tls_ids}

        for episode in range(start_episode, num_episodes):
            state_dict = env.reset()
            episode_reward = 0.0
            last_info: dict[str, dict[str, float]] = {
                "step_metrics": {
                    "teleports": 0.0,
                    "cars_passed": 0.0,
                    "waiting_time": 0.0,
                    "stopped_cars": 0.0,
                    "avg_speed": 0.0,
                },
                "episode_metrics": dict(env.episode_metrics),
            }

            for step in range(config.max_steps):
                if step % config.action_interval == 0:
                    for tls_id in tls_ids:
                        state_vector = state_dict[tls_id]

                        if random.random() < epsilon:
                            action = random.randrange(action_dim)
                        else:
                            with torch.no_grad():
                                q_values = model(state_vector)
                            action = int(torch.argmax(q_values.squeeze(0)).item())

                        action_dict[tls_id] = action

                previous_state_dict = state_dict
                next_state_dict, rewards, done, info = env.step(action_dict)
                last_info = info
                episode_reward += sum(rewards.values())

                for tls_id in tls_ids:
                    replay_buffer.append(
                        (
                            previous_state_dict[tls_id],
                            action_dict[tls_id],
                            rewards[tls_id],
                            next_state_dict[tls_id],
                            done,
                        )
                    )

                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, batch_rewards, next_states, dones = zip(*batch)

                    states_tensor = torch.tensor(states, dtype=torch.float32)
                    actions_tensor = torch.tensor(actions, dtype=torch.long)
                    rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
                    next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
                    dones_tensor = torch.tensor(dones, dtype=torch.float32)

                    q_values = model(states_tensor)
                    chosen_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_q_values = target_model(next_states_tensor)
                        max_next_q_values = next_q_values.max(dim=1).values
                        target_q_values = rewards_tensor + gamma * max_next_q_values * (
                            1.0 - dones_tensor
                        )

                    loss = F.mse_loss(chosen_q_values, target_q_values)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                global_step += 1

                if global_step % target_update_interval == 0:
                    target_model.load_state_dict(model.state_dict())

                state_dict = next_state_dict
                if done:
                    break

            recent_rewards.append(episode_reward)
            rolling_reward = sum(recent_rewards) / len(recent_rewards)
            episode_metrics = last_info["episode_metrics"]
            avg_speed = (
                episode_metrics["avg_speed_total"] / episode_metrics["metric_steps"]
                if episode_metrics["metric_steps"]
                else 0.0
            )
            print(
                "episode="
                f"{episode} reward={episode_reward:.2f} avg_reward_50={rolling_reward:.2f} "
                f"epsilon={epsilon:.3f} teleports={episode_metrics['teleports']:.0f} "
                f"cars_passed={episode_metrics['cars_passed']:.0f} "
                f"waiting_time={episode_metrics['waiting_time']:.1f} "
                f"stopped_cars={episode_metrics['stopped_cars']:.1f} "
                f"avg_speed={avg_speed:.2f}"
            )

            if (episode + 1) % checkpoint_interval == 0:
                checkpoint_path = output_dir / f"model_ep{episode + 1}.pth"
                print(f"Saving checkpoint at episode {episode + 1}")
                torch.save(model.state_dict(), checkpoint_path)

            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(model.state_dict(), output_dir / "best_model.pth")
                print("New best model saved")

            torch.save(model.state_dict(), output_dir / "latest_model.pth")
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
    finally:
        if traci.isLoaded():
            traci.close()


if __name__ == "__main__":
    evaluate_model()
