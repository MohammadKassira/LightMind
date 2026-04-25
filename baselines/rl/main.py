"""Entry point for training or evaluating the traffic-light RL system.

This module runs a basic inference-only interaction loop between a
shared RL model and the SUMO traffic light environment.
"""

import copy
import os
import random
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import traci

from baselines.rl.config import Config
from baselines.rl.env import TrafficLightEnv
from baselines.rl.model import TrafficModel


def main() -> None:
    """Run a simple SUMO control loop with one shared Q-network model."""
    config = Config()
    sumo_cmd = [config.sumo_binary, "-c", config.sumo_config_path]
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
    main()
