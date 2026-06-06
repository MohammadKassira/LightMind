"""Agent logic for the traffic-light RL system.

This module defines the reinforcement learning agent that connects
the environment, model, and training behavior.
"""

import copy
import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from baselines.rl.config import Config
from baselines.rl.env import TrafficLightEnv
from baselines.rl.model import TrafficModel


class TrafficAgent:
    """DQN agent coordinating decisions and learning for traffic light control."""

    def __init__(
        self,
        env: TrafficLightEnv,
        model: TrafficModel,
        config: Config,
    ) -> None:
        """Initialize agent with environment, model, and training configuration."""
        self.env = env
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.replay_buffer: deque[tuple] = deque(maxlen=config.replay_buffer_size)
        self.config = config
        self.epsilon = config.epsilon_start

    def act(self, state_vector: list[float], state_dim: int) -> int:
        """Select an action using an epsilon-greedy policy."""
        padded = self._pad_state_vector(state_vector, state_dim)
        if random.random() < self.epsilon:
            return random.randrange(2)
        with torch.no_grad():
            q_values = self.model(padded)
        return int(torch.argmax(q_values.squeeze(0)).item())

    def learn(self, transition: tuple) -> float | None:
        """Store a transition and update the model if the buffer is ready."""
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states_tensor)
        chosen_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states_tensor)
            max_next_q_values = next_q_values.max(dim=1).values
            target_q_values = rewards_tensor + self.config.gamma * max_next_q_values * (
                1.0 - dones_tensor
            )

        loss = F.mse_loss(chosen_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target(self) -> None:
        """Copy online model weights into the target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self) -> None:
        """Apply exponential epsilon decay, floored at epsilon_min."""
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    @staticmethod
    def _pad_state_vector(state_vector: list[float], target_dim: int) -> list[float]:
        """Resize one traffic-light state so every TLS matches the model input width."""
        if len(state_vector) > target_dim:
            if target_dim <= 0:
                raise RuntimeError(f"Configured model input {target_dim} is invalid.")
            if target_dim == 1:
                return [state_vector[-1]]
            return state_vector[: target_dim - 1] + [state_vector[-1]]
        if len(state_vector) == target_dim:
            return state_vector
        return state_vector + [0.0] * (target_dim - len(state_vector))
