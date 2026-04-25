"""Agent logic for the traffic-light RL system.

This module will define the reinforcement learning agent that connects
the environment, model, and training behavior.
"""

from typing import Any

from baselines.rl.env import TrafficLightEnv
from baselines.rl.model import TrafficModel


class TrafficAgent:
    """Placeholder agent coordinating decisions and learning."""

    def __init__(self, env: TrafficLightEnv, model: TrafficModel) -> None:
        """Store the environment and model dependencies."""

    def act(self, observation: Any) -> Any:
        """Select an action from the current observation."""

    def learn(self, transition: Any) -> None:
        """Update the agent from an environment transition."""
