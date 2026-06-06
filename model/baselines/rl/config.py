"""Configuration definitions for the traffic-light RL system.

This module will eventually hold project-wide settings for SUMO,
environment parameters, model hyperparameters, and training options.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Container for reinforcement learning configuration values."""

    sumo_binary: str = "sumo"
    sumo_config_path: str = "traffic_networks/Debugging/grid6_smoke/grid6.sumocfg"
    max_steps: int = 1500
    action_interval: int = 10
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 32
    replay_buffer_size: int = 10000
    target_update_interval: int = 200
    checkpoint_interval: int = 50
    num_episodes: int = 1500
    learning_rate: float = 1e-4
