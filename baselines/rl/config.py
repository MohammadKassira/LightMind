"""Configuration definitions for the traffic-light RL system.

This module will eventually hold project-wide settings for SUMO,
environment parameters, model hyperparameters, and training options.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Container for reinforcement learning configuration values."""

    sumo_binary: str = "sumo"
    sumo_config_path: str = "/Users/hasanhaidar/Downloads/traffic_networks/Debugging/cross_smoke/cross.sumocfg"
    max_steps: int = 1500
    action_interval: int = 10
