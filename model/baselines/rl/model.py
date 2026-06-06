"""Model definitions for the traffic-light RL system.

This module provides a minimal DQN-style Q-network for estimating
action values from a traffic light state vector.
"""

import torch
from torch import Tensor, nn


class TrafficModel(nn.Module):
    """Q-network model for traffic-light control."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """Initialize the feed-forward Q-network."""
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state) -> Tensor:
        """Return Q-values for each available action."""
        if not isinstance(state, Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        else:
            state = state.to(dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        return self.network(state)
