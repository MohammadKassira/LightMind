"""Circular replay buffer for multi-agent DQN on a fixed graph topology.

The trainer calls pad_obs_dict once per step and passes pre-padded dicts to push().
The buffer does not call pad_obs_dict internally — no double-padding.
"""

import torch


class ReplayBuffer:
    """Pre-allocated circular ring buffer for shared-parameter multi-agent DQN.

    Stores transitions as stacked tensors ordered by graph["node_ids"]:
        _obs      [capacity, N, obs_dim]  float32
        _val      [capacity, N, obs_dim]  float32
        _next_obs [capacity, N, obs_dim]  float32
        _next_val [capacity, N, obs_dim]  float32
        _actions  [capacity, N]           int64
        _rewards  [capacity, N]           float32
        _dones    [capacity]              float32

    Args:
        capacity:  Maximum number of stored transitions.
        obs_dim:   Padded observation length — from pad_obs_dict at init.
        node_ids:  Ordered node list from graph["node_ids"]; defines column order.
        device:    Device for tensors returned by sample().
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        node_ids: list,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._capacity = capacity
        self._obs_dim = obs_dim
        self._node_ids = list(node_ids)
        self._node_to_col = {nid: i for i, nid in enumerate(self._node_ids)}
        self._device = device
        N = len(self._node_ids)

        self._obs      = torch.zeros(capacity, N, obs_dim, dtype=torch.float32)
        self._val      = torch.zeros(capacity, N, obs_dim, dtype=torch.float32)
        self._next_obs = torch.zeros(capacity, N, obs_dim, dtype=torch.float32)
        self._next_val = torch.zeros(capacity, N, obs_dim, dtype=torch.float32)
        self._actions  = torch.zeros(capacity, N, dtype=torch.int64)
        self._rewards  = torch.zeros(capacity, N, dtype=torch.float32)
        self._dones    = torch.zeros(capacity, dtype=torch.float32)

        self._write_pos = 0
        self._size = 0

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(
        self,
        padded_obs: dict,
        padded_next: dict,
        actions: dict,
        rewards: dict,
        done: bool,
    ) -> None:
        """Store one transition.

        Args:
            padded_obs:  {node_id: (obs, validity)} — pre-padded by the trainer
            padded_next: {node_id: (obs, validity)} — pre-padded by the trainer
            actions:     {node_id: int}
            rewards:     {node_id: float}
            done:        episode-terminal flag
        """
        idx = self._write_pos % self._capacity

        for node_id, col in self._node_to_col.items():
            obs, val = padded_obs[node_id]
            nobs, nval = padded_next[node_id]

            if obs.shape[-1] != self._obs_dim:
                raise ValueError(
                    f"obs_dim mismatch for '{node_id}': "
                    f"expected {self._obs_dim}, got {obs.shape[-1]}"
                )

            self._obs[idx, col]      = obs.float()
            self._val[idx, col]      = val.float()
            self._next_obs[idx, col] = nobs.float()
            self._next_val[idx, col] = nval.float()
            self._actions[idx, col]  = int(actions.get(node_id, 0))
            self._rewards[idx, col]  = float(rewards.get(node_id, 0.0))

        self._dones[idx] = float(done)
        self._write_pos += 1
        self._size = min(self._size + 1, self._capacity)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> dict:
        """Sample a random minibatch.

        Returns:
            obs:      [B, N, obs_dim] float32
            validity: [B, N, obs_dim] float32
            actions:  [B, N]          int64
            rewards:  [B, N]          float32
            next_obs: [B, N, obs_dim] float32
            next_val: [B, N, obs_dim] float32
            dones:    [B]             float32
        All tensors on self.device.

        Raises:
            ValueError: if buffer holds fewer transitions than batch_size.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Not enough transitions to sample: have {self._size}, need {batch_size}"
            )
        indices = torch.randint(0, self._size, (batch_size,))
        return {
            "obs":      self._obs[indices].to(self._device),
            "validity": self._val[indices].to(self._device),
            "actions":  self._actions[indices].to(self._device),
            "rewards":  self._rewards[indices].to(self._device),
            "next_obs": self._next_obs[indices].to(self._device),
            "next_val": self._next_val[indices].to(self._device),
            "dones":    self._dones[indices].to(self._device),
        }

    def __len__(self) -> int:
        return self._size
