"""Mock environment for offline model development without SUMO (R0).

Implements the §3.5 step API using real graph structure from graph_builder
and randomly generated observations shaped to the §3.2 schema.
"""
from pathlib import Path

import torch
import yaml

from data.graph_builder import build_graph

_REPO_ROOT = Path(__file__).parent.parent
_NETS_DIR = _REPO_ROOT / "data" / "networks"
_NORM_CFG = _REPO_ROOT / "configs" / "normalization.yaml"
_PERC_CFG = _REPO_ROOT / "configs" / "perception.yaml"


def _load_cfg():
    with open(_NORM_CFG) as f:
        norm = yaml.safe_load(f)
    with open(_PERC_CFG) as f:
        perc = yaml.safe_load(f)
    return norm, perc


class MockEnv:
    """Fake traffic env: real graph topology, random observations.

    Args:
        network_name: stem of a net.xml in data/networks/ (e.g. 'cross_smoke')
        max_steps: episode length before done=True
        missing_prob: fraction of obs features to randomly corrupt with sentinel
    """

    def __init__(self, network_name: str, max_steps: int = 100, missing_prob: float = 0.0):
        norm, perc = _load_cfg()
        self._max_phase_time = norm["max_phase_time"]
        self._sentinel = perc["perception"]["sentinel_value"]
        self._missing_prob = missing_prob
        self._max_steps = max_steps
        self._graph = build_graph(_NETS_DIR / f"{network_name}.net.xml")
        self._step_count = 0

    # ------------------------------------------------------------------
    # §3.5 API
    # ------------------------------------------------------------------

    def reset(self, network_cfg=None, seed=None):
        """Return (obs_dict, graph). network_cfg ignored — network fixed at init."""
        self._step_count = 0
        if seed is not None:
            torch.manual_seed(seed)
        obs_dict = self._sample_obs()
        return obs_dict, self._graph

    def step(self, actions):
        """Return (obs_dict, graph, reward_dict, done, info).

        actions: dict[node_id -> phase_idx]
        reward_dict: dict[node_id -> float], negative (efficient pressure sign)
        """
        self._step_count += 1
        obs_dict = self._sample_obs()
        reward_dict = {nid: -torch.rand(1).item() for nid in self._graph["node_ids"]}
        done = self._step_count >= self._max_steps
        return obs_dict, self._graph, reward_dict, done, {
            "step_mean_waiting_time": float(torch.rand(1).item() * 30.0),
            "step_throughput":        int(torch.randint(0, 4, (1,)).item()),
            "step_queue_length":      float(torch.rand(1).item() * 20.0),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_obs(self):
        return {
            nid: self._make_node_obs(i)
            for i, nid in enumerate(self._graph["node_ids"])
        }

    def _make_node_obs(self, node_idx: int):
        """Build (obs, validity) tensors for one node matching §3.2.

        obs layout:
          phase_onehot          [num_phases]
          time_in_phase         [1]
          per incoming lane:    queue/q_max, running/q_max  [2 * num_incoming]
          per outgoing lane:    queue/q_max                 [num_outgoing]

        validity: same length as obs; 1=real, 0=missing (sentinel in obs).
        num_outgoing approximated as num_incoming (mock; real env reads from SUMO).
        """
        meta = self._graph["node_meta"][node_idx]
        phase_feats = self._graph["phase_features"][node_idx]
        num_phases = meta["num_phases"]
        num_incoming = len(phase_feats[0]) if phase_feats else 0
        num_outgoing = num_incoming  # symmetric intersection approximation

        # phase one-hot: pick a random current phase
        phase_onehot = torch.zeros(num_phases)
        if num_phases > 0:
            phase_onehot[torch.randint(num_phases, (1,)).item()] = 1.0

        time_in_phase = torch.rand(1)
        incoming = torch.rand(num_incoming * 2)  # queue/q_max and running/q_max per lane
        outgoing = torch.rand(num_outgoing)       # queue/q_max per lane

        obs = torch.cat([phase_onehot, time_in_phase, incoming, outgoing])
        validity = torch.ones_like(obs)

        if self._missing_prob > 0.0:
            corrupted = torch.rand_like(obs) < self._missing_prob
            obs[corrupted] = self._sentinel
            validity[corrupted] = 0.0

        return obs, validity
