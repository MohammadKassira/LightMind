import torch
import torch.nn as nn
from torch import Tensor


class NodeEncoder(nn.Module):
    """Per-node obs vector → MLP → fixed-size embedding (§6.1).

    obs must be pre-padded to obs_dim with zeros; validity must be 0 on
    padded positions. The model receives torch.cat([obs, validity]) so it can
    distinguish a last-known imputed value (obs=last_known, validity=0) from a
    live reading (obs=current, validity=1) from a never-seen position
    (obs=sentinel=-1.0, validity=0). First linear layer is obs_dim*2 wide.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128, embed_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, obs: Tensor, validity: Tensor) -> Tensor:
        """
        Args:
            obs:      (..., obs_dim) pre-padded observation (imputed where missing)
            validity: (..., obs_dim) 0/1 float; 0 = sensor failed or padding
        Returns:
            (..., embed_dim) node embedding
        """
        return self.net(torch.cat([obs, validity], dim=-1))


def pad_obs_dict(obs_dict: dict, target_dim: int | None = None) -> tuple[int, dict]:
    """Pad all per-node (obs, validity) tensors to the same length.

    Returns (obs_dim, padded_obs_dict) where obs_dim is the effective padded
    length. If target_dim is given, obs_dim = max(local_max, target_dim) so
    that observations can be padded to a global ceiling across multiple networks.
    Padding positions get validity=0 and obs=0.
    """
    local_max = max(obs.shape[-1] for obs, _ in obs_dict.values())
    obs_dim   = max(local_max, target_dim) if target_dim is not None else local_max
    padded = {}
    for node_id, (obs, validity) in obs_dict.items():
        n = obs.shape[-1]
        if n == obs_dim:
            padded[node_id] = (obs, validity)
        else:
            obs_p = torch.zeros(*obs.shape[:-1], obs_dim, dtype=obs.dtype)
            val_p = torch.zeros(*validity.shape[:-1], obs_dim, dtype=validity.dtype)
            obs_p[..., :n] = obs
            val_p[..., :n] = validity
            padded[node_id] = (obs_p, val_p)
    return obs_dim, padded
