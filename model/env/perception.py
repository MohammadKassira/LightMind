"""Stateless perception masking for observation degradation (R3).

apply_perception is the only public entry point. It operates on raw (un-padded) obs_dict
and returns a new dict — it never calls pad_obs_dict and never mutates its input.
"""

import torch


def apply_perception(
    obs_dict: dict,
    severity: float,
    sentinel: float = -1.0,
    exclude_positions=None,
) -> dict:
    """Randomly corrupt sensor features according to a severity knob.

    For each node, each feature is independently corrupted with probability `severity`.
    Corrupted positions receive the sentinel value and validity=0.0.

    Args:
        obs_dict:           {node_id: (obs_tensor, validity_tensor)} — raw from env.step()/reset()
        severity:           float in [0.0, 1.0]; 0.0 = clean copy, 1.0 = all features corrupted
        sentinel:           value written to corrupted obs positions; default -1.0
        exclude_positions:  iterable of feature indices exempt from corruption (e.g. range(8)
                            to protect a phase one-hot); default None means no exclusions

    Returns:
        New dict with same structure; input dict and tensors are not mutated.
        At severity=0.0 the output values are equal to the input values (pure copy).
    """
    result = {}
    for node_id, (obs, validity) in obs_dict.items():
        new_obs = obs.clone()
        new_val = validity.clone()

        if severity > 0.0:
            corrupt = torch.rand_like(obs) < severity
            if exclude_positions is not None:
                corrupt[list(exclude_positions)] = False
            new_obs[corrupt] = sentinel
            new_val[corrupt] = 0.0

        result[node_id] = (new_obs, new_val)
    return result
