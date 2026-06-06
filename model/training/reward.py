"""Efficient pressure reward and observation imputation for traffic signal control (§5).

All functions operate on raw (un-padded) obs_dict — do not call pad_obs_dict before them.
"""

import torch


class ObservationImputer:
    """Fills sentinel obs positions with last-known sensor readings before encoding.

    Only obs is modified. validity is NEVER changed — it always reflects sensor status:
      validity=1 = live sensor reading; validity=0 = sensor failed (this step).

    NodeEncoder receives cat([obs, validity]), so it can distinguish:
      (last_known, 0) = imputed   vs   (current, 1) = live   vs   (sentinel=-1, 0) = no history.

    Maintains an independent last-known cache from PressureReward.
    Call reset() at every env.reset() to prevent episode bleed.
    """

    def __init__(self) -> None:
        self._last_obs: dict = {}   # {node_id: FloatTensor(obs_len)}

    def reset(self) -> None:
        self._last_obs = {}

    def impute(self, obs_dict: dict) -> dict:
        """Return new obs_dict with sentinel obs replaced by last-known values.

        validity is returned unchanged — sensor failure is always recorded as 0.
        Returns a new dict; does not mutate input.
        """
        result = {}
        for node_id, (obs, validity) in obs_dict.items():
            new_obs = obs.clone()

            missing = validity < 0.5
            if node_id in self._last_obs and missing.any():
                last = self._last_obs[node_id]
                new_obs[missing] = last[missing]
                # validity intentionally NOT modified

            valid = validity > 0.5
            if valid.any():
                cache = self._last_obs.get(node_id, torch.zeros_like(obs)).clone()
                cache[valid] = obs[valid]
                self._last_obs[node_id] = cache

            result[node_id] = (new_obs, validity)   # original validity, unmodified
        return result


class PressureReward:
    """Stateful pressure reward with last-known imputation for missing lanes.

    Uses its own independent last-known cache. Receives raw (pre-imputation) obs
    from the trainer so its cache reflects true sensor readings, not imputed values.
    Call reset() at every env.reset() to prevent episode bleed.
    """

    def __init__(self, queue_weight: float = 0.0, pressure_weight: float = 1.0) -> None:
        self._last_obs: dict = {}
        self._qw = queue_weight      # weight on -q_in/n
        self._pw = pressure_weight   # weight on -|q_in-q_out|/n

    def reset(self) -> None:
        self._last_obs = {}

    def compute(self, obs_dict: dict, graph: dict) -> dict:
        """Mixed queue + pressure reward with last-known imputation.

        reward_i = -(queue_weight * q_in/n  +  pressure_weight * |q_in-q_out|/n)

        queue_weight > 0    prevents balanced-gridlock exploit (any waiting vehicle costs)
        pressure_weight > 0 adds downstream awareness (penalises imbalance vs next intersection)

        For each lane:
          - validity=1: use current reading; update last_known
          - validity=0 + last_known exists: substitute last_known value
          - validity=0 + no last_known: treat as 0
        Returns {node_id: float}, all values in (-inf, 0.0].
        """
        node_to_idx = graph["node_to_idx"]
        node_meta = graph["node_meta"]
        phase_features = graph["phase_features"]
        reward_dict = {}

        for node_id, (obs, validity) in obs_dict.items():
            node_idx = node_to_idx[node_id]
            num_phases = node_meta[node_idx]["num_phases"]
            phase_feats = phase_features[node_idx]
            num_incoming = len(phase_feats[0]) if phase_feats else 0

            if num_incoming == 0:
                reward_dict[node_id] = 0.0
                # Still update last_known for valid positions
                self._update_cache(node_id, obs, validity)
                continue

            # Build effective obs and validity using last-known imputation
            if node_id in self._last_obs:
                last = self._last_obs[node_id]
                effective_obs = torch.where(validity > 0.5, obs, last)
                effective_val = torch.ones_like(validity)
            else:
                effective_obs = obs.clone()
                effective_val = validity.clone()

            self._update_cache(node_id, obs, validity)

            q_in_start = num_phases + 1
            q_out_start = q_in_start + 2 * num_incoming
            num_outgoing = obs.shape[0] - q_out_start

            q_in_vals  = effective_obs[q_in_start  : q_in_start  + num_incoming]
            q_in_valid = effective_val[q_in_start  : q_in_start  + num_incoming]
            q_out_vals = effective_obs[q_out_start : q_out_start + num_outgoing]
            q_out_valid = effective_val[q_out_start : q_out_start + num_outgoing]

            q_in_sum  = (q_in_vals  * q_in_valid).sum()
            q_out_sum = (q_out_vals * q_out_valid).sum()
            num_valid_in = (q_in_valid > 0.5).sum().item()
            norm = max(num_valid_in, 1)

            queue    = q_in_sum / norm
            pressure = torch.abs(q_in_sum - q_out_sum) / norm
            reward_dict[node_id] = -(self._qw * queue + self._pw * pressure).item()

        return reward_dict

    def _update_cache(self, node_id: str, obs: torch.Tensor, validity: torch.Tensor) -> None:
        valid = validity > 0.5
        if valid.any():
            cache = self._last_obs.get(node_id, torch.zeros_like(obs)).clone()
            cache[valid] = obs[valid]
            self._last_obs[node_id] = cache


def compute_pressure(obs_dict: dict, graph: dict) -> dict:
    """Compute per-node efficient pressure reward.

    r_i = -abs(sum_valid_q_in - sum_valid_q_out) / max(num_valid_incoming, 1)

    obs values are already normalized by q_max (§3.2), so no further scaling is needed.
    Missing lanes (validity=0) contribute 0 to both numerator and denominator.

    Args:
        obs_dict: raw (un-padded) obs_dict from env — {node_id: (obs, validity)}
        graph:    graph dict from graph_builder / env.reset()

    Returns:
        {node_id: float}, all values in (-inf, 0.0]
    """
    node_to_idx = graph["node_to_idx"]
    node_meta = graph["node_meta"]
    phase_features = graph["phase_features"]
    reward_dict = {}

    for node_id, (obs, validity) in obs_dict.items():
        node_idx = node_to_idx[node_id]
        num_phases = node_meta[node_idx]["num_phases"]
        phase_feats = phase_features[node_idx]
        num_incoming = len(phase_feats[0]) if phase_feats else 0

        if num_incoming == 0:
            reward_dict[node_id] = 0.0
            continue

        q_in_start = num_phases + 1                         # skip phase_onehot + time_in_phase
        q_out_start = q_in_start + 2 * num_incoming         # skip queue + running per incoming lane
        num_outgoing = obs.shape[0] - q_out_start

        # Extract queue values and zero-out missing lanes via validity mask
        q_in_vals = obs[q_in_start : q_in_start + num_incoming]
        q_in_valid = validity[q_in_start : q_in_start + num_incoming]
        q_out_vals = obs[q_out_start : q_out_start + num_outgoing]
        q_out_valid = validity[q_out_start : q_out_start + num_outgoing]

        q_in_sum = (q_in_vals * q_in_valid).sum()
        q_out_sum = (q_out_vals * q_out_valid).sum()

        num_valid_in = (q_in_valid > 0.5).sum().item()
        norm = max(num_valid_in, 1)

        pressure = torch.abs(q_in_sum - q_out_sum) / norm
        reward_dict[node_id] = -pressure.item()

    return reward_dict
