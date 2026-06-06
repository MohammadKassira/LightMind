"""Tests for training/reward.py — compute_pressure, ObservationImputer, PressureReward."""
import math

import pytest
import torch

from env.mock_env import MockEnv
from training.reward import ObservationImputer, PressureReward, compute_pressure


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cross_smoke_env():
    return MockEnv("cross_smoke", max_steps=50)


@pytest.fixture()
def grid_env():
    return MockEnv("grid_3x3", max_steps=50)


@pytest.fixture()
def linear_env():
    return MockEnv("linear_two", max_steps=50)


def _obs_graph(env, seed=0):
    obs_dict, graph = env.reset(seed=seed)
    return obs_dict, graph


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestComputePressureStructure:
    def test_returns_dict(self, cross_smoke_env):
        obs_dict, graph = _obs_graph(cross_smoke_env)
        result = compute_pressure(obs_dict, graph)
        assert isinstance(result, dict)

    def test_keys_match_node_ids(self, cross_smoke_env):
        obs_dict, graph = _obs_graph(cross_smoke_env)
        result = compute_pressure(obs_dict, graph)
        assert set(result.keys()) == set(graph["node_ids"])

    def test_keys_match_node_ids_grid(self, grid_env):
        obs_dict, graph = _obs_graph(grid_env)
        result = compute_pressure(obs_dict, graph)
        assert set(result.keys()) == set(graph["node_ids"])

    def test_values_are_floats(self, cross_smoke_env):
        obs_dict, graph = _obs_graph(cross_smoke_env)
        result = compute_pressure(obs_dict, graph)
        for v in result.values():
            assert isinstance(v, float)

    def test_values_nonpositive(self, cross_smoke_env):
        obs_dict, graph = _obs_graph(cross_smoke_env)
        result = compute_pressure(obs_dict, graph)
        for v in result.values():
            assert v <= 0.0

    def test_values_nonpositive_grid(self, grid_env):
        obs_dict, graph = _obs_graph(grid_env)
        result = compute_pressure(obs_dict, graph)
        for v in result.values():
            assert v <= 0.0


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestComputePressureCorrectness:
    def _make_balanced_obs(self, graph, node_id, q_val=0.5):
        """Build obs where q_in == q_out == q_val for all lanes; validity all ones."""
        node_idx = graph["node_to_idx"][node_id]
        meta = graph["node_meta"][node_idx]
        phase_feats = graph["phase_features"][node_idx]
        num_phases = meta["num_phases"]
        num_incoming = len(phase_feats[0]) if phase_feats else 0
        num_outgoing = num_incoming

        obs_len = num_phases + 1 + 2 * num_incoming + num_outgoing
        obs = torch.zeros(obs_len)
        validity = torch.ones(obs_len)

        # phase one-hot
        if num_phases > 0:
            obs[0] = 1.0

        # incoming queues = q_val
        q_in_start = num_phases + 1
        obs[q_in_start : q_in_start + num_incoming] = q_val

        # outgoing queues = q_val
        q_out_start = q_in_start + 2 * num_incoming
        obs[q_out_start : q_out_start + num_outgoing] = q_val

        return obs, validity

    def _make_imbalanced_obs(self, graph, node_id, q_in_val, q_out_val):
        obs, validity = self._make_balanced_obs(graph, node_id, q_val=0.0)
        node_idx = graph["node_to_idx"][node_id]
        meta = graph["node_meta"][node_idx]
        phase_feats = graph["phase_features"][node_idx]
        num_phases = meta["num_phases"]
        num_incoming = len(phase_feats[0]) if phase_feats else 0

        q_in_start = num_phases + 1
        obs[q_in_start : q_in_start + num_incoming] = q_in_val

        q_out_start = q_in_start + 2 * num_incoming
        obs[q_out_start : q_out_start + num_incoming] = q_out_val
        return obs, validity

    def test_balanced_queues_give_zero_reward(self, cross_smoke_env):
        obs_dict, graph = _obs_graph(cross_smoke_env)
        node_id = graph["node_ids"][0]
        obs, validity = self._make_balanced_obs(graph, node_id, q_val=0.5)
        result = compute_pressure({node_id: (obs, validity)}, graph)
        assert abs(result[node_id]) < 1e-6

    def test_high_q_in_gives_negative_reward(self, cross_smoke_env):
        obs_dict, graph = _obs_graph(cross_smoke_env)
        node_id = graph["node_ids"][0]
        obs, validity = self._make_imbalanced_obs(graph, node_id, q_in_val=1.0, q_out_val=0.0)
        result = compute_pressure({node_id: (obs, validity)}, graph)
        assert result[node_id] < 0.0

    def test_high_q_out_gives_negative_reward(self, cross_smoke_env):
        """Absolute value: backing-up traffic is also penalised."""
        obs_dict, graph = _obs_graph(cross_smoke_env)
        node_id = graph["node_ids"][0]
        obs, validity = self._make_imbalanced_obs(graph, node_id, q_in_val=0.0, q_out_val=1.0)
        result = compute_pressure({node_id: (obs, validity)}, graph)
        assert result[node_id] < 0.0

    def test_larger_imbalance_gives_more_negative(self, cross_smoke_env):
        obs_dict, graph = _obs_graph(cross_smoke_env)
        node_id = graph["node_ids"][0]
        obs_small, val = self._make_imbalanced_obs(graph, node_id, q_in_val=0.3, q_out_val=0.0)
        obs_large, _ = self._make_imbalanced_obs(graph, node_id, q_in_val=1.0, q_out_val=0.0)
        r_small = compute_pressure({node_id: (obs_small, val)}, graph)[node_id]
        r_large = compute_pressure({node_id: (obs_large, val)}, graph)[node_id]
        assert r_large < r_small


# ---------------------------------------------------------------------------
# Validity / missing data
# ---------------------------------------------------------------------------

class TestComputePressureValidity:
    def test_all_missing_validity_no_crash(self, cross_smoke_env):
        obs_dict, graph = _obs_graph(cross_smoke_env)
        node_id = graph["node_ids"][0]
        obs, _ = obs_dict[node_id]
        all_missing_validity = torch.zeros_like(obs)
        result = compute_pressure({node_id: (obs, all_missing_validity)}, graph)
        assert math.isfinite(result[node_id])

    def test_all_missing_validity_no_nan(self, cross_smoke_env):
        obs_dict, graph = _obs_graph(cross_smoke_env)
        node_id = graph["node_ids"][0]
        obs, _ = obs_dict[node_id]
        all_missing_validity = torch.zeros_like(obs)
        result = compute_pressure({node_id: (obs, all_missing_validity)}, graph)
        assert not math.isnan(result[node_id])

    def test_missing_lanes_contribute_zero_pressure(self, cross_smoke_env):
        """Fully missing obs should give 0 pressure (sentinel * 0 = 0)."""
        obs_dict, graph = _obs_graph(cross_smoke_env)
        node_id = graph["node_ids"][0]
        obs, _ = obs_dict[node_id]
        sentinel_obs = torch.full_like(obs, -1.0)          # sentinel values
        all_missing = torch.zeros_like(obs)                # all invalid
        result = compute_pressure({node_id: (sentinel_obs, all_missing)}, graph)
        assert abs(result[node_id]) < 1e-6


# ---------------------------------------------------------------------------
# Integration: runs without error on real MockEnv output
# ---------------------------------------------------------------------------

class TestComputePressureIntegration:
    def test_works_on_cross_smoke(self, cross_smoke_env):
        obs_dict, graph = _obs_graph(cross_smoke_env)
        result = compute_pressure(obs_dict, graph)
        assert len(result) == len(graph["node_ids"])
        for v in result.values():
            assert math.isfinite(v)

    def test_works_on_grid_3x3(self, grid_env):
        obs_dict, graph = _obs_graph(grid_env)
        result = compute_pressure(obs_dict, graph)
        assert len(result) == len(graph["node_ids"])
        for v in result.values():
            assert math.isfinite(v)

    def test_works_on_linear_two(self, linear_env):
        obs_dict, graph = _obs_graph(linear_env)
        result = compute_pressure(obs_dict, graph)
        assert len(result) == len(graph["node_ids"])
        for v in result.values():
            assert math.isfinite(v)

    def test_consistent_with_env_step(self, cross_smoke_env):
        obs_dict, graph = cross_smoke_env.reset(seed=7)
        actions = {nid: 0 for nid in graph["node_ids"]}
        next_obs_dict, _, _, _, _ = cross_smoke_env.step(actions)
        result = compute_pressure(next_obs_dict, graph)
        for v in result.values():
            assert math.isfinite(v)


# ---------------------------------------------------------------------------
# ObservationImputer
# ---------------------------------------------------------------------------

def _make_obs_with_missing(obs_dim, missing_indices, sentinel=-1.0):
    obs = torch.rand(obs_dim)
    validity = torch.ones(obs_dim)
    for i in missing_indices:
        obs[i] = sentinel
        validity[i] = 0.0
    return obs, validity


class TestObservationImputer:
    def test_reset_clears_state(self):
        imp = ObservationImputer()
        obs = torch.tensor([0.5, 0.3])
        val = torch.ones(2)
        imp.impute({"n0": (obs, val)})
        imp.reset()
        assert imp._last_obs == {}

    def test_valid_positions_unchanged(self):
        imp = ObservationImputer()
        obs = torch.tensor([0.5, 0.3, 0.9])
        val = torch.ones(3)
        result = imp.impute({"n0": (obs, val)})
        out_obs, out_val = result["n0"]
        assert torch.allclose(out_obs, obs)
        assert torch.allclose(out_val, val)

    def test_missing_first_step_stays_sentinel(self):
        imp = ObservationImputer()
        obs = torch.tensor([-1.0, 0.4])
        val = torch.tensor([0.0, 1.0])
        result = imp.impute({"n0": (obs, val)})
        out_obs, out_val = result["n0"]
        assert out_obs[0].item() == -1.0
        assert out_val[0].item() == 0.0

    def test_missing_after_seen_uses_last_known(self):
        imp = ObservationImputer()
        # Step 1: valid reading
        obs1 = torch.tensor([0.7, 0.3])
        val1 = torch.ones(2)
        imp.impute({"n0": (obs1, val1)})
        # Step 2: first feature missing
        obs2 = torch.tensor([-1.0, 0.5])
        val2 = torch.tensor([0.0, 1.0])
        result = imp.impute({"n0": (obs2, val2)})
        out_obs, out_val = result["n0"]
        assert abs(out_obs[0].item() - 0.7) < 1e-6   # last-known value filled in
        assert out_val[0].item() == 0.0               # validity unchanged — sensor failed

    def test_validity_stays_zero_when_imputed(self):
        """Imputer fills obs but never flips validity — sensor failure is always recorded."""
        imp = ObservationImputer()
        obs1 = torch.tensor([0.5])
        val1 = torch.ones(1)
        imp.impute({"n0": (obs1, val1)})
        obs2 = torch.tensor([-1.0])
        val2 = torch.zeros(1)
        result = imp.impute({"n0": (obs2, val2)})
        _, out_val = result["n0"]
        assert out_val[0].item() == 0.0   # stays 0 — imputed, not live

    def test_no_episode_bleed_after_reset(self):
        imp = ObservationImputer()
        obs1 = torch.tensor([0.8, 0.2])
        val1 = torch.ones(2)
        imp.impute({"n0": (obs1, val1)})
        imp.reset()
        obs2 = torch.tensor([-1.0, 0.4])
        val2 = torch.tensor([0.0, 1.0])
        result = imp.impute({"n0": (obs2, val2)})
        out_obs, out_val = result["n0"]
        # After reset, no history → sentinel stays
        assert out_obs[0].item() == -1.0
        assert out_val[0].item() == 0.0

    def test_input_not_mutated(self):
        imp = ObservationImputer()
        obs = torch.tensor([0.5, 0.3])
        val = torch.ones(2)
        obs_clone = obs.clone()
        val_clone = val.clone()
        imp.impute({"n0": (obs, val)})
        assert torch.allclose(obs, obs_clone)
        assert torch.allclose(val, val_clone)

    def test_returns_new_dict(self):
        imp = ObservationImputer()
        obs = torch.tensor([0.5])
        val = torch.ones(1)
        d = {"n0": (obs, val)}
        result = imp.impute(d)
        assert result is not d


# ---------------------------------------------------------------------------
# PressureReward
# ---------------------------------------------------------------------------

class TestPressureReward:
    def test_reset_clears_state(self, cross_smoke_env):
        pr = PressureReward()
        obs_dict, graph = cross_smoke_env.reset(seed=0)
        pr.compute(obs_dict, graph)
        pr.reset()
        assert pr._last_obs == {}

    def test_returns_finite_values(self, cross_smoke_env):
        pr = PressureReward()
        obs_dict, graph = cross_smoke_env.reset(seed=1)
        result = pr.compute(obs_dict, graph)
        for v in result.values():
            assert math.isfinite(v)

    def test_returns_nonpositive_values(self, cross_smoke_env):
        pr = PressureReward()
        obs_dict, graph = cross_smoke_env.reset(seed=2)
        result = pr.compute(obs_dict, graph)
        for v in result.values():
            assert v <= 0.0 + 1e-9

    def test_valid_readings_update_cache(self, cross_smoke_env):
        pr = PressureReward()
        obs_dict, graph = cross_smoke_env.reset(seed=3)
        pr.compute(obs_dict, graph)
        assert len(pr._last_obs) > 0

    def test_missing_lane_uses_last_known_not_zero(self, cross_smoke_env):
        """After a valid reading, zeroing validity should not change reward if we substitute last-known."""
        pr = PressureReward()
        obs_dict, graph = cross_smoke_env.reset(seed=4)
        # Step 1: all valid
        r1 = pr.compute(obs_dict, graph)
        # Step 2: set all validity to 0 — PressureReward should fill with last-known
        zeroed_dict = {}
        for node_id, (obs, validity) in obs_dict.items():
            zeroed_dict[node_id] = (obs, torch.zeros_like(validity))
        r2 = pr.compute(zeroed_dict, graph)
        # rewards should be finite and close (last-known ≈ same obs)
        for node_id in graph["node_ids"]:
            assert math.isfinite(r2[node_id])
            assert abs(r2[node_id] - r1[node_id]) < 1e-4

    def test_no_episode_bleed_after_reset(self, cross_smoke_env):
        pr = PressureReward()
        obs_dict, graph = cross_smoke_env.reset(seed=5)
        pr.compute(obs_dict, graph)
        pr.reset()
        # After reset, missing obs should treat missing as 0 (no last-known)
        zeroed_dict = {}
        for node_id, (obs, validity) in obs_dict.items():
            zeroed_dict[node_id] = (torch.full_like(obs, -1.0), torch.zeros_like(validity))
        result = pr.compute(zeroed_dict, graph)
        for v in result.values():
            assert math.isfinite(v)
            assert abs(v) < 1e-6   # all missing + no last-known → pressure ≈ 0

    def test_first_step_all_missing_no_crash(self, cross_smoke_env):
        pr = PressureReward()
        obs_dict, graph = cross_smoke_env.reset(seed=6)
        sentinel_dict = {
            node_id: (torch.full_like(obs, -1.0), torch.zeros_like(validity))
            for node_id, (obs, validity) in obs_dict.items()
        }
        result = pr.compute(sentinel_dict, graph)
        for v in result.values():
            assert math.isfinite(v)


# ---------------------------------------------------------------------------
# Mixed reward (queue_weight + pressure_weight)
# ---------------------------------------------------------------------------

class TestMixedReward:
    """Verify the combined reward formula: -(qw*q_in/n + pw*|q_in-q_out|/n)"""

    def _single_node_obs(self, graph, node_id, q_in_val, q_out_val):
        """Build a single-node obs_dict with controlled q_in and q_out."""
        node_idx  = graph["node_to_idx"][node_id]
        meta      = graph["node_meta"][node_idx]
        pf        = graph["phase_features"][node_idx]
        num_phases   = meta["num_phases"]
        num_incoming = len(pf[0]) if pf else 0

        obs_len = num_phases + 1 + 2 * num_incoming + num_incoming
        obs      = torch.zeros(obs_len)
        validity = torch.ones(obs_len)
        if num_phases > 0:
            obs[0] = 1.0
        q_in_start  = num_phases + 1
        q_out_start = q_in_start + 2 * num_incoming
        obs[q_in_start  : q_in_start  + num_incoming] = q_in_val
        obs[q_out_start : q_out_start + num_incoming] = q_out_val
        return {node_id: (obs, validity)}

    def test_pure_queue_balanced_gives_negative(self, cross_smoke_env):
        """queue_weight=1 always penalises q_in even when balanced — unlike pure pressure."""
        obs_dict, graph = cross_smoke_env.reset(seed=0)
        node_id = graph["node_ids"][0]
        pr = PressureReward(queue_weight=1.0, pressure_weight=0.0)
        obs_in = self._single_node_obs(graph, node_id, q_in_val=0.5, q_out_val=0.5)
        result = pr.compute(obs_in, graph)
        assert result[node_id] < 0.0  # balanced queue still has q_in > 0

    def test_pure_pressure_balanced_gives_zero(self, cross_smoke_env):
        """Confirm the old behaviour: balanced queues → reward = 0 with pure pressure."""
        obs_dict, graph = cross_smoke_env.reset(seed=0)
        node_id = graph["node_ids"][0]
        pr = PressureReward(queue_weight=0.0, pressure_weight=1.0)
        obs_in = self._single_node_obs(graph, node_id, q_in_val=0.5, q_out_val=0.5)
        result = pr.compute(obs_in, graph)
        assert abs(result[node_id]) < 1e-6  # balanced → zero pressure → gameable

    def test_mixed_formula_arithmetic(self, cross_smoke_env):
        """reward = -(0.5*q_in/n + 0.5*|q_in-q_out|/n); verify exact value."""
        obs_dict, graph = cross_smoke_env.reset(seed=0)
        node_id = graph["node_ids"][0]
        node_idx     = graph["node_to_idx"][node_id]
        num_incoming = len(graph["phase_features"][node_idx][0])

        q_in, q_out = 0.8, 0.2
        pr = PressureReward(queue_weight=0.5, pressure_weight=0.5)
        obs_in = self._single_node_obs(graph, node_id, q_in_val=q_in, q_out_val=q_out)
        result = pr.compute(obs_in, graph)

        expected_queue    = q_in                   # q_in_sum/n = q_in (all lanes equal)
        expected_pressure = abs(q_in - q_out)      # |q_in_sum - q_out_sum|/n
        expected = -(0.5 * expected_queue + 0.5 * expected_pressure)
        assert abs(result[node_id] - expected) < 1e-5

    def test_mixed_more_negative_than_pure_pressure_when_balanced(self, cross_smoke_env):
        """Mixed reward penalises balanced gridlock; pure pressure does not."""
        obs_dict, graph = cross_smoke_env.reset(seed=0)
        node_id = graph["node_ids"][0]
        pr_mixed    = PressureReward(queue_weight=0.5, pressure_weight=0.5)
        pr_pressure = PressureReward(queue_weight=0.0, pressure_weight=1.0)
        obs_in = self._single_node_obs(graph, node_id, q_in_val=0.5, q_out_val=0.5)
        r_mixed    = pr_mixed.compute(obs_in, graph)[node_id]
        r_pressure = pr_pressure.compute(obs_in, graph)[node_id]
        assert r_mixed < r_pressure  # mixed correctly penalises; pure pressure gives 0

    def test_mixed_nonpositive(self, cross_smoke_env):
        obs_dict, graph = cross_smoke_env.reset(seed=7)
        pr = PressureReward(queue_weight=0.5, pressure_weight=0.5)
        result = pr.compute(obs_dict, graph)
        for v in result.values():
            assert v <= 1e-9
