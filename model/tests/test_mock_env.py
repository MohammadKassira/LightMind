"""Tests for env/mock_env.py across four synthetic SUMO networks.

See docs/mock_env.md for full documentation, the §3.2 obs vector layout,
and a description of what each test class covers.
"""

from pathlib import Path

import pytest
import torch
import yaml

from env.mock_env import MockEnv

CONFIGS = Path(__file__).parent.parent / "configs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def expected_obs_size(g, node_idx):
    """Compute expected obs vector length from graph metadata (§3.2 layout)."""
    num_phases = g["node_meta"][node_idx]["num_phases"]
    phase_feats = g["phase_features"][node_idx]
    num_incoming = len(phase_feats[0]) if phase_feats else 0
    num_outgoing = num_incoming  # mock approximation: symmetric intersection
    return num_phases + 1 + 2 * num_incoming + num_outgoing


def read_sentinel():
    cfg = yaml.safe_load((CONFIGS / "perception.yaml").read_text())
    return cfg["perception"]["sentinel_value"]


# ---------------------------------------------------------------------------
# cross_smoke — single intersection, default settings (no missing data)
# ---------------------------------------------------------------------------

class TestMockEnvCrossSmoke:
    @pytest.fixture(scope="class")
    def reset_out(self):
        env = MockEnv("cross_smoke")
        obs_dict, g = env.reset(seed=0)
        return obs_dict, g

    # --- Return types and structure ---

    def test_reset_returns_two_values(self, reset_out):
        obs_dict, g = reset_out
        assert isinstance(obs_dict, dict)
        assert isinstance(g, dict)

    def test_obs_keys_match_node_ids(self, reset_out):
        obs_dict, g = reset_out
        assert set(obs_dict.keys()) == set(g["node_ids"])

    def test_obs_and_validity_are_tensors(self, reset_out):
        obs_dict, _ = reset_out
        obs_t, val = obs_dict["A0"]
        assert isinstance(obs_t, torch.Tensor)
        assert isinstance(val, torch.Tensor)

    def test_obs_dtype_float32(self, reset_out):
        obs_dict, _ = reset_out
        obs_t, _ = obs_dict["A0"]
        assert obs_t.dtype == torch.float32

    def test_validity_dtype_float32(self, reset_out):
        obs_dict, _ = reset_out
        _, val = obs_dict["A0"]
        assert val.dtype == torch.float32

    def test_obs_and_validity_same_shape(self, reset_out):
        obs_dict, _ = reset_out
        obs_t, val = obs_dict["A0"]
        assert obs_t.shape == val.shape

    def test_obs_size_matches_schema(self, reset_out):
        obs_dict, g = reset_out
        obs_t, _ = obs_dict["A0"]
        assert obs_t.shape[0] == expected_obs_size(g, 0)

    # --- Values ---

    def test_validity_all_ones_by_default(self, reset_out):
        obs_dict, _ = reset_out
        _, val = obs_dict["A0"]
        assert val.eq(1.0).all()

    def test_phase_onehot_valid(self, reset_out):
        obs_dict, g = reset_out
        obs_t, _ = obs_dict["A0"]
        num_phases = g["node_meta"][0]["num_phases"]
        onehot = obs_t[:num_phases]
        assert onehot.sum().item() == pytest.approx(1.0)
        assert set(onehot.tolist()).issubset({0.0, 1.0})

    def test_time_in_phase_in_range(self, reset_out):
        obs_dict, g = reset_out
        obs_t, _ = obs_dict["A0"]
        num_phases = g["node_meta"][0]["num_phases"]
        t = obs_t[num_phases].item()
        assert 0.0 <= t <= 1.0

    def test_lane_values_in_range(self, reset_out):
        obs_dict, g = reset_out
        obs_t, _ = obs_dict["A0"]
        num_phases = g["node_meta"][0]["num_phases"]
        lanes = obs_t[num_phases + 1:]
        assert (lanes >= 0.0).all()
        assert (lanes <= 1.0).all()

    # --- Step ---

    def test_step_returns_five_values(self):
        e = MockEnv("cross_smoke")
        e.reset(seed=0)
        result = e.step({})
        assert len(result) == 5

    def test_step_obs_keys_match(self):
        e = MockEnv("cross_smoke")
        obs_dict, g = e.reset(seed=0)
        obs2, _, _, _, _ = e.step({})
        assert set(obs2.keys()) == set(g["node_ids"])

    def test_reward_is_negative(self):
        e = MockEnv("cross_smoke")
        e.reset(seed=0)
        _, _, rew, _, _ = e.step({})
        assert all(v <= 0.0 for v in rew.values())

    def test_reward_is_float(self):
        e = MockEnv("cross_smoke")
        e.reset(seed=0)
        _, _, rew, _, _ = e.step({})
        assert all(isinstance(v, float) for v in rew.values())

    def test_done_false_before_max_steps(self):
        e = MockEnv("cross_smoke", max_steps=5)
        e.reset(seed=0)
        _, _, _, done, _ = e.step({})
        assert not done

    def test_done_true_at_max_steps(self):
        e = MockEnv("cross_smoke", max_steps=3)
        e.reset(seed=0)
        for _ in range(3):
            _, _, _, done, _ = e.step({})
        assert done

    def test_reset_clears_step_count(self):
        e = MockEnv("cross_smoke", max_steps=2)
        e.reset(seed=0)
        e.step({})
        _, _, _, done_at_2, _ = e.step({})
        assert done_at_2
        e.reset(seed=1)
        _, _, _, done_after_reset, _ = e.step({})
        assert not done_after_reset

    def test_info_keys(self):
        e = MockEnv("cross_smoke")
        e.reset(seed=0)
        _, _, _, _, info = e.step({})
        assert "step_mean_waiting_time" in info
        assert "step_throughput" in info
        assert "step_queue_length" in info

    def test_seed_determinism(self):
        e1 = MockEnv("cross_smoke")
        e2 = MockEnv("cross_smoke")
        obs1, _ = e1.reset(seed=42)
        obs2, _ = e2.reset(seed=42)
        obs1_t, _ = obs1["A0"]
        obs2_t, _ = obs2["A0"]
        assert torch.allclose(obs1_t, obs2_t)


# ---------------------------------------------------------------------------
# Missing data — sentinel and validity mask behaviour
# ---------------------------------------------------------------------------

class TestMockEnvMissingData:

    def test_all_obs_are_sentinel(self):
        e = MockEnv("cross_smoke", missing_prob=1.0)
        obs_dict, _ = e.reset(seed=0)
        obs_t, _ = obs_dict["A0"]
        assert (obs_t == read_sentinel()).all()

    def test_all_validity_zero(self):
        e = MockEnv("cross_smoke", missing_prob=1.0)
        obs_dict, _ = e.reset(seed=0)
        _, val = obs_dict["A0"]
        assert (val == 0.0).all()

    def test_sentinel_value_from_config(self):
        """Sentinel is driven by perception.yaml, not hardcoded in the env."""
        sentinel = read_sentinel()
        e = MockEnv("cross_smoke", missing_prob=1.0)
        obs_dict, _ = e.reset(seed=0)
        obs_t, _ = obs_dict["A0"]
        assert (obs_t == sentinel).all()

    def test_no_sentinel_when_prob_0(self):
        e = MockEnv("cross_smoke", missing_prob=0.0)
        obs_dict, _ = e.reset(seed=0)
        obs_t, _ = obs_dict["A0"]
        assert (obs_t != read_sentinel()).all()

    def test_no_missing_validity_all_ones(self):
        e = MockEnv("cross_smoke", missing_prob=0.0)
        obs_dict, _ = e.reset(seed=0)
        _, val = obs_dict["A0"]
        assert (val == 1.0).all()

    def test_missing_implies_sentinel(self):
        """Wherever validity==0, obs must equal the sentinel value."""
        e = MockEnv("cross_smoke", missing_prob=0.5)
        e.reset(seed=42)
        sentinel = read_sentinel()
        for _ in range(20):
            obs_dict, _, _, _, _ = e.step({})
            obs_t, val = obs_dict["A0"]
            missing = val == 0.0
            if missing.any():
                assert (obs_t[missing] == sentinel).all()

    def test_valid_implies_not_sentinel(self):
        """Wherever validity==1, obs must not equal the sentinel value."""
        e = MockEnv("cross_smoke", missing_prob=0.5)
        e.reset(seed=42)
        sentinel = read_sentinel()
        for _ in range(20):
            obs_dict, _, _, _, _ = e.step({})
            obs_t, val = obs_dict["A0"]
            valid = val == 1.0
            if valid.any():
                assert (obs_t[valid] != sentinel).all()


# ---------------------------------------------------------------------------
# linear_two — two intersections with different lane counts
# ---------------------------------------------------------------------------

class TestMockEnvLinearTwo:
    @pytest.fixture(scope="class")
    def reset_out(self):
        env = MockEnv("linear_two")
        obs_dict, g = env.reset(seed=0)
        return obs_dict, g

    def test_two_nodes_in_obs(self, reset_out):
        obs_dict, _ = reset_out
        assert len(obs_dict) == 2

    def test_obs_keys_match_graph(self, reset_out):
        obs_dict, g = reset_out
        assert set(obs_dict.keys()) == set(g["node_ids"])

    def test_each_node_obs_size(self, reset_out):
        obs_dict, g = reset_out
        for i, nid in enumerate(g["node_ids"]):
            obs_t, _ = obs_dict[nid]
            assert obs_t.shape[0] == expected_obs_size(g, i), f"Node {nid} obs size mismatch"

    def test_nodes_can_have_different_obs_sizes(self, reset_out):
        """Obs size is derived from the graph at runtime — proves network-agnostic design."""
        obs_dict, g = reset_out
        sizes = [obs_dict[nid][0].shape[0] for nid in g["node_ids"]]
        assert len(set(sizes)) > 1

    def test_reward_both_nodes(self):
        e = MockEnv("linear_two")
        _, g = e.reset(seed=0)
        _, _, rew, _, _ = e.step({})
        assert set(rew.keys()) == set(g["node_ids"])


# ---------------------------------------------------------------------------
# grid_3x3 — nine intersections
# ---------------------------------------------------------------------------

class TestMockEnvGrid3x3:
    @pytest.fixture(scope="class")
    def reset_out(self):
        env = MockEnv("grid_3x3")
        obs_dict, g = env.reset(seed=0)
        return obs_dict, g

    def test_nine_nodes_in_obs(self, reset_out):
        obs_dict, _ = reset_out
        assert len(obs_dict) == 9

    def test_obs_keys_match_graph(self, reset_out):
        obs_dict, g = reset_out
        assert set(obs_dict.keys()) == set(g["node_ids"])

    def test_all_nodes_have_valid_obs_size(self, reset_out):
        obs_dict, g = reset_out
        for i, nid in enumerate(g["node_ids"]):
            obs_t, _ = obs_dict[nid]
            assert obs_t.shape[0] == expected_obs_size(g, i), f"Node {nid} obs size mismatch"


# ---------------------------------------------------------------------------
# Cross-network invariants — parametrized over all four test networks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("network", [
    "cross_smoke", "linear_two", "pass_through", "grid_3x3",
])
def test_obs_keys_equal_graph_node_ids(network):
    e = MockEnv(network)
    obs_dict, g = e.reset(seed=0)
    assert set(obs_dict.keys()) == set(g["node_ids"])


@pytest.mark.parametrize("network", [
    "cross_smoke", "linear_two", "pass_through", "grid_3x3",
])
def test_obs_and_validity_same_shape_all_nodes(network):
    e = MockEnv(network)
    obs_dict, _ = e.reset(seed=0)
    for nid, (obs_t, val) in obs_dict.items():
        assert obs_t.shape == val.shape, f"Shape mismatch for node {nid}"


@pytest.mark.parametrize("network", [
    "cross_smoke", "linear_two", "pass_through", "grid_3x3",
])
def test_validity_binary(network):
    e = MockEnv(network)
    obs_dict, _ = e.reset(seed=0)
    for nid, (_, val) in obs_dict.items():
        assert set(val.tolist()).issubset({0.0, 1.0}), f"Non-binary validity for node {nid}"


@pytest.mark.parametrize("network", [
    "cross_smoke", "linear_two", "pass_through", "grid_3x3",
])
def test_obs_dtype_float32_all_nodes(network):
    e = MockEnv(network)
    obs_dict, _ = e.reset(seed=0)
    for nid, (obs_t, _) in obs_dict.items():
        assert obs_t.dtype == torch.float32, f"Wrong dtype for node {nid}"


@pytest.mark.parametrize("network", [
    "cross_smoke", "linear_two", "pass_through", "grid_3x3",
])
def test_reward_keys_equal_node_ids(network):
    e = MockEnv(network)
    _, g = e.reset(seed=0)
    _, _, rew, _, _ = e.step({})
    assert set(rew.keys()) == set(g["node_ids"])


@pytest.mark.parametrize("network", [
    "cross_smoke", "linear_two", "pass_through", "grid_3x3",
])
def test_reward_nonpositive(network):
    e = MockEnv(network)
    e.reset(seed=0)
    _, _, rew, _, _ = e.step({})
    assert all(v <= 0.0 for v in rew.values())


@pytest.mark.parametrize("network", [
    "cross_smoke", "linear_two", "pass_through", "grid_3x3",
])
def test_graph_is_same_object_across_steps(network):
    """The graph dict is static — reset and step return the same Python object."""
    e = MockEnv(network)
    _, g1 = e.reset(seed=0)
    _, g2, _, _, _ = e.step({})
    assert g1 is g2


@pytest.mark.parametrize("network", [
    "cross_smoke", "linear_two", "pass_through", "grid_3x3",
])
def test_step_count_resets(network):
    e = MockEnv(network, max_steps=2)
    e.reset(seed=0)
    e.step({})
    _, _, _, done_at_2, _ = e.step({})
    assert done_at_2
    e.reset(seed=1)
    _, _, _, done_after_reset, _ = e.step({})
    assert not done_after_reset
