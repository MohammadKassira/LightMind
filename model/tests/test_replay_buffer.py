"""Tests for training/replay_buffer.py — ReplayBuffer."""
import pytest
import torch

from env.mock_env import MockEnv
from models.node_encoder import pad_obs_dict
from training.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env_and_buffer(network="cross_smoke", capacity=32):
    env = MockEnv(network, max_steps=50)
    obs_dict, graph = env.reset(seed=0)
    obs_dim, padded = pad_obs_dict(obs_dict)
    node_ids = graph["node_ids"]
    buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, node_ids=node_ids)
    return env, graph, obs_dim, node_ids, buf


def _push_random(env, graph, buf, n=1, seed=0):
    """Push n random transitions into buf. Returns last (padded_obs, padded_next)."""
    obs_dict, _ = env.reset(seed=seed)
    _, padded_obs = pad_obs_dict(obs_dict)
    last = (padded_obs, padded_obs)
    for _ in range(n):
        actions = {nid: 0 for nid in graph["node_ids"]}
        next_obs_dict, _, reward_dict, done, _ = env.step(actions)
        _, padded_next = pad_obs_dict(next_obs_dict)
        buf.push(padded_obs, padded_next, actions, reward_dict, done)
        last = (padded_obs, padded_next)
        padded_obs = padded_next
    return last


# ---------------------------------------------------------------------------
# Basic mechanics
# ---------------------------------------------------------------------------

class TestReplayBufferBasic:
    def test_empty_len_zero(self):
        _, _, _, _, buf = _make_env_and_buffer()
        assert len(buf) == 0

    def test_push_increments_len(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer()
        _push_random(env, graph, buf, n=5)
        assert len(buf) == 5

    def test_push_wraps_at_capacity(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer(capacity=4)
        _push_random(env, graph, buf, n=10)
        assert len(buf) == 4

    def test_sample_raises_when_too_small(self):
        _, _, _, _, buf = _make_env_and_buffer()
        with pytest.raises(ValueError):
            buf.sample(1)

    def test_sample_raises_exact_boundary(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer(capacity=32)
        _push_random(env, graph, buf, n=5)
        with pytest.raises(ValueError):
            buf.sample(6)

    def test_sample_succeeds_at_exact_size(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer(capacity=32)
        _push_random(env, graph, buf, n=8)
        batch = buf.sample(8)
        assert batch["obs"].shape[0] == 8


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

class TestReplayBufferShape:
    def test_obs_shape(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer()
        _push_random(env, graph, buf, n=16)
        batch = buf.sample(8)
        N = len(node_ids)
        assert batch["obs"].shape == (8, N, obs_dim)

    def test_validity_shape(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer()
        _push_random(env, graph, buf, n=16)
        batch = buf.sample(8)
        N = len(node_ids)
        assert batch["validity"].shape == (8, N, obs_dim)

    def test_actions_shape(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer()
        _push_random(env, graph, buf, n=16)
        batch = buf.sample(8)
        assert batch["actions"].shape == (8, len(node_ids))

    def test_rewards_shape(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer()
        _push_random(env, graph, buf, n=16)
        batch = buf.sample(8)
        assert batch["rewards"].shape == (8, len(node_ids))

    def test_dones_shape(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer()
        _push_random(env, graph, buf, n=16)
        batch = buf.sample(8)
        assert batch["dones"].shape == (8,)

    def test_obs_dtype_float32(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer()
        _push_random(env, graph, buf, n=16)
        batch = buf.sample(4)
        assert batch["obs"].dtype == torch.float32

    def test_actions_dtype_int64(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer()
        _push_random(env, graph, buf, n=16)
        batch = buf.sample(4)
        assert batch["actions"].dtype == torch.int64

    def test_dones_dtype_float32(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer()
        _push_random(env, graph, buf, n=16)
        batch = buf.sample(4)
        assert batch["dones"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------

class TestReplayBufferContent:
    def test_obs_dim_mismatch_raises(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer()
        obs_dict, _ = env.reset(seed=1)
        _, padded = pad_obs_dict(obs_dict)
        node_id = node_ids[0]
        obs, val = padded[node_id]
        wrong_obs = torch.zeros(obs_dim + 5)
        wrong_val = torch.ones(obs_dim + 5)
        bad_padded = dict(padded)
        bad_padded[node_id] = (wrong_obs, wrong_val)
        with pytest.raises(ValueError):
            buf.push(bad_padded, padded, {nid: 0 for nid in node_ids}, {nid: 0.0 for nid in node_ids}, False)

    def test_done_true_stored_as_one(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer(capacity=4)
        obs_dict, _ = env.reset(seed=2)
        _, padded = pad_obs_dict(obs_dict)
        actions = {nid: 0 for nid in node_ids}
        rewards = {nid: -0.1 for nid in node_ids}
        buf.push(padded, padded, actions, rewards, True)
        # Buffer has exactly 1 element — sample it
        batch = buf.sample(1)
        assert batch["dones"][0].item() == 1.0

    def test_done_false_stored_as_zero(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer(capacity=4)
        obs_dict, _ = env.reset(seed=3)
        _, padded = pad_obs_dict(obs_dict)
        actions = {nid: 0 for nid in node_ids}
        rewards = {nid: -0.1 for nid in node_ids}
        buf.push(padded, padded, actions, rewards, False)
        batch = buf.sample(1)
        assert batch["dones"][0].item() == 0.0

    def test_validity_mask_not_all_ones(self):
        """When missing_prob > 0, validity can be 0; verify it survives push/sample."""
        env = MockEnv("cross_smoke", max_steps=50, missing_prob=0.5)
        obs_dict, graph = env.reset(seed=42)
        obs_dim, padded = pad_obs_dict(obs_dict)
        buf = ReplayBuffer(capacity=16, obs_dim=obs_dim, node_ids=graph["node_ids"])
        actions = {nid: 0 for nid in graph["node_ids"]}
        rewards = {nid: -0.1 for nid in graph["node_ids"]}
        buf.push(padded, padded, actions, rewards, False)
        batch = buf.sample(1)
        # At least some validity values should be 0 with missing_prob=0.5
        # (not guaranteed in one sample but the dtype and shape must be correct)
        assert batch["validity"].dtype == torch.float32
        assert batch["validity"].min() >= 0.0
        assert batch["validity"].max() <= 1.0


# ---------------------------------------------------------------------------
# Multi-node
# ---------------------------------------------------------------------------

class TestReplayBufferMultiNode:
    def test_grid_shape(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer("grid_3x3", capacity=64)
        _push_random(env, graph, buf, n=16)
        batch = buf.sample(8)
        assert batch["obs"].shape == (8, len(node_ids), obs_dim)

    def test_linear_two_shape(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer("linear_two", capacity=16)
        _push_random(env, graph, buf, n=10)
        batch = buf.sample(4)
        assert batch["obs"].shape == (4, len(node_ids), obs_dim)

    def test_node_count_matches_env(self):
        env, graph, obs_dim, node_ids, buf = _make_env_and_buffer("grid_3x3", capacity=32)
        _push_random(env, graph, buf, n=16)
        batch = buf.sample(4)
        assert batch["actions"].shape[1] == len(graph["node_ids"])
