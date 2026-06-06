"""Tests for env/perception.py — apply_perception."""
import math

import pytest
import torch

from env.mock_env import MockEnv
from env.perception import apply_perception
from models.node_encoder import NodeEncoder, pad_obs_dict
from models.phase_head import pad_phase_features
from training.trainer import DQNTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_graph(network="cross_smoke", seed=0):
    env = MockEnv(network, max_steps=20)
    obs_dict, graph = env.reset(seed=seed)
    return obs_dict, graph, env


# ---------------------------------------------------------------------------
# Clean (severity=0.0)
# ---------------------------------------------------------------------------

class TestApplyPerceptionClean:
    def test_obs_values_match_input(self):
        obs_dict, graph, _ = _obs_graph()
        out = apply_perception(obs_dict, severity=0.0)
        for node_id, (obs, _) in obs_dict.items():
            assert torch.allclose(out[node_id][0], obs)

    def test_validity_values_match_input(self):
        obs_dict, graph, _ = _obs_graph()
        out = apply_perception(obs_dict, severity=0.0)
        for node_id, (_, val) in obs_dict.items():
            assert torch.allclose(out[node_id][1], val)

    def test_returns_new_dict(self):
        obs_dict, graph, _ = _obs_graph()
        out = apply_perception(obs_dict, severity=0.0)
        assert out is not obs_dict

    @pytest.mark.parametrize("network", ["cross_smoke", "linear_two", "grid_3x3"])
    def test_works_on_multiple_networks(self, network):
        obs_dict, graph, _ = _obs_graph(network)
        out = apply_perception(obs_dict, severity=0.0)
        assert set(out.keys()) == set(obs_dict.keys())
        for node_id, (obs, _) in obs_dict.items():
            assert torch.allclose(out[node_id][0], obs)


# ---------------------------------------------------------------------------
# Full (severity=1.0)
# ---------------------------------------------------------------------------

class TestApplyPerceptionFull:
    def test_all_validity_zero(self):
        obs_dict, graph, _ = _obs_graph()
        out = apply_perception(obs_dict, severity=1.0)
        for node_id, (_, val) in out.items():
            assert (val == 0.0).all(), f"Expected all validity=0 for {node_id}"

    def test_all_obs_equal_sentinel(self):
        obs_dict, graph, _ = _obs_graph()
        out = apply_perception(obs_dict, severity=1.0, sentinel=-1.0)
        for node_id, (obs, _) in out.items():
            assert (obs == -1.0).all(), f"Expected all obs=sentinel for {node_id}"

    def test_no_nan_in_output(self):
        obs_dict, graph, _ = _obs_graph()
        out = apply_perception(obs_dict, severity=1.0)
        for node_id, (obs, val) in out.items():
            assert not torch.isnan(obs).any()
            assert not torch.isnan(val).any()

    def test_encoder_no_nan_on_full_severity(self):
        obs_dict, graph, _ = _obs_graph()
        out = apply_perception(obs_dict, severity=1.0)
        obs_dim, padded = pad_obs_dict(out)
        encoder = NodeEncoder(obs_dim, hidden_dim=32, embed_dim=16)
        encoder.eval()
        with torch.no_grad():
            for node_id, (obs, val) in padded.items():
                emb = encoder(obs, val)
                assert not torch.isnan(emb).any(), f"NaN embedding for {node_id}"


# ---------------------------------------------------------------------------
# Partial (severity=0.5)
# ---------------------------------------------------------------------------

class TestApplyPerceptionPartial:
    def test_some_corrupted_some_not(self):
        """Over 100 samples, at least some features should be corrupted and some not."""
        obs_dict, graph, _ = _obs_graph(seed=42)
        any_corrupted = False
        any_clean = False
        for _ in range(100):
            out = apply_perception(obs_dict, severity=0.5)
            for node_id, (obs, val) in out.items():
                if (val == 0.0).any():
                    any_corrupted = True
                if (val == 1.0).any():
                    any_clean = True
        assert any_corrupted, "No corruption observed over 100 samples at severity=0.5"
        assert any_clean, "No clean features observed over 100 samples at severity=0.5"

    def test_corrupted_positions_obs_equals_sentinel(self):
        obs_dict, graph, _ = _obs_graph(seed=7)
        for _ in range(20):
            out = apply_perception(obs_dict, severity=0.5)
            for node_id, (obs, val) in out.items():
                corrupt_mask = val == 0.0
                if corrupt_mask.any():
                    assert (obs[corrupt_mask] == -1.0).all()

    def test_corrupted_positions_validity_is_zero(self):
        obs_dict, graph, _ = _obs_graph(seed=8)
        orig_obs = {nid: o.clone() for nid, (o, _) in obs_dict.items()}
        for _ in range(20):
            out = apply_perception(obs_dict, severity=0.5)
            for node_id, (obs, val) in out.items():
                orig = orig_obs[node_id]
                changed = ~torch.isclose(obs, orig)
                if changed.any():
                    assert (val[changed] == 0.0).all()

    def test_uncorrupted_positions_unchanged(self):
        obs_dict, graph, _ = _obs_graph(seed=9)
        for _ in range(20):
            out = apply_perception(obs_dict, severity=0.5)
            for node_id, (out_obs, out_val) in out.items():
                orig_obs, orig_val = obs_dict[node_id]
                clean_mask = out_val == 1.0
                if clean_mask.any():
                    assert torch.allclose(out_obs[clean_mask], orig_obs[clean_mask])

    def test_already_invalid_positions_never_restored(self):
        """Positions that were invalid in the input stay invalid after perception."""
        env = MockEnv("cross_smoke", max_steps=20, missing_prob=0.5)
        obs_dict, graph = env.reset(seed=11)
        for _ in range(20):
            out = apply_perception(obs_dict, severity=0.3)
            for node_id, (out_obs, out_val) in out.items():
                _, orig_val = obs_dict[node_id]
                already_missing = orig_val < 0.5
                if already_missing.any():
                    assert (out_val[already_missing] == 0.0).all()

    def test_output_has_same_node_ids(self):
        obs_dict, graph, _ = _obs_graph()
        out = apply_perception(obs_dict, severity=0.5)
        assert set(out.keys()) == set(obs_dict.keys())


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class TestApplyPerceptionContract:
    def test_validity_dtype_float32(self):
        obs_dict, graph, _ = _obs_graph()
        out = apply_perception(obs_dict, severity=0.5)
        for node_id, (_, val) in out.items():
            assert val.dtype == torch.float32

    def test_validity_values_binary(self):
        """validity must be exactly 0.0 or 1.0 — no intermediates."""
        obs_dict, graph, _ = _obs_graph()
        for _ in range(10):
            out = apply_perception(obs_dict, severity=0.5)
            for node_id, (_, val) in out.items():
                assert ((val == 0.0) | (val == 1.0)).all()

    def test_obs_dtype_float32(self):
        obs_dict, graph, _ = _obs_graph()
        out = apply_perception(obs_dict, severity=0.5)
        for node_id, (obs, _) in out.items():
            assert obs.dtype == torch.float32

    def test_input_not_mutated(self):
        obs_dict, graph, _ = _obs_graph()
        orig = {nid: (o.clone(), v.clone()) for nid, (o, v) in obs_dict.items()}
        apply_perception(obs_dict, severity=0.9)
        for node_id, (o_orig, v_orig) in orig.items():
            assert torch.allclose(obs_dict[node_id][0], o_orig)
            assert torch.allclose(obs_dict[node_id][1], v_orig)

    def test_exclude_positions_never_corrupted(self):
        """Excluded positions must survive severity=1.0 uncorrupted."""
        obs_dict, graph, _ = _obs_graph()
        node_meta = graph["node_meta"]
        node_to_idx = graph["node_to_idx"]
        # Use the largest num_phases across nodes so the range covers all phase one-hots.
        max_phases = max(meta["num_phases"] for meta in node_meta)
        excl = range(max_phases)
        orig = {nid: o.clone() for nid, (o, _) in obs_dict.items()}
        for _ in range(10):
            out = apply_perception(obs_dict, severity=1.0, exclude_positions=excl)
            for node_id, (obs, val) in out.items():
                nphases = node_meta[node_to_idx[node_id]]["num_phases"]
                assert torch.allclose(obs[:nphases], orig[node_id][:nphases]), (
                    f"Phase one-hot corrupted for {node_id}"
                )
                assert (val[:nphases] == 1.0).all(), (
                    f"Phase validity corrupted for {node_id}"
                )


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------

BASE_CFG = {
    "trainer": {
        "batch_size": 8,
        "replay_buffer_size": 200,
        "target_update_steps": 20,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 50,
        "warmup_steps": 8,
        "gamma": 0.99,
        "grad_clip": 10.0,
        "num_episodes": 5,
        "checkpoint_every": 100,
        "checkpoint_dir": "",
        "lr": 1e-3,
    },
    "model": {"hidden_dim": 32, "embed_dim": 16, "head_hidden_dim": 16},
    "env":   {"network_name": "cross_smoke", "max_steps": 20, "missing_prob": 0.0},
    "reward": {"use_pressure": False},
    "seed": 0,
}


def _cfg_with_severity(severity):
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in BASE_CFG.items()}
    cfg["trainer"] = dict(BASE_CFG["trainer"])
    cfg["model"]   = dict(BASE_CFG["model"])
    cfg["env"]     = dict(BASE_CFG["env"])
    cfg["reward"]  = dict(BASE_CFG["reward"])
    cfg["perception"] = {"severity": severity, "sentinel_value": -1.0}
    return cfg


class TestApplyPerceptionTrainerIntegration:
    def test_severity_zero_no_crash(self):
        cfg = _cfg_with_severity(0.0)
        env = MockEnv("cross_smoke", max_steps=20)
        trainer = DQNTrainer(cfg, env)
        metrics = trainer.train(num_episodes=5)
        assert len(metrics["episode_returns"]) == 5

    def test_all_losses_finite_severity_01(self):
        cfg = _cfg_with_severity(0.1)
        env = MockEnv("cross_smoke", max_steps=20)
        trainer = DQNTrainer(cfg, env)
        metrics = trainer.train(num_episodes=5)
        for loss in metrics["losses"]:
            assert math.isfinite(loss), f"Non-finite loss at severity=0.1: {loss}"

    def test_all_losses_finite_severity_05(self):
        cfg = _cfg_with_severity(0.5)
        env = MockEnv("cross_smoke", max_steps=20)
        trainer = DQNTrainer(cfg, env)
        metrics = trainer.train(num_episodes=5)
        for loss in metrics["losses"]:
            assert math.isfinite(loss), f"Non-finite loss at severity=0.5: {loss}"
