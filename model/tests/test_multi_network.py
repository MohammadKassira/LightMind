"""Tests for training/multi_network.py — MultiNetworkTrainer (R8).

All tests are SUMO-free, using MockEnv with cross_smoke (N=1), linear_two (N=2),
and grid_3x3 (N=9). MULTI_CFG sets max_obs_dim=32 and max_phase_feat_dim=8,
both above the actual probed maxima (15 and 4 respectively), so the ceiling
assertion path is exercised rather than the equality path.
"""

import math
import random

import pytest
import torch

from env.mock_env import MockEnv
from models.node_encoder import pad_obs_dict
from models.phase_head import pad_phase_features
from training.multi_network import MultiNetworkTrainer


# ---------------------------------------------------------------------------
# Shared config and helpers
# ---------------------------------------------------------------------------

# max_obs_dim=32 > actual max(15,12,15)=15
# max_phase_feat_dim=8 > actual max(4,3,4)=4
MULTI_CFG = {
    "trainer": {
        "batch_size": 8,
        "replay_buffer_size": 200,
        "target_update_steps": 50,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 100,
        "warmup_steps": 8,
        "gamma": 0.99,
        "grad_clip": 10.0,
        "num_episodes": 9,
        "checkpoint_every": 100,
        "checkpoint_dir": "",
        "lr": 1e-3,
    },
    "model": {
        "hidden_dim": 32,
        "embed_dim": 16,
        "head_hidden_dim": 16,
        "max_obs_dim": 32,
        "max_phase_feat_dim": 8,
        "gat": {
            "num_heads": 2,
            "out_per_head": 8,
            "num_layers": 2,
            "l2_out_per_head": 4,
            "typed_edges": True,
            "zero_hop": False,
            "neighbor_masking": True,
        },
    },
    "reward":  {"use_pressure": False},
    "perception": {"severity": 0.0, "sentinel_value": -1.0},
    "seed": 0,
}

_NAMES = ["cross_smoke", "linear_two", "grid_3x3"]


def _make_trainer():
    envs  = [MockEnv(n, max_steps=10) for n in _NAMES]
    return MultiNetworkTrainer(
        MULTI_CFG, envs, device=torch.device("cpu"), network_names=_NAMES
    )


def _local_dims():
    """Compute per-network local obs_dim and pf_dim by probing each network."""
    result = {}
    for name in _NAMES:
        env = MockEnv(name, max_steps=10)
        obs_dict, graph = env.reset(seed=0)
        local_obs, _ = pad_obs_dict(obs_dict)
        local_pf, _  = pad_phase_features(graph)
        result[name] = (local_obs, local_pf)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiNetworkTrainer:

    def test_global_obs_dim_equals_config_ceiling(self):
        """trainer.global_obs_dim equals the config ceiling, not the probed max."""
        trainer = _make_trainer()
        dims = _local_dims()
        probed_max = max(d[0] for d in dims.values())

        assert trainer.global_obs_dim == MULTI_CFG["model"]["max_obs_dim"]
        assert trainer.global_obs_dim >= probed_max

    def test_global_phase_feat_dim_equals_config_ceiling(self):
        """trainer.global_phase_feat_dim equals the config ceiling."""
        trainer = _make_trainer()
        dims = _local_dims()
        probed_max = max(d[1] for d in dims.values())

        assert trainer.global_phase_feat_dim == MULTI_CFG["model"]["max_phase_feat_dim"]
        assert trainer.global_phase_feat_dim >= probed_max

    def test_encoder_input_size_matches_global_obs_dim(self):
        """NodeEncoder first Linear in_features == global_obs_dim * 2."""
        trainer = _make_trainer()
        assert trainer.encoder.net[0].in_features == trainer.global_obs_dim * 2

    def test_head_input_size_matches_gat_out_plus_global_pf_dim(self):
        """PhaseHead scorer[0] in_features == gat.out_channels + global_phase_feat_dim."""
        trainer = _make_trainer()
        expected = trainer.gat.out_channels + trainer.global_phase_feat_dim
        assert trainer.head.scorer[0].in_features == expected

    def test_per_network_buffers_have_global_obs_dim(self):
        """All per-network replay buffers are allocated with global_obs_dim."""
        trainer = _make_trainer()
        for buf in trainer.buffers:
            assert buf._obs_dim == trainer.global_obs_dim

    def test_per_network_buffers_have_network_specific_node_count(self):
        """Buffer shape[1] reflects each network's N, not a global value."""
        trainer = _make_trainer()
        # cross_smoke=N=1, linear_two=N=2, grid_3x3=N=9
        assert trainer.buffers[0]._obs.shape[1] == 1
        assert trainer.buffers[1]._obs.shape[1] == 2
        assert trainer.buffers[2]._obs.shape[1] == 9

    def test_padded_pf_uses_global_phase_feat_dim(self):
        """Every phase feature tensor in every network's padded_pf has the global ceiling length.

        For linear_two (local pf_dim=3, ceiling=8), positions 3+ must be 0.0.
        """
        trainer = _make_trainer()
        global_pf = trainer.global_phase_feat_dim
        for k, pf_list in enumerate(trainer.padded_pf):
            for node_feats in pf_list:
                for feat in node_feats:
                    assert feat.shape[0] == global_pf

        # linear_two is index 1 (local pf_dim=3); padding beyond position 3 must be 0.0
        dims = _local_dims()
        local_pf_linear_two = dims["linear_two"][1]  # 3
        for node_feats in trainer.padded_pf[1]:
            for feat in node_feats:
                assert feat[local_pf_linear_two:].eq(0.0).all(), \
                    "padding positions must be 0.0"

    def test_pad_obs_dict_with_target_dim(self):
        """pad_obs_dict with target_dim pads observations to the target size.

        Uses linear_two (local obs_dim=12) padded to global_obs_dim=32.
        Validity at padding positions must be 0.0.
        """
        trainer = _make_trainer()
        env = MockEnv("linear_two", max_steps=10)
        obs_dict, _ = env.reset(seed=0)

        returned_dim, padded = pad_obs_dict(obs_dict, target_dim=trainer.global_obs_dim)

        assert returned_dim == trainer.global_obs_dim
        dims = _local_dims()
        local_obs_linear_two = dims["linear_two"][0]  # 12

        for node_id, (obs, val) in padded.items():
            assert obs.shape[-1] == trainer.global_obs_dim
            assert val.shape[-1] == trainer.global_obs_dim
            # padding validity must be 0.0
            assert val[..., local_obs_linear_two:].eq(0.0).all(), \
                f"padding validity for node '{node_id}' must be 0.0"

    def test_train_does_not_crash_and_covers_all_networks(self):
        """train() completes 9 episodes with finite returns and all 3 networks visited.

        seed=0 gives randrange(3) sequence [1,1,0,1,2,1,1,1,1] covering {0,1,2}.
        """
        trainer = _make_trainer()
        random.seed(0)
        metrics = trainer.train(num_episodes=9)

        assert len(metrics["episode_returns"]) == 9
        assert all(math.isfinite(r) for r in metrics["episode_returns"])
        assert len(metrics["network_sequence"]) == 9
        assert set(metrics["network_sequence"]) == set(_NAMES)

    def test_gradient_steps_produce_finite_loss(self):
        """With warmup_steps=8, at least one gradient step occurs and all losses are finite.

        Confirms [B, N_k, global_obs_dim] batches from any network produce valid gradients.
        """
        cfg = {**MULTI_CFG, "trainer": {**MULTI_CFG["trainer"], "warmup_steps": 8, "num_episodes": 12}}
        envs = [MockEnv(n, max_steps=10) for n in _NAMES]
        trainer = MultiNetworkTrainer(cfg, envs, device=torch.device("cpu"), network_names=_NAMES)
        metrics = trainer.train(num_episodes=12)

        assert len(metrics["losses"]) >= 1
        assert all(math.isfinite(l) for l in metrics["losses"])
