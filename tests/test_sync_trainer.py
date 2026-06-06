"""Tests for training/sync_trainer.py — SyncParallelTrainer.

All tests are SUMO-free. Uses MockEnv("grid_3x3", N=9) with 2 workers.
SYNC_CFG sets max_obs_dim=32 and max_phase_feat_dim=8 (both above the
actual probed maxima of ~15 and ~4 for grid_3x3).
"""

import math

import pytest
import torch

from env.mock_env import MockEnv
from models.node_encoder import pad_obs_dict
from models.phase_head import pad_phase_features
from training.sync_trainer import SyncParallelTrainer


# ---------------------------------------------------------------------------
# Shared config — 2 workers, grid_3x3 network, tiny model for speed
# ---------------------------------------------------------------------------

SYNC_CFG = {
    "trainer": {
        "batch_size": 8,
        "replay_buffer_size": 500,
        "target_update_steps": 50,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 100,
        "warmup_steps": 8,
        "gamma": 0.99,
        "grad_clip": 10.0,
        "num_episodes": 4,
        "num_workers": 2,
        "checkpoint_every": 0,   # disable checkpointing in tests
        "checkpoint_dir": "",
        "lr": 1e-3,
    },
    "model": {
        "hidden_dim": 32,
        "embed_dim": 16,
        "head_hidden_dim": 16,
        "max_obs_dim": 32,           # > actual grid_3x3 obs_dim (~15)
        "max_phase_feat_dim": 8,     # > actual grid_3x3 pf_dim (~4)
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
    "reward":      {"use_pressure": False},
    "perception":  {"severity": 0.0, "sentinel_value": -1.0},
    "seed": 0,
}

_NETWORK = "grid_3x3"
_NUM_WORKERS = 2


def _make_trainer():
    envs = [MockEnv(_NETWORK, max_steps=10) for _ in range(_NUM_WORKERS)]
    return SyncParallelTrainer(
        SYNC_CFG, envs, device=torch.device("cpu"), network_name=_NETWORK
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSyncParallelTrainer:

    def test_obs_dim_matches_config_ceiling(self):
        """global_obs_dim equals the config ceiling (not just the probed local max)."""
        trainer = _make_trainer()
        env     = MockEnv(_NETWORK, max_steps=10)
        obs_dict, _ = env.reset(seed=0)
        local_obs_dim, _ = pad_obs_dict(obs_dict)

        assert trainer.global_obs_dim == SYNC_CFG["model"]["max_obs_dim"]
        assert trainer.global_obs_dim >= local_obs_dim

    def test_encoder_and_head_dimensions(self):
        """NodeEncoder and PhaseHead are built using the config ceiling dims."""
        trainer  = _make_trainer()
        expected_enc  = trainer.global_obs_dim * 2
        expected_head = trainer.gat.out_channels + trainer.global_phase_feat_dim

        assert trainer.encoder.net[0].in_features == expected_enc
        assert trainer.head.scorer[0].in_features == expected_head

    def test_shared_buffer_uses_global_obs_dim(self):
        """Single shared buffer is allocated with global_obs_dim and grid_3x3's N=9."""
        trainer = _make_trainer()
        assert trainer.buffer._obs_dim == trainer.global_obs_dim
        assert trainer.buffer._obs.shape[1] == 9   # grid_3x3 has 9 nodes

    def test_train_returns_correct_episode_count(self):
        """train(num_episodes=4) with 2 workers returns exactly 4 episode records."""
        trainer = _make_trainer()
        metrics = trainer.train(num_episodes=4)

        assert len(metrics["episode_returns"])  == 4
        assert len(metrics["episode_lengths"])  == 4
        assert len(metrics["avg_waiting_time"]) == 4
        assert len(metrics["throughput"])       == 4
        assert len(metrics["avg_queue_length"]) == 4

    def test_eval_metric_fields_are_finite(self):
        """All per-episode metric values are finite floats (no NaN/Inf from env)."""
        trainer = _make_trainer()
        metrics = trainer.train(num_episodes=4)

        for field in ("episode_returns", "avg_waiting_time", "avg_queue_length"):
            assert all(math.isfinite(v) for v in metrics[field]), \
                f"non-finite value in {field}"
        assert all(v >= 0 for v in metrics["throughput"]), \
            "throughput must be non-negative"

    def test_gradient_steps_produce_finite_loss(self):
        """With warmup_steps=8 and 10 steps/episode, at least one gradient step fires.

        grid_3x3 × 2 workers × 10 steps/ep = 20 transitions per round > warmup_steps=8.
        All gradient-step losses must be finite.
        """
        trainer = _make_trainer()
        metrics = trainer.train(num_episodes=4)

        assert len(metrics["losses"]) >= 1, "expected at least one gradient step"
        assert all(math.isfinite(l) for l in metrics["losses"]), \
            "gradient step produced non-finite loss"
