"""R2 done-when integration tests.

Each test must pass for R2 to be considered complete.
All run on MockEnv — SUMO not required.
"""
import math
import os
import tempfile

import pytest

from env.mock_env import MockEnv
from models.node_encoder import pad_obs_dict
from models.phase_head import pad_phase_features
from training.trainer import DQNTrainer


SMALL_CFG = {
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
        "num_episodes": 10,
        "checkpoint_every": 5,
        "checkpoint_dir": "",   # overridden per test
        "lr": 1e-3,
    },
    "model": {"hidden_dim": 32, "embed_dim": 16, "head_hidden_dim": 16},
    "env":   {"network_name": "cross_smoke", "max_steps": 20, "missing_prob": 0.0},
    "reward": {"use_pressure": False},
    "seed": 0,
}


def _cfg(network="cross_smoke", tmpdir=None):
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in SMALL_CFG.items()}
    cfg["trainer"] = dict(SMALL_CFG["trainer"])
    cfg["model"]   = dict(SMALL_CFG["model"])
    cfg["env"]     = dict(SMALL_CFG["env"])
    cfg["reward"]  = dict(SMALL_CFG["reward"])
    if tmpdir:
        cfg["trainer"]["checkpoint_dir"] = str(tmpdir)
    return cfg


class TestR2Integration:
    def test_10_episodes_cross_smoke_no_crash(self):
        """Training loop runs 10 episodes on cross_smoke without raising."""
        cfg = _cfg()
        env = MockEnv("cross_smoke", max_steps=20)
        trainer = DQNTrainer(cfg, env)
        metrics = trainer.train(num_episodes=10)
        assert len(metrics["episode_returns"]) == 10

    def test_all_losses_finite(self):
        """Every gradient step produces a finite loss — no NaN or inf."""
        cfg = _cfg()
        env = MockEnv("cross_smoke", max_steps=20)
        trainer = DQNTrainer(cfg, env)
        metrics = trainer.train(num_episodes=10)
        assert len(metrics["losses"]) > 0, "No gradient steps were taken"
        for loss in metrics["losses"]:
            assert math.isfinite(loss), f"Non-finite loss encountered: {loss}"

    def test_q_mean_bounded(self):
        """Mean max-Q stays finite — divergence check."""
        cfg = _cfg()
        env = MockEnv("cross_smoke", max_steps=20)
        trainer = DQNTrainer(cfg, env)
        metrics = trainer.train(num_episodes=10)
        for q_m in metrics["q_mean"]:
            assert math.isfinite(q_m), f"Non-finite q_mean: {q_m}"
            assert abs(q_m) < 1e4, f"Q-values appear diverged: q_mean={q_m}"

    def test_checkpoint_save_and_load(self):
        """Checkpoint written at episode 5, loaded, inference produces valid actions."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmpdir=tmp)
            cfg["trainer"]["checkpoint_every"] = 5
            env = MockEnv("cross_smoke", max_steps=20)
            trainer = DQNTrainer(cfg, env)
            trainer.train(num_episodes=5)

            ckpt_file = os.path.join(tmp, "checkpoint_ep5.pt")
            assert os.path.exists(ckpt_file), f"Expected checkpoint at {ckpt_file}"

            env2 = MockEnv("cross_smoke", max_steps=20)
            loaded = DQNTrainer.load_checkpoint(ckpt_file, cfg, env2)

            obs_dict, graph = env2.reset(seed=99)
            _, padded_obs = pad_obs_dict(obs_dict)
            _, padded_pf  = pad_phase_features(graph)
            actions = loaded._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)

            assert set(actions.keys()) == set(graph["node_ids"])
            for node_id, phase_idx in actions.items():
                node_idx   = graph["node_to_idx"][node_id]
                num_phases = graph["node_meta"][node_idx]["num_phases"]
                assert 0 <= phase_idx < num_phases

    def test_5_episodes_grid_3x3_stable(self):
        """Training on a 9-node grid runs stably with finite losses."""
        cfg = _cfg(network="grid_3x3")
        env = MockEnv("grid_3x3", max_steps=20)
        trainer = DQNTrainer(cfg, env)
        metrics = trainer.train(num_episodes=5)
        assert len(metrics["episode_returns"]) == 5
        for loss in metrics["losses"]:
            assert math.isfinite(loss), f"Non-finite loss on grid_3x3: {loss}"
