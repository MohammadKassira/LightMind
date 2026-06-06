"""Tests for training/trainer.py — DQNTrainer."""
import math
import os
import tempfile

import pytest
import torch

from env.mock_env import MockEnv
from models.node_encoder import pad_obs_dict
from models.phase_head import pad_phase_features
from training.trainer import DQNTrainer, _linear_decay


# ---------------------------------------------------------------------------
# Minimal config for tests (dict, not Hydra)
# ---------------------------------------------------------------------------

BASE_CFG = {
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
        "num_episodes": 3,
        "checkpoint_every": 2,
        "checkpoint_dir": "",      # overridden per test
        "lr": 1e-3,
    },
    "model": {"hidden_dim": 32, "embed_dim": 16, "head_hidden_dim": 16},
    "env":   {"network_name": "cross_smoke", "max_steps": 20, "missing_prob": 0.0},
    "reward": {"use_pressure": False},
    "seed": 0,
}


def _make_trainer(network="cross_smoke", cfg_overrides=None):
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in BASE_CFG.items()}
    cfg["trainer"] = dict(BASE_CFG["trainer"])
    cfg["model"]   = dict(BASE_CFG["model"])
    cfg["env"]     = dict(BASE_CFG["env"])
    cfg["reward"]  = dict(BASE_CFG["reward"])
    if cfg_overrides:
        for dotted, val in cfg_overrides.items():
            parts = dotted.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = obj[part]
            obj[parts[-1]] = val
    env = MockEnv(network, max_steps=cfg["env"]["max_steps"])
    return DQNTrainer(cfg, env, device=torch.device("cpu")), env, cfg


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestDQNTrainerInit:
    def test_encoder_constructed(self):
        trainer, _, _ = _make_trainer()
        assert trainer.encoder is not None

    def test_head_constructed(self):
        trainer, _, _ = _make_trainer()
        assert trainer.head is not None

    def test_target_initially_equal_to_online_encoder(self):
        trainer, _, _ = _make_trainer()
        for p_on, p_tg in zip(trainer.encoder.parameters(), trainer.target_encoder.parameters()):
            assert torch.allclose(p_on, p_tg)

    def test_target_initially_equal_to_online_head(self):
        trainer, _, _ = _make_trainer()
        for p_on, p_tg in zip(trainer.head.parameters(), trainer.target_head.parameters()):
            assert torch.allclose(p_on, p_tg)

    def test_target_has_no_grad(self):
        trainer, _, _ = _make_trainer()
        for p in trainer.target_encoder.parameters():
            assert not p.requires_grad
        for p in trainer.target_head.parameters():
            assert not p.requires_grad

    def test_buffer_empty_at_init(self):
        trainer, _, _ = _make_trainer()
        assert len(trainer.buffer) == 0


# ---------------------------------------------------------------------------
# Epsilon decay
# ---------------------------------------------------------------------------

class TestEpsilonDecay:
    def test_decay_starts_at_start(self):
        eps = _linear_decay(0, 1.0, 0.05, 100)
        assert abs(eps - 1.0) < 1e-6

    def test_decay_ends_at_end(self):
        eps = _linear_decay(100, 1.0, 0.05, 100)
        assert abs(eps - 0.05) < 1e-6

    def test_decay_monotonically_decreasing(self):
        eps_vals = [_linear_decay(t, 1.0, 0.05, 100) for t in range(110)]
        for a, b in zip(eps_vals, eps_vals[1:]):
            assert a >= b - 1e-9

    def test_decay_saturates_after_decay_steps(self):
        eps_200 = _linear_decay(200, 1.0, 0.05, 100)
        eps_100 = _linear_decay(100, 1.0, 0.05, 100)
        assert abs(eps_200 - eps_100) < 1e-9


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

class TestSelectActions:
    def _get_padded_obs_graph(self, trainer, env):
        obs_dict, graph = env.reset(seed=1)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf = pad_phase_features(graph)
        return padded_obs, graph, padded_pf

    def test_returns_all_node_ids(self):
        trainer, env, _ = _make_trainer()
        padded_obs, graph, padded_pf = self._get_padded_obs_graph(trainer, env)
        actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)
        assert set(actions.keys()) == set(graph["node_ids"])

    def test_actions_are_ints(self):
        trainer, env, _ = _make_trainer()
        padded_obs, graph, padded_pf = self._get_padded_obs_graph(trainer, env)
        actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)
        for v in actions.values():
            assert isinstance(v, int)

    def test_greedy_actions_within_num_phases(self):
        trainer, env, _ = _make_trainer()
        padded_obs, graph, padded_pf = self._get_padded_obs_graph(trainer, env)
        actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)
        for node_id, phase_idx in actions.items():
            node_idx = graph["node_to_idx"][node_id]
            num_phases = graph["node_meta"][node_idx]["num_phases"]
            assert 0 <= phase_idx < num_phases

    def test_random_actions_within_num_phases(self):
        trainer, env, _ = _make_trainer()
        padded_obs, graph, padded_pf = self._get_padded_obs_graph(trainer, env)
        for _ in range(20):
            actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=1.0)
            for node_id, phase_idx in actions.items():
                node_idx = graph["node_to_idx"][node_id]
                num_phases = graph["node_meta"][node_idx]["num_phases"]
                assert 0 <= phase_idx < num_phases


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestComputeLoss:
    def _fill_buffer(self, trainer, env, n=16):
        obs_dict, graph = env.reset(seed=0)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf = pad_phase_features(graph)
        done = False
        step = 0
        while step < n:
            actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=1.0)
            next_obs_dict, _, reward_dict, done, _ = env.step(actions)
            _, padded_next = pad_obs_dict(next_obs_dict)
            trainer.buffer.push(padded_obs, padded_next, actions, reward_dict, done)
            padded_obs = padded_next
            step += 1
            if done:
                obs_dict, graph = env.reset()
                _, padded_obs = pad_obs_dict(obs_dict)
        return graph, padded_pf

    def test_loss_is_finite(self):
        trainer, env, cfg = _make_trainer()
        graph, padded_pf = self._fill_buffer(trainer, env, n=16)
        batch = trainer.buffer.sample(8)
        loss, q_m = trainer._compute_loss(batch, graph, padded_pf, gamma=0.99)
        assert math.isfinite(loss.item())

    def test_q_mean_is_finite(self):
        trainer, env, cfg = _make_trainer()
        graph, padded_pf = self._fill_buffer(trainer, env, n=16)
        batch = trainer.buffer.sample(8)
        _, q_m = trainer._compute_loss(batch, graph, padded_pf, gamma=0.99)
        assert math.isfinite(q_m)

    def test_loss_is_non_negative(self):
        trainer, env, cfg = _make_trainer()
        graph, padded_pf = self._fill_buffer(trainer, env, n=16)
        batch = trainer.buffer.sample(8)
        loss, _ = trainer._compute_loss(batch, graph, padded_pf, gamma=0.99)
        assert loss.item() >= 0.0

    def test_gradients_flow_after_backward(self):
        trainer, env, cfg = _make_trainer()
        graph, padded_pf = self._fill_buffer(trainer, env, n=16)
        batch = trainer.buffer.sample(8)
        loss, _ = trainer._compute_loss(batch, graph, padded_pf, gamma=0.99)
        trainer.optimizer.zero_grad()
        loss.backward()
        enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in trainer.encoder.parameters())
        head_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in trainer.head.parameters())
        assert enc_grad, "No gradient in encoder"
        assert head_grad, "No gradient in head"


# ---------------------------------------------------------------------------
# Target network update
# ---------------------------------------------------------------------------

class TestTargetNetworkUpdate:
    def test_target_changes_after_load_state_dict(self):
        trainer, env, _ = _make_trainer()
        # Perturb online params
        with torch.no_grad():
            for p in trainer.encoder.parameters():
                p.add_(0.5)
        # Initially target != online
        online_p = next(trainer.encoder.parameters()).clone()
        target_p = next(trainer.target_encoder.parameters()).clone()
        assert not torch.allclose(online_p, target_p)
        # Hard copy
        trainer.target_encoder.load_state_dict(trainer.encoder.state_dict())
        target_p_after = next(trainer.target_encoder.parameters()).clone()
        assert torch.allclose(online_p, target_p_after)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_creates_file(self):
        trainer, env, _ = _make_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            trainer.save_checkpoint(path, step=0, metrics={})
            assert os.path.exists(path)

    def test_load_restores_encoder_weights(self):
        trainer, env, cfg = _make_trainer()
        # Perturb online params
        with torch.no_grad():
            for p in trainer.encoder.parameters():
                p.fill_(3.14)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            trainer.save_checkpoint(path, step=0, metrics={})
            env2 = MockEnv("cross_smoke", max_steps=20)
            loaded = DQNTrainer.load_checkpoint(path, cfg, env2)
            for p_orig, p_loaded in zip(trainer.encoder.parameters(),
                                        loaded.encoder.parameters()):
                assert torch.allclose(p_orig, p_loaded)

    def test_load_restores_head_weights(self):
        trainer, env, cfg = _make_trainer()
        with torch.no_grad():
            for p in trainer.head.parameters():
                p.fill_(2.71)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            trainer.save_checkpoint(path, step=0, metrics={})
            env2 = MockEnv("cross_smoke", max_steps=20)
            loaded = DQNTrainer.load_checkpoint(path, cfg, env2)
            for p_orig, p_loaded in zip(trainer.head.parameters(),
                                        loaded.head.parameters()):
                assert torch.allclose(p_orig, p_loaded)

    def test_loaded_trainer_produces_same_actions(self):
        trainer, env, cfg = _make_trainer()
        obs_dict, graph = env.reset(seed=5)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        orig_actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            trainer.save_checkpoint(path, step=0, metrics={})
            env2 = MockEnv("cross_smoke", max_steps=20)
            loaded = DQNTrainer.load_checkpoint(path, cfg, env2)

        obs_dict2, graph2 = env.reset(seed=5)
        _, padded_obs2 = pad_obs_dict(obs_dict2)
        _, padded_pf2  = pad_phase_features(graph2)
        loaded_actions = loaded._select_actions(padded_obs2, graph2, padded_pf2, epsilon=0.0)
        assert orig_actions == loaded_actions
