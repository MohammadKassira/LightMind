"""Tests for models/gat_policy.py and the forward_batch addition to PhaseHead.

Also exercises DQNTrainer end-to-end with the GAT wired in.
All tests are SUMO-free — they use MockEnv or hand-built tensors.
"""
import os
import tempfile

import pytest
import torch

from env.mock_env import MockEnv
from models.gat_policy import GATPolicy
from models.node_encoder import NodeEncoder, pad_obs_dict
from models.phase_head import PhaseHead, pad_phase_features
from training.trainer import DQNTrainer


# ---------------------------------------------------------------------------
# Shared config for trainer tests (matches test_trainer.py BASE_CFG shape)
# ---------------------------------------------------------------------------

TRAINER_CFG = {
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
        "num_episodes": 2,
        "checkpoint_every": 2,
        "checkpoint_dir": "",
        "lr": 1e-3,
    },
    "model": {
        "hidden_dim": 32,
        "embed_dim": 16,
        "head_hidden_dim": 16,
        "gat": {"num_heads": 2, "out_per_head": 8, "zero_hop": False},
    },
    "env":     {"network_name": "cross_smoke", "max_steps": 10, "missing_prob": 0.0},
    "reward":  {"use_pressure": False},
    "perception": {"severity": 0.0, "sentinel_value": -1.0},
    "seed": 0,
}


def _make_trainer(zero_hop=False, network="cross_smoke"):
    cfg = {k: v for k, v in TRAINER_CFG.items()}
    cfg["model"] = {**TRAINER_CFG["model"], "gat": {**TRAINER_CFG["model"]["gat"], "zero_hop": zero_hop}}
    env = MockEnv(network, max_steps=cfg["env"]["max_steps"])
    return DQNTrainer(cfg, env, device=torch.device("cpu")), env, cfg


# ---------------------------------------------------------------------------
# GATPolicy output shape
# ---------------------------------------------------------------------------

class TestGATPolicyShape:
    def _make_graph(self, N=4, in_ch=64):
        x = torch.randn(N, in_ch)
        # Chain: 0→1, 1→2, 2→3 (bidirectional)
        ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        return x, ei

    def test_one_hop_output_shape(self):
        x, ei = self._make_graph(N=4, in_ch=64)
        model = GATPolicy(in_channels=64, num_heads=4, out_per_head=32, zero_hop=False)
        out = model(x, ei)
        assert out.shape == (4, 128)

    def test_zero_hop_output_shape(self):
        x, ei = self._make_graph(N=4, in_ch=64)
        model = GATPolicy(in_channels=64, num_heads=4, out_per_head=32, zero_hop=True)
        out = model(x, ei)
        assert out.shape == (4, 128)

    def test_out_channels_property(self):
        model = GATPolicy(in_channels=64, num_heads=4, out_per_head=32)
        assert model.out_channels == 128

    def test_small_in_channels(self):
        """Works for small embed_dim (as used in test configs)."""
        x = torch.randn(3, 16)
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        model = GATPolicy(in_channels=16, num_heads=2, out_per_head=8)
        out = model(x, ei)
        assert out.shape == (3, 16)

    def test_single_node_empty_edge_index(self):
        """Single-node graph (cross_smoke) with empty edge_index must not crash."""
        x = torch.randn(1, 64)
        ei = torch.zeros(2, 0, dtype=torch.long)
        model = GATPolicy(in_channels=64, num_heads=4, out_per_head=32, zero_hop=False)
        out = model(x, ei)
        assert out.shape == (1, 128)


# ---------------------------------------------------------------------------
# zero_hop vs one_hop difference
# ---------------------------------------------------------------------------

class TestZeroHopDifference:
    def test_zero_hop_differs_from_one_hop(self):
        """With real neighbors, 0-hop and 1-hop outputs must differ.

        The graph has node 0 connected to node 1 — a genuine neighbor exists.
        Without this, the test would be vacuous (same self-loop aggregation).
        """
        torch.manual_seed(42)
        # 3 nodes: 0↔1 are neighbors, node 2 is isolated (still has self-loop via GATConv)
        x = torch.randn(3, 64)
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        model_1hop = GATPolicy(in_channels=64, num_heads=4, out_per_head=32, zero_hop=False)
        model_0hop = GATPolicy(in_channels=64, num_heads=4, out_per_head=32, zero_hop=True)
        # Share weights so the only difference is message passing
        model_0hop.load_state_dict(model_1hop.state_dict())

        with torch.no_grad():
            out_1hop = model_1hop(x, ei)
            out_0hop = model_0hop(x, ei)

        # Node 0 aggregates from node 1 under 1-hop; should differ from 0-hop
        assert not torch.allclose(out_1hop, out_0hop), (
            "1-hop and 0-hop outputs are identical — check that the test graph has real edges"
        )

    def test_zero_hop_isolated_node_same_as_self(self):
        """A node with no neighbors produces identical output regardless of zero_hop."""
        torch.manual_seed(7)
        x = torch.randn(2, 64)
        # No edges between nodes — only self-loops will be added by GATConv
        ei = torch.zeros(2, 0, dtype=torch.long)

        model_1hop = GATPolicy(in_channels=64, num_heads=4, out_per_head=32, zero_hop=False)
        model_0hop = GATPolicy(in_channels=64, num_heads=4, out_per_head=32, zero_hop=True)
        model_0hop.load_state_dict(model_1hop.state_dict())

        with torch.no_grad():
            out_1hop = model_1hop(x, ei)
            out_0hop = model_0hop(x, ei)

        assert torch.allclose(out_1hop, out_0hop, atol=1e-5)


# ---------------------------------------------------------------------------
# PhaseHead.forward_batch
# ---------------------------------------------------------------------------

class TestForwardBatch:
    def _make_head_and_inputs(self, embed_dim=32, phase_feat_dim=4, num_phases=3, B=5):
        head = PhaseHead(embed_dim, phase_feat_dim, hidden_dim=16)
        torch.manual_seed(0)
        embeddings = torch.randn(B, embed_dim)
        phase_feats = [torch.randn(phase_feat_dim) for _ in range(num_phases)]
        mask = torch.ones(num_phases, dtype=torch.bool)
        return head, embeddings, phase_feats, mask

    def test_batch_output_shape(self):
        head, embs, pf, mask = self._make_head_and_inputs(B=5, num_phases=3)
        out = head.forward_batch(embs, pf, mask)
        assert out.shape == (5, 3)

    def test_batch_matches_individual_forward(self):
        """forward_batch must produce the same scores as B individual forward() calls."""
        head, embs, pf, mask = self._make_head_and_inputs(B=4, num_phases=3)
        with torch.no_grad():
            batch_out = head.forward_batch(embs, pf, mask)
            individual = torch.stack([head.forward(embs[b], pf, mask) for b in range(4)])
        assert torch.allclose(batch_out, individual, atol=1e-5)

    def test_invalid_phases_masked_to_neginf(self):
        head, embs, pf, mask = self._make_head_and_inputs(B=3, num_phases=3)
        partial_mask = torch.tensor([True, False, True])
        out = head.forward_batch(embs, pf, partial_mask)
        assert torch.all(out[:, 1] == float("-inf"))
        assert torch.all(out[:, 0] != float("-inf"))
        assert torch.all(out[:, 2] != float("-inf"))

    def test_forward_unchanged(self):
        """Original forward() still works after adding forward_batch."""
        head = PhaseHead(32, 4, 16)
        emb  = torch.randn(32)
        pf   = [torch.randn(4) for _ in range(3)]
        mask = torch.ones(3, dtype=torch.bool)
        out  = head.forward(emb, pf, mask)
        assert out.shape == (3,)


# ---------------------------------------------------------------------------
# DQNTrainer with GAT wired in
# ---------------------------------------------------------------------------

class TestGATTrainerSelectActions:
    def test_returns_all_node_ids(self):
        trainer, env, _ = _make_trainer()
        obs_dict, graph = env.reset(seed=1)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)
        assert set(actions.keys()) == set(graph["node_ids"])

    def test_greedy_actions_within_num_phases(self):
        trainer, env, _ = _make_trainer()
        obs_dict, graph = env.reset(seed=1)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)
        for node_id, phase_idx in actions.items():
            node_idx   = graph["node_to_idx"][node_id]
            num_phases = graph["node_meta"][node_idx]["num_phases"]
            assert 0 <= phase_idx < num_phases

    def test_zero_hop_select_actions(self):
        """zero_hop=True must also return valid actions (same interface)."""
        trainer, env, _ = _make_trainer(zero_hop=True)
        obs_dict, graph = env.reset(seed=2)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)
        assert set(actions.keys()) == set(graph["node_ids"])


class TestGATTrainerComputeLoss:
    def test_compute_loss_runs(self):
        trainer, env, _ = _make_trainer()
        obs_dict, graph = env.reset(seed=0)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)

        # Fill buffer past warmup
        for _ in range(20):
            actions = {nid: 0 for nid in graph["node_ids"]}
            next_obs, _, rew, done, _ = env.step(actions)
            _, padded_next = pad_obs_dict(next_obs)
            trainer.buffer.push(padded_obs, padded_next, actions, rew, done)
            padded_obs = padded_next
            if done:
                obs_dict, graph = env.reset()
                _, padded_obs = pad_obs_dict(obs_dict)

        batch = trainer.buffer.sample(8)
        loss, q_mean = trainer._compute_loss(batch, graph, padded_pf, gamma=0.99)
        assert loss.shape == ()
        assert loss.item() >= 0.0
        assert isinstance(q_mean, float)

    def test_loss_is_finite(self):
        trainer, env, _ = _make_trainer()
        obs_dict, graph = env.reset(seed=0)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        for _ in range(20):
            actions = {nid: 0 for nid in graph["node_ids"]}
            next_obs, _, rew, done, _ = env.step(actions)
            _, padded_next = pad_obs_dict(next_obs)
            trainer.buffer.push(padded_obs, padded_next, actions, rew, done)
            padded_obs = padded_next
            if done:
                obs_dict, graph = env.reset()
                _, padded_obs = pad_obs_dict(obs_dict)
        batch = trainer.buffer.sample(8)
        loss, _ = trainer._compute_loss(batch, graph, padded_pf, gamma=0.99)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Checkpoint round-trip (verifies "gat" key is saved and loaded)
# ---------------------------------------------------------------------------

class TestGATCheckpoint:
    def test_checkpoint_roundtrip_produces_same_actions(self):
        trainer, env, cfg = _make_trainer()
        obs_dict, graph = env.reset(seed=5)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        orig_actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            trainer.save_checkpoint(path, step=0, metrics={})

            env2   = MockEnv("cross_smoke", max_steps=10)
            loaded = DQNTrainer.load_checkpoint(path, cfg, env2)

        obs_dict2, graph2 = env.reset(seed=5)
        _, padded_obs2 = pad_obs_dict(obs_dict2)
        _, padded_pf2  = pad_phase_features(graph2)
        loaded_actions = loaded._select_actions(padded_obs2, graph2, padded_pf2, epsilon=0.0)

        assert orig_actions == loaded_actions

    def test_checkpoint_contains_gat_key(self):
        trainer, env, _ = _make_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            trainer.save_checkpoint(path, step=0, metrics={})
            ckpt = torch.load(path, map_location="cpu")
        assert "gat" in ckpt, "checkpoint must contain 'gat' state dict"

    def test_load_restores_gat_weights(self):
        trainer, env, cfg = _make_trainer()
        with torch.no_grad():
            for p in trainer.gat.parameters():
                p.fill_(1.23)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            trainer.save_checkpoint(path, step=0, metrics={})
            env2   = MockEnv("cross_smoke", max_steps=10)
            loaded = DQNTrainer.load_checkpoint(path, cfg, env2)

        for p_orig, p_loaded in zip(trainer.gat.parameters(), loaded.gat.parameters()):
            assert torch.allclose(p_orig, p_loaded)


# ---------------------------------------------------------------------------
# R5: Two-layer GATPolicy
# ---------------------------------------------------------------------------

# Small config for 2-layer trainer tests:
# Layer 1: 2 heads × 8 → 16; Layer 2: 2 heads × 4 → 8
TRAINER_CFG_2L = {
    **TRAINER_CFG,
    "model": {
        **TRAINER_CFG["model"],
        "gat": {"num_heads": 2, "out_per_head": 8, "num_layers": 2,
                "l2_out_per_head": 4, "zero_hop": False},
    },
}


def _make_trainer_2l(zero_hop=False, network="cross_smoke"):
    cfg = {k: v for k, v in TRAINER_CFG_2L.items()}
    cfg["model"] = {**TRAINER_CFG_2L["model"],
                    "gat": {**TRAINER_CFG_2L["model"]["gat"], "zero_hop": zero_hop}}
    env = MockEnv(network, max_steps=cfg["env"]["max_steps"])
    return DQNTrainer(cfg, env, device=torch.device("cpu")), env, cfg


class TestTwoLayerGATPolicy:
    def _graph(self, N=4, in_ch=64):
        x  = torch.randn(N, in_ch)
        ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        return x, ei

    def test_two_layer_output_shape(self):
        x, ei = self._graph(N=4, in_ch=64)
        model = GATPolicy(in_channels=64, num_heads=4, out_per_head=32,
                          num_layers=2, l2_out_per_head=16)
        out = model(x, ei)
        assert out.shape == (4, 64)   # 4 heads × 16 = 64

    def test_one_layer_output_still_128(self):
        """num_layers=1 is identical to R4: out_channels=128, no gat2 attribute."""
        x, ei = self._graph(N=4, in_ch=64)
        model = GATPolicy(in_channels=64, num_heads=4, out_per_head=32, num_layers=1)
        out = model(x, ei)
        assert out.shape == (4, 128)
        assert not hasattr(model, "gat2"), "num_layers=1 must not create gat2"

    def test_two_layer_zero_hop_shape(self):
        x, ei = self._graph(N=4, in_ch=64)
        model = GATPolicy(in_channels=64, num_heads=4, out_per_head=32,
                          num_layers=2, l2_out_per_head=16, zero_hop=True)
        out = model(x, ei)
        assert out.shape == (4, 64)

    def test_two_layer_differs_from_one_layer(self):
        """With real neighbors, 2-layer and 1-layer outputs must differ."""
        torch.manual_seed(0)
        x, ei = self._graph(N=4, in_ch=64)
        m1 = GATPolicy(in_channels=64, num_heads=4, out_per_head=32, num_layers=1)
        m2 = GATPolicy(in_channels=64, num_heads=4, out_per_head=32,
                       num_layers=2, l2_out_per_head=16)
        with torch.no_grad():
            out1 = m1(x, ei)   # [4, 128]
            out2 = m2(x, ei)   # [4, 64]
        # Different shapes already prove they differ; also verify out2 is not all-zero
        assert out1.shape != out2.shape
        assert out2.abs().max().item() > 1e-6

    def test_embeddings_not_collapsed(self):
        """After 2-layer forward on N=4 nodes with distinct inputs, all pairwise
        cosine similarities must be < 0.999 — no two nodes produce near-identical
        embeddings (over-smoothing / collapse guard)."""
        torch.manual_seed(42)
        N = 4
        x  = torch.randn(N, 64)
        ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        model = GATPolicy(in_channels=64, num_heads=4, out_per_head=32,
                          num_layers=2, l2_out_per_head=16)
        with torch.no_grad():
            out = model(x, ei)   # [N, 64]

        # Normalise rows for cosine similarity
        normed = out / (out.norm(dim=1, keepdim=True) + 1e-8)
        for i in range(N):
            for j in range(i + 1, N):
                cos_sim = (normed[i] * normed[j]).sum().item()
                assert cos_sim < 0.999, (
                    f"Nodes {i} and {j} have cosine similarity {cos_sim:.4f} ≥ 0.999 "
                    f"— embeddings are collapsing to the same vector"
                )

    def test_out_channels_property_two_layer(self):
        model = GATPolicy(in_channels=64, num_heads=4, out_per_head=32,
                          num_layers=2, l2_out_per_head=16)
        assert model.out_channels == 64

    def test_two_layer_trainer_select_actions(self):
        trainer, env, _ = _make_trainer_2l()
        obs_dict, graph = env.reset(seed=1)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)
        assert set(actions.keys()) == set(graph["node_ids"])
        for node_id, phase_idx in actions.items():
            node_idx   = graph["node_to_idx"][node_id]
            num_phases = graph["node_meta"][node_idx]["num_phases"]
            assert 0 <= phase_idx < num_phases

    def test_two_layer_trainer_compute_loss(self):
        trainer, env, _ = _make_trainer_2l()
        obs_dict, graph = env.reset(seed=0)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        for _ in range(20):
            actions = {nid: 0 for nid in graph["node_ids"]}
            next_obs, _, rew, done, _ = env.step(actions)
            _, padded_next = pad_obs_dict(next_obs)
            trainer.buffer.push(padded_obs, padded_next, actions, rew, done)
            padded_obs = padded_next
            if done:
                obs_dict, graph = env.reset()
                _, padded_obs = pad_obs_dict(obs_dict)
        batch = trainer.buffer.sample(8)
        loss, q_mean = trainer._compute_loss(batch, graph, padded_pf, gamma=0.99)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert isinstance(q_mean, float)

    def test_two_layer_checkpoint_roundtrip(self):
        """save → load → identical predictions; checkpoint includes gat2 weights."""
        trainer, env, cfg = _make_trainer_2l()
        obs_dict, graph = env.reset(seed=3)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        orig_actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt2l.pt")
            trainer.save_checkpoint(path, step=0, metrics={})

            # Verify gat2 keys are present in the saved state dict
            ckpt = torch.load(path, map_location="cpu")
            assert any(k.startswith("gat2.") for k in ckpt["gat"].keys()), (
                "2-layer checkpoint must contain gat2 weights"
            )

            env2   = MockEnv("cross_smoke", max_steps=10)
            loaded = DQNTrainer.load_checkpoint(path, cfg, env2)

        obs_dict2, graph2 = env.reset(seed=3)
        _, padded_obs2 = pad_obs_dict(obs_dict2)
        _, padded_pf2  = pad_phase_features(graph2)
        loaded_actions = loaded._select_actions(padded_obs2, graph2, padded_pf2, epsilon=0.0)
        assert orig_actions == loaded_actions


# ---------------------------------------------------------------------------
# R6: Typed-edge GATPolicy (separate weight matrices per edge type)
# ---------------------------------------------------------------------------

# Small config for R6 trainer tests (2-layer typed):
# Layer 1: 2 heads × 8 → 16; Layer 2: 2 heads × 4 → 8
TRAINER_CFG_R6 = {
    **TRAINER_CFG,
    "model": {
        **TRAINER_CFG["model"],
        "gat": {
            "num_heads": 2,
            "out_per_head": 8,
            "num_layers": 2,
            "l2_out_per_head": 4,
            "typed_edges": True,
            "zero_hop": False,
        },
    },
}


def _make_trainer_typed(network="cross_smoke"):
    cfg = {k: v for k, v in TRAINER_CFG_R6.items()}
    env = MockEnv(network, max_steps=cfg["env"]["max_steps"])
    return DQNTrainer(cfg, env, device=torch.device("cpu")), env, cfg


class TestTypedEdgesGATPolicy:
    def _graph(self, N=4, in_ch=16, include_coord=True):
        """Build x, edge_index, edge_type with flow and optionally coord edges."""
        x = torch.randn(N, in_ch)
        # Flow edges (type 0)
        ei_flow = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        et_flow = torch.zeros(4, dtype=torch.long)
        if include_coord:
            # Coord edges (type 1): 2→0, 3→1
            ei_coord = torch.tensor([[2, 3], [0, 1]], dtype=torch.long)
            et_coord = torch.ones(2, dtype=torch.long)
            ei = torch.cat([ei_flow, ei_coord], dim=1)
            et = torch.cat([et_flow, et_coord])
        else:
            ei = ei_flow
            et = et_flow
        return x, ei, et

    def test_typed_output_shape_two_layer(self):
        x, ei, et = self._graph(N=4, in_ch=16)
        model = GATPolicy(in_channels=16, num_heads=2, out_per_head=8,
                          num_layers=2, l2_out_per_head=4, typed_edges=True)
        out = model(x, ei, et)
        assert out.shape == (4, 8)   # 2 heads × 4 = 8

    def test_typed_output_shape_one_layer(self):
        x, ei, et = self._graph(N=4, in_ch=16)
        model = GATPolicy(in_channels=16, num_heads=2, out_per_head=8,
                          num_layers=1, typed_edges=True)
        out = model(x, ei, et)
        assert out.shape == (4, 16)  # 2 heads × 8 = 16

    def test_untyped_path_unchanged(self):
        """Passing edge_type to typed_edges=False must be silently ignored.

        Calling with and without edge_type must produce identical outputs.
        """
        torch.manual_seed(0)
        x, ei, et = self._graph(N=4, in_ch=16)
        model = GATPolicy(in_channels=16, num_heads=2, out_per_head=8,
                          num_layers=1, typed_edges=False)
        with torch.no_grad():
            out_without = model(x, ei)
            out_with    = model(x, ei, et)
        assert torch.allclose(out_without, out_with, atol=1e-6)

    def test_typed_flow_only_matches_untyped(self):
        """With gat_coord weights zeroed and only flow edges, typed output equals untyped.

        Validates the flow stream is architecturally equivalent to the untyped single stream:
        same weights in gat_flow as in gat, zero coord contribution → identical outputs.
        """
        torch.manual_seed(0)
        N, in_ch, nh, oph = 4, 16, 2, 8
        x = torch.randn(N, in_ch)
        ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        et = torch.zeros(4, dtype=torch.long)  # all flow (type 0)

        untyped = GATPolicy(in_channels=in_ch, num_heads=nh, out_per_head=oph,
                            num_layers=1, typed_edges=False)
        typed   = GATPolicy(in_channels=in_ch, num_heads=nh, out_per_head=oph,
                            num_layers=1, typed_edges=True)

        # Copy gat weights → gat_flow (identical constructor, identical parameter shapes)
        typed.gat_flow.load_state_dict(untyped.gat.state_dict())
        # Zero ALL gat_coord parameters (weights + bias) so it contributes nothing
        with torch.no_grad():
            for p in typed.gat_coord.parameters():
                p.zero_()

        with torch.no_grad():
            out_untyped = untyped(x, ei)       # [N, 16]
            out_typed   = typed(x, ei, et)     # [N, 16]

        assert torch.allclose(out_untyped, out_typed, atol=1e-6), (
            "With zeroed gat_coord and flow-only edges, typed must equal untyped"
        )

    def test_typed_coord_edges_change_output(self):
        """Adding coord edges (type 1) changes the output — validates coord routing is active."""
        torch.manual_seed(0)
        x, ei_both, et_both = self._graph(N=4, in_ch=16, include_coord=True)
        # Use the same x but flow-only edges
        ei_flow = ei_both[:, et_both == 0]
        et_flow = et_both[et_both == 0]

        model = GATPolicy(in_channels=16, num_heads=2, out_per_head=8,
                          num_layers=1, typed_edges=True)
        with torch.no_grad():
            out_flow_only = model(x, ei_flow, et_flow)
            out_both      = model(x, ei_both, et_both)

        assert not torch.allclose(out_flow_only, out_both), (
            "Adding coord edges must change the output — coord routing appears inactive"
        )

    def test_typed_zero_hop_shape(self):
        """zero_hop=True: correct output shape, no crash.

        Additional assertion: gat_coord with empty edge_index produces x-independent output
        (only the constant bias contributes; no node-specific coord signal flows).
        """
        torch.manual_seed(0)
        N, in_ch = 4, 16
        x  = torch.randn(N, in_ch)
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        et = torch.zeros(2, dtype=torch.long)

        model = GATPolicy(in_channels=in_ch, num_heads=2, out_per_head=8,
                          num_layers=2, l2_out_per_head=4,
                          zero_hop=True, typed_edges=True)
        with torch.no_grad():
            out = model(x, ei, et)
        assert out.shape == (N, 2 * 4)   # [4, 8]

        # gat_coord(x, empty_ei) with add_self_loops=False has no edges to propagate.
        # Output = zeros (aggregation) + bias (constant) — entirely x-independent.
        empty = ei.new_zeros(2, 0)
        x2 = torch.randn(N, in_ch)
        with torch.no_grad():
            coord1 = model.gat_coord(x, empty)
            coord2 = model.gat_coord(x2, empty)
        assert torch.allclose(coord1, coord2, atol=1e-6), (
            "gat_coord with empty edge_index must produce x-independent output "
            "(no coord routing; only constant bias contributes)"
        )

    def test_coord_only_nodes_get_zero_coord_signal(self):
        """With no coord edges in edge_type, total output = gat_flow + gat_coord(empty).

        Verifies the decomposition: full typed forward equals the sum of its two streams.
        """
        torch.manual_seed(0)
        N, in_ch = 4, 16
        x = torch.randn(N, in_ch)
        # All edges are flow type (no coord edges)
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        et = torch.zeros(2, dtype=torch.long)

        model = GATPolicy(in_channels=in_ch, num_heads=2, out_per_head=8,
                          num_layers=1, typed_edges=True)

        empty = ei.new_zeros(2, 0)
        with torch.no_grad():
            out_full  = model(x, ei, et)               # typed forward: ei_coord = empty
            flow_out  = model.gat_flow(x, ei)           # flow with all edges
            coord_out = model.gat_coord(x, empty)       # coord with empty ei

        # Since all edges are type 0, ei_coord inside forward is empty → same as coord_out
        assert torch.allclose(out_full, flow_out + coord_out, atol=1e-6), (
            "With no coord edges, output must equal gat_flow(ei) + gat_coord(empty)"
        )

    def test_typed_trainer_select_actions(self):
        """DQNTrainer with typed_edges=True returns valid phase indices for all nodes."""
        trainer, env, _ = _make_trainer_typed()
        obs_dict, graph = env.reset(seed=1)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)
        assert set(actions.keys()) == set(graph["node_ids"])
        for node_id, phase_idx in actions.items():
            node_idx   = graph["node_to_idx"][node_id]
            num_phases = graph["node_meta"][node_idx]["num_phases"]
            assert 0 <= phase_idx < num_phases

    def test_typed_trainer_compute_loss(self):
        """_compute_loss with typed_edges=True returns finite loss."""
        trainer, env, _ = _make_trainer_typed()
        obs_dict, graph = env.reset(seed=0)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        for _ in range(20):
            actions = {nid: 0 for nid in graph["node_ids"]}
            next_obs, _, rew, done, _ = env.step(actions)
            _, padded_next = pad_obs_dict(next_obs)
            trainer.buffer.push(padded_obs, padded_next, actions, rew, done)
            padded_obs = padded_next
            if done:
                obs_dict, graph = env.reset()
                _, padded_obs = pad_obs_dict(obs_dict)
        batch = trainer.buffer.sample(8)
        loss, q_mean = trainer._compute_loss(batch, graph, padded_pf, gamma=0.99)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert isinstance(q_mean, float)

    def test_typed_checkpoint_roundtrip(self):
        """save → load → same predictions; checkpoint keys include gat_flow."""
        trainer, env, cfg = _make_trainer_typed()
        obs_dict, graph = env.reset(seed=3)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        orig_actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt_r6.pt")
            trainer.save_checkpoint(path, step=0, metrics={})

            ckpt = torch.load(path, map_location="cpu")
            assert any(k.startswith("gat_flow.") for k in ckpt["gat"].keys()), (
                "Typed checkpoint must contain gat_flow weights"
            )

            env2   = MockEnv("cross_smoke", max_steps=10)
            loaded = DQNTrainer.load_checkpoint(path, cfg, env2)

        obs_dict2, graph2 = env.reset(seed=3)
        _, padded_obs2 = pad_obs_dict(obs_dict2)
        _, padded_pf2  = pad_phase_features(graph2)
        loaded_actions = loaded._select_actions(padded_obs2, graph2, padded_pf2, epsilon=0.0)
        assert orig_actions == loaded_actions


# ---------------------------------------------------------------------------
# R7: Neighbor-masking GATPolicy (failed-source suppression)
# ---------------------------------------------------------------------------

# Small config for R7 trainer tests: R6 typed + neighbor_masking=True, severity=0.1
TRAINER_CFG_R7 = {
    **TRAINER_CFG_R6,
    "model": {
        **TRAINER_CFG_R6["model"],
        "gat": {
            **TRAINER_CFG_R6["model"]["gat"],
            "neighbor_masking": True,
        },
    },
    "perception": {"severity": 0.1, "sentinel_value": -1.0},
}


def _make_trainer_masked(network="cross_smoke"):
    cfg = {k: v for k, v in TRAINER_CFG_R7.items()}
    env = MockEnv(network, max_steps=cfg["env"]["max_steps"])
    return DQNTrainer(cfg, env, device=torch.device("cpu")), env, cfg


class TestNeighborMaskingGATPolicy:
    def _graph(self, N=4, in_ch=16, include_coord=True):
        """Build x, edge_index, edge_type with flow and optionally coord edges."""
        x = torch.randn(N, in_ch)
        # Flow edges (type 0): 0→1, 1→0, 1→2, 2→1
        ei_flow = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        et_flow = torch.zeros(4, dtype=torch.long)
        if include_coord:
            # Coord edges (type 1): 2→0, 3→1
            ei_coord = torch.tensor([[2, 3], [0, 1]], dtype=torch.long)
            et_coord = torch.ones(2, dtype=torch.long)
            ei = torch.cat([ei_flow, ei_coord], dim=1)
            et = torch.cat([et_flow, et_coord])
        else:
            ei = ei_flow
            et = et_flow
        return x, ei, et

    def _validity(self, N=4, obs_dim=8, failed_nodes=None):
        """Build [N, obs_dim] validity tensor; failed_nodes get all-zero rows (mean=0 < 0.75)."""
        v = torch.ones(N, obs_dim)
        for j in (failed_nodes or []):
            v[j] = 0.0
        return v

    def test_masking_output_shape(self):
        """node_validity with some zeros → output still [N, out_channels], no crash."""
        torch.manual_seed(0)
        x, ei, et = self._graph(N=4, in_ch=16)
        validity = self._validity(N=4, obs_dim=8, failed_nodes=[1, 3])
        model = GATPolicy(in_channels=16, num_heads=2, out_per_head=8,
                          num_layers=2, l2_out_per_head=4, typed_edges=True)
        with torch.no_grad():
            out = model(x, ei, et, validity)
        assert out.shape == (4, 8)
        assert torch.isfinite(out).all()

    def test_failed_neighbor_has_no_effect(self):
        """Masking failed source j == manually removing edges where j is SOURCE.

        Setup invariant: 'manually removed' means edge_index[:, edge_index[0] != j]
        (remove where j is SOURCE). NOT edge_index[1] != j, which removes incoming
        edges to j and would silently pass for the wrong reason.
        """
        torch.manual_seed(0)
        j = 0  # node 0 has flow edge 0→1 as source
        x, ei, et = self._graph(N=4, in_ch=16, include_coord=True)
        validity = self._validity(N=4, obs_dim=8, failed_nodes=[j])

        model = GATPolicy(in_channels=16, num_heads=2, out_per_head=8,
                          num_layers=2, l2_out_per_head=4, typed_edges=True)
        with torch.no_grad():
            out_masked = model(x, ei, et, validity)

        # Remove edges where j is the SOURCE (not destination)
        keep = ei[0] != j
        with torch.no_grad():
            out_manual = model(x, ei[:, keep], et[keep])

        assert torch.allclose(out_masked, out_manual, atol=1e-6), (
            "Masking must be equivalent to removing j's source edges"
        )

    def test_all_valid_same_as_no_masking(self):
        """All validity=1 (mean=1 >= 0.75) → no edges removed → output unchanged."""
        torch.manual_seed(0)
        x, ei, et = self._graph(N=4, in_ch=16)
        validity = self._validity(N=4, obs_dim=8, failed_nodes=[])

        model = GATPolicy(in_channels=16, num_heads=2, out_per_head=8,
                          num_layers=2, l2_out_per_head=4, typed_edges=True)
        with torch.no_grad():
            out_masked   = model(x, ei, et, validity)
            out_unmasked = model(x, ei, et)

        assert torch.allclose(out_masked, out_unmasked, atol=1e-6), (
            "All-valid node_validity must not change the output"
        )

    def test_all_nodes_marked_failed_by_threshold(self):
        """All validity=0 → mean(0)=0 < 0.75 → all external edges removed.

        Output still has correct shape and no NaN: self-loops via gat_flow survive
        inside GATConv (they are never in the user's edge_index).
        """
        torch.manual_seed(0)
        N, in_ch, obs_dim = 4, 16, 8
        x = torch.randn(N, in_ch)
        ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        et = torch.zeros(4, dtype=torch.long)
        validity = torch.zeros(N, obs_dim)  # all zero → mean=0 < 0.75 → all failed

        model = GATPolicy(in_channels=in_ch, num_heads=2, out_per_head=8,
                          num_layers=2, l2_out_per_head=4, typed_edges=True)
        with torch.no_grad():
            out = model(x, ei, et, validity)

        assert out.shape == (N, 8)
        assert torch.isfinite(out).all(), "No NaN/Inf even when all external edges removed"

    def test_masking_with_typed_edges(self):
        """Masking applies before flow/coord split — both streams filtered consistently.

        Node 0 is source in both a flow edge (0→1) and a coord edge (0→2).
        Masked output must equal output with both edges manually removed.
        """
        torch.manual_seed(0)
        N, in_ch, obs_dim = 4, 16, 8
        x = torch.randn(N, in_ch)
        # Node 0 is source of flow 0→1 (type 0) and coord 0→2 (type 1)
        ei = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
        et = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        validity = torch.ones(N, obs_dim)
        validity[0] = 0.0  # node 0 failed

        model = GATPolicy(in_channels=in_ch, num_heads=2, out_per_head=8,
                          num_layers=2, l2_out_per_head=4, typed_edges=True)
        with torch.no_grad():
            out_masked = model(x, ei, et, validity)

        keep = ei[0] != 0  # remove ALL edges from node 0 (both flow and coord)
        with torch.no_grad():
            out_manual = model(x, ei[:, keep], et[keep])

        assert torch.allclose(out_masked, out_manual, atol=1e-6), (
            "Masking must remove edges from both flow and coord streams"
        )

    def test_masking_does_not_affect_zero_hop(self):
        """zero_hop=True skips the masking block — node_validity has no effect."""
        torch.manual_seed(0)
        x, ei, et = self._graph(N=4, in_ch=16)
        validity = self._validity(N=4, obs_dim=8, failed_nodes=[0, 2])

        model = GATPolicy(in_channels=16, num_heads=2, out_per_head=8,
                          num_layers=2, l2_out_per_head=4,
                          typed_edges=True, zero_hop=True)
        with torch.no_grad():
            out_with_mask    = model(x, ei, et, validity)
            out_without_mask = model(x, ei, et)

        assert torch.allclose(out_with_mask, out_without_mask, atol=1e-6), (
            "zero_hop=True must be unaffected by node_validity"
        )

    def test_trainer_select_actions_with_masking(self):
        """DQNTrainer with neighbor_masking=True, severity=0.1 returns valid actions."""
        trainer, env, _ = _make_trainer_masked()
        obs_dict, graph = env.reset(seed=1)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)
        assert set(actions.keys()) == set(graph["node_ids"])
        for node_id, phase_idx in actions.items():
            node_idx   = graph["node_to_idx"][node_id]
            num_phases = graph["node_meta"][node_idx]["num_phases"]
            assert 0 <= phase_idx < num_phases

    def test_trainer_compute_loss_with_masking(self):
        """_compute_loss with neighbor_masking=True and degraded obs → finite loss."""
        from env.perception import apply_perception
        trainer, env, _ = _make_trainer_masked()
        obs_dict, graph = env.reset(seed=0)
        # Use high severity to force validity zeros and exercise the masking code path
        obs_dict = apply_perception(obs_dict, severity=0.5, sentinel=-1.0)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        for _ in range(20):
            actions = {nid: 0 for nid in graph["node_ids"]}
            next_obs, _, rew, done, _ = env.step(actions)
            next_obs = apply_perception(next_obs, severity=0.5, sentinel=-1.0)
            _, padded_next = pad_obs_dict(next_obs)
            trainer.buffer.push(padded_obs, padded_next, actions, rew, done)
            padded_obs = padded_next
            if done:
                obs_dict, graph = env.reset()
                obs_dict = apply_perception(obs_dict, severity=0.5, sentinel=-1.0)
                _, padded_obs = pad_obs_dict(obs_dict)
        batch = trainer.buffer.sample(8)
        loss, q_mean = trainer._compute_loss(batch, graph, padded_pf, gamma=0.99)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert isinstance(q_mean, float)

    def test_partial_failure_changes_output(self):
        """Some nodes failed, masking enabled → output differs from unmasked run."""
        torch.manual_seed(0)
        # Nodes 0 and 2 are sources of edges in this graph; marking them failed
        # removes those edges and changes the output
        x, ei, et = self._graph(N=4, in_ch=16, include_coord=True)
        validity = self._validity(N=4, obs_dim=8, failed_nodes=[0, 2])

        model = GATPolicy(in_channels=16, num_heads=2, out_per_head=8,
                          num_layers=2, l2_out_per_head=4, typed_edges=True)
        with torch.no_grad():
            out_masked   = model(x, ei, et, validity)
            out_unmasked = model(x, ei, et)

        assert not torch.allclose(out_masked, out_unmasked), (
            "Masking failed source nodes must change the output"
        )

    def test_masking_checkpoint_roundtrip(self):
        """Save trainer with neighbor_masking=True; reload; same predictions."""
        trainer, env, cfg = _make_trainer_masked()
        obs_dict, graph = env.reset(seed=5)
        _, padded_obs = pad_obs_dict(obs_dict)
        _, padded_pf  = pad_phase_features(graph)
        orig_actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt_r7.pt")
            trainer.save_checkpoint(path, step=0, metrics={})
            env2   = MockEnv("cross_smoke", max_steps=10)
            loaded = DQNTrainer.load_checkpoint(path, cfg, env2)

        obs_dict2, graph2 = env.reset(seed=5)
        _, padded_obs2 = pad_obs_dict(obs_dict2)
        _, padded_pf2  = pad_phase_features(graph2)
        loaded_actions = loaded._select_actions(padded_obs2, graph2, padded_pf2, epsilon=0.0)
        assert orig_actions == loaded_actions
