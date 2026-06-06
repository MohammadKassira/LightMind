"""Tests for models/phase_head.py — PhaseHead.

Covers: output shape/dtype, -inf masking, select_action, NaN safety,
handcrafted gradient-flow sensibility check, and end-to-end with MockEnv.
"""

import pytest
import torch
import torch.nn as nn

from env.mock_env import MockEnv
from models.node_encoder import NodeEncoder, pad_obs_dict
from models.phase_head import PhaseHead, pad_phase_features

ALL_NETWORKS = ["cross_smoke", "linear_two", "grid_3x3"]

EMBED_DIM = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_head_and_inputs(num_phases: int, phase_feat_dim: int, embed_dim: int = EMBED_DIM):
    """Return (head, embedding, phase_features, all_valid_mask)."""
    torch.manual_seed(0)
    head = PhaseHead(embed_dim=embed_dim, phase_feat_dim=phase_feat_dim)
    embedding = torch.rand(embed_dim)
    phase_features = [torch.rand(phase_feat_dim) for _ in range(num_phases)]
    mask = torch.ones(num_phases, dtype=torch.bool)
    return head, embedding, phase_features, mask


# ---------------------------------------------------------------------------
# PhaseHead — output shape and dtype
# ---------------------------------------------------------------------------

class TestPhaseHeadOutput:
    @pytest.mark.parametrize("num_phases,phase_feat_dim", [(2, 4), (3, 2), (4, 6)])
    def test_output_shape(self, num_phases, phase_feat_dim):
        head, emb, feats, mask = make_head_and_inputs(num_phases, phase_feat_dim)
        scores = head(emb, feats, mask)
        assert scores.shape == (num_phases,), f"expected ({num_phases},), got {scores.shape}"

    @pytest.mark.parametrize("num_phases,phase_feat_dim", [(2, 4), (3, 2)])
    def test_output_dtype(self, num_phases, phase_feat_dim):
        head, emb, feats, mask = make_head_and_inputs(num_phases, phase_feat_dim)
        assert head(emb, feats, mask).dtype == torch.float32


# ---------------------------------------------------------------------------
# PhaseHead — invalid phase masking
# ---------------------------------------------------------------------------

class TestPhaseHeadMasking:
    def test_invalid_phases_are_neg_inf(self):
        head, emb, feats, mask = make_head_and_inputs(3, 4)
        mask[1] = False   # mark phase 1 as invalid
        scores = head(emb, feats, mask)
        assert scores[1].item() == float("-inf"), "invalid phase must score -inf"
        assert scores[0].item() != float("-inf")
        assert scores[2].item() != float("-inf")

    def test_all_valid_no_neg_inf(self):
        head, emb, feats, mask = make_head_and_inputs(2, 3)
        scores = head(emb, feats, mask)
        assert not torch.isinf(scores).any(), "no phase should be -inf when all valid"

    def test_all_invalid_returns_all_neg_inf(self):
        head, emb, feats, mask = make_head_and_inputs(2, 3)
        mask[:] = False
        scores = head(emb, feats, mask)
        assert torch.all(torch.isinf(scores) & (scores < 0)), (
            "all phases invalid → all scores must be -inf"
        )

    def test_multiple_invalid(self):
        head, emb, feats, mask = make_head_and_inputs(4, 4)
        mask[0] = False
        mask[2] = False
        scores = head(emb, feats, mask)
        assert scores[0].item() == float("-inf")
        assert scores[2].item() == float("-inf")
        assert scores[1].item() != float("-inf")
        assert scores[3].item() != float("-inf")


# ---------------------------------------------------------------------------
# PhaseHead — select_action
# ---------------------------------------------------------------------------

class TestPhaseHeadSelectAction:
    def test_action_is_valid_phase(self):
        head, emb, feats, mask = make_head_and_inputs(3, 4)
        mask[0] = False   # phase 0 invalid
        action = head.select_action(emb, feats, mask)
        assert action in (1, 2), f"action {action} is not a valid phase"

    def test_action_returns_int(self):
        head, emb, feats, mask = make_head_and_inputs(2, 3)
        assert isinstance(head.select_action(emb, feats, mask), int)

    def test_action_consistent_with_argmax(self):
        head, emb, feats, mask = make_head_and_inputs(4, 4)
        head.eval()
        with torch.no_grad():
            scores = head(emb, feats, mask)
            expected = int(scores.argmax().item())
        action = head.select_action(emb, feats, mask)
        assert action == expected

    def test_action_never_selects_invalid_phase(self):
        """Over many random embeddings, select_action must always return a valid index."""
        head = PhaseHead(embed_dim=EMBED_DIM, phase_feat_dim=4)
        mask = torch.tensor([False, True, True, False])
        for seed in range(20):
            torch.manual_seed(seed)
            emb = torch.rand(EMBED_DIM)
            feats = [torch.rand(4) for _ in range(4)]
            action = head.select_action(emb, feats, mask)
            assert mask[action].item(), f"seed {seed}: selected invalid phase {action}"


# ---------------------------------------------------------------------------
# PhaseHead — NaN safety
# ---------------------------------------------------------------------------

class TestPhaseHeadNaN:
    def test_no_nan_clean_input(self):
        head, emb, feats, mask = make_head_and_inputs(2, 4)
        scores = head(emb, feats, mask)
        # Valid phases only (ignore -inf from masking)
        valid_scores = scores[mask]
        assert not torch.isnan(valid_scores).any()

    def test_no_nan_zero_embedding(self):
        head = PhaseHead(embed_dim=EMBED_DIM, phase_feat_dim=4)
        emb = torch.zeros(EMBED_DIM)
        feats = [torch.rand(4) for _ in range(2)]
        mask = torch.ones(2, dtype=torch.bool)
        scores = head(emb, feats, mask)
        assert not torch.isnan(scores).any()


# ---------------------------------------------------------------------------
# PhaseHead — handcrafted gradient-flow sensibility (R1 done-when)
# ---------------------------------------------------------------------------

class TestPhaseHeadSensibility:
    def test_gradient_flows_to_phase_features(self):
        """Phase feature vectors must have non-zero gradients after backward."""
        torch.manual_seed(5)
        head = PhaseHead(embed_dim=8, phase_feat_dim=2)
        emb = torch.rand(8)
        feats = [torch.rand(2, requires_grad=True) for _ in range(2)]
        mask = torch.ones(2, dtype=torch.bool)
        scores = head(emb, feats, mask)
        scores.sum().backward()
        for i, pf in enumerate(feats):
            assert pf.grad is not None and pf.grad.abs().sum() > 0, (
                f"No gradient on phase_features[{i}]"
            )

    def test_head_can_learn_to_prefer_high_pressure_phase(self):
        """After a few gradient steps the head should score the target phase higher.

        Setup: 2 phases, phase_feat_dim=2.
          Phase 0 features = [1, 0]  (lane 0 green)
          Phase 1 features = [0, 1]  (lane 1 green)
        We push the head to score phase 0 higher by minimising -score[0].
        After 20 steps, score[0] > score[1] should hold.
        """
        torch.manual_seed(42)
        embed_dim, phase_feat_dim = 16, 2
        head = PhaseHead(embed_dim=embed_dim, phase_feat_dim=phase_feat_dim)
        optimiser = torch.optim.Adam(head.parameters(), lr=1e-2)

        emb = torch.rand(embed_dim)
        phase_feats = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        mask = torch.ones(2, dtype=torch.bool)

        # Precondition: random init must NOT already prefer phase 0.
        # If this assertion fails, the post-training check would prove nothing.
        with torch.no_grad():
            init_scores = head(emb, phase_feats, mask)
        assert init_scores[0] <= init_scores[1], (
            "Precondition failed: random init already scores phase 0 higher — "
            "change the seed so training is what produces the preference"
        )

        for _ in range(20):
            optimiser.zero_grad()
            scores = head(emb, phase_feats, mask)
            loss = -scores[0]   # maximise score for phase 0
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            final_scores = head(emb, phase_feats, mask)
        assert final_scores[0] > final_scores[1], (
            "Head failed to learn to prefer phase 0 over 20 steps — "
            f"scores: {final_scores.tolist()}"
        )


# ---------------------------------------------------------------------------
# PhaseHead — end-to-end with MockEnv + NodeEncoder (R1 done-when)
# ---------------------------------------------------------------------------

class TestPhaseHeadEndToEnd:
    @pytest.mark.parametrize("network", ALL_NETWORKS)
    def test_forward_pass_completes(self, network):
        env = MockEnv(network)
        obs_dict, graph = env.reset(seed=0)
        obs_dim, padded_obs = pad_obs_dict(obs_dict)
        phase_feat_dim, padded_phase_feats = pad_phase_features(graph)

        encoder = NodeEncoder(obs_dim=obs_dim)
        head = PhaseHead(embed_dim=EMBED_DIM, phase_feat_dim=phase_feat_dim)

        for node_id, (obs, val) in padded_obs.items():
            node_idx = graph["node_to_idx"][node_id]
            embedding = encoder(obs, val)
            num_phases = graph["node_meta"][node_idx]["num_phases"]

            # Current phase from obs (phase_onehot is the first segment, §3.2).
            # Do NOT hardcode 0 — in R2 current_phase changes every step.
            raw_obs = obs_dict[node_id][0]
            current_phase = raw_obs[:num_phases].argmax().item()
            mask = graph["node_meta"][node_idx]["valid_transition_mask"][current_phase]

            scores = head(embedding, padded_phase_feats[node_idx], mask)
            action = head.select_action(embedding, padded_phase_feats[node_idx], mask)

            assert scores.shape == (num_phases,)
            assert not torch.isnan(scores[mask]).any()
            assert 0 <= action < num_phases

    @pytest.mark.parametrize("network", ALL_NETWORKS)
    def test_action_in_valid_range_over_multiple_steps(self, network):
        env = MockEnv(network, max_steps=5)
        obs_dict, graph = env.reset(seed=1)
        obs_dim, _ = pad_obs_dict(obs_dict)
        # phase_feat_dim and padded_phase_feats are fixed for the episode (static graph)
        phase_feat_dim, padded_phase_feats = pad_phase_features(graph)
        encoder = NodeEncoder(obs_dim=obs_dim)
        head = PhaseHead(embed_dim=EMBED_DIM, phase_feat_dim=phase_feat_dim)

        done = False
        actions = {node_id: 0 for node_id in graph["node_ids"]}
        while not done:
            obs_dict, graph, _, done, _ = env.step(actions)
            _, padded_obs = pad_obs_dict(obs_dict)
            for node_id, (obs, val) in padded_obs.items():
                node_idx = graph["node_to_idx"][node_id]
                emb = encoder(obs, val)
                num_phases = graph["node_meta"][node_idx]["num_phases"]
                raw_obs = obs_dict[node_id][0]
                current_phase = raw_obs[:num_phases].argmax().item()
                mask = graph["node_meta"][node_idx]["valid_transition_mask"][current_phase]
                action = head.select_action(emb, padded_phase_feats[node_idx], mask)
                assert 0 <= action < num_phases
                actions[node_id] = action
