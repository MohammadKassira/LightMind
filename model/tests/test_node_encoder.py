"""Tests for models/node_encoder.py — NodeEncoder and pad_obs_dict.

Covers: output shape/dtype, validity masking, NaN safety, variable obs sizes
across networks, batch forward, and an end-to-end pass through MockEnv.
"""

import pytest
import torch

from env.mock_env import MockEnv
from models.node_encoder import NodeEncoder, pad_obs_dict

# Networks available in data/networks/
ALL_NETWORKS = ["cross_smoke", "linear_two", "grid_3x3"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=ALL_NETWORKS)
def env_reset(request):
    """(obs_dim, padded_obs_dict, graph) for each test network."""
    env = MockEnv(request.param)
    obs_dict, graph = env.reset(seed=0)
    obs_dim, padded = pad_obs_dict(obs_dict)
    return obs_dim, padded, graph


@pytest.fixture(scope="module")
def cross_smoke_reset():
    env = MockEnv("cross_smoke")
    obs_dict, graph = env.reset(seed=0)
    obs_dim, padded = pad_obs_dict(obs_dict)
    return obs_dim, padded, graph


@pytest.fixture(scope="module")
def linear_two_reset():
    env = MockEnv("linear_two")
    obs_dict, graph = env.reset(seed=0)
    obs_dim, padded = pad_obs_dict(obs_dict)
    return obs_dim, padded, graph


# ---------------------------------------------------------------------------
# pad_obs_dict
# ---------------------------------------------------------------------------

class TestPadObsDict:
    def test_all_obs_same_length(self, env_reset):
        obs_dim, padded, _ = env_reset
        for obs, val in padded.values():
            assert obs.shape[-1] == obs_dim

    def test_validity_zero_on_padding(self, linear_two_reset):
        """linear_two has two nodes with different obs lengths; shorter one gets padding."""
        obs_dim, padded, graph = linear_two_reset
        for node_id, (obs, val) in padded.items():
            node_idx = graph["node_to_idx"][node_id]
            n_orig = graph["node_meta"][node_idx]["num_phases"]
            # We can't know original obs_dim without re-constructing, but we can
            # verify that padded regions (if any) have validity=0.
            # Find the un-padded length from a fresh reset.
            env = MockEnv("linear_two")
            raw_obs_dict, _ = env.reset(seed=0)
            orig_len = raw_obs_dict[node_id][0].shape[-1]
            if orig_len < obs_dim:
                assert (val[..., orig_len:] == 0.0).all()

    def test_obs_values_preserved_up_to_original_length(self, linear_two_reset):
        obs_dim, padded, graph = linear_two_reset
        env = MockEnv("linear_two")
        raw_obs_dict, _ = env.reset(seed=0)
        for node_id, (obs, val) in padded.items():
            orig_obs, orig_val = raw_obs_dict[node_id]
            n = orig_obs.shape[-1]
            assert torch.allclose(obs[..., :n], orig_obs)
            assert torch.allclose(val[..., :n], orig_val)


# ---------------------------------------------------------------------------
# NodeEncoder — output shape and dtype
# ---------------------------------------------------------------------------

class TestNodeEncoderOutput:
    def test_output_shape_default_embed_dim(self, env_reset):
        obs_dim, padded, _ = env_reset
        encoder = NodeEncoder(obs_dim=obs_dim)
        for obs, val in padded.values():
            out = encoder(obs, val)
            assert out.shape == (64,), f"expected (64,), got {out.shape}"

    def test_output_shape_custom_embed_dim(self, cross_smoke_reset):
        obs_dim, padded, _ = cross_smoke_reset
        encoder = NodeEncoder(obs_dim=obs_dim, embed_dim=32)
        obs, val = next(iter(padded.values()))
        assert encoder(obs, val).shape == (32,)

    def test_output_dtype(self, env_reset):
        obs_dim, padded, _ = env_reset
        encoder = NodeEncoder(obs_dim=obs_dim)
        obs, val = next(iter(padded.values()))
        assert encoder(obs, val).dtype == torch.float32


# ---------------------------------------------------------------------------
# NodeEncoder — NaN safety
# ---------------------------------------------------------------------------

class TestNodeEncoderNaN:
    def test_no_nan_clean_input(self, env_reset):
        obs_dim, padded, _ = env_reset
        encoder = NodeEncoder(obs_dim=obs_dim)
        for obs, val in padded.values():
            out = encoder(obs, val)
            assert not torch.isnan(out).any(), "NaN in output for clean input"

    def test_no_nan_missing_input(self):
        """Sentinel (-1.0) values with validity=0 must not produce NaN."""
        obs_dim = 15
        encoder = NodeEncoder(obs_dim=obs_dim)
        obs = torch.full((obs_dim,), -1.0)   # all sentinel
        val = torch.zeros(obs_dim)            # all missing
        out = encoder(obs, val)
        assert not torch.isnan(out).any()

    def test_no_nan_partial_missing(self):
        obs_dim = 15
        encoder = NodeEncoder(obs_dim=obs_dim)
        obs = torch.rand(obs_dim)
        val = torch.ones(obs_dim)
        # corrupt half
        obs[:5] = -1.0
        val[:5] = 0.0
        out = encoder(obs, val)
        assert not torch.isnan(out).any()

    def test_no_nan_mock_env_missing_data(self):
        env = MockEnv("cross_smoke", missing_prob=0.5)
        obs_dict, graph = env.reset(seed=7)
        obs_dim, padded = pad_obs_dict(obs_dict)
        encoder = NodeEncoder(obs_dim=obs_dim)
        for obs, val in padded.values():
            assert not torch.isnan(encoder(obs, val)).any()


# ---------------------------------------------------------------------------
# NodeEncoder — validity masking
# ---------------------------------------------------------------------------

class TestNodeEncoderValidityMask:
    def test_sentinel_distinct_from_imputed_when_missing(self):
        """With cat([obs, validity]), sentinel obs and last-known obs with validity=0
        produce different embeddings — the model can tell them apart."""
        obs_dim = 12
        torch.manual_seed(0)
        encoder = NodeEncoder(obs_dim=obs_dim)
        encoder.eval()

        val = torch.ones(obs_dim)
        val[3] = 0.0   # position 3 is missing for both
        obs_sentinel = torch.rand(obs_dim)
        obs_sentinel[3] = -1.0    # never-seen case
        obs_imputed = obs_sentinel.clone()
        obs_imputed[3] = 0.6      # last-known imputed value

        with torch.no_grad():
            out_sentinel = encoder(obs_sentinel, val)
            out_imputed  = encoder(obs_imputed, val)

        assert not torch.allclose(out_sentinel, out_imputed), (
            "Encoder should distinguish sentinel from last-known imputed value"
        )

    def test_all_zeros_validity_gives_zero_input_to_mlp(self):
        """When all validity=0, obs*validity=0; encoder should not NaN."""
        obs_dim = 10
        encoder = NodeEncoder(obs_dim=obs_dim)
        obs = torch.rand(obs_dim) * 5   # arbitrary large values
        val = torch.zeros(obs_dim)
        out = encoder(obs, val)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# NodeEncoder — variable obs sizes (padding path)
# ---------------------------------------------------------------------------

class TestNodeEncoderVariableObsSizes:
    def test_same_encoder_handles_two_nodes_with_different_raw_obs_lengths(self):
        """linear_two has A0 and B0 with different raw obs lengths.
        After padding to obs_dim, both must produce (embed_dim,) embeddings."""
        env = MockEnv("linear_two")
        obs_dict, graph = env.reset(seed=0)
        obs_dim, padded = pad_obs_dict(obs_dict)

        node_ids = list(padded.keys())
        assert len(node_ids) == 2, "linear_two should have exactly 2 nodes"

        obs_a, val_a = padded[node_ids[0]]
        obs_b, val_b = padded[node_ids[1]]
        # They must differ in raw length but match after padding
        raw_a = obs_dict[node_ids[0]][0].shape[-1]
        raw_b = obs_dict[node_ids[1]][0].shape[-1]
        assert raw_a != raw_b, "test assumes A0 and B0 have different raw obs lengths"

        encoder = NodeEncoder(obs_dim=obs_dim)
        out_a = encoder(obs_a, val_a)
        out_b = encoder(obs_b, val_b)
        assert out_a.shape == out_b.shape == (64,)


# ---------------------------------------------------------------------------
# NodeEncoder — batch forward
# ---------------------------------------------------------------------------

class TestNodeEncoderBatch:
    def test_batch_output_shape(self, cross_smoke_reset):
        obs_dim, padded, _ = cross_smoke_reset
        encoder = NodeEncoder(obs_dim=obs_dim)
        obs, val = next(iter(padded.values()))
        batch = 8
        obs_batch = obs.unsqueeze(0).expand(batch, -1)
        val_batch = val.unsqueeze(0).expand(batch, -1)
        out = encoder(obs_batch, val_batch)
        assert out.shape == (batch, 64)

    def test_batch_matches_individual(self, cross_smoke_reset):
        obs_dim, padded, _ = cross_smoke_reset
        torch.manual_seed(1)
        encoder = NodeEncoder(obs_dim=obs_dim)
        encoder.eval()
        obs, val = next(iter(padded.values()))

        obs_batch = obs.unsqueeze(0).repeat(3, 1)
        val_batch = val.unsqueeze(0).repeat(3, 1)
        with torch.no_grad():
            batch_out = encoder(obs_batch, val_batch)
            single_out = encoder(obs, val)
        for i in range(3):
            assert torch.allclose(batch_out[i], single_out, atol=1e-6)


# ---------------------------------------------------------------------------
# NodeEncoder — end-to-end with MockEnv (R1 done-when)
# ---------------------------------------------------------------------------

class TestNodeEncoderEndToEnd:
    @pytest.mark.parametrize("network", ALL_NETWORKS)
    def test_forward_pass_completes(self, network):
        env = MockEnv(network)
        obs_dict, graph = env.reset(seed=0)
        obs_dim, padded = pad_obs_dict(obs_dict)
        encoder = NodeEncoder(obs_dim=obs_dim)
        for node_id, (obs, val) in padded.items():
            out = encoder(obs, val)
            assert out.shape == (64,)
            assert not torch.isnan(out).any()

    def test_embeddings_differ_across_nodes(self):
        """Distinct obs vectors should (almost certainly) produce distinct embeddings."""
        env = MockEnv("grid_3x3")
        obs_dict, graph = env.reset(seed=42)
        obs_dim, padded = pad_obs_dict(obs_dict)
        encoder = NodeEncoder(obs_dim=obs_dim)
        encoder.eval()
        embeddings = []
        with torch.no_grad():
            for obs, val in padded.values():
                embeddings.append(encoder(obs, val))
        # At least two nodes should have different embeddings
        all_same = all(torch.allclose(embeddings[0], e) for e in embeddings[1:])
        assert not all_same, "All node embeddings are identical — obs may be degenerate"
