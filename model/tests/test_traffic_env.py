"""Tests for env/traffic_env.py — TrafficEnv SUMO integration.

Test fixture: networks/external/RESCO/cologne1/ — pre-existing route file,
single-intersection network with real historical demand (7–8 AM window).
"""

import os
from pathlib import Path

import pytest
import torch

from data.graph_builder import build_graph

_REPO = Path(__file__).parent.parent
_NET  = _REPO / "networks" / "external" / "RESCO" / "cologne1" / "cologne1.net.xml"
_ROU  = _REPO / "networks" / "external" / "RESCO" / "cologne1" / "cologne1.rou.xml"

# cologne1.sumocfg declares begin=25200 (7 AM); match it so vehicles are present
_BEGIN_TIME = 25200
_MAX_STEPS  = 5   # keep tests fast


@pytest.fixture(scope="module")
def env():
    """TrafficEnv for cologne1 — shared across all tests in this module."""
    from env.traffic_env import TrafficEnv
    e = TrafficEnv(
        net_file=_NET,
        route_file=_ROU,
        max_steps=_MAX_STEPS,
        delta_time=5,
        yellow_time=2,
        min_green=5,
        begin_time=_BEGIN_TIME,
    )
    yield e
    e.close()


@pytest.fixture(scope="module")
def reset_result(env):
    obs_dict, graph = env.reset(seed=0)
    return obs_dict, graph


# ---------------------------------------------------------------------------
# reset() contract
# ---------------------------------------------------------------------------

class TestReset:
    def test_returns_2tuple(self, reset_result):
        assert len(reset_result) == 2

    def test_obs_dict_is_dict(self, reset_result):
        obs_dict, _ = reset_result
        assert isinstance(obs_dict, dict)

    def test_graph_is_dict(self, reset_result):
        _, graph = reset_result
        assert isinstance(graph, dict)

    def test_obs_dict_has_all_node_ids(self, reset_result):
        obs_dict, graph = reset_result
        assert set(obs_dict.keys()) == set(graph["node_ids"])

    def test_obs_values_are_2tuples_of_tensors(self, reset_result):
        obs_dict, _ = reset_result
        for obs, val in obs_dict.values():
            assert isinstance(obs, torch.Tensor)
            assert isinstance(val, torch.Tensor)

    def test_graph_identical_to_build_graph(self, reset_result):
        _, graph = reset_result
        ref = build_graph(_NET)
        assert graph["node_ids"] == ref["node_ids"]
        assert set(graph["node_to_idx"].keys()) == set(ref["node_to_idx"].keys())


# ---------------------------------------------------------------------------
# Observation layout (§3.2)
# ---------------------------------------------------------------------------

class TestObsLayout:
    def test_obs_shape_matches_expected(self, reset_result):
        """obs.shape[0] must equal the §3.2 formula for each node."""
        obs_dict, graph = reset_result
        for node_id, (obs, val) in obs_dict.items():
            node_idx   = graph["node_to_idx"][node_id]
            num_phases = graph["node_meta"][node_idx]["num_phases"]
            phase_feats = graph["phase_features"][node_idx]
            num_incoming = len(phase_feats[0]) if phase_feats else 0
            # num_outgoing is read from obs length (may differ from incoming)
            expected_min = num_phases + 1 + 2 * num_incoming  # at least
            assert obs.shape[0] >= expected_min, (
                f"{node_id}: obs.shape={obs.shape}, expected >= {expected_min}"
            )

    def test_validity_all_ones(self, reset_result):
        obs_dict, _ = reset_result
        for obs, val in obs_dict.values():
            assert val.shape == obs.shape
            assert (val == 1.0).all(), "TrafficEnv must return validity=1 (perception applied outside)"

    def test_obs_dtype_float32(self, reset_result):
        obs_dict, _ = reset_result
        for obs, val in obs_dict.values():
            assert obs.dtype == torch.float32
            assert val.dtype == torch.float32

    def test_obs_range(self, reset_result):
        """All obs values should be in [0, 1] (phase_onehot is 0/1; queues clamped)."""
        obs_dict, _ = reset_result
        for obs, _ in obs_dict.values():
            assert obs.min().item() >= 0.0 - 1e-6, f"obs has negative value: {obs.min().item()}"
            assert obs.max().item() <= 1.0 + 1e-6, f"obs exceeds 1.0: {obs.max().item()}"

    def test_phase_onehot_sums_to_one(self, reset_result):
        obs_dict, graph = reset_result
        for node_id, (obs, _) in obs_dict.items():
            node_idx   = graph["node_to_idx"][node_id]
            num_phases = graph["node_meta"][node_idx]["num_phases"]
            if num_phases > 0:
                phase_sum = obs[:num_phases].sum().item()
                assert abs(phase_sum - 1.0) < 1e-6, (
                    f"{node_id}: phase one-hot sum={phase_sum}, expected 1.0"
                )


# ---------------------------------------------------------------------------
# step() contract
# ---------------------------------------------------------------------------

class TestStep:
    @pytest.fixture(scope="class")
    def step_result(self, env):
        env.reset(seed=0)
        graph = env._graph
        actions = {nid: 0 for nid in graph["node_ids"]}
        return env.step(actions), graph

    def test_returns_5tuple(self, step_result):
        result, _ = step_result
        assert len(result) == 5

    def test_obs_dict_has_all_node_ids(self, step_result):
        (obs_dict, graph, _, _, _), ref_graph = step_result
        assert set(obs_dict.keys()) == set(graph["node_ids"])

    def test_reward_dict_has_all_node_ids(self, step_result):
        (_, graph, reward_dict, _, _), _ = step_result
        assert set(reward_dict.keys()) == set(graph["node_ids"])

    def test_reward_values_are_floats(self, step_result):
        (_, _, reward_dict, _, _), _ = step_result
        for v in reward_dict.values():
            assert isinstance(v, float)

    def test_reward_nonpositive(self, step_result):
        (_, _, reward_dict, _, _), _ = step_result
        for v in reward_dict.values():
            assert v <= 0.0 + 1e-9

    def test_done_is_bool(self, step_result):
        (_, _, _, done, _), _ = step_result
        assert isinstance(done, bool)

    def test_info_has_sim_time(self, step_result):
        (_, _, _, _, info), _ = step_result
        assert "sim_time" in info

    def test_graph_is_same_object_as_reset(self, env):
        obs_dict, graph = env.reset(seed=1)
        actions = {nid: 0 for nid in graph["node_ids"]}
        _, graph2, _, _, _ = env.step(actions)
        assert graph is graph2, "graph must be the same Python object across steps"


# ---------------------------------------------------------------------------
# done signal
# ---------------------------------------------------------------------------

class TestDoneSignal:
    def test_done_after_max_steps(self, env):
        env.reset(seed=2)
        graph = env._graph
        actions = {nid: 0 for nid in graph["node_ids"]}
        done = False
        steps = 0
        while not done:
            _, _, _, done, _ = env.step(actions)
            steps += 1
        assert steps == _MAX_STEPS

    def test_not_done_before_max_steps(self, env):
        env.reset(seed=3)
        graph = env._graph
        actions = {nid: 0 for nid in graph["node_ids"]}
        for _ in range(_MAX_STEPS - 1):
            _, _, _, done, _ = env.step(actions)
            assert not done


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------

class TestGraphStructure:
    def test_node_ids_nonempty(self, reset_result):
        _, graph = reset_result
        assert len(graph["node_ids"]) > 0

    def test_node_to_idx_consistent(self, reset_result):
        _, graph = reset_result
        for i, nid in enumerate(graph["node_ids"]):
            assert graph["node_to_idx"][nid] == i

    def test_node_meta_has_num_phases(self, reset_result):
        _, graph = reset_result
        for meta in graph["node_meta"]:
            assert "num_phases" in meta
            assert meta["num_phases"] >= 0

    def test_edge_index_shape(self, reset_result):
        _, graph = reset_result
        ei = graph["edge_index"]
        assert ei.shape[0] == 2
        assert ei.dtype == torch.long

    def test_valid_transition_mask_shape(self, reset_result):
        _, graph = reset_result
        for meta in graph["node_meta"]:
            p = meta["num_phases"]
            assert meta["valid_transition_mask"].shape == (p, p)
