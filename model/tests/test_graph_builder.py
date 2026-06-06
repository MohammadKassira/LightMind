"""Tests for data/graph_builder.py across four synthetic SUMO networks.

See docs/graph_builder.md for full documentation, output schema, and
a description of what each test class covers.
"""

from pathlib import Path

import pytest
import torch

from data.graph_builder import build_graph
from env.traffic_env import _parse_net_xml

NETS = Path(__file__).parent.parent / "data" / "networks"
COLOGNE3 = Path(__file__).parent.parent / "networks" / "external" / "RESCO" / "cologne3" / "cologne3.net.xml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flow_edges(g):
    """Return set of (src, dst) tuples for Type-0 (flow) edges."""
    mask = g["edge_type"] == 0
    return set(zip(g["edge_index"][0][mask].tolist(), g["edge_index"][1][mask].tolist()))


def coord_edges(g):
    """Return set of (src, dst) tuples for Type-1 (coordination) edges."""
    mask = g["edge_type"] == 1
    return set(zip(g["edge_index"][0][mask].tolist(), g["edge_index"][1][mask].tolist()))


# ---------------------------------------------------------------------------
# cross_smoke — single 4-way intersection, no inter-signal roads
# ---------------------------------------------------------------------------

class TestCrossSmoke:
    @pytest.fixture(scope="class")
    def g(self):
        return build_graph(NETS / "cross_smoke.net.xml")

    def test_single_node(self, g):
        assert g["node_ids"] == ["A0"]

    def test_node_to_idx(self, g):
        assert g["node_to_idx"] == {"A0": 0}

    def test_no_edges(self, g):
        """No two signalized junctions share a road, so graph has zero edges."""
        assert g["edge_index"].shape == (2, 0)
        assert g["edge_type"].shape == (0,)

    def test_edge_tensor_dtypes(self, g):
        assert g["edge_index"].dtype == torch.long
        assert g["edge_type"].dtype == torch.long

    def test_two_actionable_phases(self, g):
        """Yellow phases (no G/g) are excluded; only phases 0 and 2 are actionable."""
        assert g["node_meta"][0]["num_phases"] == 2

    def test_valid_transition_mask_shape(self, g):
        mask = g["node_meta"][0]["valid_transition_mask"]
        assert mask.shape == (2, 2)
        assert mask.dtype == torch.bool

    def test_valid_transition_mask_all_true(self, g):
        assert g["node_meta"][0]["valid_transition_mask"].all()

    def test_phase_features_count(self, g):
        """One node with 2 actionable phases."""
        assert len(g["phase_features"]) == 1
        assert len(g["phase_features"][0]) == 2

    def test_phase_features_shape(self, g):
        """4 unique incoming lanes (top, bottom, left, right each with lane 0)."""
        for feat in g["phase_features"][0]:
            assert feat.shape == (4,)
            assert feat.dtype == torch.float32

    def test_phase_0_ns_green(self, g):
        """Phase 0 state 'GgGGgGrrrrrr': N/S approaches (bottom0A0, top0A0) are green.
        Sorted incoming lanes: (bottom0A0,0)=idx0  (left0A0,0)=idx1
                               (right0A0,0)=idx2  (top0A0,0)=idx3
        """
        feat = g["phase_features"][0][0]
        # N/S green  → bottom=1, left=0, right=0, top=1
        expected = torch.tensor([1.0, 0.0, 0.0, 1.0])
        assert torch.allclose(feat, expected), f"Phase 0 features: {feat}"

    def test_phase_1_ew_green(self, g):
        """Phase 2 state 'rrrrrrGgGGgG': E/W approaches (left0A0, right0A0) are green."""
        feat = g["phase_features"][0][1]
        # E/W green → bottom=0, left=1, right=1, top=0
        expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
        assert torch.allclose(feat, expected), f"Phase 1 features: {feat}"


# ---------------------------------------------------------------------------
# linear_two — two directly connected signalized intersections (A0 <-> B0)
# ---------------------------------------------------------------------------

class TestLinearTwo:
    @pytest.fixture(scope="class")
    def g(self):
        return build_graph(NETS / "linear_two.net.xml")

    def test_two_nodes(self, g):
        assert len(g["node_ids"]) == 2

    def test_node_ids_sorted(self, g):
        assert g["node_ids"] == ["A0", "B0"]

    def test_node_to_idx(self, g):
        assert g["node_to_idx"] == {"A0": 0, "B0": 1}

    def test_four_total_edges(self, g):
        """2 flow (A0->B0, B0->A0) + 2 coord (reverse) = 4 total."""
        assert g["edge_index"].shape[1] == 4
        assert g["edge_type"].shape[0] == 4

    def test_edge_index_coord_consistency(self, g):
        assert g["edge_index"].shape[1] == g["edge_type"].shape[0]

    def test_flow_edges(self, g):
        """Both road directions between A0 and B0 become flow edges."""
        assert flow_edges(g) == {(0, 1), (1, 0)}

    def test_coord_edges(self, g):
        """Coordination edges are exactly the reverse of flow edges."""
        assert coord_edges(g) == {(1, 0), (0, 1)}

    def test_coord_is_reverse_of_flow(self, g):
        fe = flow_edges(g)
        ce = coord_edges(g)
        assert ce == {(b, a) for a, b in fe}

    def test_two_phases_per_node(self, g):
        for meta in g["node_meta"]:
            assert meta["num_phases"] == 2

    def test_phase_features_per_node(self, g):
        for node_feats in g["phase_features"]:
            assert len(node_feats) == 2


# ---------------------------------------------------------------------------
# pass_through — A0 -> M0 (non-signalized) -> B0; tests BFS path resolution
# ---------------------------------------------------------------------------

class TestPassThrough:
    @pytest.fixture(scope="class")
    def g(self):
        return build_graph(NETS / "pass_through.net.xml")

    def test_two_nodes(self, g):
        assert len(g["node_ids"]) == 2

    def test_node_ids(self, g):
        assert g["node_ids"] == ["A0", "B0"]

    def test_two_total_edges(self, g):
        """1 flow edge + 1 coord edge."""
        assert g["edge_index"].shape[1] == 2

    def test_flow_edge_a0_to_b0(self, g):
        """BFS through non-signalized M0 must find A0->B0 as a flow edge."""
        assert flow_edges(g) == {(0, 1)}

    def test_coord_edge_b0_to_a0(self, g):
        assert coord_edges(g) == {(1, 0)}

    def test_one_actionable_phase_per_node(self, g):
        """Each TL has one 'G' phase and one 'y' and one 'r'; only 'G' is actionable."""
        for meta in g["node_meta"]:
            assert meta["num_phases"] == 1

    def test_phase_features_fully_green(self, g):
        """Single phase 'G' with one incoming lane — the whole lane vector should be 1."""
        for node_feats in g["phase_features"]:
            assert len(node_feats) == 1
            assert node_feats[0].shape == (1,)
            assert node_feats[0].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# grid_3x3 — nine signalized intersections in a 3×3 bidirectional grid
# ---------------------------------------------------------------------------

class TestGrid3x3:
    @pytest.fixture(scope="class")
    def g(self):
        return build_graph(NETS / "grid_3x3.net.xml")

    def test_nine_nodes(self, g):
        assert len(g["node_ids"]) == 9

    def test_node_ids_sorted(self, g):
        assert g["node_ids"] == ["A0", "A1", "A2", "B0", "B1", "B2", "C0", "C1", "C2"]

    def test_node_to_idx(self, g):
        expected = {n: i for i, n in enumerate(["A0","A1","A2","B0","B1","B2","C0","C1","C2"])}
        assert g["node_to_idx"] == expected

    def test_48_total_edges(self, g):
        """24 flow (12 bidirectional pairs × 2) + 24 coord = 48."""
        assert g["edge_index"].shape == (2, 48)
        assert g["edge_type"].shape == (48,)

    def test_24_flow_24_coord(self, g):
        assert (g["edge_type"] == 0).sum().item() == 24
        assert (g["edge_type"] == 1).sum().item() == 24

    def test_all_adjacent_flow_edges_present(self, g):
        """Every adjacent pair of intersections has a flow edge in both directions."""
        idx = g["node_to_idx"]
        expected_flow = {
            # horizontal pairs
            (idx["A0"], idx["A1"]), (idx["A1"], idx["A0"]),
            (idx["A1"], idx["A2"]), (idx["A2"], idx["A1"]),
            (idx["B0"], idx["B1"]), (idx["B1"], idx["B0"]),
            (idx["B1"], idx["B2"]), (idx["B2"], idx["B1"]),
            (idx["C0"], idx["C1"]), (idx["C1"], idx["C0"]),
            (idx["C1"], idx["C2"]), (idx["C2"], idx["C1"]),
            # vertical pairs
            (idx["A0"], idx["B0"]), (idx["B0"], idx["A0"]),
            (idx["B0"], idx["C0"]), (idx["C0"], idx["B0"]),
            (idx["A1"], idx["B1"]), (idx["B1"], idx["A1"]),
            (idx["B1"], idx["C1"]), (idx["C1"], idx["B1"]),
            (idx["A2"], idx["B2"]), (idx["B2"], idx["A2"]),
            (idx["B2"], idx["C2"]), (idx["C2"], idx["B2"]),
        }
        assert flow_edges(g) == expected_flow

    def test_coord_is_reverse_of_flow(self, g):
        assert coord_edges(g) == {(b, a) for a, b in flow_edges(g)}

    def test_no_self_loops(self, g):
        srcs = g["edge_index"][0].tolist()
        dsts = g["edge_index"][1].tolist()
        assert all(s != d for s, d in zip(srcs, dsts))

    def test_two_phases_all_nodes(self, g):
        for meta in g["node_meta"]:
            assert meta["num_phases"] == 2

    def test_phase_features_shape_all_nodes(self, g):
        """Every node has 4 unique incoming lanes (N, S, W, E each with lane 0)."""
        for node_feats in g["phase_features"]:
            assert len(node_feats) == 2
            for feat in node_feats:
                assert feat.shape == (4,)

    def test_b1_phase0_ns_green(self, g):
        """B1 is fully interior: N=A1, S=C1, W=B0, E=B2.
        Phase 0 'GGrr': linkIndex 0 (A1_B1, N) and 1 (C1_B1, S) are green.
        Incoming lanes sorted: A1_B1(pos0), B0_B1(pos1), B2_B1(pos2), C1_B1(pos3).
        Expect vec = [1, 0, 0, 1]."""
        b1_idx = g["node_to_idx"]["B1"]
        feat = g["phase_features"][b1_idx][0]
        assert torch.allclose(feat, torch.tensor([1.0, 0.0, 0.0, 1.0])), f"B1 phase0: {feat}"

    def test_b1_phase1_ew_green(self, g):
        """Phase 2 'rrGG': linkIndex 2 (B0_B1, W) and 3 (B2_B1, E) are green.
        Expect vec = [0, 1, 1, 0]."""
        b1_idx = g["node_to_idx"]["B1"]
        feat = g["phase_features"][b1_idx][1]
        assert torch.allclose(feat, torch.tensor([0.0, 1.0, 1.0, 0.0])), f"B1 phase1: {feat}"


# ---------------------------------------------------------------------------
# Cross-network invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("net_file", [
    "cross_smoke.net.xml",
    "linear_two.net.xml",
    "pass_through.net.xml",
    "grid_3x3.net.xml",
])
def test_output_keys(net_file):
    g = build_graph(NETS / net_file)
    required = {"node_ids", "node_to_idx", "edge_index", "edge_type",
                "phase_features", "node_meta"}
    assert required.issubset(g.keys())


@pytest.mark.parametrize("net_file", [
    "cross_smoke.net.xml",
    "linear_two.net.xml",
    "pass_through.net.xml",
    "grid_3x3.net.xml",
])
def test_edge_index_and_type_length_match(net_file):
    g = build_graph(NETS / net_file)
    assert g["edge_index"].shape[1] == g["edge_type"].shape[0]


@pytest.mark.parametrize("net_file", [
    "cross_smoke.net.xml",
    "linear_two.net.xml",
    "pass_through.net.xml",
    "grid_3x3.net.xml",
])
def test_coord_edges_are_reverse_of_flow(net_file):
    g = build_graph(NETS / net_file)
    fe = flow_edges(g)
    ce = coord_edges(g)
    assert ce == {(b, a) for a, b in fe}


@pytest.mark.parametrize("net_file", [
    "cross_smoke.net.xml",
    "linear_two.net.xml",
    "pass_through.net.xml",
])
def test_node_meta_and_phase_features_count_match(net_file):
    g = build_graph(NETS / net_file)
    n = len(g["node_ids"])
    assert len(g["node_meta"]) == n
    assert len(g["phase_features"]) == n


@pytest.mark.parametrize("net_file", [
    "cross_smoke.net.xml",
    "linear_two.net.xml",
    "pass_through.net.xml",
])
def test_valid_transition_mask_shape_matches_num_phases(net_file):
    g = build_graph(NETS / net_file)
    for meta in g["node_meta"]:
        p = meta["num_phases"]
        assert meta["valid_transition_mask"].shape == (p, p)


# ---------------------------------------------------------------------------
# Cologne3 phase-count alignment — locks the graph_builder / traffic_env
# _is_actionable invariant that broke on real RESCO networks.
#
# graph_builder and traffic_env._parse_net_xml both filter phases with
# _is_actionable(). They must agree exactly, otherwise graph["node_meta"]
# and env._green_states report different num_phases for the same node,
# causing IndexError in _tick_phase_states when the trainer's action index
# exceeds len(green_states). Cologne3 has G+y mixed phases (e.g.
# 'yyggrrryyyg') that triggered this mismatch — synthetic nets do not.
# No SUMO needed: both functions are pure XML parsers.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not COLOGNE3.exists(), reason="cologne3 network not present")
def test_cologne3_num_phases_matches_green_states():
    graph = build_graph(COLOGNE3)
    green_states, _, _, _ = _parse_net_xml(COLOGNE3)

    for node_id in graph["node_ids"]:
        node_idx   = graph["node_to_idx"][node_id]
        graph_n    = graph["node_meta"][node_idx]["num_phases"]
        env_n      = len(green_states.get(node_id, []))
        assert graph_n == env_n, (
            f"Node {node_id!r}: graph_builder says {graph_n} phases, "
            f"traffic_env says {env_n}. "
            f"_is_actionable in graph_builder and traffic_env._parse_net_xml must match exactly."
        )
