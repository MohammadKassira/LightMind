"""Convert SUMO net.xml network structure into a typed-edge graph."""
#check: docs/graph_builder.md for design notes and assumptions.
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch


def build_graph(net_path: Path) -> dict:
    """Read a SUMO net.xml and return the static typed-edge graph structure.

    Returns:
        node_ids        List[str]               idx -> junction_id
        node_to_idx     Dict[str, int]          junction_id -> idx
        edge_index      LongTensor[2, E]        COO; E = 2 * num_flow_edges
        edge_type       LongTensor[E]           0=flow (up->down), 1=coordination (down->up)
        phase_features  List[List[Tensor]]      [N][P] each FloatTensor[num_incoming_lanes]
        node_meta       List[Dict]              num_phases, valid_transition_mask per node

    node_features and validity are absent — filled each env step by observation_encoder.
    """
    root = ET.parse(net_path).getroot()

    # --- Step 1: Parse ---

    signalized_ids: Set[str] = {tl.get("id") for tl in root.findall("tlLogic")}

    # Junction adjacency over non-internal road edges only
    adj: Dict[str, List[str]] = defaultdict(list)
    for edge in root.findall("edge"):
        if edge.get("function") == "internal" or (edge.get("id") or "").startswith(":"):
            continue
        from_j, to_j = edge.get("from"), edge.get("to")
        if from_j and to_j:
            adj[from_j].append(to_j)

    tl_phases: Dict[str, List[str]] = {
        tl.get("id"): [p.get("state", "") for p in tl.findall("phase")]
        for tl in root.findall("tlLogic")
    }

    # Connections grouped by controlling tl: (from_edge, from_lane_idx, link_index)
    tl_connections: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    for conn in root.findall("connection"):
        tl_id = conn.get("tl")
        link_idx = conn.get("linkIndex")
        if tl_id is None or link_idx is None:
            continue
        tl_connections[tl_id].append((
            conn.get("from", ""),
            int(conn.get("fromLane", "0")),
            int(link_idx),
        ))

    # --- Step 2: Node index ---

    node_ids: List[str] = sorted(signalized_ids)
    node_to_idx: Dict[str, int] = {jid: i for i, jid in enumerate(node_ids)}

    # --- Step 3: Flow edges (Type 0, upstream -> downstream) ---
    # BFS from each signalized junction through non-signalized junctions until
    # reaching the next signalized junction(s). Stop at signalized junctions so
    # we never skip over an intermediate signal.

    flow_edge_set: Set[Tuple[int, int]] = set()
    flow_edges: List[Tuple[int, int]] = []

    for start_jid in node_ids:
        start_idx = node_to_idx[start_jid]
        visited: Set[str] = {start_jid}
        queue: deque[str] = deque(adj.get(start_jid, []))
        visited.update(queue)

        while queue:
            cur = queue.popleft()
            if cur in node_to_idx:
                pair = (start_idx, node_to_idx[cur])
                if pair not in flow_edge_set:
                    flow_edge_set.add(pair)
                    flow_edges.append(pair)
                # Don't traverse further through a signalized junction
            else:
                for nxt in adj.get(cur, []):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)

    # --- Step 4: Coordination edges (Type 1, reverse of flow) ---

    coord_edges: List[Tuple[int, int]] = [(dst, src) for src, dst in flow_edges]

    # --- Step 5: Assemble edge_index and edge_type ---

    num_flow, num_coord = len(flow_edges), len(coord_edges)
    if num_flow or num_coord:
        src = [s for s, _ in flow_edges] + [s for s, _ in coord_edges]
        dst = [d for _, d in flow_edges] + [d for _, d in coord_edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor([0] * num_flow + [1] * num_coord, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)

    # --- Steps 6 & 7: node_meta and phase_features ---

    def _is_actionable(state: str) -> bool:
        # Exclude states containing 'y' — those are yellow transition phases, not stable green.
        # Must match traffic_env._parse_net_xml._is_actionable exactly so num_phases agrees.
        return any(c in ("G", "g") for c in state) and "y" not in state

    node_meta: List[Dict] = []
    phase_features: List[List[torch.Tensor]] = []

    for jid in node_ids:
        active_states = [s for s in tl_phases.get(jid, []) if _is_actionable(s)]
        p = len(active_states)

        node_meta.append({
            "num_phases": p,
            # All transitions between actionable phases are valid at the graph level.
            # Min-green enforcement is the env's responsibility.
            "valid_transition_mask": torch.ones((p, p), dtype=torch.bool),
        })

        conns = tl_connections.get(jid, [])
        incoming_lanes = sorted({(fe, fl) for fe, fl, _ in conns})
        lane_to_pos = {lane: i for i, lane in enumerate(incoming_lanes)}
        num_lanes = len(incoming_lanes)

        node_phases: List[torch.Tensor] = []
        for state in active_states:
            vec = torch.zeros(num_lanes, dtype=torch.float32)
            for from_edge, from_lane, link_idx in conns:
                if link_idx < len(state) and state[link_idx] in ("G", "g"):
                    vec[lane_to_pos[(from_edge, from_lane)]] = 1.0
            node_phases.append(vec)
        phase_features.append(node_phases)

    return {
        "node_ids": node_ids, #list of the nodes in order of their index
        "node_to_idx": node_to_idx, #opposite mapping of node_ids for easy lookup
        "edge_index": edge_index, #list "Long Tensor" of 2 lists: source and destination indices for each edge. to index: edge_index[0][i] is source node index, edge_index[1][i] is destination node index for edge i. this format is for PyG's MessagePassing API "MessagePassing.propagate()".
        "edge_type": edge_type, #list "Long Tensor" of edge types corresponding to edge_index: 0 for flow edges, 1 for coordination edges
        "phase_features": phase_features, #3d array, indexing it: phase_features[node_idx][phase_idx] gives a FloatTensor of shape [num_incoming_lanes] with 1.0 for lanes that are green in that phase and 0.0 otherwise (at node 0, phase 1, the tensor might look like [0.0, 1.0, 0.0, 1.0] meaning lanes 1 and 3 are green in that phase)
        "node_meta": node_meta, #list of dicts: node_meta[node_idx]["num_phases"] gives the number of actionable phases at that node, and node_meta[node_idx]["valid_transition_mask"] is a [num_phases x num_phases] boolean tensor where True indicates that transitioning from the row phase to the column phase is valid at the graph level (min-green enforcement is not handled here)
    }
