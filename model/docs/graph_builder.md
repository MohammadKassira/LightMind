# graph_builder — SUMO net.xml → Typed-Edge Graph

## What this module does

`data/graph_builder.py` reads a SUMO road-network file (`net.xml`) and converts the static road topology into a Python dict that the GAT policy uses for message-passing at every decision step.

It runs **once at startup**, not on every step. It produces the fixed graph structure of the network — which intersections exist, how they are connected, and what signal phases are available at each one. It does **not** include live traffic state (queue lengths, waiting times); those are added at each environment step by `observation_encoder.py`.

---

## Where this fits in the pipeline

```
net.xml ──► graph_builder.build_graph() ──► static graph dict ──┐
                                                                  │
SUMO step ──► observation_encoder.py ──► node_features ──────────┤
                                                                  ▼
                                               models/gat_policy.py
                                               (GAT message passing)
```

| Module | Runs when | Produces |
|---|---|---|
| `graph_builder.py` | Once at startup | Fixed graph topology + phase features |
| `observation_encoder.py` | Every env step | `node_features` (live traffic state) |
| `models/gat_policy.py` | Every decision step | Action logits over phases |

---

## Background: SUMO net.xml format

A SUMO network file is an XML file describing road junctions, road segments between them, turn connections, and traffic-light programs. This module reads four element types:

### `<junction>`

```xml
<junction id="A0" type="traffic_light" x="100" y="200"/>
<junction id="M0" type="priority"      x="200" y="200"/>
```

- `type="traffic_light"` → **signalized intersection** — becomes a node in the graph.
- Any other type (`priority`, `dead_end`, `right_before_left`, …) → **non-signalized** — used only for BFS path resolution, not a graph node.

### `<edge>`

```xml
<edge id="A0_B0" from="A0" to="B0" numLanes="1" speed="13.89"/>
```

A directed road segment from one junction to another. Edges whose `id` starts with `:` or whose `function="internal"` are SUMO-internal geometry (junction entry/exit curves) and are skipped.

### `<connection>`

```xml
<connection from="A0_B0" to="B0_C0" fromLane="0" toLane="0"
            tl="B0" linkIndex="2" dir="s"/>
```

A specific turn movement through a junction. `tl="B0"` means junction B0's traffic light controls this movement. `linkIndex` is the character position in the phase state string that governs this connection. Multiple connections can share the same `linkIndex` (e.g. straight + turning movements that run simultaneously).

### `<tlLogic>` and `<phase>`

```xml
<tlLogic id="B0" type="static" programID="0">
    <phase duration="30" state="GGrr"/>
    <phase duration="5"  state="yyrr"/>
    <phase duration="30" state="rrGG"/>
    <phase duration="5"  state="rryy"/>
</tlLogic>
```

The fixed-time signal program for a junction. Each character in `state` maps to a `linkIndex`. Character meanings:

| Character | Meaning |
|---|---|
| `G` | Green (protected) |
| `g` | Green (permissive/yield) |
| `y` / `Y` | Yellow (transition) |
| `r` / `R` | Red |

Only phases containing at least one `G` or `g` are **actionable** — the RL agent chooses among them. Yellow phases are short transition intervals managed by the environment; they are filtered out and never appear as RL actions.

---

## Output schema

`build_graph(net_path)` returns a single dict with six keys:

### `node_ids` — `List[str]`

Sorted list of all signalized junction IDs in the network. The position of an ID in this list is that node's integer index everywhere else in the dict and in the model.

```python
g["node_ids"]  # → ["A0", "A1", "B0", "B1", ...]
g["node_ids"][0]  # → "A0" (node index 0)
```

### `node_to_idx` — `Dict[str, int]`

Reverse lookup from junction ID to node index.

```python
g["node_to_idx"]["B1"]  # → 4  (for example)
```

### `edge_index` — `LongTensor[2, E]`

COO-format edge list. Row 0 holds source node indices, row 1 holds destination node indices. `E = 2 × num_flow_edges` because every flow edge has a matching coordination edge.

```python
g["edge_index"]
# tensor([[0, 1, 1, 0],   ← source indices
#         [1, 0, 0, 1]])  ← destination indices
```

Flow edges are stored first, then coordination edges. This layout is directly compatible with PyTorch Geometric's `MessagePassing.propagate(edge_index, ...)`.

### `edge_type` — `LongTensor[E]`

One integer per edge, same length as `edge_index.shape[1]`:

| Value | Type | Direction | Purpose |
|---|---|---|---|
| `0` | Flow | Upstream → downstream (follows road) | Propagate queue/wait state forward |
| `1` | Coordination | Downstream → upstream (reverse of road) | Propagate congestion signals backward |

```python
(g["edge_type"] == 0).sum()  # number of flow edges
(g["edge_type"] == 1).sum()  # number of coordination edges
```

### `phase_features` — `List[List[FloatTensor]]`

Indexed as `phase_features[node_idx][phase_idx]`, each entry is a `FloatTensor` of shape `[num_incoming_lanes]`.

- **1.0** at position `i` means lane `i` gets a green signal in this phase.
- **0.0** means the lane is red in this phase.

Lane order within each vector is alphabetical on `(from_edge_id, from_lane_index)`, which makes it deterministic across runs.

```python
# Example: interior intersection with 4 incoming lanes
# Phase 0 = N/S green:  [1, 0, 0, 1]
# Phase 1 = E/W green:  [0, 1, 1, 0]
g["phase_features"][b1_idx][0]  # → tensor([1., 0., 0., 1.])
g["phase_features"][b1_idx][1]  # → tensor([0., 1., 1., 0.])
```

This is the static half of each phase's feature vector — it encodes what the phase does geometrically and never changes during simulation. The phase-scoring head requires a complete feature vector = this static green/red mask concatenated with live pressure values for those lanes, assembled at decision time by observation_encoder.py. The graph builder is only responsible for the static mask; the pressure concatenation happens at the point where the head is called.

### `node_meta` — `List[Dict]`

One dict per node (same order as `node_ids`):

| Key | Type | Description |
|---|---|---|
| `num_phases` | `int` | Number of actionable phases (yellow phases excluded) |
| `valid_transition_mask` | `BoolTensor[P, P]` | `True` everywhere — all phase-to-phase transitions are topologically valid. Min-green enforcement (you can't switch away from a phase that just started) is the environment's responsibility, not the graph's. |

```python
meta = g["node_meta"][0]
meta["num_phases"]              # → 2
meta["valid_transition_mask"]   # → tensor([[True, True], [True, True]])
```

### What is NOT in this dict

`node_features` is intentionally absent. It holds live traffic state (queue lengths, density, waiting time per lane) and is added at every environment step by `observation_encoder.py`. The graph structure returned here is static.

---

## Edge types in detail

### Flow edges (type 0)

A flow edge `(A → B)` is added whenever there is a road path from signalized junction A to signalized junction B, regardless of whether non-signalized junctions sit in between.

The GAT uses these edges to propagate queue lengths and waiting times **forward** through the network — a downstream junction can see that upstream traffic is backing up and pre-emptively give green to clear it.

### Coordination edges (type 1)

A coordination edge `(B → A)` is added for every flow edge `(A → B)`. They are exactly the reverse set.

They carry congestion signals **backwards** — a downstream junction that is saturated can signal upstream junctions to hold traffic back. These edges are stored in `edge_index` now but are not yet wired up in message-passing; that happens in `models/gat_policy.py` when the GAT ring is implemented.

**Invariant enforced by tests:**
```python
coord_edges == {(b, a) for (a, b) in flow_edges}
```

---

## BFS through non-signalized junctions

Real networks place non-signalized junctions (yield nodes, merge points) between signalized ones. A simple adjacency lookup would miss the flow edge `A → B` when the actual road path is `A → M0 (priority) → B`.

The builder runs a BFS from every signalized junction:

1. Enqueue all direct road-neighbours of the starting junction.
2. **If a neighbour is signalized:** record a flow edge `(start → neighbour)` and **stop BFS through it**. Stopping here prevents skipping over intermediate signals — if the path is `A → B → C`, we get `A→B` and `B→C`, not `A→C` directly.
3. **If a neighbour is non-signalized:** enqueue its neighbours and keep searching.

This produces exactly one flow edge per pair of adjacent signalized junctions, regardless of how many non-signalized junctions lie between them.

**Tested by `TestPassThrough`:** network `pass_through.net.xml` has `A0 → M0 (priority) → B0`. The expected result is one flow edge `A0 → B0`, not zero.

---

## Synthetic test networks

Four networks in `data/networks/` cover the key topological cases:

| File | Topology | Nodes | Flow edges | Key scenario tested |
|---|---|---|---|---|
| `cross_smoke.net.xml` | Single 4-way intersection | 1 | 0 | No edges; phase features only |
| `linear_two.net.xml` | A0 ↔ B0 direct connection | 2 | 2 | Bidirectional flow + coord |
| `pass_through.net.xml` | A0 → M0 (non-sig) → B0 | 2 | 1 | BFS through non-signalized M0 |
| `grid_3x3.net.xml` | 3×3 grid, 9 intersections | 9 | 24 | Large network; all adjacencies |

---

## Test results

**58 tests, all passing** (last run 2026-05-24, Python 3.10.8, pytest 9.0.3).

Tests live in `tests/test_graph_builder.py` and are split into:

- **Per-network class tests (40)** — each class targets one network and checks topology-specific properties: node count, edge count, exact flow/coord edge pairs, phase-feature vectors for specific junctions.
- **Cross-network invariants (18)** — five properties parametrized over all four networks. Catch regressions where a change to `build_graph()` fixes one network but silently breaks another.

To run:

```bash
pytest tests/test_graph_builder.py -v
```

### Diagnosing a failure

The failing test's class name tells you which network is broken; the test name tells you which property. For example:

- `TestGrid3x3::test_all_adjacent_flow_edges_present` → BFS is not finding all flow edges in the 3×3 grid. Check Steps 1 and 3 in `build_graph()`.
- `TestCrossSmoke::test_phase_0_ns_green` → Phase feature vector is wrong. Check Step 7 (the `vec` assembly loop).
- Any `test_coord_edges_are_reverse_of_flow` → Coordination edges are out of sync with flow edges. Check Step 4.

---

## How to extend

### Add edge weights (distance, travel time)

After Step 3 in `build_graph()`, parse `<edge length="..." speed="..."/>` and build a parallel `edge_attr` FloatTensor. Pass it to PyG as `data.edge_attr`. The GAT attention mechanism can then use distance as a prior.

### Support multiple lanes between junctions

BFS currently records at most one edge per `(src, dst)` pair. To model per-lane capacity, read `numLanes` from `<edge numLanes="N"/>` and store it as an edge attribute rather than duplicating edges.

### 2-hop neighbourhood for the GAT

The GAT's 2-hop receptive field comes from stacking 2 message-passing layers over the 1-hop edges already in the graph — do not pre-compute 2-hop edges. Adding explicit 2-hop edges to edge_index here would be wrong: it bypasses the intermediate intersection entirely, breaks the layered aggregation the GAT is designed around, and creates redundant long-range edges that distort attention. Leave edge_index as 1-hop only; the depth comes from the model, not the graph.

### Add a new synthetic network

1. Create `data/networks/<name>.net.xml` following the format in existing network files.
2. Add a `Test<Name>` class to `tests/test_graph_builder.py`.
3. Add `"<name>.net.xml"` to the parametrize lists of the five cross-network invariant tests at the bottom of the test file.

---

## Related files

| File | Role |
|---|---|
| `data/graph_builder.py` | This module — parses net.xml, builds the graph dict |
| `data/observation_encoder.py` | Adds `node_features` (live state) at each env step |
| `models/gat_policy.py` | Consumes `edge_index`, `edge_type`, `node_features` |
| `visualize_graph.py` | Renders the graph and phase-feature table as an image |
| `tests/test_graph_builder.py` | 58 tests across the four synthetic networks |
| `data/networks/` | Synthetic SUMO network files used for development and testing |
