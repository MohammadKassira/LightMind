# GAT Policy (R4 + R5 + R6 + R7)

## What R4 builds

R4 replaces the 0-hop DQN (NodeEncoder → PhaseHead, no message passing) with a one-layer GAT
so each intersection can incorporate its neighbors' state when choosing a phase.

**Architecture after R4:**
```
obs, validity
  → NodeEncoder(obs_dim)         → [N, 64]   (embed_dim)
  → GATPolicy(in=64, heads=4×32) → [N, 128]  (4 heads × 32 per head, concat)
  → PhaseHead(embed_dim=128)     → [N, P]    (Q-scores per phase)
```

The `zero_hop=True` flag on `GATPolicy` is the sole toggle between the 0-hop ablation and the
full R4 policy. Both paths share identical weights; the only difference is whether real edges
or self-loops reach the GAT.

**Done-when:** `zero_hop=False` must beat `zero_hop=True` on mean waiting time after convergence
on a multi-intersection network. On cologne3 (3 intersections):

| | 1-hop GAT (`zero_hop=False`) | 0-hop ablation (`zero_hop=True`) |
|---|---|---|
| mean waiting time | **0.900 s** | 0.972 s |
| p95 waiting time | **3.08 s** | 3.875 s |
| mean return | **-2.646** | -2.757 |
| phase change rate | **1.52** | 1.99 |

1-hop beats 0-hop by 7.7% on mean waiting time and 20% on the p95 tail. The lower phase change
rate on 1-hop (1.52 vs 1.99) indicates more stable, coordinated switching.

---

## `GATPolicy` — `models/gat_policy.py`

```python
from models.gat_policy import GATPolicy

gat = GATPolicy(in_channels=64, num_heads=4, out_per_head=32, zero_hop=False)
# out_channels = num_heads * out_per_head = 128
out = gat(x, edge_index)   # x: [N, 64], edge_index: [2, E] → [N, 128]
```

### `zero_hop` flag

| `zero_hop` | What happens |
|---|---|
| `False` | Real edges from `graph["edge_index"]` are passed to `GATConv`. Self-loops added by `GATConv` automatically (`add_self_loops=True` default). |
| `True` | An empty `edge_index` (shape `[2, 0]`) is passed. `GATConv` infers N from `x.size(0)` and adds all N self-loops — each node attends only to itself. |

Both paths run the same weight matrix. The 0-hop ablation is not a separate model; it is the
same `GATPolicy` with `zero_hop=True`. This means every training condition (optimizer, LR,
epsilon schedule, replay buffer) is identical between the two runs.

### Self-loops

`graph_builder` produces no self-loops — `edge_index` contains only flow edges (upstream →
downstream) and coordination edges (downstream → upstream). `GATConv(add_self_loops=True)`
adds the missing self-loops at call time. No manual `add_self_loops` call is needed anywhere.

For `zero_hop=True`, the empty edge_index triggers `GATConv` to add all N self-loops from
`x.size(0)` — this is standard `GATConv` behavior, not a special case.

### R7 extension point

`GATPolicy.forward` has a comment marking where R7 will extend the signature:

```python
# R7 will extend forward to: forward(x, edge_index, node_validity=None)
# where node_validity: [N, obs_dim] masks attention to failed/stale neighbors.
```

Do not add the `node_validity` argument before R7. The comment is the reservation.

---

## `PhaseHead.forward_batch` — `models/phase_head.py`

R4 adds a batched scoring method alongside the existing `forward`:

```python
# Existing (R1, unchanged)
scores = head.forward(node_embedding, phase_features, valid_transition_mask)
# node_embedding: [embed_dim]  →  scores: [P]

# New (R4)
scores = head.forward_batch(node_embeddings, phase_features, valid_transition_mask)
# node_embeddings: [B, embed_dim]  →  scores: [B, P]
```

`forward_batch` is used in `DQNTrainer._compute_loss` to score all B batch samples for one
node in a single matrix operation instead of looping over B. `forward` is unchanged and still
used in `_select_actions`.

**Masking:** `forward_batch` accepts the same `valid_transition_mask: [P] bool` as `forward`.
It broadcasts the mask across the batch dimension using `unsqueeze(0)`.

---

## `DQNTrainer` changes (R4)

### Construction

```python
# obs_dim comes from a probe reset — pass it directly, NOT obs_dim * 2
# NodeEncoder doubles it internally: nn.Linear(obs_dim * 2, hidden_dim)
encoder = NodeEncoder(obs_dim, hidden_dim, embed_dim)           # embed_dim = 64

gat = GATPolicy(in_channels=embed_dim, num_heads=4, out_per_head=32, zero_hop=False)
# gat.out_channels = 128

# PhaseHead input = gat.out_channels (128), NOT encoder embed_dim (64)
head = PhaseHead(gat.out_channels, phase_feat_dim, head_hidden)

target_gat = copy.deepcopy(gat)   # third target network alongside target_encoder, target_head
```

**Critical:** `PhaseHead` now receives `gat.out_channels = 128`, not `embed_dim = 64`. Passing
`embed_dim` gives a 64-dim head that will silently compute nonsense scores on 128-dim GAT output.

### Optimizer

```python
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(gat.parameters()) + list(head.parameters()),
    lr=cfg.trainer.lr,
)
```

All three networks are trained jointly. `target_gat` parameters are frozen (`requires_grad_(False)`).

### Target sync

```python
# Every target_update_steps, sync all three target networks
self.target_encoder.load_state_dict(self.encoder.state_dict())
self.target_gat.load_state_dict(self.gat.state_dict())
self.target_head.load_state_dict(self.head.state_dict())
```

### `_select_actions` (vectorized)

```python
obs_stack = torch.stack([padded_obs[nid][0] for nid in node_ids])   # [N, obs_dim]
val_stack = torch.stack([padded_obs[nid][1] for nid in node_ids])   # [N, obs_dim]
emb     = encoder(obs_stack, val_stack)    # [N, embed_dim]  — one vectorized encode
gat_emb = gat(emb, edge_index)            # [N, 128]         — one GAT call
# then per-node: head(gat_emb[col], pf, mask) for ε-greedy
```

The entire graph is encoded and propagated in two calls before any action is selected.
In R2 there was a per-node loop over the encoder; that loop is gone.

### `_compute_loss` (B-loop GAT)

```python
emb_flat = encoder(obs_all.view(B*N, -1), val_all.view(B*N, -1)).view(B, N, -1)  # [B, N, 64]
gat_out  = torch.stack([gat(emb_flat[b], edge_index) for b in range(B)])          # [B, N, 128]
# R8 will replace the B-loop with PyG Batch for vectorized graph inference
```

The encoder is vectorized over B×N. The GAT loops over B because PyG's `GATConv` operates on
a single graph at a time; batching requires `torch_geometric.data.Batch`, deferred to R8.

For each node column: `head.forward_batch(gat_out[:, col, :], pf, all_true)` → `[B, P]`.

### Checkpoint format

```python
torch.save({
    "encoder":   encoder.state_dict(),
    "gat":       gat.state_dict(),        # added in R4
    "head":      head.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step":      total_steps,
    "cfg":       cfg_dict,
    "metrics":   metrics,
}, path)
```

`load_checkpoint` loads `ckpt["gat"]` into both `trainer.gat` and `trainer.target_gat` (target
starts as a copy of the online network, which is correct).

### Per-episode logging

`DQNTrainer.train()` now prints one line per episode:

```
[Ep    1/ 300]  return=  -123.45  len= 200  avg_loss=(warming up)  ε=1.000  elapsed=0m45s
[Ep    2/ 300]  return=   -98.30  len= 200  avg_loss=0.3421         ε=0.987  elapsed=1m23s
```

`avg_loss` shows the rolling average over the last 100 gradient steps. `(warming up)` appears
until the replay buffer reaches `warmup_steps` (default 1000).

---

## `configs/gat.yaml`

```yaml
model:
  hidden_dim: 128
  embed_dim: 64          # NodeEncoder output; GAT input
  head_hidden_dim: 64
  gat:
    num_heads: 4
    out_per_head: 32     # GAT output = 4*32 = 128; PhaseHead embed_dim = 128
    zero_hop: false      # true → 0-hop ablation (same weights, self-loops only)

trainer:
  lr: 0.0003             # NOT 3e-4 — PyYAML parses 3e-4 as a string without a decimal point
  ...
```

**PyYAML float gotcha:** `3e-4` without a decimal is parsed as the string `'3e-4'`. Use
`0.0003` or `3.0e-4`. This applies to all scientific-notation floats in YAML.

---

## Bug fixed in R4: `graph_builder._is_actionable`

`graph_builder._is_actionable` previously counted phases containing both `G` and `y` as
actionable:

```python
# BEFORE (bug)
def _is_actionable(state: str) -> bool:
    return any(c in ("G", "g") for c in state)

# AFTER (fix)
def _is_actionable(state: str) -> bool:
    return any(c in ("G", "g") for c in state) and "y" not in state
```

`traffic_env._parse_net_xml._is_actionable` had always excluded `y`-states. The mismatch
caused `num_phases` to disagree between the graph and the env on real networks like cologne3
(which has mixed `G`+`y` phases). The result was `IndexError: list index out of range` in
`_tick_phase_states` when the trainer tried to access a phase index that the env did not expose.

**Both functions must stay in sync.** The comment in `graph_builder._is_actionable` says:
`# Must match traffic_env._parse_net_xml._is_actionable exactly so num_phases agrees.`

Synthetic test networks (cross_smoke, grid_3x3, linear_two) have no `G`+`y` mixed phases,
so this bug was invisible in tests but fatal on real RESCO networks.

---

## Evaluation helpers (R4)

### `evaluation/eval_runner.py`

```python
from evaluation.eval_runner import evaluate, print_summary

metrics = evaluate(
    trainer,
    env,
    num_episodes=5,
    perception_severity=0.0,
    use_pressure=True,
    sentinel=-1.0,
    seed=999,        # ep_seed = seed + ep; None → random
)
print_summary("1-hop GAT", metrics)
```

Runs greedy (ε=0) episodes, collects `step_mean_waiting_time`, `step_throughput`, and
`step_num_vehicles` from `env.step()` info dict. Returns `compute_metrics(episode_records)`.

### `evaluation/metrics.py`

`compute_metrics(episode_records)` returns:

| Key | Description |
|---|---|
| `n_episodes` | Number of eval episodes |
| `mean_return` / `std_return` | Undiscounted return across episodes |
| `mean_ep_length` | Average steps per episode |
| `mean_waiting_time` | Mean per-step average waiting time (s) |
| `p95_waiting_time` | 95th-percentile per-step waiting time |
| `max_waiting_time` | Maximum per-step waiting time |
| `mean_throughput_per_step` | Mean arrived vehicles per step |
| `mean_vehicles_in_net` / `p95_vehicles_in_net` | Network load |
| `mean_phase_change_rate` | Avg phase changes per step across nodes |

Waiting-time fields are `NaN` when using `MockEnv` (which has no real vehicles).

### Vehicle metrics in `TrafficEnv.step()`

`env.step()` now returns an `info` dict:

```python
info = {
    "sim_time":               float,  # current SUMO simulation time (s)
    "step_mean_waiting_time": float,  # mean waiting time over all vehicles in net
    "step_num_vehicles":      int,    # active vehicles
    "step_throughput":        int,    # vehicles that completed their trip this step
}
```

`step_mean_waiting_time` is the mean of `vehicle.getWaitingTime(v)` over all active vehicles,
or `0.0` if the network is empty.

---

## Training R4 (via `train.py`)

`train_r4.py` is deleted — use the unified script:

```bash
python train.py --config configs/gat.yaml                  # 1-hop GAT, cologne3, 300 ep
python train.py --config configs/gat.yaml --zero-hop       # 0-hop ablation
python train.py --config configs/gat.yaml --episodes 300 --eval-episodes 5
```

---

## R5: Two-Layer GAT (2-Hop Receptive Field)

### What R5 builds

R5 stacks a second `GATConv` on top of R4's single layer, extending each node's receptive
field from 1-hop (direct neighbors) to 2-hop (neighbors' neighbors).

**Architecture after R5:**
```
obs, validity
  → NodeEncoder(obs_dim)                    → [N, 64]   (embed_dim, unchanged)
  → GATPolicy layer 1 (4 heads × 32) + ELU → [N, 128]
  → GATPolicy layer 2 (4 heads × 16)        → [N, 64]
  → PhaseHead(embed_dim=64)                 → [N, P]    (embed_dim: 128 → 64 vs R4)
```

`PhaseHead` input dimension changes from 128 (R4) to 64 (R5) — automatically handled via
`gat.out_channels`. `NodeEncoder` and `PhaseHead` internals are otherwise unchanged.

**ELU placement:** Applied after layer 1, not after layer 2. The final layer output feeds
directly into `PhaseHead`'s scorer — no non-linearity at the output, consistent with R4.

**Done-when:** 2-layer GAT beats 1-layer GAT on mean waiting time after convergence on
cologne3 under identical training conditions. Training to be run once R5 is merged.

### `num_layers` parameter

```python
# R4 behavior (default, backward compatible)
gat = GATPolicy(in_channels=64, num_heads=4, out_per_head=32, num_layers=1)
# gat.out_channels = 128; state dict keys: gat.xxx (identical to R4)

# R5 two-layer
gat = GATPolicy(in_channels=64, num_heads=4, out_per_head=32,
                num_layers=2, l2_out_per_head=16)
# gat.out_channels = 64; state dict keys: gat.xxx + gat2.xxx
```

The first layer attribute is `self.gat` in both cases — R4 checkpoint state dict keys are
preserved. `self.gat2` only exists when `num_layers=2`, so `num_layers=1` state dicts are
identical to R4 and can be loaded by R4-era code without modification.

`zero_hop` still works with 2 layers: both layers receive an empty `edge_index`, so each node
attends only to itself at both hops — the 0-hop ablation remains valid.

### `configs/r5.yaml`

```yaml
model:
  hidden_dim: 128
  embed_dim: 64
  head_hidden_dim: 64
  gat:
    num_heads: 4
    out_per_head: 32       # Layer 1: 4*32 = 128
    num_layers: 2
    l2_out_per_head: 16    # Layer 2: 4*16 = 64 → PhaseHead input
    zero_hop: false

trainer:
  lr: 0.0003
  ...
  checkpoint_dir: "checkpoints/r5_2hop"
```

### Training R5 (via `train.py`)

`train_r5.py` is deleted — use the unified script:

```bash
python train.py --config configs/r5.yaml                    # 2-hop, checkpoints/r5_2hop
python train.py --config configs/r5.yaml --one-layer        # 1-hop rerun, checkpoints/r5_1hop
python train.py --config configs/r5.yaml --episodes 300 --eval-episodes 5
```

`--one-layer` sets `num_layers=1` for a direct comparison under identical R5 training
conditions. Compare `r5_2hop/eval_metrics.json` vs `r5_1hop/eval_metrics.json`.

### Over-smoothing guard

`test_embeddings_not_collapsed` in `tests/test_gat_policy.py` checks that after a 2-layer
forward pass on N=4 nodes with distinct inputs, all pairwise cosine similarities are < 0.999.
This catches the pathology where all nodes collapse to the same embedding vector after repeated
aggregation — which would make the GAT useless regardless of whether edges are present.

### `DQNTrainer` changes (R5)

Two new config keys read in `__init__`, passed to `GATPolicy`:

```python
gat_num_layers = _cfg(cfg, "model.gat.num_layers",       1)   # default 1 = R4 behavior
gat_l2_out_ph  = _cfg(cfg, "model.gat.l2_out_per_head", 16)
```

No other trainer code changes — `_select_actions`, `_compute_loss`, `save_checkpoint`, and
`load_checkpoint` all use `gat.out_channels` and `self.head` which are constructed with the
right dimensions at init time.

---

## R6: Typed Edges (Separate Weight Matrices per Edge Type)

### What R6 builds

R5 passes a single mixed `edge_index` to one `GATConv`. Every edge — whether a flow edge
(upstream → downstream, type 0) or a coordination edge (downstream → upstream, type 1) —
gets the same weight matrix. R6 gives each edge type its own `GATConv` so the model can
learn distinct attention patterns for traffic flow vs downstream coordination.

**Architecture after R6:**
```
obs, validity
  → NodeEncoder(obs_dim)                            → [N, 64]
  → Layer 1: gat_flow(x,  ei_flow)  + ELU          → [N, 128]   (flow edges + self-loops)
           + gat_coord(x, ei_coord)                 → [N, 128]   (coord edges, no self-loops)
           = sum → ELU                              → [N, 128]
  → Layer 2: gat2_flow(x,  ei_flow)                → [N, 64]
           + gat2_coord(x, ei_coord)                → [N, 64]
           = sum                                    → [N, 64]
  → PhaseHead(embed_dim=64)                         → [N, P]     (unchanged from R5)
```

The outputs of the two streams are **summed**, not concatenated — `out_channels` stays the
same (64 for 2-layer, 128 for 1-layer). `PhaseHead` is unchanged.

**Done-when (R6):** All 10 new `TestTypedEdgesGATPolicy` tests pass; 344 total tests pass;
10-episode smoke run produces finite loss. Full typed-vs-untyped performance comparison is
**R9 work**.

---

### `typed_edges` flag

```python
# R4/R5 behavior (default, backward compatible)
gat = GATPolicy(in_channels=64, num_heads=4, out_per_head=32,
                num_layers=2, l2_out_per_head=16)
# state dict keys: gat.xxx, gat2.xxx (same as R5)

# R6 typed
gat = GATPolicy(in_channels=64, num_heads=4, out_per_head=32,
                num_layers=2, l2_out_per_head=16, typed_edges=True)
# state dict keys: gat_flow.xxx, gat_coord.xxx, gat2_flow.xxx, gat2_coord.xxx
```

`typed_edges=False` (default) keeps `self.gat` and `self.gat2` — R4/R5 checkpoints load
without key remapping. `typed_edges=True` creates `self.gat_flow`, `self.gat_coord`
(and `gat2_flow`, `gat2_coord` when `num_layers=2`); `self.gat` does not exist in the
typed model. Loading a typed checkpoint into an untyped trainer fails intentionally —
they are different models.

---

### `forward` signature (R6)

```python
out = gat(x, edge_index, edge_type)
# x:          [N, in_channels]
# edge_index: [2, E]           — from graph["edge_index"]
# edge_type:  [E]              — from graph["edge_type"]; 0=flow, 1=coord
# returns:    [N, out_channels]
```

When `typed_edges=False`, `edge_type` is **silently ignored** — passing it is safe and
requires no conditional logic in callers. The trainer always passes `edge_type` regardless
of the `typed_edges` setting:

```python
# trainer._select_actions and _compute_loss both do this:
edge_index = graph["edge_index"].to(self.device)
edge_type  = graph["edge_type"].to(self.device)
gat_emb = self.gat(emb, edge_index, edge_type)
```

**R7 will extend this signature** by adding a fourth argument:

```python
# R7 extension (reservation — do not add before R7):
# forward(self, x, edge_index, edge_type=None, node_validity=None)
# where node_validity: [N, obs_dim] suppresses attention to failed/stale neighbors.
```

The `edge_type` parameter was added in R6 specifically to keep this slot free for R7
without breaking existing callers.

---

### Self-loop asymmetry (deliberate design choice)

| Stream | `add_self_loops` | Handles |
|---|---|---|
| `gat_flow` | `True` | upstream flow neighbors + self-loops |
| `gat_coord` | `False` | downstream coordination neighbors only |

Self-loops are part of the flow stream. This means they appear exactly once across both
streams. **Do not change this.** If both streams had `add_self_loops=True`, each node
would attend to itself twice (double-counted self-signal). If neither had self-loops,
isolated nodes would have no signal at all.

This also defines zero_hop behavior under typed edges cleanly:

| `zero_hop=True` | What happens |
|---|---|
| `gat_flow(x, empty_ei)` | GATConv adds N self-loops → each node attends only to itself |
| `gat_coord(x, empty_ei)` | No edges, no self-loops → output = bias vector (same for all nodes, x-independent) |
| Sum | Pure self-loop attention, same semantics as R4/R5 `zero_hop` |

The `gat_coord(x, empty_ei)` x-independence is tested in `test_typed_zero_hop_shape`:
calling `gat_coord` with two different inputs `x1` and `x2` on an empty edge_index must
produce identical outputs. This is the reliable way to assert "no coord information flows"
without depending on whether bias happens to be zero.

---

### `edge_type` in the graph

`graph["edge_type"]` is a `LongTensor[E]` that has existed since R0:

```python
# data/graph_builder.py — already computed before R6:
edge_type = torch.tensor([0] * num_flow + [1] * num_coord, dtype=torch.long)
# flow edges always first in edge_index; coord edges immediately after
```

- Type 0 (flow): upstream → downstream. Corresponds to traffic direction — a car traveling
  from junction A to junction B makes A upstream of B.
- Type 1 (coord): downstream → upstream. The reverse — B signals back to A. Enables
  anticipatory coordination (downstream queue state informs upstream phase choice).

`graph_builder` guarantees edge order: all flow edges first, then all coord edges. So
`edge_index[:, edge_type == 0]` and `edge_index[:, edge_type == 1]` are always valid
index operations.

---

### `DQNTrainer` changes (R6)

One new config key in `__init__`:

```python
gat_typed = _cfg(cfg, "model.gat.typed_edges", False)
self.gat   = GATPolicy(..., typed_edges=gat_typed)
```

`_select_actions` and `_compute_loss` both extract and pass `edge_type` unconditionally:

```python
edge_type = graph["edge_type"].to(self.device)
# _select_actions:
gat_emb = self.gat(emb, edge_index, edge_type)
# _compute_loss:
gat_out      = torch.stack([self.gat(emb_flat[b],        edge_index, edge_type) for b in range(B)])
next_gat_out = torch.stack([self.target_gat(next_emb[b], edge_index, edge_type) for b in range(B)])
```

When R8 replaces the B-loop with PyG Batch, `edge_type` will need to be included in the
batched graph alongside `edge_index` — both tensors have E entries and must stay aligned.

---

### `configs/r6.yaml`

```yaml
model:
  gat:
    num_heads: 4
    out_per_head: 32       # Layer 1: 4*32 = 128
    num_layers: 2
    l2_out_per_head: 16    # Layer 2: 4*16 = 64
    typed_edges: true      # R6: separate weight matrices per edge type
    zero_hop: false

trainer:
  checkpoint_dir: "checkpoints/r6_typed"
  ...
```

---

### `train.py` — unified training script (R6 replaces train_r4.py + train_r5.py)

`train_r4.py` and `train_r5.py` are deleted. All rings from R6 forward use:

```bash
python train.py --config configs/r6.yaml                      # R6 typed edges
python train.py --config configs/r6.yaml --untyped            # override typed_edges=False
python train.py --config configs/r4.yaml --zero-hop           # 0-hop ablation
python train.py --config configs/r5.yaml --one-layer          # 1-hop (R5 comparison)
python train.py --config configs/r6.yaml --episodes 10        # smoke run
```

CLI flags and the config keys they override:

| Flag | Config key overridden |
|---|---|
| `--untyped` | `model.gat.typed_edges = False` |
| `--zero-hop` | `model.gat.zero_hop = True` |
| `--one-layer` | `model.gat.num_layers = 1` |
| `--episodes N` | `trainer.num_episodes = N` |
| `--eval-episodes N` | greedy eval episodes after training |

The script: load YAML → apply CLI overrides → build env → build `DQNTrainer` → train →
save checkpoint + metrics → run eval. No ring-specific logic in the script.

---

See [R7: Neighbor Masking](#r7-neighbor-masking-failedstale-node-suppression) for the
extension built on top of this architecture.

---

## R7: Neighbor Masking (Failed/Stale Node Suppression)

### What R7 builds

R6 has a robustness gap: if neighbor node j has degraded sensors (validity all zeros or
mostly zeros), its NodeEncoder embedding carries noise — but message passing still routes
j's bad state into node i's Q-values. R7 closes this gap by suppressing edges whose
**source is a failed node** before GATConv ever sees them. Bad neighbors cannot poison
their neighbors through message passing.

**No new parameters.** Masking is pure runtime edge filtering. A node with
`neighbor_masking: false` (R6 config) behaves identically to R6 — zero overhead, no code
path change.

**Done-when:**
1. All 10 new `TestNeighborMaskingGATPolicy` tests pass
2. `pytest tests/ -v` — all 354 tests pass (344 existing + 10 new)
3. 10-episode smoke run `python train.py --config configs/r7.yaml --episodes 10` produces
   finite loss, no NaN
4. `test_failed_neighbor_has_no_effect` passes — the core correctness proof
5. `test_all_valid_same_as_no_masking` passes — zero overhead with healthy sensors

---

### Architecture

**Architecture after R7 (same shape as R6):**
```
obs, validity
  → NodeEncoder(obs_dim)                             → [N, 64]
  → node_valid = validity.mean(dim=-1) >= 0.75       → [N] bool
  → filter edge_index (remove edges from failed src) → [2, E'] where E' ≤ E
  → Layer 1: gat_flow(x, ei_flow) + gat_coord(x, ei_coord)  → [N, 128]
  → ELU
  → Layer 2: gat2_flow(x, ei_flow) + gat2_coord(x, ei_coord) → [N, 64]
  → PhaseHead(embed_dim=64)                          → [N, P]
```

Output shape is unchanged. The only difference is that edge_index is filtered to `E'`
edges before GATConv; filtered edges carry no information.

---

### Failed node definition

```python
def _node_valid_from_validity(node_validity: Tensor) -> Tensor:
    """[N, obs_dim] → [N] bool: True if >= 75% of sensors in the node are valid."""
    return node_validity.mean(dim=-1) >= 0.75
```

A node is **failed** when fewer than 75% of its sensors report valid readings
(validity < 1.0). At `severity=0.1` (the R7 config value), each sensor fails
independently with probability 0.1, so on cologne3's ~17-feature nodes, roughly 2% of
nodes will fall below the 0.75 threshold per step.

**Why not `validity.sum() > 0` (all-fail)?** At severity=0.1 with n=17 sensors,
P(all fail) = 0.1^17 ≈ 1e-17. Masking would never fire during training. The 0.75
threshold fires when >25% of sensors fail, which happens at a meaningful frequency and
provides useful gradient signal.

**Why not a higher threshold like 0.9?** At 0.9 the masking rate would be ~52% per node
per step at severity=0.1 — too aggressive; most nodes would lose their outgoing edges
even with minor sensor noise. The 0.75 threshold (~2% masking rate) is the right balance
between robustness and connectivity preservation.

**Test coverage:** The low fire rate (~2%) at severity=0.1 means stochastic failures
alone give poor code-path coverage during training. The test suite uses synthetic inputs
with **forced** failures (validity=all-zeros for specific nodes) to exercise the masking
path unconditionally regardless of the random failure rate.

---

### Source-side masking — the critical invariant

```python
def _mask_edge_index(
    edge_index: Tensor,
    edge_type: Tensor | None,
    node_valid: Tensor,
) -> tuple[Tensor, Tensor | None]:
    """Remove edges whose source (edge_index[0]) is a failed node.

    Failed nodes are sources of suppression, not sinks: only outgoing edges from
    failed nodes are removed. Incoming edges to a failed node from valid neighbors
    survive — this is the graph-based imputation path.
    """
    if edge_index.shape[1] == 0:
        return edge_index, edge_type
    src  = edge_index[0]
    keep = node_valid[src]
    return edge_index[:, keep], (edge_type[keep] if edge_type is not None else None)
```

**Critical invariant — failed nodes are sources of suppression, not sinks.**

| Edge | What happens |
|---|---|
| j → i (j failed) | Removed. j cannot poison i through message passing. |
| k → j (k valid, j failed) | **Kept.** Valid neighbors can still inform j. |
| j → j self-loop | Never in user's `edge_index` — GATConv adds it internally. Preserved. |

This is the **graph-based imputation path**: even when j's own sensors are dead, valid
neighbors carry information about j's intersection. Removing j's incoming edges would
break this path. Do not accidentally implement masking as mutual isolation
(`edge_index[1] != j`) — that removes incoming edges to j, which is the wrong direction
and breaks imputation.

**`edge_index[0]` is the source**: confirmed from `data/graph_builder.py` line 100:
`edge_index = torch.tensor([src, dst], dtype=torch.long)`. The first row is always the
source.

---

### Masking before the typed split

The masking block runs at the **top** of `forward`, before any branch:

```python
def forward(
    self,
    x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor = None,
    node_validity: Tensor = None,
) -> Tensor:
    if node_validity is not None and not self.zero_hop:
        node_valid            = _node_valid_from_validity(node_validity)
        edge_index, edge_type = _mask_edge_index(edge_index, edge_type, node_valid)

    if not self.typed_edges:
        # R4/R5 untyped path — sees already-filtered edge_index
        ...

    # R6 typed path — flow/coord split on already-filtered edge_index
    ei_flow  = edge_index[:, edge_type == 0]
    ei_coord = edge_index[:, edge_type == 1]
    ...
```

Because masking happens before the `edge_type == 0` / `== 1` split, **both flow and
coord streams are filtered consistently from the same `node_valid` mask**. This avoids
the bug where the flow stream suppresses j's edges but the coord stream still lets j
send coord signals to its upstream neighbors.

`_mask_edge_index` filters `edge_type` in lockstep with `edge_index`, preserving the
one-to-one correspondence between edges and their types.

---

### zero_hop interaction

```python
if node_validity is not None and not self.zero_hop:   # ← note: and not self.zero_hop
    ...
```

When `zero_hop=True`, there are no external edges to filter (GATConv receives an empty
`edge_index`). The guard skips the masking block entirely. The test
`test_masking_does_not_affect_zero_hop` verifies this: passing `node_validity` with
failures when `zero_hop=True` produces identical output to not passing it.

---

### Self-loops — preserved by design

GATConv adds self-loops **internally at runtime** via `add_self_loops=True`. They never
appear in the user's `edge_index`. Since `_mask_edge_index` only filters entries in
`edge_index`, self-loops are never removed.

A failed node i still attends to itself (its own degraded embedding). This is correct:
the failed node should incorporate whatever signal it has, even if noisy. Removing the
self-loop would produce a fully input-independent output for isolated failed nodes, which
would propagate a constant bias into the Q-value and destabilize training.

---

### `DQNTrainer` changes (R7)

#### `__init__`

One new config key:

```python
self._neighbor_masking = _cfg(cfg, "model.gat.neighbor_masking", False)
```

Default `False` → identical to R6 behavior; `None` always passed to `gat.forward`.

#### `_select_actions`

```python
node_validity = val_stack if self._neighbor_masking else None
gat_emb       = self.gat(emb, edge_index, edge_type, node_validity)
```

`val_stack` is `[N, obs_dim]` — already computed for the encoder call on the same step.
No extra data collection needed.

#### `_compute_loss`

```python
val_all      = batch["validity"]       # [B, N, obs_dim]
next_val_all = batch["next_val"]       # [B, N, obs_dim]

gat_out = torch.stack([
    self.gat(
        emb_flat[b], edge_index, edge_type,
        val_all[b] if self._neighbor_masking else None,
    ) for b in range(B)
])
next_gat_out = torch.stack([
    self.target_gat(
        next_emb_flat[b], edge_index, edge_type,
        next_val_all[b] if self._neighbor_masking else None,
    ) for b in range(B)
])
```

`val_all[b]` is `[N, obs_dim]` — the per-node validity slice for one batch sample. The
target network uses `next_val_all[b]` from the next-state observation.

**No checkpoint format changes.** `neighbor_masking` is a runtime flag, not a parameter
— no new state dict keys. R4/R5/R6 checkpoints load unchanged.

---

### `configs/r7.yaml`

```yaml
model:
  hidden_dim: 128
  embed_dim: 64          # NodeEncoder output; GAT layer-1 input
  head_hidden_dim: 64
  gat:
    num_heads: 4
    out_per_head: 32       # Layer 1: 4*32 = 128
    num_layers: 2
    l2_out_per_head: 16    # Layer 2: 4*16 = 64 → PhaseHead input
    typed_edges: true      # keep R6 typed edges
    zero_hop: false
    neighbor_masking: true # R7: suppress attention from failed neighbors

trainer:
  lr: 0.0003
  replay_buffer_size: 50000
  batch_size: 64
  warmup_steps: 1000
  gamma: 0.99
  grad_clip: 10.0
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_steps: 20000
  target_update_steps: 1000
  num_episodes: 300
  checkpoint_every: 100
  checkpoint_dir: "checkpoints/r7_masked"

reward:
  use_pressure: true

perception:
  severity: 0.1        # R7: 10% per-sensor failure rate; tests use synthetic forced
                       # failures for full code-path coverage
  sentinel_value: -1.0

seed: 1
```

---

### `train.py` — R7 additions

One new CLI flag:

```python
parser.add_argument("--no-masking", action="store_true",
                    help="Override model.gat.neighbor_masking=False")
```

Applied before trainer construction:

```python
if args.no_masking:
    cfg["model"]["gat"]["neighbor_masking"] = False
```

Full flag table after R7:

| Flag | Config key overridden |
|---|---|
| `--untyped` | `model.gat.typed_edges = False` |
| `--zero-hop` | `model.gat.zero_hop = True` |
| `--one-layer` | `model.gat.num_layers = 1` |
| `--no-masking` | `model.gat.neighbor_masking = False` |
| `--episodes N` | `trainer.num_episodes = N` |
| `--eval-episodes N` | greedy eval episodes after training |

---

### Tests — `TestNeighborMaskingGATPolicy` (10 tests)

Test infrastructure: small `GATPolicy(num_heads=2, out_per_head=8, num_layers=2,
l2_out_per_head=4, typed_edges=True)`. `TRAINER_CFG_R7` mirrors `TRAINER_CFG_R6` with
`neighbor_masking: true` and `perception.severity: 0.1`.

| Test | What it verifies |
|---|---|
| `test_masking_output_shape` | `node_validity` with some zeros → output still `[N, out_channels]`, no crash |
| `test_failed_neighbor_has_no_effect` | Mark node j fully failed (validity=0); masked output == output with `edge_index[:, edge_index[0] != j]`. **Setup invariant:** "manually removed" uses `edge_index[0] != j` (source), NOT `[1] != j` (destination — that would be wrong). Proves masking is source-edge removal. |
| `test_all_valid_same_as_no_masking` | All validity=1 → masked output equals unmasked output exactly. Zero overhead when no sensors fail. |
| `test_all_nodes_marked_failed_by_threshold` | Validity=all-zeros for every node → `mean(0.0) = 0.0 < 0.75` → all nodes failed → all external edges removed. Output has correct shape, no NaN (self-loops via `gat_flow` survive internally). Confirms the threshold handles the all-zero corner case. |
| `test_masking_with_typed_edges` | `typed_edges=True` + node j failed → output for node i differs from unmasked run. Both flow and coord streams affected — masking applies before the typed split. |
| `test_masking_does_not_affect_zero_hop` | `zero_hop=True` + failures in node_validity → output equals `zero_hop=True` without masking. No-op; the `and not self.zero_hop` guard works. |
| `test_trainer_select_actions_with_masking` | `DQNTrainer` with `neighbor_masking=True` and `perception_severity=0.1` → `_select_actions` returns valid phase indices, no crash |
| `test_trainer_compute_loss_with_masking` | `_compute_loss` with masking + degraded observations (severity=0.1) → finite scalar loss, no NaN |
| `test_partial_failure_changes_output` | Some neighbors valid, some failed → masked output ≠ unmasked output. Confirms masking actually changes results when failures exist (not silently skipped). |
| `test_masking_checkpoint_roundtrip` | Save trainer with `neighbor_masking=True`; reload; pass same `node_validity` → same predictions. |

---

### Running R7

```bash
# New tests only
pytest tests/test_gat_policy.py -v -k TestNeighborMasking

# Full suite
pytest tests/ -v     # 354 tests expected

# Smoke run under sensor failure
python train.py --config configs/r7.yaml --episodes 10 --eval-episodes 0

# Backward-compat check (R6 unmodified)
python train.py --config configs/r6.yaml --episodes 10 --eval-episodes 0

# Masking-off comparison run (same arch, no masking)
python train.py --config configs/r7.yaml --no-masking --episodes 300
```

---

### R8 forward pointer

R8 replaces the B-loop with `torch_geometric.data.Batch` for vectorized graph inference.
R7 introduces two R8-relevant constraints:

**Pre-filter before batching.** The current B-loop calls `_mask_edge_index` once per
batch sample per forward pass — 64 small tensor allocations per gradient step (at B=64).
This adds GC pressure but is not a correctness issue. R8 should call `_mask_edge_index`
once per graph **before** `Batch.from_data_list`, not inside the batched forward. Slicing
into a concatenated batched `edge_index` to apply per-graph masks requires un-batching
logic, which is slower and more complex than pre-filtering individual graphs.

**`node_validity` must be batched alongside `edge_index`.** In the R8 batched path,
`val_all[b]` and `next_val_all[b]` are per-graph tensors. They should be included in the
`torch_geometric.data.Data` objects before `Batch.from_data_list` so the mask is
available during the batched forward. Alternatively, pre-filter per-graph before batching
so the batched `edge_index` already has failed edges removed — in that case
`node_validity` does not need to travel through the batch at all.

**Recommended R8 strategy:** for each graph in the batch, call `_mask_edge_index`
immediately after encoding (before building the `Batch`) and store the filtered
`edge_index` + `edge_type` into a `Data` object. `Batch.from_data_list` then assembles
the pre-filtered graphs. This approach:
- Eliminates per-step tensor allocations inside the batched forward
- Keeps `_mask_edge_index` as-is (no changes needed)
- Avoids the complexity of slicing batched edge tensors by graph membership
