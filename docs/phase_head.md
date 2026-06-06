# R1 — phase_head — FRAP Phase-Scoring Head

## What this module does

`models/phase_head.py` takes a node embedding produced by `NodeEncoder` and scores every candidate phase at that intersection, returning one Q-value per phase. The phase with the highest score becomes the chosen action.

Instead of emitting Q-values over a fixed array of action indices — where "action 2" means a different physical movement at every intersection — it scores each phase by its **feature vector**: which lanes go green, and how much pressure those lanes carry. The same learned scorer generalises across intersections with different phase configurations and across networks of any topology. This design is FRAP-style (from the FRAP paper on phase-feature-based action representations) and is what makes one model weight set work across all intersections in the network.

The phase score **is** the Q-value in the DQN loop (R2). There is no separate value head; the phase head directly produces the target for the Bellman update.

---

## Where this fits in the pipeline

```
NodeEncoder.forward(obs, validity)
        │
        │  node_embedding (embed_dim,)
        ▼
PhaseHead.forward(node_embedding, phase_features, valid_transition_mask)
        │
        │  scores (num_phases,)  ← -inf on invalid phases
        ▼
argmax / ε-greedy  ──►  action (phase index)
        │
        ▼
env.step(actions)
```

| Module | Role |
|---|---|
| `models/phase_head.py` | This module — (embedding, phase features, mask) → Q-values per phase |
| `models/node_encoder.py` | Upstream — produces `node_embedding` from `(obs, validity)` |
| `data/graph_builder.py` | Source of `phase_features` and `valid_transition_mask` via the graph dict |
| `training/trainer.py` | Calls `forward` to get Q-values; calls `select_action` during rollout (R2) |
| `models/gat_policy.py` | Wraps `NodeEncoder` + `PhaseHead` with GAT message-passing in between (R4+) |

---

## Background: FRAP-style phase scoring

Classical DQN outputs a Q-value for each action index: `Q(s) ∈ R^|A|`. This is fine for a single fixed intersection, but breaks for a shared-parameter multi-intersection model because "action 2" is N/S green at one junction and E/W green at another.

FRAP resolves this by treating each phase as an input, not an output slot. The model learns a function `f(node_embedding, phase_feature_vector) → scalar`. At decision time, it is called once per candidate phase and the scores are stacked into a vector before argmax. The phase feature vector encodes what the phase *does* — which lanes get a green signal — so the model scores phase *semantics* rather than phase *indices*.

In this codebase, `phase_feature_vector = graph["phase_features"][node_idx][phase_idx]`, a `FloatTensor[num_incoming_lanes]` where `1.0` means that lane gets a green signal in this phase and `0.0` means red (from `graph_builder.py`). The same MLP scores phases at any intersection because it always sees `(embedding, green-mask)` regardless of how many phases or lanes the intersection has.

---

## API

### `pad_phase_features(graph) -> (phase_feat_dim, padded_phase_features)`

Call this **before** constructing `PhaseHead`. It scans every node's phase feature vectors, finds the longest one, and pads all shorter vectors to that length. The padding value is `0.0`, which means "no green on this lane" — the same convention as a red signal in §3.4, so no spurious activations are introduced.

| Argument | Type | Description |
|---|---|---|
| `graph` | `dict` | Graph dict from `graph_builder.build_graph()` or `env.reset()` — must contain `"phase_features"` |

**Returns:**

| Value | Type | Description |
|---|---|---|
| `phase_feat_dim` | `int` | The padded length — max `num_incoming_lanes` across all nodes |
| `padded_phase_features` | `list[list[Tensor]]` | Same `[node_idx][phase_idx]` indexing as `graph["phase_features"]`; all feature tensors are now length `phase_feat_dim` |

```python
from models.phase_head import PhaseHead, pad_phase_features
from env.mock_env import MockEnv

env = MockEnv("linear_two")
obs_dict, graph = env.reset(seed=0)

phase_feat_dim, padded_phase_feats = pad_phase_features(graph)
head = PhaseHead(embed_dim=64, phase_feat_dim=phase_feat_dim)

# All nodes can now use the same head:
for node_id in graph["node_ids"]:
    node_idx = graph["node_to_idx"][node_id]
    scores = head(embedding, padded_phase_feats[node_idx], mask)
```

`phase_feat_dim` (= number of incoming lanes) varies per intersection within a network. Without padding, you would need one `PhaseHead` instance per distinct lane count — creating an incompatible weight set per intersection and breaking shared-parameter training. Padding resolves this the same way `pad_obs_dict` handles variable obs lengths: one fixed input size, one shared set of weights.

The original `graph` dict is not mutated. `padded_phase_features` is a new list; nodes whose raw `phase_feat_dim` is already at the max are returned by reference (no copy).

---

### `PhaseHead(embed_dim, phase_feat_dim, hidden_dim=64)`

Constructs the scoring MLP. Both `embed_dim` and `phase_feat_dim` must be known at construction time — they determine the first `Linear` layer's input size.

| Argument | Type | Description |
|---|---|---|
| `embed_dim` | `int` | Size of the node embedding — must match `NodeEncoder`'s `embed_dim` (default 64) |
| `phase_feat_dim` | `int` | **Padded** max incoming-lane count across all nodes — the value returned by `pad_phase_features` |
| `hidden_dim` | `int` | Width of the hidden layer. Default 64. |

```python
phase_feat_dim, padded_phase_feats = pad_phase_features(graph)
head = PhaseHead(embed_dim=64, phase_feat_dim=phase_feat_dim)
```

`phase_feat_dim` comes from `pad_phase_features` at runtime — never hardcode it (§1.1). One `PhaseHead` instance covers all nodes in the network because every phase feature vector has been padded to the same length.

---

### `PhaseHead.forward(node_embedding, phase_features, valid_transition_mask) -> Tensor`

Scores all candidate phases for one node.

| Argument | Type | Description |
|---|---|---|
| `node_embedding` | `FloatTensor (embed_dim,)` | Output of `NodeEncoder.forward` for this node |
| `phase_features` | `list[FloatTensor]` | `padded_phase_feats[node_idx]` — one `(phase_feat_dim,)` padded tensor per candidate phase |
| `valid_transition_mask` | `BoolTensor (num_phases,)` | `True` = this phase is a legal choice right now; `False` → score set to `-inf` |

**Returns:**

| Value | Type | Description |
|---|---|---|
| scores | `FloatTensor (num_phases,)` | Q-value for each phase; invalid phases = `-inf` |

```python
# phase_feat_dim and padded_phase_feats come from pad_phase_features(graph)
num_phases = graph["node_meta"][node_idx]["num_phases"]

# Get the current phase from the observation (phase_onehot is the first segment, §3.2)
current_phase = obs[:num_phases].argmax().item()
mask = graph["node_meta"][node_idx]["valid_transition_mask"][current_phase]

scores = head(embedding, padded_phase_feats[node_idx], mask)
print(scores)   # e.g. tensor([0.42, -inf])  — phase 1 was masked
```

Invalid phases receive `-inf` before the scores are returned. This means `softmax(scores)` assigns them probability 0 and `argmax(scores)` never selects them, without any special casing in the caller.

---

### `PhaseHead.select_action(node_embedding, phase_features, valid_transition_mask) -> int`

Convenience wrapper: runs `forward` under `torch.no_grad()` and returns `argmax` as a plain Python `int`.

```python
action = head.select_action(embedding, phase_feats, mask)
# action is always a valid phase (mask[action] is True)
```

Used during rollout in the DQN training loop (R2). For ε-greedy exploration, the caller replaces this with a random valid phase with probability ε; this method is the greedy branch.

---

## valid_transition_mask — how to index it correctly

`graph["node_meta"][node_idx]["valid_transition_mask"]` is a `BoolTensor[num_phases, num_phases]`. Each row `[i]` gives the set of phases that are legal to transition *to* when the current phase is `i`.

**You must index by the current phase, not by a hardcoded constant:**

```python
# Correct
current_phase = obs[:num_phases].argmax().item()   # from phase_onehot, §3.2
mask = graph["node_meta"][node_idx]["valid_transition_mask"][current_phase]

# Wrong — only works when the intersection is always in phase 0
mask = graph["node_meta"][node_idx]["valid_transition_mask"][0]
```

The phase_onehot is the **first segment** of the obs vector (§3.2): `obs[:num_phases]` gives the one-hot encoding of the currently active phase. `argmax()` recovers the phase index.

In R1 all entries in `valid_transition_mask` are `True` (the spec's R0 constraint — all transitions are topologically valid; minimum-green enforcement is the environment's job, not the graph's). The mask still exists and must be used correctly so R2 can add min-green enforcement without touching model code.

---

## Architecture

```
for each candidate phase p:
    concat(node_embedding, phase_features[p])        # (embed_dim + phase_feat_dim,)
    → Linear(embed_dim + phase_feat_dim, hidden_dim) # shared MLP weights across all phases
    → ReLU
    → Linear(hidden_dim, 1)
    → squeeze                                        # scalar Q-value for phase p

torch.stack([score_0, score_1, ...])                 # (num_phases,)
masked_fill(~valid_transition_mask, -inf)            # invalid phases → -inf
return scores
```

The MLP is called once per candidate phase per node per decision step. For a 2-phase intersection this is 2 forward passes through a small MLP; even for a 4-phase intersection this is negligible compute. The key property is that the same MLP weights process every phase at every intersection — it is not a separate head per phase index.

---

## Test results

**23 tests, all passing** (last run 2026-05-24, Python 3.10.8, pytest 9.0.3).

Tests live in `tests/test_phase_head.py` and are split into:

- **`TestPhaseHeadOutput` (5)** — output shape `(num_phases,)` for three different `(num_phases, phase_feat_dim)` combinations; dtype is `float32`.
- **`TestPhaseHeadMasking` (4)** — masked phase scores exactly `-inf`; all-valid scores have no `-inf`; all-invalid returns all `-inf` without crashing; multiple simultaneous masked phases.
- **`TestPhaseHeadSelectAction` (4)** — action is always a valid-mask index; return type is `int`; consistent with `forward`'s argmax; never selects an invalid phase over 20 random seeds.
- **`TestPhaseHeadNaN` (2)** — no NaN for clean input; no NaN when embedding is all zeros.
- **`TestPhaseHeadSensibility` (2)** — gradients flow back to phase feature tensors; after 20 Adam steps the head scores a target phase higher than its competitor (R1 done-when: *phase scores respond sensibly to handcrafted inputs*).
- **`TestPhaseHeadEndToEnd` (6)** — `MockEnv → NodeEncoder → PhaseHead` pipeline completes on `cross_smoke`, `linear_two`, and `grid_3x3`; action stays in `[0, num_phases)` over multiple `env.step()` calls.

To run:

```bash
pytest tests/test_phase_head.py -v
```

To run both R1 model tests:

```bash
pytest tests/test_node_encoder.py tests/test_phase_head.py -v
```

### Diagnosing a failure

| Failing test | Likely cause |
|---|---|
| `test_output_shape` | The `torch.stack` of per-phase scalars is not being squeezed correctly — check `.squeeze(-1)` after the final `Linear` |
| `test_invalid_phases_are_neg_inf` | `masked_fill` is using `valid_transition_mask` instead of its complement — use `~valid_transition_mask` |
| `test_all_invalid_returns_all_neg_inf` | `masked_fill` is conditional on at least one valid phase — it must run unconditionally |
| `test_action_never_selects_invalid_phase` | `select_action` is calling `argmax` on raw scores before masking |
| `test_gradient_flows_to_phase_features` | Phase features are not included in the computation graph — check `requires_grad` on the test inputs; check that `torch.cat` is used (not `torch.stack` or `copy`) |
| `test_head_can_learn_to_prefer_high_pressure_phase` | The concat input order is `(embedding, phase_feat)` — if reversed the head still works but this test's specific 20-step setup may not converge; also check learning rate |
| `test_forward_pass_completes` (end-to-end) | `phase_feat_dim` mismatch — verify `pad_phase_features(graph)` is called and its return value is passed to `PhaseHead`; raw `graph["phase_features"]` without padding will crash when node lane counts differ |

---

## Related files

| File | Role |
|---|---|
| `models/phase_head.py` | This module — `PhaseHead`, `pad_phase_features` |
| `models/node_encoder.py` | Upstream — produces the `node_embedding` input |
| `data/graph_builder.py` | Source of `phase_features` (green-lane masks) and `valid_transition_mask` |
| `models/gat_policy.py` | Will wrap `NodeEncoder` + this module with GAT layers in between (R4+) |
| `training/trainer.py` | Calls `forward` for Q-values; calls `select_action` during rollout (R2) |
| `env/mock_env.py` | Provides `obs_dict` and `graph` for testing and R1–R2 development |
| `tests/test_phase_head.py` | 23 tests including the R1 done-when sensibility check |
| `docs/node_encoder.md` | Documents the upstream encoder that produces `node_embedding` |

---

## What R1 built — R4 starting point

`pad_phase_features` and `PhaseHead` are **complete as of R1**. Both are fully tested (23 tests
passing). Do not modify `models/phase_head.py` in R2.

### What R2 uses from this module

R2 uses `pad_phase_features` and `PhaseHead.forward` exactly as built. The trainer calls:

```python
phase_feat_dim, padded_pf = pad_phase_features(graph)   # once at init
head = PhaseHead(embed_dim, phase_feat_dim, head_hidden_dim)

# per node per batch element in _compute_loss:
scores = head(embedding_1d, padded_pf[node_idx], all_true_mask)  # (embed_dim,) input
```

Note: `forward` takes a 1D `(embed_dim,)` embedding. R2's loss computation loops over batch
elements to satisfy this constraint. This is correct behaviour — do not attempt to batch here.

### R4's only responsibility for this module

R4 adds one new method to `PhaseHead`:

```python
def forward_batch(
    self,
    emb_batch: Tensor,        # (B, embed_dim)
    phase_stack: Tensor,      # (P, phase_feat_dim)
    mask: Tensor,             # (P,) bool
) -> Tensor:                  # (B, P)
    """Vectorised forward over a batch of embeddings for one node's phases."""
```

This method is **additive only** — it does not replace or modify `forward` or `select_action`.
The 23 existing R1 tests must pass without change after R4 adds `forward_batch`. R4 wires it
into `_compute_loss` to eliminate the per-batch-element loop; R2 does not need it.
