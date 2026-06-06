# R1 — node_encoder — Per-Node Observation Encoder

## What this module does

`models/node_encoder.py` takes a single intersection's observation vector and validity mask and produces a fixed-size node embedding that the policy head (and, in later rings, the GAT backbone) can work with.

It is the first learned component in the pipeline. Everything upstream of it — the environment, the graph builder, the observation schema — is already in place from R0. Everything downstream — the phase-scoring head, the DQN loop, the GAT layers — depends on the embedding this module produces.

The module also exports `pad_obs_dict`, a helper that pads all per-node observation vectors in a network to the same length before the encoder is constructed. This is necessary because the obs vector length varies per node and per network (it depends on the number of phases and incoming/outgoing lanes), but `nn.Linear` requires a fixed input size.

---

## Where this fits in the pipeline

```
obs_dict  ──► pad_obs_dict() ──► (obs_dim, padded_obs_dict) ──┐
                                                                │
graph["node_meta"], graph["phase_features"] ───────────────────┤
                                                                ▼
                                         NodeEncoder.forward(obs, validity)
                                                                │
                                                                ▼
                                              node_embedding  (embed_dim,)
                                                                │
                                                                ▼
                                         models/phase_head.py  (R1)
                                         models/gat_policy.py  (R4+)
```

| Module | Role |
|---|---|
| `models/node_encoder.py` | This module — obs + validity → fixed-size embedding |
| `env/mock_env.py` | Produces `obs_dict` during R0–R2 development |
| `env/traffic_env.py` | Produces `obs_dict` when SUMO is integrated |
| `models/phase_head.py` | Consumes the node embedding to score candidate phases |
| `models/gat_policy.py` | Consumes the node embedding for graph message-passing (R4+) |

---

## API

### `pad_obs_dict(obs_dict) -> (obs_dim, padded_obs_dict)`

Call this **before** constructing `NodeEncoder`. It scans every node's raw observation, finds the longest one, and pads all shorter observations to that length. Padding positions receive `obs=0.0` and `validity=0.0`.

| Argument | Type | Description |
|---|---|---|
| `obs_dict` | `dict[str, tuple[Tensor, Tensor]]` | Raw output of `env.reset()` or `env.step()` — maps each node ID to `(obs, validity)` |

**Returns:**

| Value | Type | Description |
|---|---|---|
| `obs_dim` | `int` | The padded length — max raw obs length across all nodes |
| `padded_obs_dict` | `dict[str, tuple[Tensor, Tensor]]` | Same structure; all `(obs, validity)` tensors are now length `obs_dim` |

```python
from env.mock_env import MockEnv
from models.node_encoder import pad_obs_dict

env = MockEnv("linear_two")
obs_dict, graph = env.reset(seed=0)

obs_dim, padded = pad_obs_dict(obs_dict)
print(obs_dim)               # 12  (A0's raw obs length; B0 raw = 9, padded to 12)
for node_id, (obs, val) in padded.items():
    print(node_id, obs.shape)  # all (12,)
```

Nodes whose raw obs is already `obs_dim` are returned unchanged (no copy). Shorter nodes get new zero-padded tensors; the original `obs_dict` is not mutated.

**Why padding, not `LazyLinear`:** `LazyLinear` fixes its weight shape on the first forward call. If that call happens to be a longer-obs node, the encoder silently becomes incompatible with shorter nodes in the same network — or in a different network during multi-network training (R8). Explicit padding makes the input contract visible at construction time and is safe across all rings.

---

### `NodeEncoder(obs_dim, hidden_dim=128, embed_dim=64)`

Constructs a two-layer MLP that encodes one node's (padded, masked) observation into a fixed-size embedding.

| Argument | Type | Description |
|---|---|---|
| `obs_dim` | `int` | Padded obs length — the value returned by `pad_obs_dict` |
| `hidden_dim` | `int` | Width of the single hidden layer. Default 128 matches the GAT layer-1 output size (§10). |
| `embed_dim` | `int` | Output embedding size. Default 64 matches the GAT layer-2 output size (§10). |

```python
from models.node_encoder import NodeEncoder, pad_obs_dict

obs_dim, padded = pad_obs_dict(obs_dict)
encoder = NodeEncoder(obs_dim=obs_dim)           # defaults: hidden=128, embed=64
encoder = NodeEncoder(obs_dim=obs_dim, embed_dim=32)  # smaller for quick experiments
```

`obs_dim` is never hardcoded. It comes from the environment at runtime so the same encoder class works on any network (§1.1 constraint).

---

### `NodeEncoder.forward(obs, validity) -> Tensor`

Encodes one node's observation into an embedding. The validity mask is applied before the MLP: `masked_obs = obs * validity`. This zeros out any positions where `validity = 0` (missing sensor or padding), so the sentinel value `-1.0` never reaches the network weights (§3.3).

| Argument | Type | Description |
|---|---|---|
| `obs` | `FloatTensor` | `(obs_dim,)` or `(batch, obs_dim)` — pre-padded to `obs_dim`; may contain sentinel `-1.0` at missing/padding positions |
| `validity` | `FloatTensor` | Same shape as `obs` — `1.0` = real reading, `0.0` = missing or padding |

**Returns:**

| Value | Type | Description |
|---|---|---|
| embedding | `FloatTensor` | `(embed_dim,)` or `(batch, embed_dim)` — never NaN |

```python
# Single node
obs, validity = padded["A0"]
embedding = encoder(obs, validity)
print(embedding.shape)   # torch.Size([64])

# Batch of nodes (e.g. for replay buffer training in R2)
obs_batch  = torch.stack([padded[n][0] for n in graph["node_ids"]])
val_batch  = torch.stack([padded[n][1] for n in graph["node_ids"]])
embeddings = encoder(obs_batch, val_batch)
print(embeddings.shape)  # torch.Size([num_nodes, 64])
```

The module is `eval()`-safe (no dropout, no batch norm) — results are deterministic at inference time.

---

## Padding design in detail

`pad_obs_dict` is called once per environment reset, not per step. After the first reset, the `obs_dim` is fixed for the lifetime of that episode (the graph topology does not change). At each subsequent `step()`, re-run `pad_obs_dict` on the new `obs_dict`; the returned `obs_dim` will be identical because the graph is the same.

```python
obs_dim, padded = pad_obs_dict(obs_dict)
encoder = NodeEncoder(obs_dim=obs_dim)

done = False
actions = {node_id: 0 for node_id in graph["node_ids"]}
while not done:
    obs_dict, graph, reward, done, info = env.step(actions)
    _, padded = pad_obs_dict(obs_dict)   # obs_dim unchanged; same encoder works
    for node_id, (obs, val) in padded.items():
        emb = encoder(obs, val)
        # ...
```

Obs lengths by network and node (for reference):

| Network | Node | Raw obs length | Notes |
|---|---|---|---|
| `cross_smoke` | A0 | 15 | Only node; `obs_dim = 15` |
| `linear_two` | A0 | 12 | Longer node; `obs_dim = 12` |
| `linear_two` | B0 | 9 | Shorter node; padded to 12 |
| `grid_3x3` | interior nodes | 17 | 4 incoming, 4 outgoing |
| `grid_3x3` | corner / edge nodes | varies | 2–3 incoming; padded to interior size |

Exact obs length formula (§3.2): `num_phases + 1 + 2 × num_incoming + num_outgoing`.

---

## Validity masking (§3.3)

The encoder's only contract with the missing-data system is this single line:

```python
masked_obs = obs * validity
```

Before this multiplication, `obs[i]` may be `-1.0` (the sentinel for a missing reading). After it, `obs[i]` is `0.0` if `validity[i] == 0.0`. The MLP never sees the sentinel value.

**What the encoder does NOT do:**

- It does not propagate the validity mask to the output. The embedding is always a plain `FloatTensor` with no attached mask. Downstream modules (phase head, GAT layers) do not need to know which input features were missing.
- It does not impute missing values. A position with `validity=0` contributes `0.0` to the linear layer's dot products — equivalent to "no signal from that feature." Structured imputation (using neighbor embeddings to fill in a missing intersection) is a later-ring concern handled by the GAT's neighbor masking (R7).

---

## Architecture

```
obs * validity                       # zero-mask sentinels and padding
    → Linear(obs_dim, hidden_dim)    # obs_dim from pad_obs_dict; hidden_dim = 128
    → ReLU
    → Linear(hidden_dim, embed_dim)  # embed_dim = 64
    → ReLU
    → (embed_dim,) embedding
```

Two hidden layers match the encoding depth used by PressLight and Advanced-XLight for their per-intersection MLP. The ReLU after the output layer keeps embeddings non-negative, which is consistent with how attention weights operate over them in the GAT (R4+).

Default `embed_dim = 64` aligns with the GAT backbone's layer-2 output (§10: 4 heads × 16 → concat 64). If you change `embed_dim` here, update the GAT layer-2 head count/width accordingly.

---

## Test results

**27 tests, all passing** (last run 2026-05-24, Python 3.10.8, pytest 9.0.3).

Tests live in `tests/test_node_encoder.py` and are split into:

- **`TestPadObsDict` (5)** — padding correctness: all nodes reach `obs_dim`; padding positions have `validity=0`; original obs values are preserved up to raw length. Verified on `linear_two` where A0 and B0 have different raw lengths.
- **`TestNodeEncoderOutput` (7)** — output shape and dtype across `cross_smoke`, `linear_two`, `grid_3x3`; custom `embed_dim` respected.
- **`TestNodeEncoderNaN` (6)** — no NaN for clean input, all-missing input (full sentinel), partial missing, and `MockEnv` with `missing_prob=0.5`.
- **`TestNodeEncoderValidityMask` (2)** — sentinel `-1.0` with `validity=0` produces the same embedding as `0.0` with `validity=0`; all-zero validity does not cause NaN.
- **`TestNodeEncoderVariableObsSizes` (1)** — A0 and B0 from `linear_two` (different raw lengths) both produce `(64,)` embeddings after padding.
- **`TestNodeEncoderBatch` (2)** — batch shape correct; batch result matches individual result within float32 tolerance.
- **`TestNodeEncoderEndToEnd` (4)** — forward pass completes on all three networks; embeddings differ across nodes (non-degenerate output).

To run:

```bash
pytest tests/test_node_encoder.py -v
```

To run all R1 and earlier tests together:

```bash
pytest tests/ -v
```

### Diagnosing a failure

| Failing test | Likely cause |
|---|---|
| `test_all_obs_same_length` | `pad_obs_dict` is not using the max length across all nodes — check the `max(...)` call |
| `test_validity_zero_on_padding` | Padding is setting `obs=0` but not `validity=0` — padding positions must have both zeroed |
| `test_obs_values_preserved_up_to_original_length` | `pad_obs_dict` is mutating the original tensors instead of writing into a new zero buffer |
| `test_no_nan_missing_input` | The MLP is receiving the raw sentinel `-1.0` — check that `obs * validity` precedes the first `Linear` |
| `test_sentinel_same_as_zero_when_masked` | `obs * validity` is applied after, not before, the first layer |
| `test_batch_matches_individual` | Float32 BLAS rounding — use `atol=1e-6` in `torch.allclose` |
| `test_embeddings_differ_across_nodes` | All obs vectors are zero after masking (degenerate mock output) — check MockEnv seed |

---

## Related files

| File | Role |
|---|---|
| `models/node_encoder.py` | This module — `NodeEncoder`, `pad_obs_dict` |
| `models/phase_head.py` | Consumes node embeddings to score candidate phases (R1) |
| `models/gat_policy.py` | Uses embeddings for graph message-passing; 2-layer GAT (R4+) |
| `env/mock_env.py` | Supplies `obs_dict` during R0–R2 development; identical API to `traffic_env` |
| `env/traffic_env.py` | Supplies `obs_dict` when SUMO is wired in (later ring) |
| `data/observation_encoder.py` | Converts raw SUMO state into the §3.2 obs schema |
| `configs/normalization.yaml` | §3.1 constants (`q_max`, `max_phase_time`) used upstream in obs construction |
| `configs/perception.yaml` | Sentinel value (`-1.0`); consumed upstream; encoder only sees post-sentinel obs |
| `tests/test_node_encoder.py` | 27 tests across all three synthetic networks |
| `docs/phase_head.md` | Documents the phase-scoring head that consumes this module's output |
