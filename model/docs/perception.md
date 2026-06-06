# Perception Model (R3)

## What R3 builds

R3 wires a crude severity-based masking layer into the training loop so the model learns to
handle degraded, partially-missing sensor data. A single `severity` knob controls the fraction
of features randomly masked per step.

The full structured perception model (weather regimes, distance decay, occlusion, lane
misassignment, latency buffers) is aspirational architecture — that is not R3. R3 just proves
the obs contract and training loop handle degraded input correctly, and lays the data-shape
foundation for R4.

---

## The obs contract

Every consumer of `obs_dict` (trainer, encoder, buffer, reward) must respect these rules:

| Field | Type | Meaning |
|---|---|---|
| `obs[i]` | float32 | sensor reading, or `-1.0` (sentinel) if missing |
| `validity[i]` | float32 | `1.0` = real reading; `0.0` = missing/corrupted |

**Invariant:** `validity[i] == 0.0` implies `obs[i] == -1.0` (sentinel). The converse is not
required — a genuine reading of -1.0 would be distinguished by `validity[i] == 1.0`, though
this is unlikely for normalized traffic counts.

`validity` values are exactly `0.0` or `1.0` — no intermediate floats.

`NodeEncoder.forward(obs, validity)` concatenates `obs` and `validity` into a single
`2 × obs_dim` vector before the first linear layer — `torch.cat([obs, validity], dim=-1)`.
The model therefore sees the validity flag explicitly alongside each observation value,
letting it distinguish live, imputed, and never-seen positions rather than only zeroing
sentinels. This is applied to the already-imputed obs (see below).

---

## `apply_perception` interface

```python
from env.perception import apply_perception

masked = apply_perception(obs_dict, severity, sentinel=-1.0)
```

**Parameters:**
- `obs_dict`: `{node_id: (obs_tensor, validity_tensor)}` — raw from `env.step()` or `env.reset()`
- `severity`: `float` in `[0.0, 1.0]` — fraction of features corrupted per step (Bernoulli)
- `sentinel`: value placed in obs when a position is corrupted; default `-1.0`

**Algorithm (per node, per feature):**
```
corrupt = Bernoulli(severity)       # independent per feature
obs[corrupt]      = sentinel
validity[corrupt] = 0.0
```

**Invariants:**
- `severity=0.0`: returns a new dict with values identical to input — no corruption
- `severity=1.0`: all positions → sentinel, all validity → 0.0
- Already-invalid positions (`validity=0` from env) are never restored — `corrupt` only adds
- Does **not** call `pad_obs_dict`; operates on raw (un-padded) obs
- Returns a new dict; input dict and tensors are not mutated

---

## Observation imputation (`ObservationImputer`)

**Problem:** After masking, `NodeEncoder(obs * validity)` produces `0.0` for missing features
(`sentinel × 0 = 0`). This is ambiguous: `0` means both "sensor failed" and "genuinely empty
lane." The model cannot distinguish them.

**Fix:** Apply last-known imputation to the obs *before* padding and encoding.
`ObservationImputer` maintains a per-node cache of the last valid sensor reading for each
feature. Missing positions are filled with the last-known value, and `validity` is flipped to
`1.0` for those positions.

After imputation:
- `validity=1` positions — current reading (unchanged)
- `validity=0` positions with prior history — replaced with last-known value; `validity→1.0`
- `validity=0` positions with no prior history (first-step failure) — sentinel stays; `validity`
  stays `0.0` (rare, and now unambiguous: it means "we have truly never seen this sensor")

`NodeEncoder` then sees last-known values instead of zeros for almost all missing features.
`NodeEncoder` code is unchanged — the fix is upstream.

**Why `PressureReward` gets raw (pre-imputation) obs:**
`PressureReward` maintains its own independent last-known cache for pressure computation.
Passing pre-imputed obs to it would contaminate its cache with `ObservationImputer`'s values
instead of true sensor readings. Both classes call `reset()` at each `env.reset()` to prevent
episode bleed.

---

## Where perception sits in the pipeline

Applied after `env.reset()` / `env.step()`, before imputation and `pad_obs_dict`:

```
env.reset() / env.step()
  → apply_perception(obs_dict, severity, sentinel)   # stateless masking
  → PressureReward.compute(raw_obs_dict, graph)      # reward uses raw obs; own imputation
  → ObservationImputer.impute(obs_dict)              # fills sentinels with last-known
  → pad_obs_dict(imputed_obs_dict)                   # pads for encoder + buffer
  → NodeEncoder(cat([obs, validity]))                 # sees last-known value + validity flag
```

---

## Why perception is applied at both reset AND step

`_select_actions` is called immediately after `env.reset()` using the reset observation.
If perception is only applied at `env.step()`, the first action of every episode is chosen
from clean obs even when training under degraded conditions. This would cause a train/eval
distribution mismatch — at deployment, sensors fail from the very first timestep.

Both sites use the same `severity` value. At `severity=0.0` both are no-ops.

---

## Backward compatibility

`severity=0.0` (default, matches R2 config) makes `apply_perception` a pure copy with no
corruption. `ObservationImputer.impute()` sees all `validity=1` and makes no changes.
`use_pressure=false` (R2 default) skips `PressureReward` entirely. All 241 R2 tests pass
unchanged.

---

## R4 starting point

After imputation, `validity=0` is rare and semantically precise: it means a feature was never
observed in the current episode. This clean signal is available to R4's GAT as an edge feature.

Two options for R4 to handle degraded neighbors:
- **Implicit**: let attention weights learn to down-weight degraded embeddings on their own
- **Explicit**: pass a validity summary scalar (e.g. fraction of valid features) as an edge
  feature so the attention head has a direct degradation signal

`apply_perception` already returns `(obs, validity)` per node — the validity vector is the
right shape for either approach. R4 decides; R3 does not need to change.

Note: outgoing queue of node A and incoming queue of node B are **different observations**
at different physical locations. A's outgoing queue is useful for A's own spillback detection
only. B's incoming state is inside B's own observation vector, sent to A via GAT edges in R4.
