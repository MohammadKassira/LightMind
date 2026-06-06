# R0 mock_env — Offline Development Environment (No SUMO)

## What this module does

`env/mock_env.py` provides a SUMO-free environment for writing and unit-testing model code during rings R0–R2 (and optionally R3). It wraps a **real graph** built from a SUMO `net.xml` file via `graph_builder.py`, but generates **randomly shaped observations** instead of running a traffic simulation.

The observation vectors match the §3.2 schema exactly — correct lengths, correct normalization ranges, correct dtypes, correct validity mask convention. This means the node encoder, phase head, and DQN training loop can be implemented and tested before SUMO is installed or integrated.

When SUMO integration arrives, swap `MockEnv` for `traffic_env.TrafficEnv` — the `reset` / `step` API is identical.

---

## Where this fits in the pipeline

```
net.xml ──► graph_builder.build_graph() ──► static graph dict ──┐
                                                                  │
MockEnv._make_node_obs() ──► random obs (§3.2 shaped) ───────────┤
                                                                  ▼
                                               models/node_encoder.py
                                               models/phase_head.py
                                               training/trainer.py
```

| Module | Role |
|---|---|
| `env/mock_env.py` | This module — fake obs, real graph, SUMO-free |
| `env/traffic_env.py` | SUMO version with identical `reset`/`step` API (later ring) |
| `data/graph_builder.py` | Builds the real graph topology loaded at `__init__` |
| `data/observation_encoder.py` | Will replace `_make_node_obs()` when SUMO is wired in |

---

## API

### `MockEnv(network_name, max_steps=100, missing_prob=0.0)`

Loads configs and builds the graph. Must be called before `reset`.

| Argument | Type | Description |
|---|---|---|
| `network_name` | `str` | Stem of a `net.xml` in `data/networks/` — e.g. `"cross_smoke"`, `"grid_3x3"` |
| `max_steps` | `int` | Episode length. `done=True` is returned after this many `step()` calls. |
| `missing_prob` | `float` | Per-feature probability of corruption. `0.0` = no missing data (default). |

```python
from env.mock_env import MockEnv

e = MockEnv("cross_smoke")
e = MockEnv("grid_3x3", max_steps=200, missing_prob=0.1)
```

---

### `reset(network_cfg=None, seed=None) -> (obs_dict, graph)`

Resets the episode step counter and returns a fresh observation.

| Argument | Type | Description |
|---|---|---|
| `network_cfg` | any | Accepted for API compatibility with `traffic_env`; **ignored** — network is fixed at `__init__`. |
| `seed` | `int \| None` | If given, calls `torch.manual_seed(seed)` before sampling. Produces deterministic obs. |

**Returns:**

| Value | Type | Description |
|---|---|---|
| `obs_dict` | `dict[str, tuple[Tensor, Tensor]]` | Maps each node ID to `(obs, validity)` — see §3.2 layout below. |
| `graph` | `dict` | Static typed-edge graph from `graph_builder.build_graph()`. Same object every call — do not mutate. |

```python
obs_dict, graph = e.reset(seed=42)
obs_tensor, validity = obs_dict["A0"]
print(obs_tensor.shape)   # e.g. torch.Size([15]) for cross_smoke
print(validity.shape)     # same shape as obs_tensor
```

---

### `step(actions) -> (obs_dict, graph, reward_dict, done, info)`

Advances the episode by one step. Generates new random observations and rewards.

| Argument | Type | Description |
|---|---|---|
| `actions` | `dict[str, int]` | Maps each node ID to a chosen phase index. **Ignored in the mock** — has no effect on the next obs. |

**Returns:**

| Value | Type | Description |
|---|---|---|
| `obs_dict` | `dict[str, tuple[Tensor, Tensor]]` | Fresh `(obs, validity)` per node (same schema as `reset`). |
| `graph` | `dict` | Same static graph object as returned by `reset`. |
| `reward_dict` | `dict[str, float]` | Maps each node ID to a random negative float in `(-1, 0]`. |
| `done` | `bool` | `True` when `step_count >= max_steps`. |
| `info` | `dict` | Always `{}` in the mock. |

```python
obs_dict, graph, reward, done, info = e.step({"A0": 0})
print(reward)   # e.g. {"A0": -0.37}
print(done)     # False until max_steps reached
```

---

## Observation vector layout (§3.2)

Each node's observation is a `FloatTensor` built by concatenating these segments in order:

| Segment | Length | Values | Description |
|---|---|---|---|
| `phase_onehot` | `num_phases` | {0.0, 1.0} one-hot | Which phase is currently active |
| `time_in_phase` | 1 | [0, 1] | How long the current phase has been active, normalised by `max_phase_time` |
| `queue / q_max` per incoming lane | `num_incoming` | [0, 1] | Normalised queue length for each incoming lane |
| `running / q_max` per incoming lane | `num_incoming` | [0, 1] | Normalised count of vehicles within lookahead distance L |
| `queue / q_max` per outgoing lane | `num_outgoing` | [0, 1] | Normalised queue length on each outgoing lane |

**Total length:** `num_phases + 1 + 2 × num_incoming + num_outgoing`

Observation length **varies per node** — it is computed from the graph at runtime and never hardcoded. Nodes in the same network can have different lengths if their lane counts differ (see `linear_two`: A0 = 12, B0 = 9).

---

## Validity mask convention (§3.3)

`validity` is a `FloatTensor` of the **same length** as `obs`:

| Value | Meaning |
|---|---|
| `1.0` | Real reading — the corresponding obs feature is valid |
| `0.0` | Missing — the corresponding obs feature holds the sentinel value |

**Sentinel: `-1.0`** (from `configs/perception.yaml`). Missing ≠ zero. Zero means "empty lane" — a real observation of no vehicles. `-1.0` is below the normalised range `[0, 1]` and unambiguously signals missing data.

Consumers must branch on the **validity flag**, never the value:

```python
obs_t, val = obs_dict["B1"]
missing = val == 0.0
valid_obs = obs_t[~missing]  # safe to use
```

---

## Missing data mode

Controlled by the `missing_prob` argument to `__init__`. Each feature in the obs vector is independently corrupted with this probability on every call to `reset()` or `step()`.

Corruption sets `obs[i] = sentinel_value` and `validity[i] = 0.0`.

```python
# 30% of features missing on average
e = MockEnv("grid_3x3", missing_prob=0.3)
obs_dict, _ = e.reset(seed=0)
obs_t, val = obs_dict["A0"]
print((val == 0.0).sum().item(), "missing features out of", len(val))
```

Use `missing_prob=1.0` for deterministic all-missing tests. Use `missing_prob=0.0` (default) for clean training during R1–R2.

---

## Configuration loaded

The constructor reads two config files. Constants are **never redefined locally** — they come from these single sources of truth (§3.1).

| Constant | File | Key path | Used for |
|---|---|---|---|
| `max_phase_time` | `configs/normalization.yaml` | `max_phase_time` | Normalises time-in-phase scalar to [0, 1] |
| `sentinel_value` | `configs/perception.yaml` | `perception.sentinel_value` | Value written to obs at missing positions |

`q_max`, `v_max`, `t_duration` are also in `normalization.yaml` (§3.1) but are not used by the mock — it generates normalised values directly via `torch.rand`. These constants will be used by `observation_encoder.py` in the SUMO-backed `traffic_env`.

---

## Approximations and limitations

1. **Symmetric intersection**: `num_outgoing = num_incoming`. Real intersections may differ. `traffic_env.py` will read the actual outgoing lane count from SUMO state.

2. **No temporal dynamics**: observations are freshly random at every `reset()` and `step()` call — queue lengths do not accumulate, phases do not cycle. A model trained exclusively on `MockEnv` will have only random-policy experience.

3. **Reward is random**: `reward = -Uniform(0, 1)` per node, always negative. Not meaningful for learning. Its purpose is to confirm the DQN loop runs without crashing, not to drive convergence.

4. **`network_cfg` ignored**: `reset(network_cfg=...)` accepts the parameter for API compatibility but has no effect. The network is fixed at construction time.

5. **Static graph returned by reference**: `reset()` and `step()` return `self._graph` — the same Python dict object every call. Callers must not mutate it.

---

## Test results

**67 tests, all passing** (last run 2026-05-24, Python 3.10.8, pytest 9.0.3).

Tests live in `tests/test_mock_env.py` and are split into:

- **Per-network class tests (35)** — four classes targeting `cross_smoke`, `linear_two`, `grid_3x3`, and missing-data behaviour. Cover return types, tensor shapes and dtypes, obs vector layout, validity mask, step counter, seeded determinism, reward sign.
- **Cross-network invariants (32)** — eight properties parametrized over all four test networks. Catch regressions where a change fixes one network but silently breaks another.

To run:

```bash
pytest tests/test_mock_env.py -v
```

To run both `mock_env` and `graph_builder` together:

```bash
pytest tests/ -v
```

### Diagnosing a failure

| Failing test | Likely cause |
|---|---|
| `test_obs_size_matches_schema` | `_make_node_obs` is building the wrong length — check each segment's slice |
| `test_phase_onehot_valid` | One-hot sampling is broken — check `torch.randint` / `torch.zeros` logic |
| `test_missing_implies_sentinel` | Corruption sets value but doesn't set validity to 0, or vice versa |
| `test_graph_is_same_object_across_steps` | `step()` or `reset()` is rebuilding the graph instead of returning `self._graph` |
| `test_nodes_can_have_different_obs_sizes` | `_make_node_obs` is reading a hardcoded lane count instead of from the graph |
| `test_step_count_resets` | `reset()` is not setting `self._step_count = 0` |

---

## Related files

| File | Role |
|---|---|
| `env/mock_env.py` | This module |
| `env/traffic_env.py` | SUMO/TraCI wrapper with identical `reset`/`step` API (later ring) |
| `env/perception.py` | Sensor degradation model (R3); replaces the `missing_prob` mechanism with a structured failure model |
| `data/graph_builder.py` | Provides `build_graph()` used in `MockEnv.__init__` |
| `data/observation_encoder.py` | Will encode real SUMO state into the same §3.2 schema |
| `configs/normalization.yaml` | §3.1 constants — `q_max`, `max_phase_time`, etc. |
| `configs/perception.yaml` | Sentinel value (`-1.0`) and perception flags |
| `tests/test_mock_env.py` | 67 tests across all four synthetic networks |
| `docs/graph_builder.md` | Documents the graph dict schema returned by `reset`/`step` |
