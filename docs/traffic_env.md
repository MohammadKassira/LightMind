# TrafficEnv — Real SUMO/TraCI Environment

## What this module does

`env/traffic_env.py` replaces `MockEnv` with a live SUMO simulation. It implements the
**identical `reset`/`step` API** so `DQNTrainer`, `compute_pressure`, `ObservationImputer`,
and `NodeEncoder` work unchanged — swap one env for the other without touching any other file.

Where MockEnv generates random observations, TrafficEnv extracts real per-lane vehicle counts
from SUMO via TraCI, applies yellow-phase transitions, enforces min-green constraints, and
advances the simulation by a configurable number of seconds per action step.

`env/demand_generator.py` is a companion utility for networks that lack pre-existing route
files. It wraps SUMO's bundled `randomTrips.py` to generate vehicle demand before training.

---

## Where this fits in the pipeline

```
net.xml ──► graph_builder.build_graph() ──► static graph dict ─────────────────┐
                                                                                 │
SUMO simulation ──► TraCI ──► TrafficEnv._extract_obs() ──► obs_dict (§3.2) ───►├──► DQNTrainer
                                                                                 │
apply_perception() ──► ObservationImputer.impute() ──► pad_obs_dict() ──────────┘
```

| Module | Role |
|---|---|
| `env/traffic_env.py` | This module — live SUMO obs, real phase control |
| `env/mock_env.py` | Drop-in mock with random obs, no SUMO required (R0) |
| `env/demand_generator.py` | Generates `.rou.xml` demand files for SUMO networks |
| `env/perception.py` | Sensor degradation applied *outside* the env by the trainer |
| `data/graph_builder.py` | Builds the static graph loaded at `__init__` |

---

## Prerequisites

SUMO must be installed and `SUMO_HOME` must point to the installation directory:

```bash
# Linux / macOS
export SUMO_HOME="/usr/share/sumo"          # adjust to your install path
export PATH="$SUMO_HOME/bin:$PATH"

# Windows (PowerShell)
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
```

`traffic_env.py` raises `RuntimeError` at import time if `SUMO_HOME` is unset, so the
failure is immediate and explicit rather than a confusing `ModuleNotFoundError` later.

---

## API

### `TrafficEnv(net_file, route_file, ...)`

```python
from env.traffic_env import TrafficEnv

env = TrafficEnv(
    net_file   = "networks/external/RESCO/cologne1/cologne1.net.xml",
    route_file = "networks/external/RESCO/cologne1/cologne1.rou.xml",
    max_steps  = 200,
    delta_time = 5,
    yellow_time = 2,
    min_green  = 5,
    use_gui    = False,
    begin_time = 25200,   # 7 AM for cologne1; 0 for synthetic networks
)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `net_file` | `str \| Path` | required | Path to SUMO `.net.xml` |
| `route_file` | `str \| Path` | required | Path to SUMO `.rou.xml`; must exist (use `demand_generator` to create one) |
| `max_steps` | `int` | 200 | Episode length in **action steps** (not simulation seconds) |
| `delta_time` | `int` | 5 | Simulation seconds advanced per `step()` call |
| `yellow_time` | `int` | 2 | Seconds the yellow phase is held on a phase transition |
| `min_green` | `int` | 5 | Minimum green seconds before a phase change is allowed |
| `use_gui` | `bool` | False | Open SUMO-GUI window (slow; useful for debugging) |
| `begin_time` | `int` | 0 | Simulation start time in seconds (`25200` = 7 AM for cologne1) |

SUMO is **not started in `__init__`** — the `.net.xml` is only parsed statically (to build the
graph and extract lane/phase tables). SUMO starts on the first `reset()` call.

---

### `reset(seed=None) → (obs_dict, graph)`

Starts a new SUMO episode. If SUMO is already running from a previous episode, it is closed
first.

| Argument | Type | Description |
|---|---|---|
| `seed` | `int \| None` | Passed to SUMO as `--seed`. `None` → `--random` (different demand each episode) |

**Returns:**

| Value | Type | Description |
|---|---|---|
| `obs_dict` | `dict[str, tuple[Tensor, Tensor]]` | `{node_id: (obs, validity)}` at simulation time `begin_time` |
| `graph` | `dict` | Static typed-edge graph from `build_graph()`. Same object every call — do not mutate. |

```python
obs_dict, graph = env.reset(seed=0)
obs, val = obs_dict["n1"]
print(obs.shape)     # e.g. torch.Size([17])
print(val.all())     # True — TrafficEnv always returns validity=1.0
```

---

### `step(actions) → (obs_dict, graph, reward_dict, done, info)`

Applies the chosen phases, advances the simulation by `delta_time` seconds, then reads SUMO
state.

| Argument | Type | Description |
|---|---|---|
| `actions` | `dict[str, int]` | `{node_id: phase_idx}` — index into actionable phases for each node |

**What happens inside one step:**

1. **Action application** — for each TLS junction:
   - If the junction is mid-yellow: ignore the action (let yellow finish).
   - If `new_phase == current_phase` or `time_in_phase < min_green + yellow_time`: hold current green (re-asserts SUMO state).
   - Otherwise: set yellow state in SUMO, start yellow countdown.
2. **Simulation advance** — `simulationStep()` is called `delta_time` times. Each sim-second the yellow countdown is decremented; when it reaches zero the target green phase is activated.
3. **Observation extraction** — read per-lane vehicle counts from SUMO.
4. **Reward** — returns `{node_id: 0.0}` for all nodes. The env does not compute pressure. The trainer owns all reward logic: when `use_pressure=False` it uses `reward_dict` from `env.step()` directly; when `use_pressure=True` it calls `PressureReward.compute()` on the raw obs and discards `reward_dict`. Keeping reward computation out of the env means the env imports nothing from `training/` and carries no knowledge of pressure.

**Returns:**

| Value | Type | Description |
|---|---|---|
| `obs_dict` | `dict[str, tuple[Tensor, Tensor]]` | Fresh `(obs, validity)` per node |
| `graph` | `dict` | Same static graph object from `reset` |
| `reward_dict` | `dict[str, float]` | `{node_id: 0.0}` — placeholder; trainer computes the real reward externally |
| `done` | `bool` | `True` when `step_count >= max_steps` |
| `info` | `dict` | `{"sim_time": float}` — current SUMO simulation time in seconds |

```python
actions = {nid: 0 for nid in graph["node_ids"]}
obs_dict, graph, reward, done, info = env.step(actions)
print(info["sim_time"])   # e.g. 25205.0 (begin_time + delta_time)
```

---

### `close()`

Closes the TraCI connection. Always call this when the env is no longer needed to avoid
leaving SUMO processes running.

```python
env.close()

# Or use as a context manager:
with TrafficEnv(net_file, route_file) as env:
    obs_dict, graph = env.reset()
    ...
```

---

## Observation vector layout (§3.2)

Identical to MockEnv — same segment order, same normalization, same dtypes:

| Segment | Length | Values | Description |
|---|---|---|---|
| `phase_onehot` | `num_phases` | {0.0, 1.0} one-hot | Which actionable phase is active |
| `time_in_phase` | 1 | [0, 1] | Seconds in current phase / `max_phase_time` (clamped) |
| `queue_in / q_max` | `num_incoming` | [0, 1] | Halting vehicles / `q_max` per incoming lane |
| `running_in / q_max` | `num_incoming` | [0, 1] | Moving vehicles / `q_max` per incoming lane |
| `queue_out / q_max` | `num_outgoing` | [0, 1] | Halting vehicles / `q_max` per outgoing lane |

**Total length:** `num_phases + 1 + 2 × num_incoming + num_outgoing`

`num_outgoing` is read from the actual SUMO network topology and may differ from
`num_incoming` — this is exact, not the symmetric approximation used by MockEnv.

---

## Validity mask

TrafficEnv always returns `validity = torch.ones_like(obs)`. Sensor degradation is applied
**outside** the env by the trainer:

```
env.step()
  → apply_perception(obs_dict, severity, sentinel)   # adds validity=0 for corrupted features
  → PressureReward.compute(raw_obs_dict)             # reward sees raw obs
  → ObservationImputer.impute(obs_dict)              # fills sentinels with last-known
  → pad_obs_dict()                                   # pads for encoder + buffer
  → NodeEncoder(cat([obs, validity]))                # encodes obs + validity jointly
```

This separation means `TrafficEnv` is perception-agnostic — the same env works for both clean
training (`severity=0`) and degraded training (`severity=0.1`, etc.) without any env change.

---

## Phase control

### Actionable phases

Each TLS junction has a set of **actionable (green) phases** — states where at least one lane
has `G` or `g` and no lane has `y`. These are the only phases the agent controls; yellow and
all-red intermediate phases are managed by the env.

The list of actionable phases for each junction comes from the `.net.xml` `<tlLogic>`
elements — the same source as `graph_builder.py`, ensuring that `graph["node_meta"][i]["num_phases"]`
always equals the number of actionable phases for node `i`.

### Yellow transition logic

Phase changes go through an intermediate yellow state. The yellow state for the transition
`i → j` is computed at `__init__` time from the `.net.xml`:

```
for each signal position k:
    if phase_i[k] is green (G/g) and phase_j[k] is red/stop (r/s):
        yellow_state[k] = 'y'
    else:
        yellow_state[k] = phase_i[k]   # keep current (preserve partial greens)
```

This matches SUMO's standard yellow convention and the sumo-rl reference implementation.

### Min-green enforcement

A phase change is refused if `time_in_phase < min_green + yellow_time`. The action is
silently held (current state re-asserted in SUMO). This prevents rapid oscillation.

| Condition | Result |
|---|---|
| `new == current` | Hold current green; increment timer |
| `time_in_phase < min_green + yellow_time` | Hold current green (min-green guard) |
| else | Start yellow transition; set `yellow_timer = yellow_time` |

---

## Lane ordering (critical for obs consistency)

The obs vector's lane segments must align with the `phase_features` entries in the graph dict.
Both are derived from the same source and use the same canonical sort:

**Incoming lanes:** sorted by `(from_edge_id, from_lane_index)` — ascending, lexicographic on
the edge id string. SUMO lane ID: `f"{from_edge}_{from_lane_idx}"`.

**Outgoing lanes:** sorted by `(to_edge_id, to_lane_index)` — same rule.

This ordering is computed once at `__init__` from the `.net.xml` (no SUMO connection needed).
It is stored in `env._incoming_lanes[node_id]` and `env._outgoing_lanes[node_id]`.

`compute_pressure` extracts `q_in` and `q_out` slices by counting from fixed offsets derived
from `num_phases` and `num_incoming`. If lane ordering is wrong, pressure computation will be
silently incorrect. The canonical sort guarantees graph and env agree.

---

## Configuration

`configs/traffic.yaml` holds defaults for real-env training runs. Read with `OmegaConf` or
`yaml.safe_load` and pass to `TrafficEnv`:

```yaml
env:
  net_file: "networks/generated/cross_smoke/cross_smoke.net.xml"
  route_file: null         # provide or generate with demand_generator
  max_steps: 200
  delta_time: 5
  yellow_time: 2
  min_green: 5
  use_gui: false
  begin_time: 0

demand:
  period: 2.0              # seconds between vehicle insertions
  duration: 3600
  seed: 42
```

Normalization constants (`q_max`, `max_phase_time`) are read from `configs/normalization.yaml`
at import time — TrafficEnv never redefines them locally.

---

## Normalization constants

| Constant | Source | Value | Used for |
|---|---|---|---|
| `q_max` | `configs/normalization.yaml` | 30 | Divides all halting/vehicle counts → [0, 1] |
| `max_phase_time` | `configs/normalization.yaml` | 90.0 | Divides `time_in_phase` → [0, 1] |

Values above `q_max` are clamped to 1.0. A future improvement would use per-lane capacity
from SUMO lane lengths: `capacity = lane_length / (MIN_GAP + avg_vehicle_length)`.

---

## Demand generation (`env/demand_generator.py`)

Networks generated by `netgenerate` (e.g. `networks/generated/cross_smoke/`) ship with empty
route files. Use `generate_demand` to create vehicle flows before training:

```python
from env.demand_generator import generate_demand
from env.traffic_env import TrafficEnv

rou = generate_demand(
    net_file    = "networks/generated/cross_smoke/cross_smoke.net.xml",
    output_rou  = "networks/generated/cross_smoke/cross_smoke.demand.rou.xml",
    period      = 2.0,     # ~30 vehicles/min
    duration    = 3600,    # 1 hour of demand
    seed        = 42,
)
env = TrafficEnv("networks/generated/cross_smoke/cross_smoke.net.xml", rou)
```

`generate_demand` calls SUMO's `randomTrips.py` (at `$SUMO_HOME/tools/randomTrips.py`) as a
subprocess. The intermediate trips file is written to a temp path and deleted automatically.

| Argument | Type | Default | Description |
|---|---|---|---|
| `net_file` | `str \| Path` | required | SUMO `.net.xml` |
| `output_rou` | `str \| Path` | required | Output `.rou.xml` path |
| `period` | `float` | 2.0 | Seconds between vehicle insertions (lower = denser traffic) |
| `duration` | `int` | 3600 | End time for trip generation (seconds) |
| `seed` | `int` | 42 | Reproducible demand |

**RESCO networks** (`networks/external/RESCO/`) already include calibrated `.rou.xml` files
from real traffic counts. No demand generation needed for these.

---

## Limitations and known approximations

1. **Global `q_max`**: uses 30 (from config) for all lanes. Lanes with very short or long
   lengths will produce values slightly above or below [0, 1]. Values are clamped to [0, 1].

2. **Running vehicles approximation**: `running = total_vehicles - halting_vehicles` per lane.
   The strict §3.2 definition counts only vehicles within `L = v_max × t_duration = 139 m`
   of the stop line. This approximation is equivalent for short lanes and slightly loose for
   long lanes.

3. **Single-episode SUMO instance**: each `reset()` closes and restarts SUMO. On slow machines
   this adds ~0.5–2 s per episode. Multi-episode reuse via `begin_time` cycling is a future
   optimisation.

4. **SUMO zombie processes if `close()` is skipped**: if `close()` is never called (e.g. a
   test crashes mid-episode), the SUMO process keeps running until the OS kills it. `TrafficEnv`
   provides three safety nets: (a) `close()` itself, (b) context-manager `__exit__`, and (c) a
   `__del__` destructor for best-effort cleanup when the Python object is garbage-collected. In
   practice, use the context manager in scripts and call `env.close()` explicitly in test
   teardown. Check for orphaned processes with `tasklist | findstr sumo` (Windows) or
   `pgrep sumo` (Linux/macOS).

5. **No SUMO-RL multi-client support**: uses a single TraCI connection per env instance. Running
   two `TrafficEnv` instances simultaneously requires distinct `net_file`/`route_file` pairs
   (the label counter ensures unique connection labels).

6. **All transitions valid**: `valid_transition_mask` in the graph is all-ones (any phase can
   follow any other). Min-green enforcement happens in the env, not the mask. This matches the
   graph_builder behaviour and the trainer's action selection.

---

## Using TrafficEnv with DQNTrainer

`TrafficEnv` is a drop-in replacement for `MockEnv`:

```python
from env.traffic_env import TrafficEnv
from training.trainer import DQNTrainer
import yaml

cfg = yaml.safe_load(open("configs/train.yaml"))

env = TrafficEnv(
    net_file   = "networks/external/RESCO/cologne1/cologne1.net.xml",
    route_file = "networks/external/RESCO/cologne1/cologne1.rou.xml",
    max_steps  = cfg["env"]["max_steps"],
    begin_time = 25200,
)

trainer = DQNTrainer(cfg, env)
metrics = trainer.train(num_episodes=100)
```

The trainer's `apply_perception → PressureReward → ObservationImputer → pad_obs_dict` pipeline
is unchanged. `TrafficEnv` is perception-agnostic; the trainer applies degradation from outside.

---

## Tests

**26 tests, all passing** (`tests/test_traffic_env.py`). Test fixture: RESCO cologne1
(pre-existing demand, single-intersection, 7–8 AM demand window).

| Test class | What it covers |
|---|---|
| `TestReset` | Return types, obs_dict keys, graph structure, graph matches `build_graph()` |
| `TestObsLayout` | §3.2 shape formula, validity all-ones, dtype float32, values in [0,1], phase one-hot sums to 1 |
| `TestStep` | 5-tuple return, reward keys/types/sign, done type, `sim_time` in info, graph same object |
| `TestDoneSignal` | `done=True` after exactly `max_steps`, `done=False` before |
| `TestGraphStructure` | node_ids nonempty, node_to_idx consistent, num_phases present, edge_index shape, valid_transition_mask shape |

```bash
pytest tests/test_traffic_env.py -v
```

---

## Related files

| File | Role |
|---|---|
| `env/traffic_env.py` | This module |
| `env/mock_env.py` | Drop-in mock env (R0) with identical API, no SUMO |
| `env/demand_generator.py` | Generates `.rou.xml` for networks without demand |
| `env/perception.py` | Sensor degradation applied outside the env by the trainer |
| `data/graph_builder.py` | `build_graph()` — builds the static graph used by both envs |
| `configs/traffic.yaml` | Real-env config (net_file, route_file, delta_time, etc.) |
| `configs/normalization.yaml` | `q_max`, `max_phase_time` — normalization constants |
| `tests/test_traffic_env.py` | 26 API and layout tests against RESCO cologne1 |
| `docs/mock_env.md` | MockEnv reference — API is identical |
| `docs/graph_builder.md` | Graph dict schema used by both envs |
| `docs/perception.md` | Perception pipeline that wraps the env output |
