# Multi-Network Shared-Parameter Training (R8)

## What R8 builds

R7 proved the model works on a single real network under sensor failure. R8 answers the
generalization question: can **one** set of weights train across topologically different
networks simultaneously?

The GAT is already inductive — shared weights, message-passing over any graph — so no
architecture changes are needed. The work is entirely in the training loop and data
pipeline: cycling through multiple networks per episode while keeping a single optimizer
and shared model, handling the variable observation and phase-feature dimensions across
networks, and providing a starting-point mechanism (the "factory model" checkpoint) for
fine-tuning on new deployment networks.

**Done-when:** One parameter set trains across ≥3 configured networks without crashing.

**What R8 is NOT:**
- Not a performance comparison — convergence analysis is R9 work.
- Not PyG Batch. The B-loop over the batch dimension is preserved from R7. R8 does one
  gradient step per network sample; within any step, batch + graph come from the same
  network (consistent N). Cross-network batch mixing requires PyG Batch (variable-N
  batching via node offset) — deferred to post-R9.
- Not a new model architecture. `GATPolicy`, `NodeEncoder`, and `PhaseHead` are unchanged.

---

## The Core Problem

Different networks have different structural dimensions:

| Dimension | Varies by | Example values |
|---|---|---|
| **N** (num intersections) | network topology | cross_smoke=1, cologne8=8, grid4×4=16 |
| **obs_dim** per node | num phases + num lanes | 12–20 in practice |
| **phase_feat_dim** | max incoming lanes | 3–8 in practice |

`NodeEncoder` has `nn.Linear(obs_dim*2, hidden_dim)` — `obs_dim` must be fixed at model
construction time, not per-episode. Similarly, `PhaseHead` has
`nn.Linear(gat_out + phase_feat_dim, head_hidden)`. A model built for one network cannot
be reused on another unless dimensions match.

`ReplayBuffer` pre-allocates `[capacity, N, obs_dim]` — N also varies per network.

**Solution (config ceiling + probe validation):**

1. `r8.yaml` declares `max_obs_dim` and `max_phase_feat_dim` as explicit deployment ceilings.
2. At init, `MultiNetworkTrainer` probes all networks to get their actual dims.
3. Any network that exceeds a ceiling → crash with a clear error.
4. The model is built with the **ceiling** dims, not the probed max. This makes the
   checkpoint deployment-safe for any new network within the ceiling without rebuilding
   the model.
5. All observations and phase features are zero-padded to the ceiling before use.

---

## Architecture

### `MultiNetworkTrainer` — `training/multi_network.py`

Standalone class, does not inherit `DQNTrainer`. The shared model and per-network buffers
live here. Episode-level network selection is random uniform over the configured set.

```
Init:
  cfg → read max_obs_dim, max_phase_feat_dim (ceilings)
  For each env k:
    probe reset → local_obs_dim, local_pf_dim
    assert local_obs_dim ≤ max_obs_dim, local_pf_dim ≤ max_phase_feat_dim
  For each network k:
    pad_phase_features(graph, target_dim=max_phase_feat_dim) → padded_pf[k]
    ReplayBuffer(capacity, max_obs_dim, graph["node_ids"]) → buffers[k]
  NodeEncoder(max_obs_dim, hidden_dim, embed_dim) → encoder
  GATPolicy(...) → gat
  PhaseHead(gat.out_channels, max_phase_feat_dim, ...) → head
  Adam(encoder + gat + head) → optimizer

Per episode:
  k = random.randrange(len(envs))
  env[k].reset() → obs_dict, graph[k]
  apply_perception; obs_imputer
  pad_obs_dict(obs_dict, target_dim=max_obs_dim) → padded_obs

  Per step:
    _select_actions(padded_obs, graph[k], padded_pf[k], epsilon)
    env[k].step() → next_obs_dict
    pad_obs_dict(next_obs_dict, target_dim=max_obs_dim) → padded_next
    buffers[k].push(padded_obs, padded_next, actions, rewards, done)
    total_steps += 1                              ← global counter
    epsilon = _linear_decay(total_steps, ...)    ← global, not per-network
    if len(buffers[k]) >= warmup_steps:          ← per-buffer, not global
      batch = buffers[k].sample(B)               ← [B, N_k, max_obs_dim]
      _compute_loss(batch, graph[k], padded_pf[k], gamma)
      optimizer.step()
```

### Key invariants (do not change these)

**Global epsilon.** One `total_steps` counter increments once per env step regardless of
which network is active. `epsilon = _linear_decay(total_steps, ...)` uses this counter.
There is no per-network epsilon tracker. With 4 networks and `epsilon_decay_steps=30000`,
each network sees ~7500 decay steps on average.

**Per-buffer warmup.** `warmup_steps` applies independently to each buffer. With 4
networks and `warmup_steps=1000`, the first gradient steps start after roughly
`warmup_steps * num_networks ≈ 4000` total env steps — the first buffer to reach 1000
transitions fires; others follow as they fill up. This is safer than a global warmup
counter because each buffer has genuine replay coverage before it is sampled.

**Intra-batch consistency.** Within any one gradient step, `batch + graph` come from the
same network (same N). Cross-network batch mixing would require either padding all networks
to `max_N` + masking padding nodes, or PyG Batch — both deferred to post-R8.

---

## `pad_obs_dict` and `pad_phase_features` extensions

### `models/node_encoder.py` — `pad_obs_dict`

Added `target_dim: int | None = None`. When provided, the effective padded size is
`max(local_max, target_dim)` rather than just the local max:

```python
def pad_obs_dict(obs_dict: dict, target_dim: int | None = None) -> tuple[int, dict]:
    local_max = max(obs.shape[-1] for obs, _ in obs_dict.values())
    obs_dim   = max(local_max, target_dim) if target_dim is not None else local_max
    # zero-pad all tensors to obs_dim; padding positions get validity=0
    ...
    return obs_dim, padded
```

**Non-breaking.** `target_dim=None` (the default) preserves the exact previous behavior.
All existing callers in `DQNTrainer` and tests are unchanged.

### `models/phase_head.py` — `pad_phase_features`

Same extension:

```python
def pad_phase_features(graph: dict, target_dim: int | None = None) -> tuple[int, list]:
    local_max      = max(feats[0].shape[0] for feats in all_feats if feats)
    phase_feat_dim = max(local_max, target_dim) if target_dim is not None else local_max
    # zero-pad each phase feature vector to phase_feat_dim
    ...
    return phase_feat_dim, padded
```

Zero padding maps to "no-green" — same as red in the §3.4 feature convention.
Non-breaking for the same reason.

---

## `MultiNetworkTrainer` internals

### Init: probing and ceiling validation

```python
# Read deployment ceilings
max_obs_dim        = cfg["model"]["max_obs_dim"]        # required; no default
max_phase_feat_dim = cfg["model"]["max_phase_feat_dim"]  # required; no default

# Probe all envs
for k, env in enumerate(envs):
    obs_dict, graph = env.reset(seed=cfg_seed)
    local_obs_dim, _ = pad_obs_dict(obs_dict)
    local_pf_dim, _  = pad_phase_features(graph)
    if local_obs_dim > max_obs_dim:
        raise ValueError(
            f"Network '{network_names[k]}' obs_dim={local_obs_dim} "
            f"exceeds cfg model.max_obs_dim={max_obs_dim}. Raise max_obs_dim."
        )
    # same for pf_dim
```

The ceilings are stored as `self.global_obs_dim` and `self.global_phase_feat_dim`. These
are the values the model is built with — not `max(probed dims)`, but the config ceilings.

### Model construction

```python
# All three components use ceiling dims, not probed dims
encoder = NodeEncoder(max_obs_dim, hidden_dim, embed_dim)
gat     = GATPolicy(embed_dim, num_heads, out_per_head, ...)
head    = PhaseHead(gat.out_channels, max_phase_feat_dim, head_hidden)

# Per-network buffers: own N but global obs_dim
buffers = [
    ReplayBuffer(capacity, max_obs_dim, graph["node_ids"])
    for graph in graphs
]
```

Per-network `PressureReward` and `ObservationImputer` instances are created at init
(one per env) so that episode-level state (running averages) is isolated per network.

### `_select_actions` and `_compute_loss`

These are structurally identical to `DQNTrainer._select_actions` and `_compute_loss`.
The only differences are:

1. The active graph, padded_pf, and buffer are passed in as arguments (not stored as
   `self.buffer`, `self.graph`, etc.) — they vary per episode.
2. `val_stack` / `val_all[b]` is passed to `gat(...)` as `node_validity` when
   `self._neighbor_masking` is `True` — same R7 logic.

The B-loop over the batch dimension is preserved. PyG Batch optimization is post-R8.

---

## Checkpoint format

`save_checkpoint` adds three extra keys beyond `DQNTrainer`'s format:

```python
torch.save({
    "encoder":            encoder.state_dict(),
    "gat":                gat.state_dict(),
    "head":               head.state_dict(),
    "optimizer":          optimizer.state_dict(),
    "step":               total_steps,
    "cfg":                cfg_dict,
    "metrics":            metrics,
    # R8 additions:
    "max_obs_dim":        self.global_obs_dim,
    "max_phase_feat_dim": self.global_phase_feat_dim,
    "network_names":      self.network_names,
}, path)
```

`max_obs_dim` and `max_phase_feat_dim` are saved so that `load_checkpoint` can validate
the config matches the checkpoint without reprobing all networks.

### `load_checkpoint` — hard dimension assertion

```python
@classmethod
def load_checkpoint(cls, path, cfg, envs, device, network_names=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    cfg_obs_dim = cfg["model"]["max_obs_dim"]
    if ckpt["max_obs_dim"] != int(cfg_obs_dim):
        raise ValueError(
            f"Checkpoint max_obs_dim={ckpt['max_obs_dim']} does not match "
            f"cfg model.max_obs_dim={cfg_obs_dim}. Update the config or retrain."
        )
    # same check for max_phase_feat_dim

    trainer = cls(cfg, envs, device, network_names=network_names)
    trainer.encoder.load_state_dict(ckpt["encoder"])
    ...
    return trainer
```

**Why the assertion:** If someone edits `finetune_template.yaml` and accidentally changes
`max_obs_dim`, `load_checkpoint` would build a model with wrong dims and silently produce
bad results — wrong-shaped Linear layers would still initialize without error but compute
nonsense. The assertion catches this at load time with a clear error message.

---

## `metrics` output

`train()` returns a dict with these keys:

| Key | Type | Description |
|---|---|---|
| `episode_returns` | `list[float]` | Total reward per episode (summed across all nodes and steps) |
| `episode_lengths` | `list[int]` | Steps per episode |
| `losses` | `list[float]` | Huber loss per gradient step |
| `epsilons` | `list[float]` | Epsilon value at each gradient step |
| `q_mean` | `list[float]` | Mean max-Q per gradient step |
| `network_sequence` | `list[str]` | Network name for each episode, in order |

`network_sequence` enables per-network convergence curves in R9 — filter episodes where
`network_sequence[i] == "cologne8"` to get the cologne8 return curve, etc.

---

## `configs/r8.yaml`

```yaml
# R8: multi-network shared-parameter training
# TRAINING SET: cologne8 (real/EU), bologna_pasubio (real/EU different city),
#               toronto (real/NA North American grid), grid4x4 (synthetic)
# HELD-OUT eval nets (must NOT appear here): Ing1, Ing7, Ing21, Art4x4

networks:
  - net:        networks/external/RESCO/cologne8/cologne8.net.xml
    rou:        networks/external/RESCO/cologne8/cologne8.rou.xml
    begin_time: 25200
    name:       cologne8
  - net:        networks/external/bologna_pasubio/pasubio/pasubio_buslanes.net.xml
    rou:        networks/external/bologna_pasubio/pasubio/pasubio.rou.xml
    begin_time: 0
    name:       bologna_pasubio
  - net:        networks/generated/toronto/toronto_scarborough_agincourt_arterial.net.xml
    rou:        networks/generated/toronto/toronto_generated.rou.xml
    begin_time: 0
    name:       toronto
  - net:        networks/external/RESCO/grid4x4/grid4x4.net.xml
    rou:        networks/external/RESCO/grid4x4/grid4x4_1.rou.xml
    begin_time: 0
    name:       grid4x4

model:
  max_obs_dim: 64          # deployment ceiling — model built with this, not probed max
  max_phase_feat_dim: 32   # raise if adding a network that exceeds this
  ...

trainer:
  warmup_steps: 1000       # PER BUFFER — effective start ≈ warmup_steps × num_networks
  epsilon_decay_steps: 30000  # GLOBAL — counts total steps across all networks
  ...
```

**Training set rationale:**

| Network | Type | Diversity contribution |
|---|---|---|
| `cologne8` | real city, EU | reference real-city network |
| `bologna_pasubio` | real city, EU different country | different European road design |
| `toronto` | real city, NA | North American grid, very different from European layouts |
| `grid4x4` | synthetic | structural diversity, perfect regularity |

**Held-out networks (never in training):** `ingolstadt1`, `ingolstadt7`, `ingolstadt21`,
`arterial4x4`. These are the R9 cold-start evaluation targets.

**toronto prerequisite:** `toronto_generated.rou.xml` does not ship in the repo — it must
be generated once before training:

```bash
python -c "
from env.demand_generator import generate_demand
generate_demand(
    net_file='networks/generated/toronto/toronto_scarborough_agincourt_arterial.net.xml',
    output_rou='networks/generated/toronto/toronto_generated.rou.xml',
    period=2.0, duration=3600, seed=42
)
"
```

`demand_generator.py` wraps SUMO's `randomTrips.py`. Requires `SUMO_HOME` to be set.
`period=2.0` means one vehicle inserted every 2 seconds on average (1800 vehicles over
3600 s). Re-run with a different seed to get a different demand pattern.

---

## `configs/finetune_template.yaml`

The R8 checkpoint is the **factory model** — every new deployment starts from it. The fine-tune
config loads the factory model's weights and trains on a single new (possibly held-out)
network with lower LR and epsilon_start to avoid destroying pretrained representations.

```yaml
# Usage:
#   python train.py --config configs/finetune_template.yaml \
#                   --checkpoint checkpoints/r8_multi/final.pt

model:
  max_obs_dim: 64         # must match checkpoint's saved value — load_checkpoint asserts this
  max_phase_feat_dim: 32  # must match checkpoint's saved value — load_checkpoint asserts this
  ...

trainer:
  lr: 0.00005          # 6× lower than R8 training lr
  epsilon_start: 0.3   # start near-greedy — pretrained policy already reasonable
  epsilon_decay_steps: 5000
  warmup_steps: 200
  num_episodes: 50
  ...
```

**Why lower lr and epsilon_start:** The factory model already learned a reasonable policy.
Starting with `epsilon_start=1.0` would re-explore randomly and overwrite pretrained weights.
`epsilon_start=0.3` means the agent exploits its pretrained knowledge 70% of the time from
episode 1, converging faster on the new network.

---

## `train.py` changes (R8)

### Multi-network branch

`train.py` detects `cfg["networks"]` (a list). When present, it constructs one `TrafficEnv`
per entry and hands them to `MultiNetworkTrainer`. The existing `DQNTrainer` path is
completely unchanged.

```python
network_list = cfg.get("networks", None)
if network_list:
    from training.multi_network import MultiNetworkTrainer
    envs = [TrafficEnv(net_file=n["net"], route_file=n["rou"], begin_time=n.get("begin_time", 0),
                       max_steps=n.get("max_steps", cfg.get("env", {}).get("max_steps", 200)))
            for n in network_list]
    names   = [n.get("name", n["net"].split("/")[-1].replace(".net.xml","")) for n in network_list]
    trainer = MultiNetworkTrainer(cfg, envs, device=device, network_names=names)
    if args.checkpoint:
        MultiNetworkTrainer.load_checkpoint(args.checkpoint, cfg, envs, device, network_names=names)
    metrics = trainer.train()
    # ... standard checkpoint/eval/close loop
    return

# Single-network DQNTrainer path unchanged
env = TrafficEnv(...)
trainer = DQNTrainer(cfg, env, device=device)
```

### `--checkpoint` flag

Added to both paths:

```python
parser.add_argument("--checkpoint", default=None,
                    help="Path to .pt checkpoint to resume or fine-tune from")
```

Single-network path:
```python
trainer = DQNTrainer(cfg, env, device=device)
if args.checkpoint:
    trainer = DQNTrainer.load_checkpoint(args.checkpoint, cfg, env, device)
```

Multi-network path: same pattern with `MultiNetworkTrainer.load_checkpoint`.

Full CLI flag table after R8:

| Flag | Effect |
|---|---|
| `--config` | YAML config path (required) |
| `--checkpoint` | Restore weights from .pt before training (resume or fine-tune) |
| `--untyped` | Override `model.gat.typed_edges=False` |
| `--zero-hop` | Override `model.gat.zero_hop=True` |
| `--one-layer` | Override `model.gat.num_layers=1` |
| `--no-masking` | Override `model.gat.neighbor_masking=False` |
| `--episodes N` | Override `trainer.num_episodes` |
| `--eval-episodes N` | Greedy eval episodes after training (0 to skip) |

---

## Tests — `TestMultiNetworkTrainer` (10 tests)

All SUMO-free, using `MockEnv` with three networks:

| Network | N | obs_dim | pf_dim |
|---|---|---|---|
| `cross_smoke` | 1 | 15 | 4 |
| `linear_two` | 2 | 12 | 3 |
| `grid_3x3` | 9 | 15 | 4 |

`MULTI_CFG` sets `max_obs_dim=32`, `max_phase_feat_dim=8` — both above the probed maxima
(15 and 4 respectively) so the ceiling path is exercised.

| Test | What it verifies |
|---|---|
| `test_global_obs_dim_equals_config_ceiling` | `trainer.global_obs_dim == MULTI_CFG["model"]["max_obs_dim"]` (32, not the probed max 15); AND `trainer.global_obs_dim >= max(probed_dims)` |
| `test_global_phase_feat_dim_equals_config_ceiling` | Same pattern for phase_feat_dim; ceiling=8, probed max=4 |
| `test_encoder_input_size_matches_global_obs_dim` | `encoder.net[0].in_features == global_obs_dim * 2` |
| `test_head_input_size_matches_gat_out_plus_global_pf_dim` | `head.scorer[0].in_features == gat.out_channels + global_phase_feat_dim` |
| `test_per_network_buffers_have_global_obs_dim` | All 3 buffers: `buf._obs_dim == trainer.global_obs_dim` |
| `test_per_network_buffers_have_network_specific_node_count` | `buffers[0]._obs.shape[1]==1`, `[1]==2`, `[2]==9` — each buffer's N matches its network |
| `test_padded_pf_uses_global_phase_feat_dim` | Every phase feature tensor across all networks has `shape[0]==global_phase_feat_dim`; for `linear_two` (local pf=3), positions 3+ are exactly 0.0 |
| `test_pad_obs_dict_with_target_dim` | `pad_obs_dict(obs_dict_linear_two, target_dim=32)` returns `obs_dim=32`; all tensors padded to 32; padding validity==0.0 |
| `test_train_does_not_crash_and_covers_all_networks` | `random.seed(0)` then `train(9 episodes)`: all 9 returns finite; `set(network_sequence) == {"cross_smoke", "linear_two", "grid_3x3"}` — seed=0 guarantees all 3 appear in 9 picks |
| `test_gradient_steps_produce_finite_loss` | `warmup_steps=8`, `num_episodes=12`: at least 1 gradient step fires; all losses finite — confirms `[B, N_k, global_obs_dim]` batches from any of the 3 networks produce valid gradients |

---

## Running R8

```bash
# 0a. Verify imports (pre-flight)
python -c "from training.trainer import _cfg, _cfg_to_dict, _linear_decay; print('OK')"

# 0b. Generate toronto route file (requires SUMO_HOME, one-time)
python -c "
from env.demand_generator import generate_demand
generate_demand(
    'networks/generated/toronto/toronto_scarborough_agincourt_arterial.net.xml',
    'networks/generated/toronto/toronto_generated.rou.xml',
    period=2.0, duration=3600, seed=42
)
print('toronto_generated.rou.xml written')
"

# 1. R8-only tests
pytest tests/test_multi_network.py -v      # 10 tests

# 2. Full suite (364 tests)
pytest tests/ -v

# 3. Smoke run (requires SUMO)
python train.py --config configs/r8.yaml --episodes 10 --eval-episodes 0

# 4. Verify all 4 networks appear in training_metrics.json
python -c "
import json
m = json.load(open('checkpoints/r8_multi/training_metrics.json'))
print('Networks seen:', set(m['network_sequence']))
"

# 5. Full 300-episode training run
python train.py --config configs/r8.yaml

# 6. Fine-tune from R8 checkpoint onto a new network
python train.py --config configs/finetune_template.yaml \
                --checkpoint checkpoints/r8_multi/final.pt \
                --episodes 50 --eval-episodes 5
```

---

## R9 forward pointer

R9 uses the R8 factory model as the starting point for held-out evaluation.

**Cold-start evaluation:** Load `checkpoints/r8_multi/final.pt`, construct a
`MultiNetworkTrainer.load_checkpoint` against a held-out network (e.g., `ingolstadt7`),
run greedy evaluation without any fine-tuning, compare against fixed-time and MaxPressure
baselines.

**Per-network convergence curves:** `metrics["network_sequence"]` enables filtering
episode returns by network. Feed this into the evaluation suite to get separate convergence
plots for each training network — useful for diagnosing which networks are harder to learn.

**`network_sequence` in the checkpoint:** The `metrics` dict is saved inside the checkpoint
(`ckpt["metrics"]["network_sequence"]`). R9 can recover the full training trajectory from
a checkpoint without re-running training.

**What R9 needs from R8:**
- `MultiNetworkTrainer.load_checkpoint` must accept a single held-out env (not a list of
  training envs) — it already does, since `envs` is a user-provided list.
- `global_obs_dim` and `global_phase_feat_dim` from the checkpoint must be respected by
  the eval config — `load_checkpoint` asserts this.
- The evaluation suite must call `pad_obs_dict(obs_dict, target_dim=global_obs_dim)` and
  `pad_phase_features(graph, target_dim=global_phase_feat_dim)` when probing a held-out
  network — `eval_runner.py` will need to be updated to accept these dims.
