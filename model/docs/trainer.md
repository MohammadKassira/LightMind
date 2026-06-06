# trainer.md — R2 DQN Training Loop

## How the trainer consumes R0/R1

### Import order and value flow

```
env/mock_env.py          → MockEnv (R0)          provides obs_dict, graph, reward_dict
data/graph_builder.py    → build_graph (R0)       called by MockEnv internally
models/node_encoder.py   → NodeEncoder (R1)       obs → embedding
models/node_encoder.py   → pad_obs_dict (R1)      variable-length obs → fixed obs_dim
models/phase_head.py     → PhaseHead (R1)         embedding + phase_features → Q-values
models/phase_head.py     → pad_phase_features (R1) variable phase_feat_dim → fixed dim
training/reward.py       → compute_pressure (R2)  obs_dict + graph → reward_dict (used in R3+)
training/replay_buffer.py → ReplayBuffer (R2)     stores transitions
training/trainer.py      → DQNTrainer (R2)        coordinates all of the above
```

### Call sequence at startup

```python
obs_dict, graph = env.reset(seed=cfg.seed)

obs_dim, padded_obs   = pad_obs_dict(obs_dict)         # R1 utility — call once at init
phase_feat_dim, padded_pf = pad_phase_features(graph)  # R1 utility — call once at init
node_ids = graph["node_ids"]                           # fixed for this network

encoder = NodeEncoder(obs_dim, cfg.model.hidden_dim, cfg.model.embed_dim)
head    = PhaseHead(cfg.model.embed_dim, phase_feat_dim, cfg.model.head_hidden_dim)
```

`pad_obs_dict` and `pad_phase_features` are called **once at init** (not per-step) because the
network topology — and therefore obs_dim and phase_feat_dim — is fixed for the duration of
training on a single network. Per-step calls are needed only to pad the new obs_dict each step
(obs values change; shape does not).

### Per-step call sequence

```python
_, padded_obs  = pad_obs_dict(obs_dict)       # pad current obs (shape fixed, values change)
actions = _select_actions(padded_obs, ...)    # uses encoder + head
next_obs_dict, _, reward_dict, done, _ = env.step(actions)
_, padded_next = pad_obs_dict(next_obs_dict)  # pad next obs
buffer.push(padded_obs, padded_next, ...)     # store pre-padded; buffer never calls pad_obs_dict
padded_obs = padded_next                      # advance — critical, see "obs advancement" below
```

**Obs advancement:** `padded_obs = padded_next` must be the last line inside the while loop.
Without it, `_select_actions` acts on the reset observation for the entire episode after step 1.

---

## Reward wiring: mock vs real

The trainer reads `cfg.reward.use_pressure` at each step:

```python
if cfg.reward.use_pressure:
    reward_dict = compute_pressure(next_obs_dict, graph)
# else: use reward_dict from env.step() directly
```

- **R2 (`use_pressure: false`):** trainer uses MockEnv's random `reward_dict` from `env.step()`.
  `compute_pressure` is implemented and tested but not wired. The training loop mechanics
  (loss, buffer, target network) can be validated even without a meaningful reward signal.

- **R3+ (`use_pressure: true`):** flip the flag in `configs/train.yaml`. No trainer code changes.
  `compute_pressure` receives `next_obs_dict` (not `obs_dict`) because pressure is computed on
  the state reached after the action, matching the standard RL convention r(s, a, s').

`compute_pressure` operates on raw (un-padded) obs_dict — it indexes by lane position using
`num_phases` and `N_in` from the graph, which are only unambiguous on the raw obs. Do not pass
padded obs to it.

---

## Batch computation: per-node loop is unbatched

`_compute_loss` loops over nodes, then over batch elements per node:

```python
for node_i, node_id in enumerate(graph["node_ids"]):
    ...
    for b in range(B):
        emb = encoder(obs[b, node_i, :], val[b, node_i, :])  # (embed_dim,)
        q   = head(emb, padded_pf[node_idx], all_true_mask)   # (num_phases,)
```

This is intentionally unbatched for R2. `PhaseHead.forward` takes a 1D `(embed_dim,)` embedding
and a list of `(phase_feat_dim,)` tensors — it cannot accept a batch dimension without modification.

**R4 will add `forward_batch(emb_batch, phase_stack)`** to `PhaseHead`. At that point the inner
loop can be vectorised. Do not attempt to vectorise or modify `PhaseHead` in R2 — the R1 tests
must continue passing unchanged.

---

## Target network masking

When computing the Bellman target:

```python
# CORRECT — all-True mask for target network
all_true = torch.ones(num_phases, dtype=torch.bool)
q_next = target_head(target_emb, padded_pf[node_idx], all_true)
q_target = reward + gamma * q_next.max() * (1 - done)

# WRONG — real mask for target network (do not do this)
q_next = target_head(target_emb, padded_pf[node_idx], real_mask)  # -inf → NaN in max()
```

The real `valid_transition_mask[current_phase]` is used **only** in `_select_actions` — it
prevents selecting illegal phase transitions at inference time. During the Bellman backup,
the target network must see all phases as candidates for the next-step max, otherwise `-inf`
values from the mask propagate into `max()` and produce NaN loss that poisons the entire
parameter update.

The all-True mask is the **only** place in the codebase where an all-True mask is used.

---

## R2 checkpoint = 0-hop ablation baseline

At the end of R2 training, the saved checkpoint is the **0-hop ablation baseline** required
for every later ring (R4–R9). "0-hop" means: shared-parameter NodeEncoder + PhaseHead trained
with DQN, no graph message passing.

**What is saved:**

```python
torch.save({
    "encoder": encoder.state_dict(),
    "head":    head.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step":    total_steps,
    "epsilon": current_epsilon,
    "metrics": metrics_history,
}, path)
```

**Path:** `checkpoints/r2_0hop/checkpoint_ep<N>.pt`

**Why this checkpoint matters for R4–R9:**
- R4 adds the first GAT layer. To claim "graph helps," GAT must outperform 0-hop on the
  coordination scenario. The R2 checkpoint is the baseline for that comparison.
- R5–R9 each strengthen the graph; each ring's ablation table shows delta vs 0-hop.
- The checkpoint must be loadable by `DQNTrainer.load_checkpoint` and by the R9 evaluation
  suite without modification to the checkpoint format.

**Loading:**

```python
trainer = DQNTrainer.load_checkpoint("checkpoints/r2_0hop/checkpoint_ep500.pt", cfg, env)
actions = trainer._select_actions(padded_obs, graph, padded_pf, epsilon=0.0)
```
