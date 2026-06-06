# Build Spec — GAT Traffic-Signal Controller + Baselines
## Agent Implementation Document (this repo only)

> **For the coding agent.** Build only what is in this document. This repo covers the **GAT model and the baselines**, plus the minimum data/env/training/evaluation needed to train them and prove the model beats the baselines. **Serving, deployment, FastAPI, Docker, watchdog, imputation, monitoring, lifecycle, and inference-topology are OUT OF SCOPE — they live in a separate repo.** Build in the §2 ring order. Lock §3 contracts before model code. Do not invent behavior not specified here; if a decision is marked open in §11, ask rather than assume.

---

## 0. Repo scope (read first)

**IN this repo:**
- Networks → graph construction + observation encoding
- Environment interface (SUMO/TraCI wrapper) + a mock env for offline model dev
- Reward (efficient pressure)
- **The GAT model**
- **Baselines:** Fixed-Time, MaxPressure, per-agent Independent DQN, 0-hop GAT ablation
- Training loop (DQN)
- Evaluation: beat-baselines table, graph-ablation, diagnostic scenarios, convergence curves, transfer
- **Perception / sensor-degradation model** — included, but sequenced as a later ring (§2 R3). It is required for the robustness/degradation results; if the team decides robustness belongs to a different effort, it can be lifted out cleanly (it is a single module behind the env).

**OUT of this repo (separate effort — do NOT build):**
- FastAPI server, `/predict` and all API endpoints
- Docker / docker-compose / containerization
- Watchdog, tiered fallback wiring, imputation service, drift monitoring service
- Deployment lifecycle (cold-start / adaptation / frozen) as a running service
- Inference-topology / stale-embedding-caching infrastructure
- Online-learning endpoints

(The model must be *exportable* and have a clean `predict(graph, obs) -> actions` function so the other repo can wrap it — but this repo does not serve it.)

---

## 1. Goal (what "done" means for this repo)

Produce a trained GAT controller and trained baselines, plus evaluation showing:
1. **The GAT beats all baselines** on average waiting time and throughput.
2. **The graph is justified** — the GAT beats the 0-hop ablation (same architecture, message passing removed), cleanly attributable to the graph alone.
3. *(If perception ring is built)* **The margin holds under sensor degradation** — GAT stays above MaxPressure across a severity sweep.

**Non-goals:** SOTA performance, fast training. We care about the **converged ceiling**, and accept long training. We make **no sample-efficiency claim** for the graph (we train everything to convergence and compare asymptotes).

---

## 1.1 CRITICAL CONSTRAINT — network-agnostic, config-driven

**Never hardcode a network.** No network name, node count, intersection count, lane count, or phase count may appear as a literal in model, training, or evaluation code. All network choice is Hydra config. Networks named anywhere here (RILSA, Cologne, NxN grids, etc.) are **non-binding placeholders** for config files. The model reads counts (phases, lanes, neighbors) from the graph/observation at runtime. Hard requirement on the training set is parametric only: **≥3 networks of varying size and connectivity**, supplied by config.

---

## 2. Build Order (rings — do not start a ring until the prior "done-when" passes)

Build inside-out. The no-graph core comes first because it **is** the 0-hop ablation baseline. Keep every ring's checkpoint (ablations come from this for free).

| Ring | Build | Done-when |
|---|---|---|
| **R0** | Lock §3 contracts; build `mock_env.py` (stub graph + shaped fake observations) | A dummy forward pass runs against the agreed schema, no SUMO needed |
| **R1** | `node_encoder.py` + `phase_head.py` (FRAP-style), **single intersection, no graph** | Forward pass on one intersection; phase scores respond sensibly to handcrafted inputs |
| **R2** | DQN loop (`trainer.py`, `replay_buffer.py`), single intersection | Trains stably; beats Fixed-Time; no divergence. **Save checkpoint = 0-hop ablation baseline** |
| **R3** | `perception.py` on in the loop (crude is fine) | Still converges under noise; handles validity-flag/sentinel inputs without NaN |
| **R4** | **One** GAT layer, single untyped edge type | Trains on a small multi-intersection net; ideally beats the 0-hop checkpoint on a coordination scenario. **MAKE-OR-BREAK: if 1-hop never beats 0-hop, stop and debug before continuing** |
| **R5** | Second GAT layer (2-hop) + multi-head | Trains without embedding collapse / over-smoothing |
| **R6** | Typed edges (2 types, separate weight matrices) | Trains; typed > untyped on a mini-ablation |
| **R7** | Neighbor masking (failed/stale/unreachable neighbors via validity flag) | Under dropout, attention on masked neighbors drops; degrades gracefully |
| **R8** | Multi-network shared-parameter training | One param set trains across ≥3 configured networks without crashing |
| **R9** | Baselines (§7) + evaluation suite (§8) | Full results table + graph ablation + diagnostic scenarios + (if R3 built) degradation sweep |

---

## 3. Data Contracts (lock BEFORE model code)

### 3.1 Normalization constants — single source of truth (`configs/normalization.yaml`)
Define once; every module imports from here. Never redefine locally.
```yaml
q_max: <int>           # lane capacity in vehicles; queue / q_max -> [0,1] ("how full")
v_max: <float>         # free-flow lane speed (m/s); lookahead L = v_max * t_duration
t_duration: <float>    # phase window (s) for the running-vehicle lookahead
max_phase_time: <float> # normalizes time-in-current-phase -> ~[0,1]
```

### 3.2 Per-intersection observation vector (identical for GAT and Independent DQN)
The env returns `obs_dict: dict[str, tuple[Tensor, Tensor]]` — always unpack as `(obs, validity) = obs_dict[node_id]`. Never index as just `obs_dict[node_id]` or you get the tuple.
```
obs(node) = concat(
  phase_onehot,                                  # len = node.num_phases  ← segment 0
  [ time_in_phase / max_phase_time ],            # 1 scalar               ← segment 1
  per incoming lane: [ queue/q_max, running_within_L/q_max ],   # 2 per incoming lane
  per outgoing lane: [ queue/q_max ],                           # 1 per outgoing lane
)
validity(node) = same-length float32 mask; 1.0 = real reading, 0.0 = no data
```
Total length: `num_phases + 1 + 2×num_incoming + num_outgoing`. **Varies per node** — never hardcode. Derive from graph at runtime.
- Running vehicles **incoming lanes only** (upstream = actionable future arrivals; downstream running vehicles are gone and duplicate the outgoing-queue signal).
- To recover the current phase: `current_phase = obs[:num_phases].argmax().item()` — this is needed to index `valid_transition_mask` correctly.

### 3.3 Missing-sensor convention (CRITICAL — perception, masking, imputation all depend on it)
- Missing ≠ zero. Zero means "empty lane" = real info.
- Missing = **sentinel `-1.0`** (from `configs/perception.yaml` key `perception.sentinel_value`) + `validity = 0.0`.
- Consumers branch on the validity flag, **never** the value: `missing = (validity == 0.0)`.
- `validity` is `float32` (not bool). Branch on `== 0.0`, not `== False`.
- The sentinel can corrupt any segment including `phase_onehot` — never assume the one-hot property holds when validity has zeros. Always check validity first.
- The `NodeEncoder` zeros out sentinels via `obs * validity` before the first linear layer — the sentinel never reaches model weights.

### 3.4 Graph schema (PyTorch Geometric compatible; use PyG `MessagePassing`, don't hand-roll scatter/gather)
```
graph = {
  node_features: Tensor[num_nodes, F]   # variable F handled per-node; absent from graph dict — added by env each step
  validity:      Tensor[num_nodes, F]   # absent from graph dict — added by env each step
  edge_index:    LongTensor[2, num_edges]   # [src, dst]; flow edges first, then coordination
  edge_type:     LongTensor[num_edges]      # 0 = flow (upstream→downstream), 1 = coordination (downstream→upstream)
  phase_features: list[list[FloatTensor]]   # [node_idx][phase_idx] → FloatTensor[num_incoming_lanes]
                                            # STATIC HALF ONLY: 1.0 = green, 0.0 = red for that lane
                                            # complete phase feature = this static mask + live pressure (concatenated at decision time)
                                            # lane count varies per node — call pad_phase_features(graph) before constructing PhaseHead
  node_meta:     list[dict]   # per-node: num_phases (int), valid_transition_mask (BoolTensor[P,P] all True)
}
```
- `node_features` and `validity` are **absent** from the graph dict — this is intentional, not a bug. They are added separately at each env step by the observation encoder.
- The graph is returned **by reference** from `reset()` and `step()` — do not mutate it.
- `valid_transition_mask[i]` gives legal transitions **from** phase `i`. Index with `current_phase` from obs, not a hardcoded constant.

### 3.5 Environment step API
```
env.reset(network_cfg, seed) -> (obs_dict, graph)
env.step(actions: dict[node_id -> phase_idx]) -> (obs_dict, graph, reward: dict[node_id -> float], done, info)

obs_dict: dict[str, tuple[FloatTensor, FloatTensor]]   # node_id -> (obs, validity)
```
- Always unpack: `obs, validity = obs_dict[node_id]`.
- When perception is enabled, `obs` is already post-perception. A `clean_mode` flag exposes ground truth (for the clean baseline of the degradation sweep only).
- `info` is always `{}` in the mock; carries ground-truth metrics in `traffic_env` for evaluation logging (NOT visible to the agent/model).
- The graph is the same object every call — do not mutate it.

### 3.6 Model inference contract (so the separate serving repo can wrap it)
Expose a pure function: `predict(graph, obs) -> {node_id: phase_idx}`. No global state, no I/O. This is the only seam the deployment repo needs.

---

## 4. Tech Stack (pin these)
Python ≥3.10 · PyTorch · **PyTorch Geometric** (graph layers + batched variable-size graphs) · SUMO + **TraCI** (per-vehicle access for perception) · **Hydra** YAML config · TorchScript/ONNX export optional (model is tiny). No FastAPI/Docker in this repo.

---

## 5. Reward (`training/reward.py`)
Local efficient pressure (Advanced-XLight), single term, no weighting, computed from observed (post-perception) state:
```
r_i = - P_efficient_i
P_efficient_i = | sum_movements ( q_in/q_in_max - q_out/q_out_max ) |   (averaged over movements)
```
Stability is from env-level constraints (min green, masking), **not** reward shaping.
Optional CO₂ term `- λ·CO2_i` behind a config flag, **off by default**.
*Done-when:* finite under missing-data inputs; pressure direction matches MaxPressure on hand-checked cases.

---

## 6. Model (`models/`)

### 6.1 Node encoder (`node_encoder.py`)
Per-node obs vector → MLP → fixed-size node embedding. Consumes the validity mask.

**Helpers exported from this module:**
- `pad_obs_dict(obs_dict) -> (obs_dim, padded_obs_dict)` — call once before constructing `NodeEncoder`. Pads all per-node obs tensors to the max length in the network; padding positions get `obs=0.0, validity=0.0`. Returns the padded length and the new dict. Re-call every step — `obs_dim` is stable (graph doesn't change), so the same encoder instance works throughout.

**Constructor:** `NodeEncoder(obs_dim, hidden_dim=128, embed_dim=64)`
- `obs_dim` comes from `pad_obs_dict` — never hardcode it.
- `hidden_dim=128` matches GAT layer-1 output; `embed_dim=64` matches GAT layer-2 output (§6.3).

**Forward:** `forward(obs, validity) -> Tensor`
- `obs`: `(obs_dim,)` or `(batch, obs_dim)` — pre-padded, may contain sentinel `-1.0`.
- `validity`: same shape, float32 0/1.
- Internally: `masked_obs = obs * validity` before the first layer (sentinel never reaches weights).
- Returns `(embed_dim,)` or `(batch, embed_dim)`, always float32, never NaN.

**Architecture:** `Linear(obs_dim, 128) → ReLU → Linear(128, 64) → ReLU`

### 6.2 Phase-scoring head (`phase_head.py`) — FRAP-style
Score each candidate phase by its features instead of emitting Q-values over fixed action indices.

**Helpers exported from this module:**
- `pad_phase_features(graph) -> (phase_feat_dim, padded_phase_features)` — call before constructing `PhaseHead`. Pads all phase feature vectors to the max `num_incoming_lanes` across all nodes; padding value `0.0` (= red, no spurious activations). Returns padded length and new nested list.

**Constructor:** `PhaseHead(embed_dim, phase_feat_dim, hidden_dim=64)`
- `phase_feat_dim` comes from `pad_phase_features` — never hardcode it.
- One `PhaseHead` instance serves all nodes because all phase vectors are now the same padded length.

**Forward:** `forward(node_embedding, phase_features, valid_transition_mask) -> Tensor`
- `node_embedding`: `(embed_dim,)` — from `NodeEncoder`.
- `phase_features`: `padded_phase_feats[node_idx]` — list of `(phase_feat_dim,)` tensors.
- `valid_transition_mask`: `BoolTensor(num_phases,)` — **index from the P×P mask by current phase**: `mask = graph["node_meta"][node_idx]["valid_transition_mask"][current_phase]` where `current_phase = obs[:num_phases].argmax().item()`. Never hardcode `[0]`.
- Returns `(num_phases,)` scores; invalid phases = `-inf`.

**Helper:** `select_action(node_embedding, phase_features, valid_transition_mask) -> int` — greedy argmax under `no_grad`. Used by the DQN rollout; caller handles ε-greedy by replacing with a random valid phase at rate ε.

**The phase score is the Q-value.** No separate value head.

**Architecture (per phase):** `concat(embedding, phase_feat) → Linear(embed_dim+phase_feat_dim, 64) → ReLU → Linear(64, 1) → squeeze`. Same MLP weights across all phases and intersections.

**Note for R4:** `forward` currently expects `(embed_dim,)` not `(batch, embed_dim)`. A batched path `forward_batch` must be added in R4 for efficient loss computation — see R2 trainer docs.

### 6.3 GAT backbone (`gat_policy.py`)
- **2 layers** → 2-hop receptive field (do not add layers to "go deeper" — over-smoothing).
- Layer 1: 4 heads × 32 → concat 128. Layer 2: 4 heads × 16 → concat 64.
- **Typed edges:** 2 edge types each with own weight matrix; attention per edge type, aggregated across types.
- **Neighbor masking:** mask attention to neighbors with validity 0 / stale / unreachable. (This is the robustness lever and prevents a dead neighbor poisoning node *i* through message passing.)
- **Shared parameters across all intersections** (inductive — no per-node weights).
- **`zero_hop` flag:** when true, skip message passing entirely → this configuration is the ablation baseline (§7).
*Done-when:* runs on batched variable-size graphs; masking a neighbor measurably lowers its attention weight; typed > untyped on the mini-ablation; embeddings don't collapse to uniform.

---

## 7. Baselines (`models/`)
Four baselines beneath the GAT. They isolate different variables — build all four.

| Baseline | What it is | Transfers? |
|---|---|---|
| Fixed-Time | Static cycle, demand-blind. Floor. | n/a |
| MaxPressure (~80 LOC) | Greedy rule-based: pick phase maximizing pressure. Strong competitor. | n/a |
| Independent DQN (per-agent) | One MLP-DQN **per intersection**, own weights, own replay/target, sees only its own obs. Strongest non-graph *local* learner. | No |
| 0-hop GAT | The `gat_policy` with `zero_hop=True` (message passing off). Clean single-variable "does the graph matter" control. | Yes |

### 7.1 Independent DQN (`independent_dqn.py`) — detail
- Per-agent: each intersection trains its own network from its own observations only.
- **Identical** observation features, normalization, reward, and **same degraded inputs under the perception model** as the GAT (it just has no graph to impute with). If its inputs or noise differ from the GAT's, the comparison is invalid.
- Plain MLP: input → 2 hidden → Q per phase. Action masking on invalid transitions; min-green enforced at env level.
- Input size fixed per-intersection (no cross-network padding; it does not transfer). Trained to convergence **per-network**.
- **Must NOT have:** neighbor features, edges, message passing, parameter sharing, transfer.

### 7.2 Comparison hygiene
All learning methods use **DQN** — never put the GAT on a different algorithm than its baseline (that changes two variables). Train everything **to convergence**; compare **asymptotic** performance.

---

## 8. Evaluation (`evaluation/`)

### 8.1 Metrics (`metrics.py`)
Average waiting time, throughput, total queue length, stability (phase-change rate), **plus tail metrics: p95 queue, spillback events, max wait** (worst case causes gridlock; means alone mislead). Report mean ± std over 1–3 seeds (5 is overkill).

### 8.2 Graph ablation (primary "does the graph matter" result)
Full GAT vs 0-hop GAT, trained to convergence on the same networks. Win attributable to message passing alone. Optionally also vs per-agent Independent DQN on fixed networks to show even the strongest local baseline loses.

### 8.3 Diagnostic scenarios (`diagnostic_scenarios.py`)
Run all four baselines + GAT. **Selection rule:** a scenario counts only if doing well *requires cross-intersection information* (else it tests adaptivity, not the graph).
- **Hero (expect graph wins):** oversaturation with spillback (gating/metering; breaks MaxPressure's no-blocking assumption); platoon anticipation on long links; corridor green-wave under directional demand; *(if perception built)* blind-node-under-high-demand.
- **Control / honesty (expect parity, state it):** uniform moderate demand (≈ MaxPressure); single isolated intersection (graph ≈ Independent DQN by construction); extreme one-sided demand → starvation **guardrail** check (fairness from constraints, NOT a graph win).
- **Discriminating knobs:** longer inter-intersection links, spatial/temporal demand correlation, capacity asymmetry. Crank → graph edge grows; flatten → parity.

### 8.4 Convergence curves
Performance vs episodes, per method per network. **Framing:** curves show (a) each method *converged* and (b) the graph plateaus *higher* — they do NOT claim faster training. Per-agent Independent DQN has per-network curves only; GAT and 0-hop GAT also have cross-network curves.

### 8.5 Degradation sweep (`degradation_sweep.py`) — only if R3 (perception) is built
All methods under the SAME perception severity, sweeping severity on x-axis vs performance on y-axis. Deliverable: GAT curve staying above MaxPressure's across the sweep. This is the deployability/robustness result.

### 8.6 Transfer (`transfer_eval.py`) — supporting, not headline
On one config-chosen held-out network: cold-start + fine-tuning curve. Frame as good-init + fast-enough convergence, NOT zero-shot. Single held-out net = weak sample; bonus result, don't stake conclusions on it.

---

## 9. Repository Structure (this repo)
```
traffic_rl/
├── configs/                  # Hydra — ALL network choice + hyperparams + normalization
│   ├── normalization.yaml
│   ├── networks/             # placeholder/example network configs (non-binding)
│   ├── train.yaml
│   └── perception.yaml
├── data/
│   ├── networks/             # SUMO nets, config-referenced, NOT hardcoded
│   ├── graph_builder.py      # net.xml → typed-edge graph (3.4)
│   └── observation_encoder.py# raw state → obs vector (3.2)
├── env/
│   ├── traffic_env.py        # SUMO/TraCI wrapper, step API (3.5)
│   ├── perception.py         # structured sensor model, severity knob (R3)
│   └── mock_env.py           # R0 stub for model dev without SUMO
├── models/
│   ├── node_encoder.py
│   ├── phase_head.py
│   ├── gat_policy.py         # full GAT; zero_hop flag = ablation baseline
│   ├── independent_dqn.py    # per-agent non-graph baseline
│   └── max_pressure.py       # rule-based baseline
├── training/
│   ├── trainer.py            # DQN loop, ε-greedy, target net
│   ├── replay_buffer.py
│   ├── reward.py             # efficient pressure
│   └── multi_network.py      # cross-network episode orchestrator
├── evaluation/
│   ├── metrics.py            # incl. tail metrics
│   ├── diagnostic_scenarios.py
│   ├── degradation_sweep.py  # if perception built
│   ├── transfer_eval.py
│   └── ablations.py
├── configs/, notebooks/, tests/
└── README.md
```
*(No `serving/` and no `deployment/` directory in this repo — those are the separate effort.)*

---

## 10. Hyperparameters (defaults — Hydra-overridable)

| Param | Default |
|---|---|
| Replay buffer | 50,000 |
| Target net update | every 1,000 steps |
| Batch size | 64 |
| ε schedule | 1.0 → 0.05 over 10,000 steps |
| Min green | 7 s (env-enforced) |
| GAT layer 1 / 2 | 4×32→128 / 4×16→64 |
| Edge types | 2 (flow, coordination) |
| Optimizer / LR | Adam, start 1e-3 → 3e-4 (tune) |
| Seeds | 1–3 |
| Training networks | ≥3, varying size/topology (config) |
| Algorithm | DQN (no MAPPO in this repo) |

---

## 11. Open decisions — ask before assuming
- **Is perception (R3 + §8.5) in this repo or the separate effort?** Default assumption: in this repo, sequenced late. Confirm.
- **Exact network set + held-out network:** chosen later by the team; code must not depend on the choice (§1.1).
- **Sentinel value for missing data** (§3.3): pick and document one.

---

## 12. Hard rules (don't violate)
1. No hardcoded networks/counts/sizes anywhere (§1.1).
2. GAT and Independent DQN see **identical** features, normalization, reward, and noise — only neighbor access differs.
3. 0-hop GAT = full GAT with `zero_hop=True`; differ in exactly one thing.
4. All learning methods on DQN; train to convergence; compare asymptotes (no speed claims).
5. Missing data uses sentinel + validity flag, never zero.
6. Build ring-by-ring; don't start a ring until the prior done-when passes; keep every checkpoint.
7. No serving/deployment code in this repo — expose `predict(graph, obs)` and stop there.
