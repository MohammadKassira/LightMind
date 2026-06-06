    # Graph-Structured RL for Urban Traffic Signal Control
## Architecture & Implementation Plan — v2 (Internal Team Reference)

**Team** Mohamad Al Aalami · Hasan Haidar · Mohamad Kassira
**Course** EECE 490 (or equivalent — graded on rigor)
**Project type** Application-oriented ML system (Option A)
**Timeline** 8 weeks (1–2 month build)
**Status** Current consolidated plan (v2 + all modifications: 4-baseline structure, ceiling-via-coordination framing, config-driven networks, model build-order rings). This is the human reference we go back to; the agent build spec is a separate scoped doc.

---

### How v2 differs from v1, in one paragraph

v1 was a clean design but it quietly assumed perfect, ground-truth state appears for free — which is exactly what SUMO hands you and exactly the trap. Our actual goal is **sensors/cameras → decision**, deployed as close to real life as we can get in the pipeline. v2 puts the missing layer — the one that turns messy sensor reality into model input — at the center, makes robustness to that messiness our headline claim, and resolves the algorithm/edge/CTDE questions we'd left open. The model itself does not need to be SOTA. It needs to **beat every baseline, justify the graph, and survive realistic input.** Everything below serves those three things.

---

## 0. What we are optimizing for (read this first)

We are **not** chasing SOTA performance. Our goal is a controller that:

1. **Beats all baselines** (Fixed-Time, MaxPressure, Independent DQN) on average waiting time and throughput.
2. **Justifies the graph structure** — the GAT must measurably beat the non-graph baseline, cleanly attributable to the graph alone.
3. **Is genuinely deployable** — survives the messy sensor→decision pipeline, not just clean simulator state.

From goal #3, two concrete optimization targets fall out, and everything in this plan traces back to one of them:

- **Robustness to structured input degradation.** Real cameras and detectors produce biased, correlated, regime-dependent errors — not zero-mean noise. The model must hold its margin over baselines as input quality drops.
- **Inference latency within the control cycle.** The decision must return inside the signal's timing window (our watchdog budget is 500 ms). Compute is cheap for our model; the real latency cost is communication between intersections, which we design around explicitly.

Fast fine-tuning, graceful fallback, and decision logging are all *consequences* of getting these two right. If the model is robust and fast, the deployment story holds. If either breaks, nothing else saves it.

---

## 1. Project Goal & Framing

### What we are building
A graph-based RL controller for urban traffic signals that is **deployment-aware**, not just a simulation experiment. Dropped onto a new road network, it behaves reasonably out of the box and fine-tunes to local conditions over the first few episodes — **while running on realistically noisy, occasionally-failing sensor input.**

### What we are NOT building
- **Not a zero-shot system.** We don't claim it works perfectly on unseen networks immediately. We claim good prior knowledge that adapts quickly.
- **Not a research paper.** No CoLight reproduction, no 5-seed significance runs, no full ablation matrices. We are graded on a working product.
- **Not production-grade infrastructure.** NTCIP protocols, formal safety verification, certified hardware — multi-year efforts beyond scope. We document the gap honestly.

### The product story (one paragraph)
We built a graph-based RL controller trained on a diverse set of networks **and under a realistic sensor-degradation model**, so it learns general traffic-control principles rather than memorizing layouts *or* over-trusting clean inputs. On a new network it starts with reasonable behavior and fine-tunes within a few episodes to match or exceed strong baselines. Our deployment layer makes this practical: the model loads, runs safely with tiered fallback, imputes missing sensor data through the graph, logs decisions, monitors its own health, and adapts to local conditions automatically.

### What success looks like
- Beats Fixed-Time and MaxPressure on average waiting time and throughput on training networks.
- **Holds its margin over baselines under realistic sensor degradation** — measured by a clean-vs-degraded severity sweep (this is our headline result).
- Cold-start performance on a held-out network (Ingolstadt7) is reasonable (better than untrained baseline).
- Fine-tuning on the new network converges within ~10 episodes to match or exceed MaxPressure.
- Deployment artifact is a working Docker container with FastAPI endpoints, **tiered watchdog fallback (model → MaxPressure → fixed-time)**, sensor imputation, health monitoring, and decision logging.
- Codebase cleanly separated into data / model / serving layers with reproducible experiments.

---

## 2. Why This Architecture (Derived From Requirements)

We did not pick the architecture because a paper said so. We derived it from requirements. Each requirement eliminates options and points toward others.

| Requirement | Implication |
|---|---|
| Must work on any road network topology | Inductive GNN with shared parameters — no per-node memorization |
| Must handle variable network sizes | Shared parameters across intersections (no fixed N) |
| Must handle variable neighborhood sizes | Message-passing GNN that aggregates over arbitrary neighbor counts |
| Must capture spatial dependencies | At least 1-hop neighbor aggregation; ideally 2-hop |
| **Must degrade gracefully under noisy / failing sensors** | **Attention-based aggregation so a node can downweight a noisy or dead neighbor (favors GAT over GCN); structured perception model in training; graph-based imputation for missing nodes** |
| Must not oscillate phases | Minimum green-time + action masking (simulator level, not reward) |
| Must converge in reasonable time | DQN with parameter sharing; avoid hierarchical/transformer methods |
| **Must run inside the control cycle on edge hardware** | **Lightweight GAT (compute is a non-issue); the real cost is message-passing communication, so we commit to an explicit inference topology** |
| Must be callable by external systems | FastAPI server in Docker |
| Must fall back safely on failure | Tiered watchdog between model and signal controller |

**What this leaves us with:** a Graph Attention Network with shared parameters, trained with DQN, using a pressure-based reward, trained and evaluated **through a structured perception model**, served via FastAPI in a Docker container with a **tiered watchdog** and **graph-based imputation**. Not a creative choice — it's what the requirements force.

**Note on attention's real job here:** the GAT-over-GCN decision is not cosmetic. Attention is our primary *sensor-robustness* lever — it lets a node learn to downweight a neighbor whose reading is noisy or stale. That capability only materializes if we train under the failure distribution (see §5).

---

## 3. Model Architecture

### 3.1 Inputs (per intersection)
Following Advanced-XLight's finding that state representation matters more than architectural sophistication:

- Current phase (one-hot)
- Time in current phase (normalized)
- For each incoming lane: queue length + effective running vehicles within `L = Vmax × t_duration`
- For each outgoing lane: queue length only

**Why running vehicles only on incoming lanes:** upstream running vehicles are future arrivals (actionable); downstream ones are already gone and duplicate the outgoing-queue signal while adding noise.

### 3.1.1 Sensor representation contract (locked — implemented in R0)
The env returns `obs_dict: dict[str, tuple[FloatTensor, FloatTensor]]`. Always unpack as `(obs, validity) = obs_dict[node_id]`.

- `validity` is **float32** (not bool). `1.0` = real reading, `0.0` = no data. Branch on `== 0.0`, not `== False`.
- A failed/missing reading is **not** zero (zero = "empty lane" = real info). It is the **sentinel `-1.0`** (from `configs/perception.yaml`, key `perception.sentinel_value`) plus `validity = 0.0`.
- The sentinel can corrupt any segment including `phase_onehot` — never assume one-hot property holds when validity has zeros.
- `NodeEncoder` zeros sentinels via `obs * validity` before the first linear layer — sentinel never reaches model weights.
- The imputation layer (serving repo) consumes the validity flag to substitute last-known or neighbor-derived values at deployment.

This contract is locked and implemented. Do not redefine the sentinel value anywhere — it lives only in `configs/perception.yaml`.

### 3.2 Graph Structure
- **Nodes:** signalized intersections only (not vehicles, not lanes — keep it simple).
- **Edges:** directed, two types — (1) upstream→downstream for traffic flow, (2) downstream→upstream for signal coordination.
- **Each edge type has its own learned weight matrix** (from IG-RL) — this is what gives the graph semantic meaning instead of plain neighbor lookup.

### 3.3 GAT Backbone (2 layers)
- Layer 1: 4 heads × 32 dim → concat to 128
- Layer 2: 4 heads × 16 dim → concat to 64
- Attention computed per edge type, then aggregated across types.
- 2 layers → 2-hop receptive field — enough to capture congestion propagation without over-smoothing.

**Execution is not fully decentralized — and we design for that.** Message passing pulls neighbor embeddings into node *i*, so a neighbor's failure *propagates into i* unless we suppress it. The same mechanism that lets the graph fill in a blind node (neighbors carry information about it) is also a path for bad data to spread. **Attention masking of unreachable/stale/failed neighbors is what resolves this tension** and must be built into the forward pass, not bolted on.

### 3.4 Phase Scoring Head (FRAP-inspired, implemented in R1)
Instead of Q-values over a fixed action-index array (where "action 5" means different things at different intersections), we **score each candidate phase by its features**:

- Each candidate phase = a feature vector (which lanes go green, current pressure on those lanes).
- Score function: `f(concat(node_embedding, phase_feature_vector), θ) → scalar` — a small shared MLP.
- Argmax across valid phases, with `valid_transition_mask` applied as `-inf` on invalid phases before argmax.
- **`valid_transition_mask` must be indexed by current phase from obs**: `mask = node_meta[i]["valid_transition_mask"][current_phase]` where `current_phase = obs[:num_phases].argmax().item()`. Never hardcode `[0]`.
- **`pad_phase_features(graph) -> (phase_feat_dim, padded_phase_features)`** exported from `models/phase_head.py`. Call before constructing `PhaseHead` — pads all phase feature vectors to the max `num_incoming_lanes` in the network so one `PhaseHead` instance serves all intersections (same pattern as `pad_obs_dict`).
- **`phase_features` in the graph dict is the static half only** (1.0=green lane, 0.0=red). The complete phase feature = static mask + live pressure concatenated at decision time.
- **Why it matters:** the same model works on intersections with different phase configurations because it scores phase *semantics*, not phase indices. This is a one-way architectural door.

---

## 4. Reward Function

Our reading revealed a clear pattern: even when waiting/travel time is the optimization goal, the reward never uses it directly. CoLight, PressLight, MPLight, Advanced-XLight, IG-RL — none use waiting time in the reward. They use queue length or pressure as proxies.

**Why pressure-based reward is correct:**
- **Local credit assignment.** Pressure is a property of one intersection; waiting time is global (smeared across many signals).
- **No reward delay.** Pressure is observable now; true waiting time is only known when a trip ends.
- **Theoretically grounded.** MaxPressure has a guarantee: minimizing pressure maximizes throughput, which bounds waiting time under flow conservation.

**Our reward:**
```
r_i = − P_efficient_i
P_efficient_i = | Σ_movements ( q_in/q_in_max − q_out/q_out_max ) |
```
(efficient pressure from Advanced-XLight, averaged over movements)

Single term. No artificial weighting. Stability is handled at the simulator level via a 7-second minimum green and action masking — **not** via reward shaping. This is what every paper we read does; we don't deviate without strong reason.

**Optional multi-objective extension (Week 7 stretch):** add CO₂ as a secondary local term `r_i = −P_efficient_i − λ·CO2_i` (the only multi-objective form the literature supports — Graph Cooperation DRL, Yan et al. 2022). Stretch goal, not core.

---

## 5. Perception / Observation Model — *the layer where our goal lives* (NEW, CENTRAL)

This is the single biggest addition over v1, and it is the part of the project that makes "deployable" mean something. v1's noise injection started from ground truth and degraded it with Gaussian noise — but real perception errors are **structured, biased, correlated, and regime-dependent**, and they are born *per vehicle* before being summed into our state. We model them at that level so the realistic noise *emerges* from aggregation instead of being faked on the aggregate.

### 5.1 The pipeline we are modeling
`camera frame → detector → tracker → lane assignment → count`. Errors originate per-detection and accumulate into our per-lane state. So instead of reading SUMO's `queue_length`, we iterate over the actual vehicles on a lane (via TraCI `getLastStepVehicleIDs`, `getLanePosition`) and decide, **per vehicle**, whether the "camera" sees it and which lane it's assigned to. We aggregate the survivors. That observed count is what the model gets.

### 5.2 Per-vehicle detection model
`p_detect` is a product of factors:
- **Distance decay** — cameras see the stop line far better than the tail of a long queue.
- **Occlusion** — `p_detect` falls with the number of vehicles between this one and the camera (more for large vehicles).
- **Environment multiplier** — the episode's weather/lighting regime scales everything.

This alone buys the most important realistic property: **undercounting that grows with congestion**, and is *biased* (occlusion only removes vehicles). Zero-mean Gaussian can never produce this.

**Lane misassignment:** each detected vehicle goes to its true lane with probability `1−ε`, else bleeds into an *adjacent* lane (higher ε near lane boundaries and in the far field). Structured leakage, not independent per-lane noise.

### 5.3 Temporal & regime layers (on top of detection)
1. **Per-episode regime.** Sample a condition at episode start — clear / rain / fog / night / low-sun glare / partial failure — fixed for the episode. Sets the detection multiplier, latency, and drop rate. This makes errors correlated *across all sensors at once* and *persistent* (rain doesn't flicker frame to frame).
2. **Latency / staleness.** Model acts on state from `t−Δt` (capture→detect→transmit takes time). A small delay buffer. Decisions lag reality — itself a deployment failure mode worth showing.
3. **Frame drops + hard failure.** Per-step probability of no new reading → hold last-known value, state goes stale; drops can come in bursts. Separately, low-probability sustained full sensor failure — this is the case that feeds the watchdog and imputation path.

### 5.4 Static layer
**Per-sensor calibration bias** — a fixed multiplicative/additive offset sampled once per "deployment," never zero-mean (mounting angle, calibration drift). Sensor 7 always reads a bit high, forever.

### 5.5 Honesty about parameters (write this into the report verbatim)
We have no real camera data, so every number above is an assumption. We do **not** pick one operating point and claim it's realistic. Instead we **sweep severity** and show our margin over baselines holds across a range of degradation (see §6). That turns "we guessed the noise levels" into our strongest result. We anchor the ranges loosely in the vehicle-detection literature (reported recall under occlusion/weather) and cite it — but the *claim* lives in robustness across the sweep, not in matching one condition.

### 5.6 The gold standard we are deliberately not doing
The fully realistic version renders SUMO scenes and runs a real detector (YOLO on rendered frames, or via CARLA). Authentic failures, but weeks of work — camera placement, 3D rendering, detector integration — out of scope. One line in "Path to Production" acknowledges it and makes our parametric model a deliberate choice, not a shortcut.

### 5.7 Where it lives in code
This is v1's `noise_injection.py` grown up → renamed **`perception.py`**. It sits between the SUMO env and `observation_encoder.py`, exposes a single **severity knob** we can sweep, and the **same module is used in both training (so the model learns robustness) and evaluation (so we can prove it).**

---

## 6. Training Protocol

### 6.1 Algorithm
- **Primary: DQN** with shared parameters across intersections (CoLight-style).
- Experience replay (50,000), target network (update every 1000 steps), batch 64.
- ε-greedy: 1.0 → 0.05 over 10,000 steps.
- **MAPPO is a diagnosed-need upgrade, not a default (Week 5+).** We adopt it *only if* DQN converges cleanly on single networks but specifically fails on cross-network credit assignment (agents thrashing / policy averaging into mush that demand randomization doesn't fix). Reasons we stay on DQN by default, spelled out in §11.

### 6.2 Cross-network training (the generalization story)
Train on multiple networks simultaneously, **through the perception model**, with random network selection per episode and demand randomization within networks. The model grows up under both network diversity *and* the sensor-failure distribution — robustness is not retrofitted later.

**Training set — config-driven, NOT fixed.** Network choice is configuration, not a commitment, and the code never hardcodes a network. The only hard requirement is parametric: **≥3 networks of varying size and connectivity.** Concrete networks (e.g. RILSA, Cologne1/3/8, synthetic NxN grids) are *non-binding examples* we may or may not use; we finalize the set later. Synthetic grids of varying size are one cheap way to inject topological diversity and are a config option, not a fixed asset.
- Whatever the final set, it should span **size** (few vs many intersections) and **connectivity/topology** so the model generalizes across the *family of standard signalized networks* rather than memorizing one layout.

**Episode structure**
- Random network selection per episode (uniform over the configured set)
- Random demand scaling: ±25% of base flow
- **Perception model active**: per-episode regime sampled, per-vehicle detection, latency, drops, calibration bias

**Demonstration set (held out)** — also config-driven
- One held-out network (e.g. Ingolstadt7) → cold-start performance + fine-tuning curve. Identity is a config choice; the code must not depend on which network it is.

**Why diversity matters.** If a candidate set is topologically monotone (e.g. all similar cities), add structurally different networks (synthetic grids of varying size) so generalization is forced rather than the German-pattern memorized. This is a property of the *set*, achieved via config — not something baked into code.

### 6.3 The healthy training signal
If the model is exceptional on training networks but poor on Ingolstadt7, that's a **warning sign**, not a success — it memorized instead of learning principles. We want:
- Decent (not exceptional) performance on training networks — no overfitting.
- Reasonable cold-start on Ingolstadt7 — transferable knowledge.
- Fast convergence when fine-tuned on Ingolstadt7 — correct inductive bias.

---

## 7. Baselines & Evaluation

### 7.1 Baselines
We removed the vanilla CoLight reproduction — we're not claiming novelty over CoLight. Four baselines beneath our method, two non-AI and two learning. They isolate *different* variables, so we keep all of them.

| Type | Baseline | Role / what it tests | Transfers? |
|---|---|---|---|
| Non-AI | Fixed-Time | Floor — demand-blind status quo | n/a |
| Non-AI | MaxPressure (~80 LOC) | Strong rule-based competitor; also our serving fallback tier | n/a |
| Non-graph AI | **Independent DQN (per-agent)** | Strongest non-graph *local* learner; insurance against "you crippled the baseline" | No |
| Ablation | **0-hop GAT** (our model minus message passing) | Clean single-variable "does the graph matter" control | Yes |
| Our method | GAT + typed edges + masking + advanced state | The full system | Yes |

**Why two learning baselines, not one.** They answer different questions and aren't redundant:
- The **0-hop GAT** is our exact architecture with message passing switched off, so it differs from the full model in *exactly one thing* — the graph. This is the clean attribution control, and because it shares parameters it can also appear in the cross-network/transfer experiments.
- The **per-agent Independent DQN** gives each intersection its own specialized weights — the strongest possible non-graph local controller on a *fixed* network. Its job is to preempt "you forced parameter sharing to make the graph look good." If the full GAT beats even this, the coordination claim has no soft spot. Limitation: per-agent weights are tied to node identity, so it **cannot transfer** — it only appears in fixed-network ceiling tests, not cold-start/transfer.

**Comparison hygiene.** All learning methods use DQN — never put the graph method on a different algorithm than its baseline, or we change two variables at once. We train everything **to convergence** and compare **asymptotic** performance (per the ceiling framing in §14).

### 7.2 The headline experiment (NEW — our money plot)
**Clean state vs. realistically degraded state**, with *all* methods (including baselines) evaluated under the same degradation. Deliverable: a plot with **degradation severity on the x-axis, performance on the y-axis**, showing our method's curve staying above MaxPressure's across the sweep. This is arguably more important than the transfer result — it's the experiment that actually demonstrates deployability.

### 7.3 Metrics
Average waiting time, throughput, total queue length, stability (phase-change rate) — **plus tail metrics: p95 queue, spillback events, max wait.** For deployment the worst case causes gridlock, so means alone are misleading. 3 seeds per network, mean ± std. Five seeds is research-grade overkill.

### 7.4 Diagnostic scenario tests (validate the model choice, not just the score)
Run every scenario below across all four methods (Fixed-Time, MaxPressure, Independent DQN, GAT). The aggregate results table tells us *who wins*; these scenarios tell us *why*, and specifically whether the graph earns its place. **Selection criterion:** a scenario only counts as diagnostic if doing well on it structurally requires cross-intersection information — otherwise it tests adaptivity (which Independent DQN also has), not the graph. We report all methods on each, both trained to convergence (no episode budget cap — per our higher-ceiling framing in §14).

**Hero scenarios — we expect the graph to win, for reasons no baseline tuning can fix:**
- **Oversaturation with spillback.** Push demand until a queue backs past an upstream intersection and blocks it. This breaks MaxPressure's optimality assumption (queues can't absorb flow under blocking). Tests whether the graph can *gate/meter* — hold upstream traffic to protect a critical intersection. Strongest single test; here MaxPressure doesn't just do worse, it acts structurally wrong (feeds a full link).
- **Platoon anticipation on long links.** Release a clear pulse down a long link. Tests whether the model uses the upstream neighbor's state to have green ready before the clump arrives. Longer link = more lead time = bigger graph edge. Cleanest demonstration of the information advantage.
- **Corridor / green-wave under directional demand.** Heavy flow along a corridor. Tests whether the model synchronizes greens across intersections; local methods break the wave.
- **Compound: blind node under high demand.** High demand + a failed/degraded sensor at one intersection (ties to the §5/§8.4 robustness story). Tests graph-based imputation + coordination simultaneously. No baseline has a mechanism for this.

**Control / honesty scenarios — we expect parity, and we say so up front:**
- **Uniform moderate demand.** Expect parity with MaxPressure (nothing to coordinate). Run as a control — the story "our margin grows with difficulty" beats "we win everywhere."
- **Single isolated intersection (e.g. Cologne1).** Graph reduces to ~Independent DQN by construction; expect parity. Claiming a graph win here would be a red flag.
- **Extreme one-sided demand → starvation check.** This is a *guardrail* test (fairness via min-green + masking), **not** a graph win. Verify we don't starve the minor approach. Any edge over MaxPressure here comes from constraint design, not the graph — attribute honestly.

**Knobs that make any scenario discriminate harder:** long inter-intersection links (more anticipation lead time), spatial/temporal demand correlation (more to coordinate), capacity asymmetry/bottlenecks (more reason to meter). Crank these and the graph's edge grows; flatten them and everything converges to parity.

**Summary line for the report:** the graph's advantage scales with congestion, irregularity, and the distance over which information is useful — so we expect to win biggest where it's hardest and tie where it's easy. *These scenarios are the selection framework; exact demand profiles and network configs are finalized after the build, once we know what the env supports.*

### 7.5 Convergence curves (and what they're allowed to claim)
Plot performance vs episodes, per method per network. **Framing rule, because we don't care about training speed:** the curves exist to show (a) each method *converged at all* (sanity — DQN didn't diverge in the multi-agent setting) and (b) the graph plateaus *higher* (the ceiling claim — the thing we're actually betting on). They do **not** claim the graph trains faster — since we allow unlimited episodes, a speed claim would be indefensible anyway. Note the asymmetry: full-GAT and 0-hop-GAT can have *cross-network* curves (they share params), but the per-agent Independent DQN only has *per-network* curves (it can't transfer). Report wording: *"all learning methods trained to convergence; we compare asymptotic performance and use learning curves only to evidence convergence and the performance ceiling."*

---

## 8. Deployment Architecture

This is where we differentiate from typical student projects. The system is a **stateful adaptive deployment manager** supporting a real lifecycle — cold start, adaptation, optimized operation — not a stateless inference function.

### 8.1 Lifecycle phases
- **Phase 1 — Cold start (Day 0).** Pretrained weights, inference only. Watchdog conservative; falls back more readily.
- **Phase 2 — Adaptation (Days 1–N).** Serves decisions and updates weights from logged experience. Performance improves over time.
- **Phase 3 — Optimized operation.** Converged; weights frozen for safety. Optionally re-enters adaptation periodically for seasonal change — **gated by the drift monitor (§8.5), not on a blind timer.**

### 8.2 API endpoints
**Inference**
- `POST /predict` — graph + observations → actions + confidence + fallback flag
- `GET /health` — status + model version + current lifecycle phase

**Lifecycle**
- `POST /deploy` — register a new network → load pretrained weights
- `POST /experience` — feed (state, action, reward, next_state) for learning
- `POST /train_step` — trigger one gradient update on collected experience
- `POST /set_mode` — inference | adaptation | frozen

**Observability**
- `GET /metrics` — throughput, wait, queue, fallback rate
- `GET /training_progress`
- `GET /attention_logs` — recent decisions with attention weights
- `GET /drift_status` — input-distribution & fallback-rate health (NEW)

### 8.3 Tiered safety layer / watchdog (CHANGED — gradient, not binary)
Every decision passes through a watchdog checking:
- Did the model respond within 500 ms?
- Is the output a valid phase for this intersection?
- Has the same phase been held longer than `MAX_PHASE_DURATION`? (oscillation/freeze detection)
- Did the input pass validation (no NaN, no impossible queue jumps)?

**Failure response is a gradient, not a cliff:**
```
healthy model decision  →  MaxPressure (rule-based, still traffic-responsive)  →  fixed-time (last resort)
```
This matters because in realistic deployment **partial sensor failure is constant**. A binary "any failure → fixed-time" would run fixed-time most of the time and make the model look useless. MaxPressure is already implemented as a baseline, so using it as the middle tier is nearly free and a large credibility win. Fixed-time is reserved for sustained hard failure.

### 8.4 Sensor imputation (NEW)
A failed/missing sensor (per the §3.1.1 validity flag) triggers **imputation before fallback**:
- last-known value (short outages), and/or
- **neighbor-derived estimate through the graph** (the graph earns its keep here — neighbors carry information about a blind node).

Only *sustained* hard failure that imputation can't cover drops the node down the watchdog tiers. Per-node isolation still holds: a partition affects only the isolated node, not the whole system — a real advantage of decentralized execution.

### 8.5 Health & drift monitoring (NEW — promoted to spine)
For a deployment-focused project, knowing *when the model is degrading* is core, not optional — without it the frozen→re-adaptation lifecycle is faith-based. Minimum viable version:
- Monitor input statistics (distribution of queue/running-vehicle features) and **fallback rate**.
- Alert / surface via `/drift_status` when they shift beyond threshold.
- This is what gates re-entry into adaptation in Phase 3.

### 8.6 Real signal safety constraints (CHANGED — "valid phase" was too thin)
"Valid phase + min-green" alone is not deployment-aware. We model a couple of real intersection constraints in the action masking so the safety claim is real rather than cosmetic:
- All-red clearance intervals
- Yellow timing
- Conflicting-movement prevention
- Pedestrian phase minimums

### 8.7 Inference topology (NEW — the real edge cost)
**Compute is a non-issue.** A 2-layer GAT at 128→64 is a handful of matmuls per intersection, sub-millisecond on CPU, far inside the 500 ms budget, quantizes/exports to ONNX/TorchScript trivially. Algorithm choice (DQN vs MAPPO) costs *nothing* at inference — both just run the actor forward and argmax; the critic only exists during training.

**The real cost is message-passing communication.** If each intersection is its own edge device, 2 GAT layers = 2-hop receptive field = data needed from two hops away = multiple synchronous comm rounds between physical devices per decision. That's network latency, clock sync, and a new failure surface. Three options:

- **Centralized inference** — one server holds the whole graph; devices send observations. Simplest, lowest latency, but single point of failure and undercuts the "decentralized" claim.
- **Truly distributed** — devices exchange embeddings; 2-hop = 2 comm rounds. Most faithful, most fragile.
- **Stale-embedding caching (our recommended default)** — each node uses neighbors' *last-known* embedding instead of synchronizing fresh every step. One forward pass, no blocking comm round, degrades gracefully if a neighbor link drops. This is the sweet spot: it makes the system tolerant of *communication* failure (distinct from sensor failure) and serves latency *and* robustness at once.

**We commit to stale-embedding caching as the default and document the trade-off.** It also dovetails with §8.4 — a stale or unreachable neighbor embedding is exactly what attention masking should downweight.

### 8.8 What we do NOT build (production gap, documented honestly)
- Formal safety verification — multi-year research effort
- NTCIP / OCIT protocol translation — certified-equipment territory
- Real sensor fusion & calibration — its own ML subsystem
- **Rendered-frame perception with a real detector (CARLA/YOLO)** — the gold-standard observation model from §5.6
- Cryptographically signed audit logs — operational/legal layer
- Multi-tenant Kubernetes orchestration — beyond academic scope

We document this gap in the final report under **"Path to Production."** Showing we understand the gap is what separates a strong project from a great one.

---

## 9. Project Structure

Three top-level layers — data / model / serving — matching the "clean separation" requirement.

```
traffic_rl/
├── data/
│   ├── networks/            (SUMO nets — config-referenced, NOT hardcoded; any net works)
│   ├── graph_builder.py     (net.xml → typed-edge graph JSON)
│   └── observation_encoder.py
│
├── models/
│   ├── gat_policy.py        (GAT + phase scoring head + neighbor masking; 0-hop flag = ablation baseline)
│   ├── independent_dqn.py   (per-agent non-graph baseline)
│   └── max_pressure.py      (rule-based baseline + fallback tier)
│
├── training/
│   ├── trainer.py
│   ├── replay_buffer.py
│   ├── reward.py            (efficient pressure)
│   ├── perception.py        (NEW — structured sensor model, severity knob)
│   └── multi_network.py     (cross-network orchestrator)
│
├── evaluation/
│   ├── metrics.py           (incl. tail metrics: p95, spillback, max wait)
│   ├── transfer_eval.py     (cold-start + fine-tuning curves)
│   ├── degradation_sweep.py (NEW — clean-vs-degraded money plot)
│   ├── error_analysis.py
│   └── ablations.py
│
├── serving/
│   ├── api.py               (FastAPI)
│   ├── deployment_manager.py (stateful, per-network)
│   ├── inference.py         (incl. stale-embedding caching)
│   ├── training.py          (online learning path)
│   ├── lifecycle.py         (cold start / adaptation / frozen)
│   ├── watchdog.py          (tiered fallback)
│   ├── imputation.py        (NEW — last-known + neighbor-derived)
│   ├── monitoring.py        (NEW — drift / fallback-rate health)
│   ├── validation.py
│   └── decision_logger.py
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── README.md
│
├── notebooks/
├── configs/                 (Hydra YAML — reproducibility)
├── tests/
└── README.md
```

---

## 10. How we build it (process — read before coding)

**Principle: design up front only what's expensive to change; discover the rest by building.** Expensive = interfaces and data contracts (the seams between A/B/C). Cheap = hyperparameters, layer sizes, reward coefficients. With three people on parallel workstreams, the only thing that *must* precede code is the contracts between us.

### 10.1 Lock these interfaces first (≈ half a day, all three sign off)
1. **Graph JSON schema** (B→A): node features, edge types, exact tensor layout.
2. **Observation contract** (B→A): field order, normalization, and the **missing-sensor sentinel + validity flag** (§3.1.1).
3. **Env step API** (B→A): what `step()` returns, what `reset()` does.
4. **`/predict` request/response shape** (A→C): already drafted in §8.2.

### 10.2 Then build a thin end-to-end vertical slice on RILSA in week 1
The dumbest version that runs the *whole* pipeline: one network, MaxPressure (or random) actions, real graph builder, real env, **perception layer already in the loop (crude is fine)**, stub `/predict`. Traffic moves sensor-state → decision → simulator → back, end to end. This de-risks integration and gives all three of us a working skeleton to develop against in parallel.

### 10.3 Then spiral outward, one ring at a time
Confirm the model converges on one network → add cross-network → strengthen the perception model → build the serving layer → add lifecycle. Return to a working end-to-end state at the end of each ring. **Never integrate big pieces for the first time late.**

### 10.4 The one trap to avoid
Do **not** build clean-state first and bolt robustness on in week 7. Because robustness is our headline claim, the noisy observation path is in the slice from the start so the model trains under it. Robustness retrofitted late is robustness we can't trust.

### 10.5 Model build order (rings — for whoever builds the GAT)
Build the model inside-out, with the **no-graph core first** because it doubles as the 0-hop ablation baseline. By the time message passing is added, everything else is known-good, so any new bug is attributable to the graph. Never start a ring until the previous one's "done-when" passes; keep every ring's checkpoint (that's how the ablations get generated for free).
- **R0** — lock contracts + a mock env/stub graph so model dev isn't blocked on SUMO. *Done: dummy forward pass runs on the agreed schema.*
- **R1** — node encoder + FRAP phase-scoring head, single intersection, no graph. *Done: scores respond sensibly to handcrafted inputs.*
- **R2** — wrap in DQN loop (replay, target net, ε-greedy, masking), single intersection. *Done: trains stably, beats Fixed-Time. **Save as the 0-hop ablation baseline.***
- **R3** — turn the perception model on in the loop. *Done: still converges under noise, handles validity/sentinel without NaN.*
- **R4** — one GAT layer, untyped edges. *Done: trains on a small multi-intersection net; ideally beats the 0-hop checkpoint on a coordination scenario. **Make-or-break — if 1-hop never beats 0-hop, stop and debug.***
- **R5** — second GAT layer (2-hop) + multi-head. *Done: trains without embedding collapse / over-smoothing.*
- **R6** — typed edges (2 types, separate weight matrices). *Done: typed > untyped on a mini-ablation.*
- **R7** — neighbor masking of failed/stale neighbors via the validity flag. *Done: under dropout, attention on masked neighbors drops, degrades gracefully.*
- **R8** — multi-network shared-parameter training. *Done: one param set trains across ≥3 networks without crashing → hand to evaluation.*

---

## 11. 8-Week Implementation Plan

| Week | Goal | Deliverable |
|---|---|---|
| 1 | **Lock interfaces; thin end-to-end slice on RILSA with crude perception in-loop**; build MaxPressure | Pipeline runs sensor→decision→sim end to end; typed-edge graphs ready |
| 2 | Core model: GAT layer + neighbor masking, phase scoring head, efficient-pressure reward | Model trains on RILSA (through perception model), beats Fixed-Time |
| 3 | Multi-network training + full perception model (regimes, latency, drops, bias) | Model trains across 4–6 networks without crashing, under degradation |
| 4 | Evaluation infra + Independent DQN baseline + degradation sweep harness | Full baseline matrix runs; degradation-sweep scaffold works |
| 5 | First real results: train all methods, run on Ingolstadt7, run the money plot | Results table + clean-vs-degraded plot; deployment-readiness check; *decide MAPPO yes/no based on observed DQN behavior* |
| 6 | Serving layer: FastAPI, tiered watchdog, imputation, monitoring, stale-embedding caching, Docker | `docker run` produces a working API with tiered fallback |
| 7 | Error analysis, ablations, attention visualizations, tail-metric reporting | Failure-case writeup + ablation results + severity-sweep figures |
| 8 | Final polish: report (incl. Path to Production), README, demo video | Submission-ready repo + PDF report |

### Division of labor (3 people)
- **Owner A — Model & Training:** GAT layer + neighbor masking, reward, training loop, replay buffer, multi-network orchestrator.
- **Owner B — Data & Environment:** graph builder, observation encoder, synthetic grids, **`perception.py`**, SUMO/TraCI integration.
- **Owner C — Evaluation & Serving:** baselines (MaxPressure, Independent DQN), metrics + tail metrics, degradation sweep, FastAPI, watchdog, imputation, monitoring, Docker.

Pair on hard parts (GAT debugging, neighbor masking, watchdog/imputation logic). Daily 15-min standup. Weekly demo.

---

## 12. Scope Discipline — What to Cut If Time Slips

### Spine (NEVER cut)
- GAT + typed edges + neighbor masking
- Efficient-pressure reward
- Cross-network training (≥3 networks)
- **Perception model (at least detection + regimes)** — without it the project's headline claim is empty
- **Degradation sweep (the money plot)** — the experiment that proves deployability
- MaxPressure + Independent DQN baselines + 0-hop GAT ablation
- Docker container with FastAPI + **tiered fallback**
- **Minimal drift/fallback-rate monitoring**

### First to drop
- MAPPO upgrade — stay with DQN
- Synthetic grid generation — fall back to fewer / simpler configured networks (still ≥3, still varied)
- CO₂ multi-objective term — keep pressure-only
- Full per-vehicle perception fidelity — fall back to a *coarser but still structured* model (regime + occlusion-correlated undercount), never to plain Gaussian

### Second to drop
- Stale-embedding caching → fall back to centralized inference (document the trade-off)
- Decision logging with full attention weights (still log decisions, simpler format)
- Some ablations — keep at minimum: with/without graph, with/without cross-network training, with/without perception-robustness training

### Acceptable to ship without
- Full attention-weight visualizations
- Multiple seeds (1–2 acceptable with caveat)
- `/train_step` online learning endpoint (simulate adaptation offline; feature it as future work)
- `/attention_logs` endpoint (keep `/predict`, `/health`, `/deploy`, `/drift_status`)

---

## 13. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| GAT doesn't converge with shared parameters across heterogeneous networks | Start single-network on RILSA (week 2) to confirm convergence; expand only after. Confront this with the *simplest* algorithm (DQN) so the failure isn't entangled with a PPO bug. |
| Cross-network training collapses (model averages instead of generalizes) | Demand randomization + diverse network sizes; if it still fails, this is the diagnosed signal to consider MAPPO (centralized critic for credit assignment). |
| Cold-start on Ingolstadt7 worse than Fixed-Time | Frame as fine-tuning demonstration: show convergence curve over ~10 episodes. |
| **Perception parameters are unvalidated guesses** | **Don't claim one operating point is realistic — sweep severity and report margin-over-baseline across the range.** Anchor ranges loosely in detection literature. |
| **Message-passing comms add latency/failure on edge** | **Stale-embedding caching (one forward pass, no blocking round); fall back to centralized inference if distributed proves fragile.** |
| Online learning in Docker too complex for week 6 | Drop `/train_step`; demo lifecycle with offline fine-tuning instead. |
| DQN instability in MARL | Aggressive target networks (update every 1000 steps); fall back to running averages if needed. |
| Imputation masks a real failure (model trusts a bad estimate) | Cap imputation duration; sustained failure escalates down the watchdog tiers; monitor fallback rate. |

---

## 14. Decisions we've locked (was "open questions")

These were genuinely open in v1. We've resolved them; recorded here so we don't relitigate.

- **Goal = good initialization + fast fine-tuning, NOT zero-shot.** Shapes framing and which experiments matter. ✅ Locked.
- **DQN is the spine; MAPPO is a diagnosed-need upgrade only.** Reasons: MAPPO improves final policy quality (the axis we don't care about), gives *zero* field robustness (the critic is discarded at deployment), and would muddy the graph-vs-non-graph comparison. Adopt only if we *observe* DQN failing specifically on cross-network credit assignment. ✅ Locked.
- **The graph is justified by a higher converged ceiling via coordination — NOT by faster training.** Independent DQN structurally cannot see neighbor state, so it can never anticipate platoons or coordinate; no number of episodes fixes that. We therefore train all methods to convergence and compare *asymptotes*. (Corollary: since we allow unlimited training, we make no sample-efficiency claim for the graph.) ✅ Locked.
- **Network choice is config-driven, never hardcoded.** Requirement is parametric (≥3 networks, varying size/topology); named networks are non-binding examples; the exact set + the held-out network are chosen later and the code must not depend on the choice. Generalization is claimed over the *family of standard signalized networks*, not literally "any network." ✅ Locked (replaces v1's "exactly 6 networks").
- **Four baselines, two of them learning.** Per-agent Independent DQN (strongest local, doesn't transfer) *and* the 0-hop GAT ablation (clean single-variable control, transfers). Don't collapse them into one. ✅ Locked.
- **`/train_step` online learning is in v1 if week 6 allows; otherwise offline fine-tuning + future-work framing.** ✅ Locked with fallback.
- **Multi-objective CO₂ stays a stretch goal.** Polishing the deployment story beats a second reward term. ✅ Locked.

### Still genuinely open — decide by end of Week 1
- **Q-A: Inference topology default.** We recommend stale-embedding caching; confirm the team is comfortable building it, or commit to centralized for v1 and document.
- **Q-B: Perception fidelity for the grade.** How realistic does the perception model need to be, and where's the cut line if week 3 runs long? (Floor = regime + occlusion-correlated undercount.)
- **Q-C: Ownership sign-off.** Confirm the A/B/C split in §11 and who pairs on GAT/watchdog.

**Decision deadline:** end of Week 1, Monday standup. After that: implementation only.

---

## 15. Final Word (for us, not a grader)

This is our shared reference. Every architectural decision here is defensible from first principles or the literature — nothing is here because "it sounded good." The three things that make this project worth being proud of, and that most student projects miss:

1. A **graph architecture chosen and justified from requirements** — and a clean experiment (GAT vs Independent DQN, single variable) that actually proves the graph earns its place.
2. A **training protocol that produces a genuinely deployable model** — cross-network *and* under a structured sensor-degradation model, with the headline result being margin-over-baseline that holds as input quality drops.
3. A **serving layer with the safety patterns of a real adaptive ML system** — tiered fallback, graph-based imputation, drift monitoring, real signal constraints, and an honest inference-topology choice.

None of these alone is novel. Together they make a project that demonstrates real understanding of building applied ML systems — which is exactly what the course is meant to teach. If anyone disagrees with a specific choice, bring the argument and we update. But the default is: **this is the plan, we execute. Build discipline beats design discipline from here.**
