# LightMind — AI Traffic Signal Control

> **Adaptive traffic signal control powered by Graph Attention Networks, trained on real city road networks and deployable to new cities via a web dashboard.**

<video src="DEMO.mp4" controls width="100%"></video>

**Team:** Mohamad Al Aalami · Hasan Haidar · Mohamad Kassira  
**Course:** EECE 490  
**Website:** [https://light-mind-beta.vercel.app](https://light-mind-beta.vercel.app)  
**Live Demo:** [https://hasanhaidarcoder-lightmind.hf.space](https://hasanhaidarcoder-lightmind.hf.space)

---

## What It Does

LightMind learns to control traffic signals across a city by treating the road network as a graph — intersections as nodes, roads as edges. A Graph Attention Network (GAT) runs at every intersection, reads queue lengths and signal states, and coordinates with neighbors via learned attention weights to decide when to change phases.

A full-stack web dashboard lets anyone upload a SUMO network `.net.xml` file, and watch the GAT model train and evaluate in real time inside SUMO.

---

## Repository Structure

```
LightMind/
├── model/                          # Core ML system
│   ├── models/
│   │   ├── gat_policy.py           # 1- or 2-layer GAT backbone (typed edges, neighbor masking)
│   │   ├── node_encoder.py         # Per-node MLP: obs + validity → 64-dim embedding
│   │   └── phase_head.py           # Q-value head per phase
│   ├── data/
│   │   ├── graph_builder.py        # SUMO net.xml → typed-edge PyG graph
│   │   └── observation_encoder.py  # Live sensor readings → padded obs tensors
│   ├── training/
│   │   ├── trainer.py              # Single-network DQN training loop
│   │   ├── sync_trainer.py         # Synchronised multi-network trainer
│   │   ├── multi_network.py        # Cross-network data loading
│   │   ├── replay_buffer.py        # Experience replay
│   │   └── reward.py               # Pressure reward + observation imputation
│   ├── evaluation/
│   │   ├── eval_runner.py          # Episode evaluation
│   │   ├── metrics.py              # Waiting time, throughput, queue, phase-change rate
│   │   ├── ablations.py            # 0-hop / typed-edge ablation runs
│   │   ├── degradation_sweep.py    # Sensor degradation robustness tests
│   │   └── transfer_eval.py        # Transfer / fine-tuning curves
│   ├── serve/
│   │   └── inference_server.py     # FastAPI inference: /decide, /health (with watchdog)
│   ├── baselines/
│   │   ├── max_pressure/           # MaxPressure controller
│   │   └── rl/                     # Independent DQN baseline (flat features, no graph)
│   ├── networks/
│   │   ├── external/
│   │   │   ├── RESCO/              # cologne*, ingolstadt*, grid4x4, arterial4x4
│   │   │   ├── bologna_pasubio/    # SUMO sample scenario
│   │   │   ├── MoST/               # Monaco Street scenario
│   │   │   └── toronto/            # Toronto (Scarborough-Agincourt)
│   │   ├── generated/              # Smoke-test networks
│   │   └── source_repos/
│   │       └── sumo-rl/            # sumo-rl submodule
│   ├── configs/                    # Per-network YAML configs (Hydra-style)
│   ├── scripts/                    # Route generation utilities
│   ├── tests/                      # 15 unit + integration tests
│   ├── train.py                    # Single-network training entry point
│   ├── run_training.py             # Batch training entry point
│   └── TRAINING.md                 # Training instructions
│
├── website/
│   ├── frontend/                   # React + Vite + Tailwind + Leaflet
│   │   └── src/components/
│   │       ├── UploadScreen.jsx    # OSM + optional demand file upload
│   │       ├── ModelSelect.jsx     # Model / baseline selection
│   │       ├── TrainingScreen.jsx  # Live training stream via WebSocket
│   │       ├── ResultsScreen.jsx   # 24-hour KPI comparison charts
│   │       ├── DeploymentScreen.jsx
│   │       └── RetrainingScreen.jsx
│   └── backend/                    # FastAPI
│       ├── main.py                 # App entry, CORS, routers
│       ├── routers/                # upload, train, real_train, results, ws, deployment…
│       ├── services/               # OSM converter, route generator, real trainer
│       └── traffic_rl/
│           └── benchmark/
│               ├── controllers/fixed_time/      # Fixed-Time benchmark runner
│               ├── controllers/independent_dqn_v2/      # DQN baseline (offline)
│               └── controllers/independent_dqn_v2_web/  # DQN web integration
│
├── checkpoints/                    # Pre-trained model weights + eval metrics
├── Dockerfile                      # Multi-stage: Node 20 frontend + Python 3.11 runtime
├── docker-compose.yml
└── supervisord.conf                # Manages Xvfb + noVNC + uvicorn inside Docker
```

---

## Architecture

### Model Pipeline

```
SUMO net.xml
    │
    ▼
graph_builder.py ──► typed-edge graph (flow edges + coordination edges)
    │
    ▼
observation_encoder.py ──► per-node (obs, validity) tensors
    │
    ▼
NodeEncoder (MLP) ──► 64-dim embeddings  [obs_dim*2 → 128 → 64]
    │
    ▼
GATPolicy (1 or 2 layers)
    ├── Layer 1: 4 heads × 32 dim → 128-dim  (typed: separate weights per edge type)
    └── Layer 2: 4 heads × 16 dim → 64-dim   (R5+; same topology as R4 when num_layers=1)
    │
    ▼
PhaseHead ──► Q-value per phase → argmax → phase decision
```

**Typed edges :** `gat_flow` (upstream → downstream, `add_self_loops=True`) and `gat_coord` (downstream → upstream, `add_self_loops=False`) use separate weight matrices whose outputs are summed. This lets the model distinguish *traffic flow* from *coordination signals*.

**Sensor robustness:** `NodeEncoder` receives `cat([obs, validity])` so it can distinguish live readings, last-known imputed values, and never-seen positions. `ObservationImputer` fills failed sensors with last-known readings; `PressureReward` maintains its own independent cache.

**Reward:** Mixed pressure + queue reward:  
`r_i = -(queue_weight × q_in/n + pressure_weight × |q_in − q_out|/n)`  
Default: `queue_weight=0, pressure_weight=1` (pure pressure, R4/R5). The queue term is available to prevent balanced-gridlock exploits.

**Watchdog (inference server):** inference > 500 ms → fall back to MaxPressure → fall back to fixed-time.

### Baselines

| Controller | Description | Location |
|---|---|---|
| **Fixed-Time** | Pre-timed cycles, no adaptation | `website/backend/traffic_rl/benchmark/controllers/fixed_time/` |
| **MaxPressure** | Greedy pressure-based phase switching | `model/baselines/max_pressure/` |
| **Independent DQN** | Per-intersection DQN, flat features, no graph | `website/backend/traffic_rl/benchmark/controllers/independent_dqn_v2/` |

---

## Networks

All networks live under `model/networks/external/`:

| Network | Intersections | Source |
|---|---|---|
| `cologne1` | 1 TLS | RESCO / DLR |
| `cologne3` | Small | RESCO / DLR |
| `cologne8` | Medium | RESCO / DLR |
| `ingolstadt1` | 1 TLS | RESCO / DLR |
| `ingolstadt7` | Small | RESCO / DLR |
| `ingolstadt21` | Large | RESCO / DLR |
| `grid4x4` | 4×4 synthetic grid | RESCO |
| `arterial4x4` | 4×4 arterial | RESCO |
| `bologna_pasubio` | Real (Bologna, Italy) | SUMO Scenarios |
| `MoST` | Large (Monaco) | Monaco Street Scenario |
| `toronto` | Real (Toronto, Canada) | Toronto Open Data / sumo-rl |

Stochastic demand variants (light (25%)/ medium (45%)/ dense (65%)/ heavy(90%)) are pre-generated per network.

---

## Quick Start

**Prerequisites:** Docker Desktop installed and running.

```bash
git clone https://github.com/MohammadKassira/LightMind.git
cd LightMind
cp .env.example .env        # then add your ANTHROPIC_API_KEY inside .env
docker-compose up --build
```

Everything is served on a single port:

| Service | URL |
|---|---|
| Web dashboard | http://localhost:7860 |
| API docs (Swagger) | http://localhost:7860/docs |
| noVNC (SUMO GUI) | http://localhost:7860/novnc/ |

```bash
docker-compose down   # stop
```

## Behind the Scenes

All hyperparameters are in `configs/train.yaml` and the network-specific YAMLs under `configs/`. Outputs are written to `model/checkpoints/<RUN_NAME>/`:

```
final.pt               # model weights + optimizer state
training_metrics.json  # episode_returns, losses, q_mean, avg_waiting_time, throughput
checkpoint_ep100.pt    # periodic checkpoints every 100 episodes
eval_metrics.json      # n_episodes, mean/p95/max waiting time, throughput, phase_change_rate
```

**Available configs:** `cologne1`, `cologne8`, `arterial4x4`, `grid4x4`, `ingolstadt1`, `ingolstadt7`, `ingolstadt21`, `pasubio_stochastic`, and ablation variants (`zero_hop`, `single_layer`, `dense`, `sync`).

---


## Running Tests

```bash
cd model
pip install pytest
pytest tests/ -v
```

15 tests covering: GAT policy, graph builder, node encoder, phase head, reward, replay buffer, observation imputation, multi-network loading, trainer convergence, and SUMO integration.

---

## Web Dashboard Flow

1. **Upload** — drop a SUMO `.net.xml` file.
2. **Training** — live reward curve and signal states streamed over WebSocket
3. **Results** — KPI comparison: LightMind vs Fixed-Time, waiting time / throughput / queue
4. **Deployment** — download the trained model checkpoint
5. **Retraining** — re-run with different demand or parameters

---

## Inference Server (standalone)

The model ships a dedicated FastAPI inference server separate from the web dashboard:
The idea is that it simulates having cameras/sensors sending data to the controller by having 2 servers communicating back and forth

```bash
cd model
python serve/inference_server.py --checkpoint checkpoints/<RUN_NAME>/final.pt --port 8001
```

**Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/decide` | POST | Phase decisions given current obs + validity per node |
| `/health` | GET | Last latency (ms), fallback tier, cycle count |

Fallback tiers: `0` = model, `1` = MaxPressure, `2` = fixed-time. Tier escalates automatically if inference exceeds 500 ms or throws an exception.

---

## Key Design Decisions

**Why GAT over GCN?** GAT learns per-neighbor attention weights, making it inductive (works on unseen topologies without retraining) and interpretable (attention = which neighbors matter, when). The typed-edge variant (R6+) further separates upstream flow from downstream coordination with independent weight matrices.

**Why shared parameters?** One model handles any number of intersections and any topology — no per-network retraining on deployment. This is what makes generalization possible.

**Why pressure reward?** Theoretically grounded (drives the network toward balanced queue states), stable, and avoids reward hacking. The optional queue term (`queue_weight > 0`) prevents edge cases where balanced gridlock earns zero penalty.

**Why `cat([obs, validity])`?** The NodeEncoder receives both the observation vector and the validity mask concatenated, so it can distinguish three states: live reading, last-known imputed value (sensor temporarily failed), and never-seen position (padded). This is the core robustness mechanism.

**Why `obs_dim * 2` as input width?** Direct consequence of the above — every obs feature gets a paired validity feature.

---

## References

- Wei et al. (2019). *CoLight: Learning Network-level Cooperation for Traffic Signal Control*
- Yoon et al. (2021). *Transferable Traffic Signal Control with Graph-Centric State Representation*
- Devailly et al. (2021). *IG-RL: Inductive Graph Reinforcement Learning for Massive-Scale Traffic Signal Control*
- Ault & Sharon (2021). *RESCO: Reinforcement Learning Extensions for SUMO*
- Chen et al. (2020). *Toward A Thousand Lights: Decentralized Deep Reinforcement Learning for Large-Scale Traffic Signal Control*

---

## License

For academic use. RESCO networks under CC BY-NC-ND 4.0 — see per-network `LICENSE` files under `model/networks/external/RESCO/`.
