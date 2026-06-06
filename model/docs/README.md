# traffic_rl

Repository structure for the traffic RL project.

This repo follows the design in section 9 of `agent_spec_gat_and_baselines.md`.

## Layout

- `configs/` — Hydra-style config files
- `data/` — SUMO networks and graph/observation builders
- `env/` — SUMO/TraCI environment wrappers and mock env
- `models/` — GAT, baselines, and encoder modules
- `training/` — training loop, replay buffer, reward, and multi-network orchestration
- `evaluation/` — metrics, diagnostics, transfer, and ablation scripts
- `notebooks/` — experiments and analysis notebooks
- `tests/` — unit and integration tests

## Documentation index

| Doc | Covers |
|---|---|
| `docs/mock_env.md` | MockEnv — SUMO-free random-obs environment (R0) |
| `docs/traffic_env.md` | TrafficEnv — real SUMO/TraCI environment + demand generator |
| `docs/graph_builder.md` | Graph dict schema (`node_ids`, `edge_index`, `phase_features`, `node_meta`) |
| `docs/node_encoder.md` | NodeEncoder (obs → embedding) and `pad_obs_dict` |
| `docs/phase_head.md` | PhaseHead (FRAP-style phase scoring) and `pad_phase_features` |
| `docs/trainer.md` | DQNTrainer — R2 shared-parameter training loop |
| `docs/perception.md` | Perception pipeline (R3): `apply_perception`, `ObservationImputer`, `PressureReward` |
| `docs/gat_policy.md` | GAT policy (R4): `GATPolicy`, `forward_batch`, trainer changes, eval results |
| `docs/agent_spec_gat_and_baselines.md` | R4+ GAT policy and baseline specifications |
| `docs/architecture_plan_v2.md` | Full project architecture and design decisions |

## Build rings completed

| Ring | Status | What it delivers |
|---|---|---|
| R0 | Done | MockEnv, graph_builder, obs contract, config files |
| R1 | Done | NodeEncoder, PhaseHead, pad utilities |
| R2 | Done | DQNTrainer, ReplayBuffer, compute_pressure |
| R3 | Done | apply_perception, ObservationImputer, PressureReward, NodeEncoder cat fix |
| Real env | Done | TrafficEnv (SUMO/TraCI), demand_generator, 26 tests |
| R4 | Done | GATPolicy (1-hop GAT, zero_hop ablation), forward_batch, eval runner, train_r4.py |
| R5 | Next | Second GAT layer (2-hop receptive field) |
