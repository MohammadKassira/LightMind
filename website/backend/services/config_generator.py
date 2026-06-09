"""Training config generator for capacity-based route training.

Creates YAML config files for model/training/trainer.py.
All hyperparameters are derived from a single value: num_episodes.
"""

from pathlib import Path

# Steps per episode: route files cover 3600s, delta_time=5s → 720 action steps.
# This is a fixed physical constant (5s = minimum safe signal phase duration).
_STEPS_PER_EPISODE = 720


def create_training_config(
    session_id: str,
    net_path: Path,
    route_files: dict[str, Path],
    output_dir: Path,
    episodes: int = 1000,
    begin_time: int = 0,
    stop_file: str = "",
    pass_threshold_pct: float = 25.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not net_path.exists():
        raise FileNotFoundError(f"Network file not found: {net_path}")
    for variant, route_path in route_files.items():
        if not route_path.exists():
            raise FileNotFoundError(f"Route file not found ({variant}): {route_path}")

    # All hyperparameters derived from episodes × steps_per_episode = total budget
    S = _STEPS_PER_EPISODE
    T = episodes * S  # total budgeted training steps

    warmup_steps        = max(1000,  T // 50)               # 2%  — fill buffer before learning
    epsilon_decay_steps = max(5000,  T * 6 // 10)           # 60% — long exploration phase
    replay_buffer_size  = max(10000, min(200000, T // 10))  # 10% — ~100 episodes of experience
    target_update_steps = max(1000,  T // 50)               # 2%  — stable target cadence
    checkpoint_every    = max(50,    episodes // 10)         # every 10% of episodes

    p = lambda path: Path(path).resolve().as_posix()  # forward slashes everywhere — safe in YAML
    config = f"""network:
  net: {p(net_path)}
  rou: {p(route_files['dense'])}
  route_files:
    - {p(route_files['light'])}
    - {p(route_files['medium'])}
    - {p(route_files['dense'])}
    - {p(route_files['heavy'])}
  begin_time: {begin_time}
  max_steps: {S}
  name: {session_id}

model:
  hidden_dim: 128
  embed_dim: 64
  head_hidden_dim: 64
  max_obs_dim: 80
  max_phase_feat_dim: 32
  gat:
    num_heads: 4
    out_per_head: 32
    num_layers: 2
    l2_out_per_head: 16
    typed_edges: true
    zero_hop: false
    neighbor_masking: true

trainer:
  lr: 0.0001
  batch_size: 64
  gamma: 0.99
  grad_clip: 10.0
  epsilon_start: 1.0
  epsilon_end: 0.05
  num_workers: 1
  num_episodes: {episodes}
  warmup_steps: {warmup_steps}
  epsilon_decay_steps: {epsilon_decay_steps}
  replay_buffer_size: {replay_buffer_size}
  target_update_steps: {target_update_steps}
  checkpoint_every: {checkpoint_every}
  checkpoint_dir: "checkpoints/{session_id}"
  stop_file: {stop_file}
  convergence_window: 10
  convergence_min_episodes: {max(20, episodes // 4)}
  convergence_delta_return: null
  convergence_delta_wait: 5.0

reward:
  use_pressure: true
  queue_weight: 0.5
  pressure_weight: 0.5
  scale: 20

perception:
  severity: 0.0
  sentinel_value: -1.0

seed: 42

eval_pass_threshold_pct: {pass_threshold_pct}
"""

    config_path = output_dir / f"{session_id}_training_config.yaml"
    config_path.write_text(config, encoding="utf-8")
    return config_path
