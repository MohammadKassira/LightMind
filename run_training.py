"""run_training.py — A-to-Z DQN training on any SUMO network.

Edit the ── CONFIG ── block at the top (network paths only), then run:
    python run_training.py

All hyperparameters (episodes, batch size, epsilon schedule, etc.) live in
configs/train.yaml — change them there, not here.
"""

import copy
import json
import math
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ── CONFIG ────────────────────────────────────────────────────────────────────
NET_FILE   = "networks/external/RESCO/cologne1/cologne1.net.xml"
ROUTE_FILE = "networks/external/RESCO/cologne1/cologne1.rou.xml"
BEGIN_TIME = 25200   # simulation start in seconds
             # cologne1 / cologne3 / cologne8  → 25200  (7 AM)
             # ingolstadt1 / 7 / 21            → 57600  (4 PM)
             # grid4x4 / arterial4x4           → 0

RUN_NAME = "cologne1_dqn"   # output goes to checkpoints/<RUN_NAME>/
# ──────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Training helpers  (mirrors training/trainer.py — kept in sync manually)
# ---------------------------------------------------------------------------

def _linear_decay(step: int, start: float, end: float, decay_steps: int) -> float:
    fraction = min(step / max(decay_steps, 1), 1.0)
    return end + (start - end) * (1.0 - fraction)


def _select_actions(encoder, head, padded_obs, graph, padded_pf, epsilon, device):
    actions     = {}
    node_to_idx = graph["node_to_idx"]
    node_meta   = graph["node_meta"]

    encoder.eval()
    head.eval()
    with torch.no_grad():
        for node_id, (obs, val) in padded_obs.items():
            node_idx   = node_to_idx[node_id]
            num_phases = node_meta[node_idx]["num_phases"]
            obs_d      = obs.to(device)
            val_d      = val.to(device)

            current_phase = int(obs_d[:num_phases].argmax().item())
            mask = node_meta[node_idx]["valid_transition_mask"][current_phase].to(device)

            if torch.rand(1).item() < epsilon:
                valid = mask.nonzero(as_tuple=False).flatten()
                actions[node_id] = (
                    int(valid[torch.randint(len(valid), (1,))].item())
                    if valid.numel() > 0 else 0
                )
            else:
                emb    = encoder(obs_d, val_d)
                scores = head(emb, padded_pf[node_idx], mask)
                actions[node_id] = int(scores.argmax().item())
    encoder.train()
    head.train()
    return actions


def _compute_loss(encoder, head, t_encoder, t_head,
                  batch, graph, padded_pf, gamma, device):
    node_to_idx = graph["node_to_idx"]
    node_meta   = graph["node_meta"]
    node_ids    = graph["node_ids"]
    B           = batch["obs"].shape[0]

    all_losses = []
    all_q_max  = []

    for col, node_id in enumerate(node_ids):
        node_idx   = node_to_idx[node_id]
        num_phases = node_meta[node_idx]["num_phases"]
        pf         = padded_pf[node_idx]

        obs_i      = batch["obs"][:, col, :].to(device)
        val_i      = batch["validity"][:, col, :].to(device)
        next_obs_i = batch["next_obs"][:, col, :].to(device)
        next_val_i = batch["next_val"][:, col, :].to(device)
        actions_i  = batch["actions"][:, col].to(device)
        rewards_i  = batch["rewards"][:, col].to(device)
        dones      = batch["dones"].to(device)

        q_online_list = []
        for b in range(B):
            emb = encoder(obs_i[b], val_i[b])
            all_true = torch.ones(num_phases, dtype=torch.bool, device=device)
            q_online_list.append(head(emb, pf, all_true))
        q_online = torch.stack(q_online_list)
        q_pred   = q_online.gather(1, actions_i.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_tgt_list = []
            for b in range(B):
                t_emb    = t_encoder(next_obs_i[b], next_val_i[b])
                all_true = torch.ones(num_phases, dtype=torch.bool, device=device)
                q_tgt_list.append(t_head(t_emb, pf, all_true).max())
            q_next_max = torch.stack(q_tgt_list)
            q_target   = rewards_i + gamma * q_next_max * (1.0 - dones)

        all_losses.append(F.huber_loss(q_pred, q_target))
        all_q_max.append(q_next_max.mean().item())

    loss   = torch.stack(all_losses).mean()
    q_mean = sum(all_q_max) / max(len(all_q_max), 1)
    return loss, q_mean


def _save_checkpoint(path: Path, encoder, head, optimizer, step, cfg, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder":   encoder.state_dict(),
        "head":      head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "cfg":       cfg,
        "metrics":   metrics,
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Deferred imports so SUMO bootstrap error (missing SUMO_HOME) surfaces cleanly
    from env.traffic_env import TrafficEnv
    from env.perception import apply_perception
    from models.node_encoder import NodeEncoder, pad_obs_dict
    from models.phase_head import PhaseHead, pad_phase_features
    from training.replay_buffer import ReplayBuffer
    from training.reward import ObservationImputer, PressureReward

    # All hyperparameters come from configs/train.yaml
    cfg = yaml.safe_load(Path("configs/train.yaml").read_text())

    def _c(dotted, default=None):
        obj = cfg
        for k in dotted.split("."):
            obj = obj.get(k) if isinstance(obj, dict) else None
            if obj is None:
                return default
        return obj

    device       = torch.device(_c("device", "cpu"))
    use_pressure = _c("reward.use_pressure", False)
    severity     = _c("perception.severity", 0.0)
    sentinel     = _c("perception.sentinel_value", -1.0)
    num_episodes = _c("trainer.num_episodes", 500)
    ckpt_dir     = Path(_c("trainer.checkpoint_dir", "checkpoints")) / RUN_NAME

    # ── Environment ──────────────────────────────────────────────────────────
    env = TrafficEnv(
        net_file   = NET_FILE,
        route_file = ROUTE_FILE,
        begin_time = BEGIN_TIME,
        max_steps  = _c("env.max_steps", 200),
    )

    # ── Model (probe reset to get obs_dim and phase_feat_dim) ─────────────
    obs_dict, graph = env.reset(seed=_c("seed", 1))
    obs_dim,  _     = pad_obs_dict(obs_dict)
    pf_dim,   _     = pad_phase_features(graph)

    encoder = NodeEncoder(
        obs_dim, _c("model.hidden_dim", 128), _c("model.embed_dim", 64)
    ).to(device)
    head = PhaseHead(
        _c("model.embed_dim", 64), pf_dim, _c("model.head_hidden_dim", 64)
    ).to(device)

    t_encoder = copy.deepcopy(encoder)
    t_head    = copy.deepcopy(head)
    for p in list(t_encoder.parameters()) + list(t_head.parameters()):
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=_c("trainer.lr", 1e-3),
    )

    # ── Auxiliary ─────────────────────────────────────────────────────────
    buffer = ReplayBuffer(
        _c("trainer.replay_buffer_size", 50000),
        obs_dim,
        graph["node_ids"],
        device=torch.device("cpu"),
    )
    pressure_fn = PressureReward()
    obs_imputer = ObservationImputer()

    # ── Hyperparameters ───────────────────────────────────────────────────
    batch_size         = _c("trainer.batch_size", 64)
    warmup_steps       = _c("trainer.warmup_steps", 1000)
    gamma              = _c("trainer.gamma", 0.99)
    grad_clip          = _c("trainer.grad_clip", 10.0)
    target_update_freq = _c("trainer.target_update_steps", 1000)
    eps_start          = _c("trainer.epsilon_start", 1.0)
    eps_end            = _c("trainer.epsilon_end", 0.05)
    eps_decay_steps    = _c("trainer.epsilon_decay_steps", 10000)
    checkpoint_every   = _c("trainer.checkpoint_every", 100)

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = {
        "episode_returns": [],
        "episode_lengths": [],
        "losses":          [],
        "epsilons":        [],
        "q_mean":          [],
    }

    _, padded_pf = pad_phase_features(graph)
    total_steps  = 0
    start_time   = time.time()

    print(f"Training {RUN_NAME!r}  —  {num_episodes} episodes  [configs/train.yaml]")
    print(f"  net:          {NET_FILE}")
    print(f"  route:        {ROUTE_FILE}")
    print(f"  nodes:        {graph['node_ids']}")
    print(f"  device:       {device}  |  use_pressure: {use_pressure}  |  severity: {severity}")
    print(f"  outputs:      {ckpt_dir}/")
    print()

    # ── Training loop ─────────────────────────────────────────────────────
    for episode in range(num_episodes):
        obs_dict, graph = env.reset()
        obs_dict = apply_perception(obs_dict, severity, sentinel)
        pressure_fn.reset()
        obs_imputer.reset()
        obs_dict = obs_imputer.impute(obs_dict)
        _, padded_obs = pad_obs_dict(obs_dict)

        done           = False
        episode_reward = 0.0
        episode_steps  = 0

        while not done:
            epsilon = _linear_decay(total_steps, eps_start, eps_end, eps_decay_steps)
            actions = _select_actions(
                encoder, head, padded_obs, graph, padded_pf, epsilon, device
            )

            next_obs_dict, _, reward_dict, done, _ = env.step(actions)
            next_obs_dict = apply_perception(next_obs_dict, severity, sentinel)
            if use_pressure:
                reward_dict = pressure_fn.compute(next_obs_dict, graph)
            next_obs_dict = obs_imputer.impute(next_obs_dict)
            _, padded_next = pad_obs_dict(next_obs_dict)

            buffer.push(padded_obs, padded_next, actions, reward_dict, done)

            episode_reward += sum(reward_dict.values())
            episode_steps  += 1
            total_steps    += 1

            if len(buffer) >= warmup_steps:
                batch = buffer.sample(batch_size)
                loss, q_m = _compute_loss(
                    encoder, head, t_encoder, t_head,
                    batch, graph, padded_pf, gamma, device,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()), grad_clip
                )
                optimizer.step()

                metrics["losses"].append(loss.item())
                metrics["epsilons"].append(epsilon)
                metrics["q_mean"].append(q_m)

            if total_steps % target_update_freq == 0:
                t_encoder.load_state_dict(encoder.state_dict())
                t_head.load_state_dict(head.state_dict())

            padded_obs = padded_next

        metrics["episode_returns"].append(episode_reward)
        metrics["episode_lengths"].append(episode_steps)

        # ── Per-episode log ──────────────────────────────────────────────
        n_losses  = len(metrics["losses"])
        avg_loss  = (
            sum(metrics["losses"][-100:]) / min(n_losses, 100)
            if n_losses else float("nan")
        )
        elapsed  = int(time.time() - start_time)
        h, rem   = divmod(elapsed, 3600)
        m, s     = divmod(rem, 60)
        t_str    = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
        loss_str = "(warming up)" if math.isnan(avg_loss) else f"{avg_loss:.4f}"

        print(
            f"[Ep {episode + 1:4d}/{num_episodes}]"
            f"  return={episode_reward:+9.2f}"
            f"  len={episode_steps:4d}"
            f"  avg_loss={loss_str}"
            f"  ε={epsilon:.3f}"
            f"  elapsed={t_str}",
            flush=True,
        )

        # ── Periodic checkpoint ────────────────────────────────────────
        if (episode + 1) % checkpoint_every == 0:
            ckpt = ckpt_dir / f"ep{episode + 1:05d}.pt"
            _save_checkpoint(ckpt, encoder, head, optimizer, total_steps, cfg, metrics)

    # ── Final saves ───────────────────────────────────────────────────────
    final_ckpt = ckpt_dir / "final.pt"
    _save_checkpoint(final_ckpt, encoder, head, optimizer, total_steps, cfg, metrics)
    print(f"\nCheckpoint → {final_ckpt}")

    metrics_path = ckpt_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics    → {metrics_path}")

    env.close()


if __name__ == "__main__":
    main()
