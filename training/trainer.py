"""DQN trainer with one-layer GAT message passing (R4).

NodeEncoder → GATPolicy → PhaseHead, one shared weight set across all intersections.
The zero_hop flag on GATPolicy is the only toggle between 0-hop ablation and full R4.
"""

import copy
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.perception import apply_perception
from models.gat_policy import GATPolicy
from models.node_encoder import NodeEncoder, pad_obs_dict
from models.phase_head import PhaseHead, pad_phase_features
from training.replay_buffer import ReplayBuffer
from training.reward import ObservationImputer, PressureReward, compute_pressure


class DQNTrainer:
    """Shared-parameter DQN trainer with GAT message passing (R4).

    Args:
        cfg:    Config dict or OmegaConf DictConfig with keys matching configs/gat.yaml
        env:    MockEnv (or TrafficEnv) — already constructed, exposes reset/step API
        device: torch.device for model parameters and training tensors
    """

    def __init__(self, cfg, env, device: torch.device = torch.device("cpu")) -> None:
        self.cfg = cfg
        self.env = env
        self.device = device

        # Derive dims and build model from a probe reset
        obs_dict, graph = env.reset(seed=_cfg(cfg, "seed", 1))
        obs_dim, _ = pad_obs_dict(obs_dict)
        phase_feat_dim, _ = pad_phase_features(graph)

        hidden_dim    = _cfg(cfg, "model.hidden_dim", 128)
        embed_dim     = _cfg(cfg, "model.embed_dim", 64)
        head_hidden   = _cfg(cfg, "model.head_hidden_dim", 64)
        gat_heads     = _cfg(cfg, "model.gat.num_heads",       4)
        gat_out_ph    = _cfg(cfg, "model.gat.out_per_head",   32)
        gat_num_layers = _cfg(cfg, "model.gat.num_layers",    1)
        gat_l2_out_ph  = _cfg(cfg, "model.gat.l2_out_per_head", 16)
        gat_zero_hop   = _cfg(cfg, "model.gat.zero_hop", False)
        gat_typed      = _cfg(cfg, "model.gat.typed_edges", False)
        self._neighbor_masking = _cfg(cfg, "model.gat.neighbor_masking", False)

        # NodeEncoder: pass obs_dim (NOT obs_dim*2) — doubling happens inside the constructor
        self.encoder = NodeEncoder(obs_dim, hidden_dim, embed_dim).to(device)
        self.gat     = GATPolicy(
            in_channels=embed_dim,
            num_heads=gat_heads,
            out_per_head=gat_out_ph,
            num_layers=gat_num_layers,
            l2_out_per_head=gat_l2_out_ph,
            zero_hop=gat_zero_hop,
            typed_edges=gat_typed,
        ).to(device)
        # PhaseHead input = GAT output dim (128 for default 4 heads × 32), not NodeEncoder embed_dim
        self.head = PhaseHead(self.gat.out_channels, phase_feat_dim, head_hidden).to(device)

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_gat     = copy.deepcopy(self.gat)
        self.target_head    = copy.deepcopy(self.head)
        for p in (
            list(self.target_encoder.parameters())
            + list(self.target_gat.parameters())
            + list(self.target_head.parameters())
        ):
            p.requires_grad_(False)

        lr = _cfg(cfg, "trainer.lr", 1e-3)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.gat.parameters())
            + list(self.head.parameters()),
            lr=lr,
        )

        capacity   = _cfg(cfg, "trainer.replay_buffer_size", 50000)
        node_ids   = graph["node_ids"]
        self.buffer = ReplayBuffer(capacity, obs_dim, node_ids, device=torch.device("cpu"))

        queue_weight    = float(_cfg(cfg, "reward.queue_weight",    0.0))
        pressure_weight = float(_cfg(cfg, "reward.pressure_weight", 1.0))
        self._reward_scale = float(_cfg(cfg, "reward.scale", 1.0))
        self.pressure_fn = PressureReward(queue_weight=queue_weight, pressure_weight=pressure_weight)
        self.obs_imputer = ObservationImputer()

        self._obs_dim        = obs_dim
        self._phase_feat_dim = phase_feat_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, num_episodes: int | None = None) -> dict:
        """Run the DQN training loop.

        Returns:
            metrics dict with keys:
                episode_returns  list[float]  total reward per episode
                episode_lengths  list[int]    steps per episode
                losses           list[float]  Huber loss per gradient step
                epsilons         list[float]  epsilon at each gradient step
                q_mean           list[float]  mean max-Q per gradient step
        """
        cfg = self.cfg
        if num_episodes is None:
            num_episodes = _cfg(cfg, "trainer.num_episodes", 500)

        batch_size         = _cfg(cfg, "trainer.batch_size", 64)
        warmup_steps       = _cfg(cfg, "trainer.warmup_steps", 1000)
        gamma              = _cfg(cfg, "trainer.gamma", 0.99)
        grad_clip          = _cfg(cfg, "trainer.grad_clip", 10.0)
        target_update_freq = _cfg(cfg, "trainer.target_update_steps", 1000)
        target_tau         = _cfg(cfg, "trainer.target_tau", None)
        eps_start          = _cfg(cfg, "trainer.epsilon_start", 1.0)
        eps_end            = _cfg(cfg, "trainer.epsilon_end", 0.05)
        eps_decay_steps    = _cfg(cfg, "trainer.epsilon_decay_steps", 10000)
        checkpoint_every   = _cfg(cfg, "trainer.checkpoint_every", 100)
        checkpoint_dir     = _cfg(cfg, "trainer.checkpoint_dir", "checkpoints/r4_gat")
        use_pressure       = _cfg(cfg, "reward.use_pressure", False)
        perception_severity = _cfg(cfg, "perception.severity", 0.0)
        sentinel           = _cfg(cfg, "perception.sentinel_value", -1.0)
        seed               = _cfg(cfg, "seed", 1)

        metrics = {
            "episode_returns":  [],
            "episode_lengths":  [],
            "losses":           [],
            "epsilons":         [],
            "q_mean":           [],
            "avg_waiting_time": [],
            "throughput":       [],
            "avg_queue_length": [],
            "episode_routes":   [],
        }
        total_steps = 0
        grad_steps  = 0
        start_time  = time.time()

        obs_dict, graph = self.env.reset(seed=seed)
        _, padded_pf   = pad_phase_features(graph)
        node_ids       = graph["node_ids"]

        for episode in range(num_episodes):
            obs_dict, graph = self.env.reset()
            active_route = Path(getattr(self.env, "active_route", "")).stem
            obs_dict = apply_perception(obs_dict, perception_severity, sentinel)
            self.pressure_fn.reset()
            self.obs_imputer.reset()
            obs_dict = self.obs_imputer.impute(obs_dict)
            _, padded_obs = pad_obs_dict(obs_dict)
            done = False
            episode_reward = 0.0
            episode_steps  = 0
            sum_wait       = 0.0
            total_arrived  = 0
            sum_queue      = 0.0

            while not done:
                epsilon = _linear_decay(grad_steps, eps_start, eps_end, eps_decay_steps)
                actions = self._select_actions(padded_obs, graph, padded_pf, epsilon)

                next_obs_dict, _, reward_dict, done, info = self.env.step(actions)
                next_obs_dict = apply_perception(next_obs_dict, perception_severity, sentinel)
                if use_pressure:
                    reward_dict = self.pressure_fn.compute(next_obs_dict, graph)
                    if self._reward_scale != 1.0:
                        reward_dict = {k: v * self._reward_scale for k, v in reward_dict.items()}
                next_obs_dict = self.obs_imputer.impute(next_obs_dict)
                _, padded_next = pad_obs_dict(next_obs_dict)

                self.buffer.push(padded_obs, padded_next, actions, reward_dict, done)

                episode_reward += sum(reward_dict.values())
                episode_steps  += 1
                total_steps    += 1
                sum_wait       += float(info.get("step_mean_waiting_time", 0.0))
                total_arrived  += int(info.get("step_throughput", 0))
                sum_queue      += float(info.get("step_queue_length", 0.0))

                if len(self.buffer) >= warmup_steps:
                    batch = self.buffer.sample(batch_size)
                    loss, q_m = self._compute_loss(batch, graph, padded_pf, gamma)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters())
                        + list(self.gat.parameters())
                        + list(self.head.parameters()),
                        grad_clip,
                    )
                    self.optimizer.step()
                    grad_steps += 1

                    if target_tau is not None:
                        for p_on, p_tgt in zip(
                            list(self.encoder.parameters())
                            + list(self.gat.parameters())
                            + list(self.head.parameters()),
                            list(self.target_encoder.parameters())
                            + list(self.target_gat.parameters())
                            + list(self.target_head.parameters()),
                        ):
                            p_tgt.data.mul_(1.0 - target_tau).add_(p_on.data * target_tau)
                    elif total_steps % target_update_freq == 0:
                        self.target_encoder.load_state_dict(self.encoder.state_dict())
                        self.target_gat.load_state_dict(self.gat.state_dict())
                        self.target_head.load_state_dict(self.head.state_dict())

                    metrics["losses"].append(loss.item())
                    metrics["epsilons"].append(epsilon)
                    metrics["q_mean"].append(q_m)

                padded_obs = padded_next

            metrics["episode_returns"].append(episode_reward)
            metrics["episode_lengths"].append(episode_steps)
            metrics["avg_waiting_time"].append(sum_wait / max(1, episode_steps))
            metrics["throughput"].append(total_arrived)
            metrics["avg_queue_length"].append(sum_queue / max(1, episode_steps))
            metrics["episode_routes"].append(active_route)

            n_losses = len(metrics["losses"])
            avg_loss = (
                sum(metrics["losses"][-100:]) / min(n_losses, 100)
                if n_losses else float("nan")
            )
            elapsed = int(time.time() - start_time)
            h, rem  = divmod(elapsed, 3600)
            m, s    = divmod(rem, 60)
            t_str   = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
            loss_str = "(warming up)" if math.isnan(avg_loss) else f"{avg_loss:.4f}"
            avg_wait = sum_wait / max(1, episode_steps)
            avg_q    = sum_queue / max(1, episode_steps)
            print(
                f"[Ep {episode + 1:4d}/{num_episodes}]"
                f"  return={episode_reward:+9.2f}"
                f"  wait={avg_wait:.1f}s"
                f"  tput={total_arrived:4d}"
                f"  q={avg_q:.1f}"
                f"  loss={loss_str}"
                f"  eps={epsilon:.3f}"
                f"  route={active_route}"
                f"  elapsed={t_str}",
                flush=True,
            )

            if (episode + 1) % checkpoint_every == 0:
                path = Path(checkpoint_dir) / f"checkpoint_ep{episode + 1}.pt"
                self.save_checkpoint(str(path), total_steps, metrics)

        return metrics

    def _select_actions(self, padded_obs: dict, graph: dict, padded_pf: list, epsilon: float) -> dict:
        """ε-greedy action selection for all nodes.

        Encodes all N nodes in one vectorized pass, runs a single GAT call,
        then selects per-node actions.

        Returns:
            {node_id: phase_idx}
        """
        node_ids    = graph["node_ids"]
        node_to_idx = graph["node_to_idx"]
        node_meta   = graph["node_meta"]
        edge_index  = graph["edge_index"].to(self.device)
        edge_type   = graph["edge_type"].to(self.device)

        self.encoder.eval()
        self.gat.eval()
        self.head.eval()
        with torch.no_grad():
            obs_stack = torch.stack([padded_obs[nid][0] for nid in node_ids]).to(self.device)
            val_stack = torch.stack([padded_obs[nid][1] for nid in node_ids]).to(self.device)
            emb           = self.encoder(obs_stack, val_stack)        # [N, embed_dim]
            node_validity = val_stack if self._neighbor_masking else None
            gat_emb       = self.gat(emb, edge_index, edge_type, node_validity)  # [N, gat_out_dim]

            actions = {}
            for col, node_id in enumerate(node_ids):
                node_idx   = node_to_idx[node_id]
                num_phases = node_meta[node_idx]["num_phases"]

                current_phase = int(obs_stack[col, :num_phases].argmax().item())
                mask = node_meta[node_idx]["valid_transition_mask"][current_phase].to(self.device)

                if torch.rand(1).item() < epsilon:
                    valid_indices = mask.nonzero(as_tuple=False).flatten()
                    if valid_indices.numel() == 0:
                        actions[node_id] = 0
                    else:
                        actions[node_id] = int(
                            valid_indices[torch.randint(len(valid_indices), (1,))].item()
                        )
                else:
                    scores = self.head(gat_emb[col], padded_pf[node_idx], mask)
                    actions[node_id] = int(scores.argmax().item())

        self.encoder.train()
        self.gat.train()
        self.head.train()
        return actions

    def _compute_loss(
        self, batch: dict, graph: dict, padded_pf: list, gamma: float
    ) -> tuple[torch.Tensor, float]:
        """Huber loss averaged over all nodes.

        Encoder is vectorized over B*N in one call.  GAT loops over B (one pass
        per sample) — PyG Batch optimization is deferred to R8.

        Returns:
            (loss tensor, mean max-Q scalar for logging)
        """
        node_to_idx = graph["node_to_idx"]
        node_meta   = graph["node_meta"]
        node_ids    = graph["node_ids"]
        edge_index  = graph["edge_index"].to(self.device)
        edge_type   = graph["edge_type"].to(self.device)
        B           = batch["obs"].shape[0]
        N           = len(node_ids)

        obs_all      = batch["obs"].to(self.device)       # [B, N, obs_dim]
        val_all      = batch["validity"].to(self.device)
        next_obs_all = batch["next_obs"].to(self.device)
        next_val_all = batch["next_val"].to(self.device)
        actions_all  = batch["actions"].to(self.device)   # [B, N] int64
        rewards_all  = batch["rewards"].to(self.device)   # [B, N]
        dones        = batch["dones"].to(self.device)      # [B]

        # Encode all B*N observations in one vectorized pass
        emb_flat = self.encoder(
            obs_all.view(B * N, -1), val_all.view(B * N, -1)
        ).view(B, N, -1)                                  # [B, N, embed_dim]

        # GAT: one forward per batch sample; R8 will replace with PyG Batch
        gat_out = torch.stack([
            self.gat(
                emb_flat[b], edge_index, edge_type,
                val_all[b] if self._neighbor_masking else None,
            ) for b in range(B)
        ])                                                # [B, N, gat_out_dim]

        with torch.no_grad():
            next_emb_flat = self.target_encoder(
                next_obs_all.view(B * N, -1), next_val_all.view(B * N, -1)
            ).view(B, N, -1)
            next_gat_out = torch.stack([
                self.target_gat(
                    next_emb_flat[b], edge_index, edge_type,
                    next_val_all[b] if self._neighbor_masking else None,
                ) for b in range(B)
            ])                                            # [B, N, gat_out_dim]

        all_losses = []
        all_q_max  = []

        for col, node_id in enumerate(node_ids):
            node_idx   = node_to_idx[node_id]
            num_phases = node_meta[node_idx]["num_phases"]
            pf         = padded_pf[node_idx]
            all_true   = torch.ones(num_phases, dtype=torch.bool, device=self.device)

            node_emb      = gat_out[:, col, :]       # [B, gat_out_dim]
            next_node_emb = next_gat_out[:, col, :]  # [B, gat_out_dim]

            q_online = self.head.forward_batch(node_emb, pf, all_true)          # [B, P]
            q_pred   = q_online.gather(1, actions_all[:, col].unsqueeze(1)).squeeze(1)  # [B]

            with torch.no_grad():
                q_next_scores = self.target_head.forward_batch(next_node_emb, pf, all_true)
                q_next_max    = q_next_scores.max(dim=1).values                  # [B]
                q_target      = rewards_all[:, col] + gamma * q_next_max * (1.0 - dones)

            loss_i = F.huber_loss(q_pred, q_target)
            all_losses.append(loss_i)
            all_q_max.append(q_next_max.mean().item())

        loss   = torch.stack(all_losses).mean()
        q_mean = sum(all_q_max) / max(len(all_q_max), 1)
        return loss, q_mean

    def save_checkpoint(self, path: str, step: int, metrics: dict) -> None:
        """Save model weights, optimizer state, and metrics history."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "encoder":    self.encoder.state_dict(),
                "gat":        self.gat.state_dict(),
                "head":       self.head.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "step":       step,
                "cfg":        _cfg_to_dict(self.cfg),
                "metrics":    metrics,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str, cfg, env, device: torch.device = torch.device("cpu")
    ) -> "DQNTrainer":
        """Restore a trainer from a checkpoint file."""
        trainer = cls(cfg, env, device)
        ckpt = torch.load(path, map_location=device)
        trainer.encoder.load_state_dict(ckpt["encoder"])
        trainer.gat.load_state_dict(ckpt["gat"])
        trainer.head.load_state_dict(ckpt["head"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.target_encoder.load_state_dict(ckpt["encoder"])
        trainer.target_gat.load_state_dict(ckpt["gat"])
        trainer.target_head.load_state_dict(ckpt["head"])
        return trainer


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _linear_decay(step: int, start: float, end: float, decay_steps: int) -> float:
    fraction = min(step / max(decay_steps, 1), 1.0)
    return end + (start - end) * (1.0 - fraction)


def _cfg(cfg, dotted_key: str, default=None):
    """Read a dotted key from a plain dict or OmegaConf DictConfig."""
    try:
        from omegaconf import OmegaConf
        node = OmegaConf.select(cfg, dotted_key)
        return node if node is not None else default
    except ImportError:
        pass
    obj = cfg
    for part in dotted_key.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return default
        if obj is None:
            return default
    return obj


def _cfg_to_dict(cfg) -> dict:
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)
    except (ImportError, Exception):
        return dict(cfg) if isinstance(cfg, dict) else {}
