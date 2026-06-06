"""Cross-network orchestration for shared-parameter DQN training (R8).

MultiNetworkTrainer reads max_obs_dim/max_phase_feat_dim ceilings from cfg,
probes all environments to validate that no network exceeds the ceiling, then
builds one shared model and per-network replay buffers.

Key design choices (documented here to prevent drift):
- Epsilon decays with a GLOBAL step counter across all networks combined.
  There is no per-network epsilon tracker.
- Warmup is PER BUFFER: gradient steps for network k start only when
  len(buffers[k]) >= warmup_steps. With K networks, effective start of
  training is ~warmup_steps * K total env steps.
- Within any single gradient step, batch + graph come from the SAME network
  (consistent N). Cross-network batch mixing is not done in R8.
"""

import copy
import math
import os
import random
import threading
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
from training.reward import ObservationImputer, PressureReward
from training.trainer import _cfg, _cfg_to_dict, _linear_decay


class MultiNetworkTrainer:
    """Shared-parameter DQN trainer that cycles across multiple networks (R8).

    Args:
        cfg:           Config dict with model/trainer/reward/perception keys.
                       Must include cfg["model"]["max_obs_dim"] and
                       cfg["model"]["max_phase_feat_dim"] as deployment ceilings.
        envs:          List of pre-constructed env instances (TrafficEnv or MockEnv).
        device:        torch.device for model parameters and training tensors.
        network_names: Optional list of names matching envs (for metrics/logging).
    """

    def __init__(
        self,
        cfg,
        envs: list,
        device: torch.device = torch.device("cpu"),
        network_names: list[str] | None = None,
    ) -> None:
        self.cfg    = cfg
        self.envs   = envs
        self.device = device
        self.network_names = (
            network_names
            if network_names is not None
            else [str(i) for i in range(len(envs))]
        )

        # --- Read deployment ceilings from config ---
        max_obs_dim       = _cfg(cfg, "model.max_obs_dim", None)
        max_phase_feat_dim = _cfg(cfg, "model.max_phase_feat_dim", None)
        if max_obs_dim is None or max_phase_feat_dim is None:
            raise ValueError(
                "cfg must include model.max_obs_dim and model.max_phase_feat_dim. "
                "These are deployment ceilings that fix the model architecture "
                "independently of which networks are actually present."
            )
        self.global_obs_dim        = int(max_obs_dim)
        self.global_phase_feat_dim = int(max_phase_feat_dim)

        seed = _cfg(cfg, "seed", 1)

        # --- Probe all envs: validate dims ≤ ceiling and cache graphs ---
        graphs          = []
        local_obs_dims  = []  # per-network actual obs_dim (< global ceiling)
        for k, env in enumerate(envs):
            obs_dict, graph = env.reset(seed=seed)
            local_obs_dim, _ = pad_obs_dict(obs_dict)
            local_pf_dim, _  = pad_phase_features(graph)
            if local_obs_dim > self.global_obs_dim:
                raise ValueError(
                    f"Network '{self.network_names[k]}' obs_dim={local_obs_dim} "
                    f"exceeds cfg model.max_obs_dim={self.global_obs_dim}. "
                    f"Raise max_obs_dim in your config."
                )
            if local_pf_dim > self.global_phase_feat_dim:
                raise ValueError(
                    f"Network '{self.network_names[k]}' phase_feat_dim={local_pf_dim} "
                    f"exceeds cfg model.max_phase_feat_dim={self.global_phase_feat_dim}. "
                    f"Raise max_phase_feat_dim in your config."
                )
            graphs.append(graph)
            local_obs_dims.append(local_obs_dim)
        self.graphs         = graphs
        # Per-network actual obs_dim for slicing validity before GAT neighbor masking.
        # Padding zeros in positions [local_obs_dim:global_obs_dim] would otherwise
        # deflate the mean below the 0.75 valid-sensor threshold on every node.
        self.local_obs_dims = local_obs_dims

        # --- Per-network padded phase features at global ceiling ---
        self.padded_pf = []
        for k, graph in enumerate(graphs):
            _, pf = pad_phase_features(graph, target_dim=self.global_phase_feat_dim)
            self.padded_pf.append(pf)

        # --- Shared model (built with ceiling dims) ---
        hidden_dim    = _cfg(cfg, "model.hidden_dim", 128)
        embed_dim     = _cfg(cfg, "model.embed_dim", 64)
        head_hidden   = _cfg(cfg, "model.head_hidden_dim", 64)
        gat_heads     = _cfg(cfg, "model.gat.num_heads",        4)
        gat_out_ph    = _cfg(cfg, "model.gat.out_per_head",    32)
        gat_num_layers = _cfg(cfg, "model.gat.num_layers",     1)
        gat_l2_out_ph  = _cfg(cfg, "model.gat.l2_out_per_head", 16)
        gat_zero_hop   = _cfg(cfg, "model.gat.zero_hop",       False)
        gat_typed      = _cfg(cfg, "model.gat.typed_edges",    False)
        self._neighbor_masking = _cfg(cfg, "model.gat.neighbor_masking", False)

        self.encoder = NodeEncoder(self.global_obs_dim, hidden_dim, embed_dim).to(device)
        self.gat     = GATPolicy(
            in_channels=embed_dim,
            num_heads=gat_heads,
            out_per_head=gat_out_ph,
            num_layers=gat_num_layers,
            l2_out_per_head=gat_l2_out_ph,
            zero_hop=gat_zero_hop,
            typed_edges=gat_typed,
        ).to(device)
        self.head = PhaseHead(
            self.gat.out_channels, self.global_phase_feat_dim, head_hidden
        ).to(device)

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

        # --- Per-network replay buffers (each uses global_obs_dim, own N) ---
        # Per-network replay_buffer_size overrides the global capacity.
        # Useful for large-N networks (e.g. toronto N=88) to avoid multi-GB buffers.
        global_capacity = int(_cfg(cfg, "trainer.replay_buffer_size", 50000))
        network_list    = cfg.get("networks") or []
        self.buffers = [
            ReplayBuffer(
                int(network_list[k].get("replay_buffer_size", global_capacity))
                if k < len(network_list) else global_capacity,
                self.global_obs_dim,
                g["node_ids"],
                device=torch.device("cpu"),
            )
            for k, g in enumerate(graphs)
        ]

        queue_weight    = float(_cfg(cfg, "reward.queue_weight",    0.0))
        pressure_weight = float(_cfg(cfg, "reward.pressure_weight", 1.0))
        self.pressure_fns  = [
            PressureReward(queue_weight=queue_weight, pressure_weight=pressure_weight)
            for _ in envs
        ]
        self.obs_imputers  = [ObservationImputer() for _ in envs]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, num_episodes: int | None = None) -> dict:
        """Run multi-network DQN training.

        Returns:
            metrics dict with keys:
                episode_returns  list[float]
                episode_lengths  list[int]
                losses           list[float]  Huber loss per gradient step
                epsilons         list[float]  epsilon at each gradient step
                q_mean           list[float]  mean max-Q per gradient step
                network_sequence list[str]    network name for each episode
        """
        cfg = self.cfg
        if num_episodes is None:
            num_episodes = _cfg(cfg, "trainer.num_episodes", 300)

        batch_size         = _cfg(cfg, "trainer.batch_size", 64)
        warmup_steps       = _cfg(cfg, "trainer.warmup_steps", 1000)
        gamma              = _cfg(cfg, "trainer.gamma", 0.99)
        grad_clip          = _cfg(cfg, "trainer.grad_clip", 10.0)
        target_update_freq = _cfg(cfg, "trainer.target_update_steps", 1000)
        eps_start          = _cfg(cfg, "trainer.epsilon_start", 1.0)
        eps_end            = _cfg(cfg, "trainer.epsilon_end", 0.05)
        eps_decay_steps    = _cfg(cfg, "trainer.epsilon_decay_steps", 10000)
        checkpoint_every   = _cfg(cfg, "trainer.checkpoint_every", 100)
        checkpoint_dir     = _cfg(cfg, "trainer.checkpoint_dir", "checkpoints/r8_multi")
        use_pressure       = _cfg(cfg, "reward.use_pressure", False)
        reward_scale       = float(_cfg(cfg, "reward.scale", 1.0))
        perception_severity = _cfg(cfg, "perception.severity", 0.0)
        sentinel           = _cfg(cfg, "perception.sentinel_value", -1.0)
        num_workers        = int(_cfg(cfg, "trainer.num_workers", 1))

        if num_workers > 1:
            return self._train_parallel(
                num_episodes, batch_size, warmup_steps, gamma, grad_clip,
                target_update_freq, eps_start, eps_end, eps_decay_steps,
                checkpoint_every, checkpoint_dir, use_pressure, reward_scale,
                perception_severity, sentinel, num_workers,
            )

        metrics = {
            "episode_returns":  [],
            "episode_lengths":  [],
            "losses":           [],
            "epsilons":         [],
            "q_mean":           [],
            "network_sequence": [],
        }
        # One global step counter — epsilon decays across all networks combined.
        total_steps = 0
        start_time  = time.time()

        for episode in range(num_episodes):
            k    = random.randrange(len(self.envs))
            name = self.network_names[k]
            env        = self.envs[k]
            graph      = self.graphs[k]
            padded_pf  = self.padded_pf[k]
            buf        = self.buffers[k]
            pressure   = self.pressure_fns[k]
            imputer    = self.obs_imputers[k]

            obs_dict, graph = env.reset()
            self.graphs[k]  = graph  # refresh graph in case env re-generates it
            _, padded_pf    = pad_phase_features(graph, target_dim=self.global_phase_feat_dim)
            self.padded_pf[k] = padded_pf

            obs_dict = apply_perception(obs_dict, perception_severity, sentinel)
            pressure.reset()
            imputer.reset()
            obs_dict = imputer.impute(obs_dict)
            _, padded_obs = pad_obs_dict(obs_dict, target_dim=self.global_obs_dim)

            done = False
            episode_reward = 0.0
            episode_steps  = 0

            while not done:
                # Epsilon decays with global total_steps (not per-network)
                epsilon = _linear_decay(total_steps, eps_start, eps_end, eps_decay_steps)
                actions = self._select_actions(padded_obs, graph, padded_pf, epsilon,
                                               self.local_obs_dims[k])

                next_obs_dict, _, reward_dict, done, _ = env.step(actions)
                next_obs_dict = apply_perception(next_obs_dict, perception_severity, sentinel)
                if use_pressure:
                    reward_dict = pressure.compute(next_obs_dict, graph)
                if reward_scale != 1.0:
                    reward_dict = {k: v * reward_scale for k, v in reward_dict.items()}
                next_obs_dict = imputer.impute(next_obs_dict)
                _, padded_next = pad_obs_dict(next_obs_dict, target_dim=self.global_obs_dim)

                buf.push(padded_obs, padded_next, actions, reward_dict, done)

                episode_reward += sum(reward_dict.values())
                episode_steps  += 1
                total_steps    += 1  # global counter

                # Warmup is per-buffer — each network's buffer must independently
                # reach warmup_steps before gradient steps begin for that network.
                if len(buf) >= warmup_steps:
                    batch = buf.sample(batch_size)
                    loss, q_m = self._compute_loss(batch, graph, padded_pf, gamma,
                                                    self.local_obs_dims[k])

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters())
                        + list(self.gat.parameters())
                        + list(self.head.parameters()),
                        grad_clip,
                    )
                    self.optimizer.step()

                    metrics["losses"].append(loss.item())
                    metrics["epsilons"].append(epsilon)
                    metrics["q_mean"].append(q_m)

                if total_steps % target_update_freq == 0:
                    self.target_encoder.load_state_dict(self.encoder.state_dict())
                    self.target_gat.load_state_dict(self.gat.state_dict())
                    self.target_head.load_state_dict(self.head.state_dict())

                padded_obs = padded_next

            metrics["episode_returns"].append(episode_reward)
            metrics["episode_lengths"].append(episode_steps)
            metrics["network_sequence"].append(name)

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
            print(
                f"[Ep {episode + 1:4d}/{num_episodes}]  net={name}"
                f"  return={episode_reward:+9.2f}"
                f"  len={episode_steps:4d}"
                f"  avg_loss={loss_str}"
                f"  eps={epsilon:.3f}"
                f"  elapsed={t_str}",
                flush=True,
            )

            if (episode + 1) % checkpoint_every == 0:
                path = Path(checkpoint_dir) / f"checkpoint_ep{episode + 1}.pt"
                self.save_checkpoint(str(path), total_steps, metrics)

        return metrics

    def _select_actions(
        self, padded_obs: dict, graph: dict, padded_pf: list, epsilon: float,
        local_obs_dim: int | None = None,
    ) -> dict:
        """ε-greedy action selection for all nodes in the active network."""
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
            emb = self.encoder(obs_stack, val_stack)
            if self._neighbor_masking:
                val_for_gat = val_stack[:, :local_obs_dim] if local_obs_dim else val_stack
            else:
                val_for_gat = None
            gat_emb = self.gat(emb, edge_index, edge_type, val_for_gat)

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

    @staticmethod
    def _select_actions_local(
        padded_obs: dict,
        graph: dict,
        padded_pf: list,
        epsilon: float,
        enc,
        gat,
        head,
        device: torch.device,
        neighbor_masking: bool,
        local_obs_dim: int | None = None,
    ) -> dict:
        """ε-greedy action selection using explicit model instances.

        Used by worker threads, which hold snapshot copies of the model
        rather than accessing self.encoder/gat/head directly.
        """
        node_ids    = graph["node_ids"]
        node_to_idx = graph["node_to_idx"]
        node_meta   = graph["node_meta"]
        edge_index  = graph["edge_index"].to(device)
        edge_type   = graph["edge_type"].to(device)

        with torch.no_grad():
            obs_stack = torch.stack([padded_obs[nid][0] for nid in node_ids]).to(device)
            val_stack = torch.stack([padded_obs[nid][1] for nid in node_ids]).to(device)
            emb = enc(obs_stack, val_stack)
            if neighbor_masking:
                val_for_gat = val_stack[:, :local_obs_dim] if local_obs_dim else val_stack
            else:
                val_for_gat = None
            gat_emb = gat(emb, edge_index, edge_type, val_for_gat)

            actions = {}
            for col, node_id in enumerate(node_ids):
                node_idx   = node_to_idx[node_id]
                num_phases = node_meta[node_idx]["num_phases"]

                current_phase = int(obs_stack[col, :num_phases].argmax().item())
                mask = node_meta[node_idx]["valid_transition_mask"][current_phase].to(device)

                if torch.rand(1).item() < epsilon:
                    valid_indices = mask.nonzero(as_tuple=False).flatten()
                    if valid_indices.numel() == 0:
                        actions[node_id] = 0
                    else:
                        actions[node_id] = int(
                            valid_indices[torch.randint(len(valid_indices), (1,))].item()
                        )
                else:
                    scores = head(gat_emb[col], padded_pf[node_idx], mask)
                    actions[node_id] = int(scores.argmax().item())

        return actions

    def _compute_loss(
        self, batch: dict, graph: dict, padded_pf: list, gamma: float,
        local_obs_dim: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Huber loss averaged over all nodes in the active network."""
        node_to_idx = graph["node_to_idx"]
        node_meta   = graph["node_meta"]
        node_ids    = graph["node_ids"]
        edge_index  = graph["edge_index"].to(self.device)
        edge_type   = graph["edge_type"].to(self.device)
        B           = batch["obs"].shape[0]
        N           = len(node_ids)

        obs_all      = batch["obs"].to(self.device)
        val_all      = batch["validity"].to(self.device)
        next_obs_all = batch["next_obs"].to(self.device)
        next_val_all = batch["next_val"].to(self.device)
        actions_all  = batch["actions"].to(self.device)
        rewards_all  = batch["rewards"].to(self.device)
        dones        = batch["dones"].to(self.device)

        emb_flat = self.encoder(
            obs_all.view(B * N, -1), val_all.view(B * N, -1)
        ).view(B, N, -1)

        lod = local_obs_dim
        gat_out = torch.stack([
            self.gat(
                emb_flat[b], edge_index, edge_type,
                val_all[b, :, :lod] if (self._neighbor_masking and lod) else (
                    val_all[b] if self._neighbor_masking else None
                ),
            ) for b in range(B)
        ])

        with torch.no_grad():
            next_emb_flat = self.target_encoder(
                next_obs_all.view(B * N, -1), next_val_all.view(B * N, -1)
            ).view(B, N, -1)
            next_gat_out = torch.stack([
                self.target_gat(
                    next_emb_flat[b], edge_index, edge_type,
                    next_val_all[b, :, :lod] if (self._neighbor_masking and lod) else (
                        next_val_all[b] if self._neighbor_masking else None
                    ),
                ) for b in range(B)
            ])

        all_losses = []
        all_q_max  = []

        for col, node_id in enumerate(node_ids):
            node_idx   = node_to_idx[node_id]
            num_phases = node_meta[node_idx]["num_phases"]
            pf         = padded_pf[node_idx]
            all_true   = torch.ones(num_phases, dtype=torch.bool, device=self.device)

            node_emb      = gat_out[:, col, :]
            next_node_emb = next_gat_out[:, col, :]

            q_online = self.head.forward_batch(node_emb, pf, all_true)
            q_pred   = q_online.gather(1, actions_all[:, col].unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next_scores = self.target_head.forward_batch(next_node_emb, pf, all_true)
                q_next_max    = q_next_scores.max(dim=1).values
                q_target      = rewards_all[:, col] + gamma * q_next_max * (1.0 - dones)

            loss_i = F.huber_loss(q_pred, q_target)
            all_losses.append(loss_i)
            all_q_max.append(q_next_max.mean().item())

        loss   = torch.stack(all_losses).mean()
        q_mean = sum(all_q_max) / max(len(all_q_max), 1)
        return loss, q_mean

    def _train_parallel(
        self,
        num_episodes: int,
        batch_size: int,
        warmup_steps: int,
        gamma: float,
        grad_clip: float,
        target_update_freq: int,
        eps_start: float,
        eps_end: float,
        eps_decay_steps: int,
        checkpoint_every: int,
        checkpoint_dir: str,
        use_pressure: bool,
        reward_scale: float,
        perception_severity: float,
        sentinel: float,
        num_workers: int,
    ) -> dict:
        """Parallel episode collection with a central learner in the main thread.

        Workers run SUMO episodes in threads. The GIL is released during
        env.step() (SUMO is an external subprocess), giving real parallelism
        for environment interaction. The main thread handles all gradient updates.

        Each worker takes a per-episode model snapshot so the policy is stable
        within one episode. Stale policies are acceptable for off-policy DQN.

        Network assignment: worker i owns networks {i, i+W, i+2W, ...} where
        W = min(num_workers, num_networks). This ensures no env is accessed by
        two threads simultaneously.

        Thread-safety contract:
            _model_lock  (RLock): deepcopy of encoder/gat/head and optimizer.step
            _buf_locks[k] (Lock): buffers[k].push and .sample
            _step_lock   (Lock):  _total_steps, _episodes_done, _epsilon
            _metrics_lock (Lock): episode_returns/lengths/network_sequence lists
        """
        self._epsilon       = float(eps_start)
        self._total_steps   = 0
        self._episodes_done = 0
        self._step_lock     = threading.Lock()
        self._env_locks     = [threading.Lock() for _ in self.envs]
        self._buf_locks     = [threading.Lock() for _ in self.buffers]
        self._model_lock    = threading.RLock()
        self._metrics_lock  = threading.Lock()

        metrics = {
            "episode_returns":  [],
            "episode_lengths":  [],
            "losses":           [],
            "epsilons":         [],
            "q_mean":           [],
            "network_sequence": [],
        }
        start_time = time.time()

        def worker_fn(worker_id: int) -> None:
            # Workers compete for any free env in random order.
            # Fast networks (few nodes, short steps) are free more often and
            # naturally acquire more episode slots, balancing wall-clock time
            # across networks rather than episode counts.
            while True:
                with self._step_lock:
                    if self._episodes_done >= num_episodes:
                        return

                # Non-blocking acquire: try envs in random order, skip busy ones.
                k = None
                for candidate in random.sample(range(len(self.envs)), len(self.envs)):
                    if self._env_locks[candidate].acquire(blocking=False):
                        k = candidate
                        break
                if k is None:
                    time.sleep(0.002)
                    continue

                # Re-check quota now that we hold the env; claim episode slot.
                with self._step_lock:
                    if self._episodes_done >= num_episodes:
                        self._env_locks[k].release()
                        return
                    ep_idx = self._episodes_done
                    self._episodes_done += 1

                try:
                    name     = self.network_names[k]
                    env      = self.envs[k]
                    pressure = self.pressure_fns[k]
                    imputer  = self.obs_imputers[k]
                    buf      = self.buffers[k]

                    # Snapshot model weights for this episode.
                    with self._model_lock:
                        enc  = copy.deepcopy(self.encoder)
                        gat  = copy.deepcopy(self.gat)
                        head = copy.deepcopy(self.head)
                    enc.eval(); gat.eval(); head.eval()

                    obs_dict, graph = env.reset()
                    self.graphs[k]    = graph
                    _, padded_pf = pad_phase_features(graph, target_dim=self.global_phase_feat_dim)
                    self.padded_pf[k] = padded_pf

                    obs_dict = apply_perception(obs_dict, perception_severity, sentinel)
                    pressure.reset()
                    imputer.reset()
                    obs_dict = imputer.impute(obs_dict)
                    _, padded_obs = pad_obs_dict(obs_dict, target_dim=self.global_obs_dim)

                    done           = False
                    episode_reward = 0.0
                    episode_steps  = 0

                    while not done:
                        epsilon = self._epsilon  # float read is GIL-safe in CPython
                        actions = MultiNetworkTrainer._select_actions_local(
                            padded_obs, graph, padded_pf, epsilon,
                            enc, gat, head, self.device, self._neighbor_masking,
                            self.local_obs_dims[k],
                        )

                        # SUMO subprocess call — GIL released during I/O
                        next_obs_dict, _, reward_dict, done, _ = env.step(actions)
                        next_obs_dict = apply_perception(next_obs_dict, perception_severity, sentinel)
                        if use_pressure:
                            reward_dict = pressure.compute(next_obs_dict, graph)
                        if reward_scale != 1.0:
                            reward_dict = {k: v * reward_scale for k, v in reward_dict.items()}
                        next_obs_dict = imputer.impute(next_obs_dict)
                        _, padded_next = pad_obs_dict(next_obs_dict, target_dim=self.global_obs_dim)

                        with self._buf_locks[k]:
                            buf.push(padded_obs, padded_next, actions, reward_dict, done)

                        episode_reward += sum(reward_dict.values())
                        episode_steps  += 1
                        with self._step_lock:
                            self._total_steps += 1

                        padded_obs = padded_next

                    elapsed = int(time.time() - start_time)
                    h, rem  = divmod(elapsed, 3600)
                    m, s    = divmod(rem, 60)
                    t_str   = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
                    print(
                        f"[Ep {ep_idx + 1:4d}/{num_episodes}]  net={name}"
                        f"  return={episode_reward:+9.2f}"
                        f"  len={episode_steps:4d}"
                        f"  eps={epsilon:.3f}"
                        f"  elapsed={t_str}  [w{worker_id}]",
                        flush=True,
                    )
                    with self._metrics_lock:
                        metrics["episode_returns"].append(episode_reward)
                        metrics["episode_lengths"].append(episode_steps)
                        metrics["network_sequence"].append(name)
                finally:
                    self._env_locks[k].release()

        # Launch worker threads — more workers than envs is fine; excess workers
        # will spin on env_locks and pick up whichever env finishes next.
        threads = [
            threading.Thread(target=worker_fn, args=(i,), daemon=True)
            for i in range(num_workers)
        ]
        for t in threads:
            t.start()

        # Learner loop: gradient updates in main thread while workers collect data
        grad_steps   = 0
        last_ckpt_ep = 0

        while any(t.is_alive() for t in threads):
            did_work = False

            for k in range(len(self.envs)):
                with self._buf_locks[k]:
                    buf_ready = len(self.buffers[k]) >= warmup_steps
                if not buf_ready:
                    continue

                with self._buf_locks[k]:
                    batch = self.buffers[k].sample(batch_size)

                graph     = self.graphs[k]
                padded_pf = self.padded_pf[k]

                with self._model_lock:
                    loss, q_m = self._compute_loss(batch, graph, padded_pf, gamma,
                                                    self.local_obs_dims[k])
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

                    with self._step_lock:
                        self._epsilon = _linear_decay(
                            self._total_steps, eps_start, eps_end, eps_decay_steps
                        )

                    if grad_steps % target_update_freq == 0:
                        self.target_encoder.load_state_dict(self.encoder.state_dict())
                        self.target_gat.load_state_dict(self.gat.state_dict())
                        self.target_head.load_state_dict(self.head.state_dict())

                with self._metrics_lock:
                    metrics["losses"].append(loss.item())
                    metrics["epsilons"].append(self._epsilon)
                    metrics["q_mean"].append(q_m)

                did_work = True

            if not did_work:
                time.sleep(0.005)

            # Checkpoint once per checkpoint_every episode boundary
            if checkpoint_every:
                with self._step_lock:
                    done_count = self._episodes_done
                ckpt_epoch = (done_count // checkpoint_every) * checkpoint_every
                if ckpt_epoch > last_ckpt_ep and done_count > 0:
                    last_ckpt_ep = ckpt_epoch
                    path = Path(checkpoint_dir) / f"checkpoint_ep{done_count}.pt"
                    with self._model_lock:
                        with self._metrics_lock:
                            self.save_checkpoint(str(path), self._total_steps, dict(metrics))

        for t in threads:
            t.join(timeout=60.0)

        return metrics

    def save_checkpoint(self, path: str, step: int, metrics: dict) -> None:
        """Save weights, optimizer, metrics, and multi-network metadata."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "encoder":              self.encoder.state_dict(),
                "gat":                  self.gat.state_dict(),
                "head":                 self.head.state_dict(),
                "optimizer":            self.optimizer.state_dict(),
                "step":                 step,
                "cfg":                  _cfg_to_dict(self.cfg),
                "metrics":              metrics,
                "max_obs_dim":          self.global_obs_dim,
                "max_phase_feat_dim":   self.global_phase_feat_dim,
                "network_names":        self.network_names,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        cfg,
        envs: list,
        device: torch.device = torch.device("cpu"),
        network_names: list[str] | None = None,
    ) -> "MultiNetworkTrainer":
        """Restore a MultiNetworkTrainer from a checkpoint.

        Raises ValueError if the checkpoint's max_obs_dim or max_phase_feat_dim
        does not match cfg — this prevents silently building a model with wrong
        dims when a fine-tune config accidentally changes the ceilings.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)

        cfg_obs_dim = _cfg(cfg, "model.max_obs_dim", None)
        cfg_pf_dim  = _cfg(cfg, "model.max_phase_feat_dim", None)
        if cfg_obs_dim is not None and ckpt["max_obs_dim"] != int(cfg_obs_dim):
            raise ValueError(
                f"Checkpoint max_obs_dim={ckpt['max_obs_dim']} does not match "
                f"cfg model.max_obs_dim={cfg_obs_dim}. Update the config to match "
                f"the checkpoint, or retrain."
            )
        if cfg_pf_dim is not None and ckpt["max_phase_feat_dim"] != int(cfg_pf_dim):
            raise ValueError(
                f"Checkpoint max_phase_feat_dim={ckpt['max_phase_feat_dim']} does not "
                f"match cfg model.max_phase_feat_dim={cfg_pf_dim}. Update the config "
                f"to match the checkpoint, or retrain."
            )

        trainer = cls(cfg, envs, device, network_names=network_names)
        trainer.encoder.load_state_dict(ckpt["encoder"])
        trainer.gat.load_state_dict(ckpt["gat"])
        trainer.head.load_state_dict(ckpt["head"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.target_encoder.load_state_dict(ckpt["encoder"])
        trainer.target_gat.load_state_dict(ckpt["gat"])
        trainer.target_head.load_state_dict(ckpt["head"])
        return trainer
