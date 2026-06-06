"""Synchronous parallel DQN trainer for a single network.

Workers run episodes in parallel. After every round (all workers finish one
episode) the learner performs num_workers gradient steps with the latest
shared model, then releases workers for the next round. Workers always start
from the same fresh weights — no stale-policy lag.

Evaluation metrics (avg_waiting_time, throughput, avg_queue_length) are
read from the info dict returned by env.step() and reported per episode.
"""

import copy
import math
import os
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


class SyncParallelTrainer:
    """Synchronous parallel DQN trainer for a single network.

    Args:
        cfg:          Config dict (model / trainer / reward / perception keys).
        envs:         num_workers pre-constructed env instances, all the same
                      network but with independent TraCI connections.
        device:       torch.device for model tensors.
        network_name: Name string for logging and checkpoints.
    """

    def __init__(
        self,
        cfg,
        envs: list,
        device: torch.device = torch.device("cpu"),
        network_name: str | None = None,
    ) -> None:
        self.cfg          = cfg
        self.envs         = envs
        self.device       = device
        self.num_workers  = len(envs)
        self.network_name = network_name or "network"

        max_obs_dim        = _cfg(cfg, "model.max_obs_dim", None)
        max_phase_feat_dim = _cfg(cfg, "model.max_phase_feat_dim", None)
        if max_obs_dim is None or max_phase_feat_dim is None:
            raise ValueError(
                "cfg must include model.max_obs_dim and model.max_phase_feat_dim."
            )
        self.global_obs_dim        = int(max_obs_dim)
        self.global_phase_feat_dim = int(max_phase_feat_dim)

        seed = _cfg(cfg, "seed", 42)

        obs_dict, graph = envs[0].reset(seed=seed)
        local_obs_dim, _ = pad_obs_dict(obs_dict)
        local_pf_dim, _  = pad_phase_features(graph)
        if local_obs_dim > self.global_obs_dim:
            raise ValueError(
                f"Network obs_dim={local_obs_dim} exceeds cfg max_obs_dim={self.global_obs_dim}."
            )
        # Actual (unpadded) obs_dim for this network. Used to slice validity before
        # passing to the GAT so that padding zeros don't trigger neighbor masking
        # for healthy nodes (see _node_valid_from_validity threshold of 0.75).
        self.local_obs_dim = local_obs_dim
        if local_pf_dim > self.global_phase_feat_dim:
            raise ValueError(
                f"Network phase_feat_dim={local_pf_dim} exceeds "
                f"cfg max_phase_feat_dim={self.global_phase_feat_dim}."
            )

        self.graph     = graph
        _, self.padded_pf = pad_phase_features(graph, target_dim=self.global_phase_feat_dim)

        # Shared model
        hidden_dim     = _cfg(cfg, "model.hidden_dim", 128)
        embed_dim      = _cfg(cfg, "model.embed_dim", 64)
        head_hidden    = _cfg(cfg, "model.head_hidden_dim", 64)
        gat_heads      = _cfg(cfg, "model.gat.num_heads",        4)
        gat_out_ph     = _cfg(cfg, "model.gat.out_per_head",    32)
        gat_num_layers = _cfg(cfg, "model.gat.num_layers",       1)
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

        capacity = int(_cfg(cfg, "trainer.replay_buffer_size", 50000))
        self.buffer = ReplayBuffer(
            capacity, self.global_obs_dim, graph["node_ids"], device=torch.device("cpu")
        )

        queue_weight    = float(_cfg(cfg, "reward.queue_weight",    0.0))
        pressure_weight = float(_cfg(cfg, "reward.pressure_weight", 1.0))
        self.pressure_fns = [PressureReward(queue_weight=queue_weight, pressure_weight=pressure_weight) for _ in envs]
        self.obs_imputers = [ObservationImputer() for _ in envs]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, num_episodes: int | None = None) -> dict:
        """Run synchronous parallel DQN training.

        Returns a metrics dict with keys:
            episode_returns     list[float]  total reward per episode
            episode_lengths     list[int]
            losses              list[float]  Huber loss per gradient step
            epsilons            list[float]
            q_mean              list[float]
            avg_waiting_time    list[float]  mean per-vehicle waiting time (s)
            throughput          list[int]    vehicles that completed their trip
            avg_queue_length    list[float]  mean halting vehicles per step
        """
        cfg = self.cfg
        if num_episodes is None:
            num_episodes = _cfg(cfg, "trainer.num_episodes", 500)

        batch_size         = _cfg(cfg, "trainer.batch_size", 64)
        warmup_steps       = _cfg(cfg, "trainer.warmup_steps", 5000)
        gamma              = _cfg(cfg, "trainer.gamma", 0.99)
        grad_clip          = _cfg(cfg, "trainer.grad_clip", 10.0)
        target_tau         = _cfg(cfg, "trainer.target_tau", 0.01)
        eps_start          = _cfg(cfg, "trainer.epsilon_start", 1.0)
        eps_end            = _cfg(cfg, "trainer.epsilon_end", 0.05)
        eps_decay_steps    = _cfg(cfg, "trainer.epsilon_decay_steps", 50000)
        checkpoint_every   = _cfg(cfg, "trainer.checkpoint_every", 100)
        checkpoint_dir     = _cfg(cfg, "trainer.checkpoint_dir", "checkpoints/sync")
        use_pressure        = _cfg(cfg, "reward.use_pressure", False)
        reward_scale        = float(_cfg(cfg, "reward.scale", 1.0))
        perception_severity = _cfg(cfg, "perception.severity", 0.0)
        sentinel            = _cfg(cfg, "perception.sentinel_value", -1.0)

        # Rounds: each round runs num_workers episodes in parallel
        num_rounds = math.ceil(num_episodes / self.num_workers)

        self._epsilon     = float(eps_start)
        self._total_steps = 0
        self._model_lock  = threading.RLock()
        self._buf_lock    = threading.Lock()
        self._step_lock   = threading.Lock()

        # Two barriers synchronise the end-of-episode and end-of-update phases
        phase1     = threading.Barrier(self.num_workers + 1)
        phase2     = threading.Barrier(self.num_workers + 1)
        stop_event = threading.Event()

        # Workers write per-episode results into their own slot — no lock needed
        round_results: list = [None] * self.num_workers

        def worker_fn(worker_id: int) -> None:
            env      = self.envs[worker_id]
            pressure = self.pressure_fns[worker_id]
            imputer  = self.obs_imputers[worker_id]

            while not stop_event.is_set():
                # Fresh model snapshot — all workers snapshot after the same update
                with self._model_lock:
                    enc  = copy.deepcopy(self.encoder)
                    gat  = copy.deepcopy(self.gat)
                    head = copy.deepcopy(self.head)
                enc.eval(); gat.eval(); head.eval()

                obs_dict, graph = env.reset()
                _, padded_pf = pad_phase_features(graph, target_dim=self.global_phase_feat_dim)

                obs_dict = apply_perception(obs_dict, perception_severity, sentinel)
                pressure.reset()
                imputer.reset()
                obs_dict = imputer.impute(obs_dict)
                _, padded_obs = pad_obs_dict(obs_dict, target_dim=self.global_obs_dim)

                done           = False
                episode_reward = 0.0
                episode_steps  = 0
                sum_wait       = 0.0
                total_arrived  = 0
                sum_queue      = 0.0

                while not done:
                    epsilon = self._epsilon  # float read is GIL-safe in CPython
                    actions = SyncParallelTrainer._select_actions_local(
                        padded_obs, graph, padded_pf, epsilon,
                        enc, gat, head, self.device, self._neighbor_masking,
                        self.local_obs_dim,
                    )

                    next_obs_dict, _, reward_dict, done, info = env.step(actions)
                    next_obs_dict = apply_perception(next_obs_dict, perception_severity, sentinel)
                    if use_pressure:
                        reward_dict = pressure.compute(next_obs_dict, graph)
                    if reward_scale != 1.0:
                        reward_dict = {k: v * reward_scale for k, v in reward_dict.items()}
                    next_obs_dict = imputer.impute(next_obs_dict)
                    _, padded_next = pad_obs_dict(next_obs_dict, target_dim=self.global_obs_dim)

                    with self._buf_lock:
                        self.buffer.push(padded_obs, padded_next, actions, reward_dict, done)

                    episode_reward += sum(reward_dict.values())
                    episode_steps  += 1
                    sum_wait       += float(info.get("step_mean_waiting_time", 0.0))
                    total_arrived  += int(info.get("step_throughput", 0))
                    sum_queue      += float(info.get("step_queue_length", 0.0))

                    with self._step_lock:
                        self._total_steps += 1

                    padded_obs = padded_next

                round_results[worker_id] = {
                    "return":           episode_reward,
                    "length":           episode_steps,
                    "avg_waiting_time": sum_wait / max(1, episode_steps),
                    "throughput":       total_arrived,
                    "avg_queue_length": sum_queue / max(1, episode_steps),
                }

                # Sync point 1: all workers done → learner can update
                try:
                    phase1.wait()
                    # Sync point 2: learner done → workers can snapshot fresh model
                    phase2.wait()
                except threading.BrokenBarrierError:
                    return

        threads = [
            threading.Thread(target=worker_fn, args=(i,), daemon=True)
            for i in range(self.num_workers)
        ]
        for t in threads:
            t.start()

        metrics = {
            "episode_returns":  [],
            "episode_lengths":  [],
            "losses":           [],
            "epsilons":         [],
            "q_mean":           [],
            "avg_waiting_time": [],
            "throughput":       [],
            "avg_queue_length": [],
        }
        grad_steps = 0
        peak_loss  = 0.0          # collapse detection: track highest loss seen
        start_time = time.time()

        try:
            for round_idx in range(num_rounds):
                try:
                    phase1.wait()  # wait for all workers to finish their episode
                except threading.BrokenBarrierError:
                    break

                # Collect episode metrics
                for rr in round_results:
                    if rr is not None:
                        metrics["episode_returns"].append(rr["return"])
                        metrics["episode_lengths"].append(rr["length"])
                        metrics["avg_waiting_time"].append(rr["avg_waiting_time"])
                        metrics["throughput"].append(rr["throughput"])
                        metrics["avg_queue_length"].append(rr["avg_queue_length"])

                # Gradient updates — num_workers steps per round to match data production rate
                with self._model_lock:
                    if len(self.buffer) >= warmup_steps:
                        for _ in range(self.num_workers):
                            with self._buf_lock:
                                batch = self.buffer.sample(batch_size)
                            loss, q_m = self._compute_loss(
                                batch, self.graph, self.padded_pf, gamma
                            )
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

                            # Soft Polyak update: target slowly tracks online network each grad step.
                            # Avoids "frozen random target" when hard-copy threshold is never reached.
                            for p_on, p_tgt in zip(
                                list(self.encoder.parameters())
                                + list(self.gat.parameters())
                                + list(self.head.parameters()),
                                list(self.target_encoder.parameters())
                                + list(self.target_gat.parameters())
                                + list(self.target_head.parameters()),
                            ):
                                p_tgt.data.mul_(1.0 - target_tau).add_(p_on.data * target_tau)

                            # Epsilon decays on gradient steps so exploration matches learning progress.
                            self._epsilon = _linear_decay(
                                grad_steps, eps_start, eps_end, eps_decay_steps
                            )

                            lv = loss.item()
                            if lv > peak_loss:
                                peak_loss = lv
                            # Collapse: loss fell to <5% of its peak after at least 20 grad steps
                            if grad_steps >= 20 and peak_loss > 1e-4 and lv < peak_loss * 0.05:
                                print(
                                    f"\n*** COLLAPSE DETECTED at grad_step={grad_steps} ***"
                                    f"  peak_loss={peak_loss:.5f}  now={lv:.7f}"
                                    f"  -> reward scale too small or target diverged\n",
                                    flush=True,
                                )
                            metrics["losses"].append(lv)
                            metrics["epsilons"].append(self._epsilon)
                            metrics["q_mean"].append(q_m)

                # Log
                ep_done  = (round_idx + 1) * self.num_workers
                n_loss   = len(metrics["losses"])
                avg_loss = (
                    sum(metrics["losses"][-50:]) / min(n_loss, 50) if n_loss else float("nan")
                )
                avg_ret  = sum(metrics["episode_returns"][-self.num_workers:]) / self.num_workers
                avg_wait = sum(metrics["avg_waiting_time"][-self.num_workers:]) / self.num_workers
                avg_q    = sum(metrics["avg_queue_length"][-self.num_workers:]) / self.num_workers
                tput     = sum(metrics["throughput"][-self.num_workers:])
                elapsed  = int(time.time() - start_time)
                h, rem   = divmod(elapsed, 3600)
                m, s     = divmod(rem, 60)
                t_str    = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
                loss_str = "(warmup)" if math.isnan(avg_loss) else f"{avg_loss:.4f}"
                print(
                    f"[Round {round_idx + 1:4d}/{num_rounds}]  ep={ep_done:4d}"
                    f"  ret={avg_ret:+9.2f}"
                    f"  wait={avg_wait:.1f}s"
                    f"  tput={tput:4d}"
                    f"  q={avg_q:.1f}"
                    f"  loss={loss_str}"
                    f"  eps={self._epsilon:.3f}"
                    f"  t={t_str}",
                    flush=True,
                )

                if checkpoint_every and ep_done % checkpoint_every == 0:
                    path = Path(checkpoint_dir) / f"checkpoint_ep{ep_done}.pt"
                    with self._model_lock:
                        self.save_checkpoint(str(path), self._total_steps, dict(metrics))

                try:
                    phase2.wait()  # release workers for next round
                except threading.BrokenBarrierError:
                    break
        finally:
            stop_event.set()
            try:
                phase1.abort()
                phase2.abort()
            except Exception:
                pass
            for t in threads:
                t.join(timeout=30.0)

        return metrics

    def save_checkpoint(self, path: str, step: int, metrics: dict) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "encoder":            self.encoder.state_dict(),
                "gat":                self.gat.state_dict(),
                "head":               self.head.state_dict(),
                "optimizer":          self.optimizer.state_dict(),
                "step":               step,
                "cfg":                _cfg_to_dict(self.cfg),
                "metrics":            metrics,
                "max_obs_dim":        self.global_obs_dim,
                "max_phase_feat_dim": self.global_phase_feat_dim,
                "network_name":       self.network_name,
            },
            path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_actions_local(
        padded_obs, graph, padded_pf, epsilon,
        enc, gat, head, device, neighbor_masking,
        local_obs_dim=None,
    ) -> dict:
        node_ids    = graph["node_ids"]
        node_to_idx = graph["node_to_idx"]
        node_meta   = graph["node_meta"]
        edge_index  = graph["edge_index"].to(device)
        edge_type   = graph["edge_type"].to(device)

        with torch.no_grad():
            obs_stack = torch.stack([padded_obs[nid][0] for nid in node_ids]).to(device)
            val_stack = torch.stack([padded_obs[nid][1] for nid in node_ids]).to(device)
            emb           = enc(obs_stack, val_stack)
            # Slice validity to actual obs_dim so padding zeros don't falsely
            # trigger neighbor masking (threshold 0.75 fails if padding deflates mean).
            if neighbor_masking:
                val_for_gat = val_stack[:, :local_obs_dim] if local_obs_dim else val_stack
            else:
                val_for_gat = None
            gat_emb       = gat(emb, edge_index, edge_type, val_for_gat)

            actions = {}
            for col, node_id in enumerate(node_ids):
                node_idx   = node_to_idx[node_id]
                num_phases = node_meta[node_idx]["num_phases"]
                cur_phase  = int(obs_stack[col, :num_phases].argmax().item())
                mask = node_meta[node_idx]["valid_transition_mask"][cur_phase].to(device)

                if torch.rand(1).item() < epsilon:
                    vi = mask.nonzero(as_tuple=False).flatten()
                    actions[node_id] = (
                        0 if vi.numel() == 0
                        else int(vi[torch.randint(len(vi), (1,))].item())
                    )
                else:
                    actions[node_id] = int(
                        head(gat_emb[col], padded_pf[node_idx], mask).argmax().item()
                    )
        return actions

    def _compute_loss(self, batch, graph, padded_pf, gamma):
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
        # Slice validity to actual obs_dim before GAT neighbor-masking check so that
        # padding zeros don't deflate the mean below the 0.75 valid-sensor threshold.
        lod = self.local_obs_dim
        gat_out = torch.stack([
            self.gat(emb_flat[b], edge_index, edge_type,
                     val_all[b, :, :lod] if self._neighbor_masking else None)
            for b in range(B)
        ])

        with torch.no_grad():
            nef = self.target_encoder(
                next_obs_all.view(B * N, -1), next_val_all.view(B * N, -1)
            ).view(B, N, -1)
            next_gat_out = torch.stack([
                self.target_gat(nef[b], edge_index, edge_type,
                                next_val_all[b, :, :lod] if self._neighbor_masking else None)
                for b in range(B)
            ])

        all_losses, all_q_max = [], []
        for col, node_id in enumerate(node_ids):
            node_idx   = node_to_idx[node_id]
            num_phases = node_meta[node_idx]["num_phases"]
            pf         = padded_pf[node_idx]
            all_true   = torch.ones(num_phases, dtype=torch.bool, device=self.device)

            q_online  = self.head.forward_batch(gat_out[:, col, :], pf, all_true)
            q_pred    = q_online.gather(1, actions_all[:, col].unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next   = self.target_head.forward_batch(next_gat_out[:, col, :], pf, all_true)
                q_target = rewards_all[:, col] + gamma * q_next.max(1).values * (1.0 - dones)

            all_losses.append(F.huber_loss(q_pred, q_target))
            all_q_max.append(q_next.max(1).values.mean().item())

        return torch.stack(all_losses).mean(), sum(all_q_max) / max(len(all_q_max), 1)
