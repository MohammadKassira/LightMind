from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DQNAgentConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_buffer_size: int = 50_000
    min_replay_size: int = 1_000
    target_update_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    train_every_steps: int = 8
    learning_starts_steps: int = 1_000
    hidden_dim_1: int = 128
    hidden_dim_2: int = 128
    loss_type: str = "huber"  # huber | mse
    grad_clip_norm: float = 10.0
    device: str = "cpu"
    seed: int = 0


@dataclass
class AgentSpec:
    obs_dim: int
    num_actions: int
    config_overrides: dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    def __init__(self, *, obs_dim: int, action_dim: int, capacity: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.next_masks = np.ones((self.capacity, action_dim), dtype=np.bool_)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        *,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        i = self._idx
        self.obs[i] = obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_obs[i] = next_obs
        self.dones[i] = 1.0 if done else 0.0
        self.next_masks[i] = np.asarray(next_action_mask, dtype=np.bool_)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample_indices(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self._size, size=batch_size, dtype=np.int64)

    def sample_tensors(self, batch_size: int, rng: np.random.Generator, torch_module: Any, device: Any) -> dict[str, Any]:
        idx = self.sample_indices(batch_size, rng)
        return {
            "obs": torch_module.as_tensor(self.obs[idx], dtype=torch_module.float32, device=device),
            "actions": torch_module.as_tensor(self.actions[idx], dtype=torch_module.int64, device=device),
            "rewards": torch_module.as_tensor(self.rewards[idx], dtype=torch_module.float32, device=device),
            "next_obs": torch_module.as_tensor(self.next_obs[idx], dtype=torch_module.float32, device=device),
            "dones": torch_module.as_tensor(self.dones[idx], dtype=torch_module.float32, device=device),
            "next_masks": torch_module.as_tensor(self.next_masks[idx], dtype=torch_module.bool, device=device),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "idx": self._idx,
            "size": self._size,
            "obs": self.obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_obs": self.next_obs,
            "dones": self.dones,
            "next_masks": self.next_masks,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self._idx = int(state["idx"])
        self._size = int(state["size"])
        self.obs = np.asarray(state["obs"], dtype=np.float32)
        self.actions = np.asarray(state["actions"], dtype=np.int64)
        self.rewards = np.asarray(state["rewards"], dtype=np.float32)
        self.next_obs = np.asarray(state["next_obs"], dtype=np.float32)
        self.dones = np.asarray(state["dones"], dtype=np.float32)
        self.next_masks = np.asarray(state["next_masks"], dtype=np.bool_)


class IndependentDQNAgent:
    def __init__(self, *, obs_dim: int, num_actions: int, config: DQNAgentConfig) -> None:
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Torch is required for IndependentDQNAgent") from exc

        self.torch = torch
        self.nn = nn
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.config = config
        self.device = torch.device(config.device)

        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(int(config.seed))

        self.online_q = nn.Sequential(
            nn.Linear(self.obs_dim, int(config.hidden_dim_1)),
            nn.ReLU(),
            nn.Linear(int(config.hidden_dim_1), int(config.hidden_dim_2)),
            nn.ReLU(),
            nn.Linear(int(config.hidden_dim_2), self.num_actions),
        ).to(self.device)
        self.target_q = nn.Sequential(
            nn.Linear(self.obs_dim, int(config.hidden_dim_1)),
            nn.ReLU(),
            nn.Linear(int(config.hidden_dim_1), int(config.hidden_dim_2)),
            nn.ReLU(),
            nn.Linear(int(config.hidden_dim_2), self.num_actions),
        ).to(self.device)
        self.target_q.load_state_dict(self.online_q.state_dict())
        self.target_q.eval()

        self.optimizer = torch.optim.Adam(self.online_q.parameters(), lr=float(config.learning_rate))
        self.replay = ReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.num_actions,
            capacity=int(config.replay_buffer_size),
        )

        self.env_steps = 0
        self.train_steps = 0
        self.target_sync_count = 0

    def _epsilon(self) -> float:
        frac = min(max(self.env_steps, 0), int(self.config.epsilon_decay_steps)) / max(
            int(self.config.epsilon_decay_steps), 1
        )
        return float(self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start))

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray, *, explore: bool) -> int:
        mask = np.asarray(action_mask, dtype=np.bool_)
        if mask.shape != (self.num_actions,):
            raise ValueError(f"Invalid action mask shape {mask.shape}, expected {(self.num_actions,)}")
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            raise ValueError("Action mask has no valid actions.")

        eps = self._epsilon()
        if explore and self.rng.random() < eps:
            return int(self.rng.choice(valid))

        with self.torch.no_grad():
            x = self.torch.as_tensor(obs, dtype=self.torch.float32, device=self.device).view(1, -1)
            q = self.online_q(x).view(-1)
            neg_inf = self.torch.full_like(q, fill_value=-1e30)
            mask_t = self.torch.as_tensor(mask, dtype=self.torch.bool, device=self.device)
            q_masked = self.torch.where(mask_t, q, neg_inf)
            return int(self.torch.argmax(q_masked).item())

    def store_transition(
        self,
        *,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        self.replay.add(
            obs=np.asarray(obs, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_obs=np.asarray(next_obs, dtype=np.float32),
            done=bool(done),
            next_action_mask=np.asarray(next_action_mask, dtype=np.bool_),
        )
        self.env_steps += 1

    def train_step(self) -> dict[str, float] | None:
        if self.env_steps < int(self.config.learning_starts_steps):
            return None
        if len(self.replay) < int(self.config.min_replay_size):
            return None

        batch = self.replay.sample_tensors(
            batch_size=int(self.config.batch_size),
            rng=self.rng,
            torch_module=self.torch,
            device=self.device,
        )

        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        next_masks = batch["next_masks"]

        q_values = self.online_q(obs)
        q_taken = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

        with self.torch.no_grad():
            next_q = self.target_q(next_obs)
            neg_inf = self.torch.full_like(next_q, fill_value=-1e30)
            masked_next_q = self.torch.where(next_masks, next_q, neg_inf)
            next_max = masked_next_q.max(dim=1).values
            target = rewards + (1.0 - dones) * float(self.config.gamma) * next_max

        if self.config.loss_type == "mse":
            loss = self.nn.functional.mse_loss(q_taken, target)
        else:
            loss = self.nn.functional.smooth_l1_loss(q_taken, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if float(self.config.grad_clip_norm) > 0:
            self.nn.utils.clip_grad_norm_(self.online_q.parameters(), float(self.config.grad_clip_norm))
        self.optimizer.step()

        self.train_steps += 1
        target_updated = 0
        # v2 contract smoke expectation: target sync interval is exercised against
        # environment transition steps (not optimizer step count).
        if self.env_steps % int(self.config.target_update_interval) == 0:
            self.target_q.load_state_dict(self.online_q.state_dict())
            self.target_sync_count += 1
            target_updated = 1

        return {
            "loss": float(loss.detach().cpu().item()),
            "epsilon": float(self._epsilon()),
            "train_steps": float(self.train_steps),
            "env_steps": float(self.env_steps),
            "target_updated": float(target_updated),
            "target_sync_count": float(self.target_sync_count),
            "replay_size": float(len(self.replay)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "config": dict(self.config.__dict__),
            "online_q": self.online_q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "env_steps": self.env_steps,
            "train_steps": self.train_steps,
            "target_sync_count": self.target_sync_count,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.online_q.load_state_dict(state["online_q"])
        self.target_q.load_state_dict(state["target_q"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.env_steps = int(state.get("env_steps", 0))
        self.train_steps = int(state.get("train_steps", 0))
        self.target_sync_count = int(state.get("target_sync_count", 0))
        try:
            self.rng.bit_generator.state = state.get("rng_state", self.rng.bit_generator.state)
        except Exception:
            pass

    def save(self, path: Path) -> None:
        self.torch.save(self.state_dict(), path)

    def load(self, path: Path, map_location: str | None = None) -> None:
        try:
            state = self.torch.load(path, map_location=map_location or self.device, weights_only=False)
        except TypeError:
            state = self.torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(state)


class IndependentDQNController:
    def __init__(self, *, agent_specs: dict[str, AgentSpec], default_config: DQNAgentConfig | None = None) -> None:
        self.default_config = default_config or DQNAgentConfig()
        self.agents: dict[str, IndependentDQNAgent] = {}

        seed_base = int(self.default_config.seed)
        for i, (agent_id, spec) in enumerate(sorted(agent_specs.items())):
            cfg_dict = dict(self.default_config.__dict__)
            cfg_dict["seed"] = seed_base + i
            cfg_dict.update(spec.config_overrides or {})
            cfg = DQNAgentConfig(**cfg_dict)
            self.agents[agent_id] = IndependentDQNAgent(
                obs_dim=int(spec.obs_dim),
                num_actions=int(spec.num_actions),
                config=cfg,
            )

    def select_actions(
        self,
        *,
        observations: dict[str, np.ndarray],
        action_masks: dict[str, np.ndarray],
        explore: bool,
    ) -> dict[str, int]:
        actions: dict[str, int] = {}
        for aid, agent in self.agents.items():
            actions[aid] = int(
                agent.select_action(
                    np.asarray(observations[aid], dtype=np.float32),
                    np.asarray(action_masks[aid], dtype=np.bool_),
                    explore=explore,
                )
            )
        return actions

    def store_transitions(
        self,
        *,
        observations: dict[str, np.ndarray],
        actions: dict[str, int],
        rewards: dict[str, float],
        next_observations: dict[str, np.ndarray],
        dones: dict[str, bool],
        next_action_masks: dict[str, np.ndarray],
    ) -> None:
        for aid, agent in self.agents.items():
            agent.store_transition(
                obs=np.asarray(observations[aid], dtype=np.float32),
                action=int(actions[aid]),
                reward=float(rewards[aid]),
                next_obs=np.asarray(next_observations[aid], dtype=np.float32),
                done=bool(dones[aid]),
                next_action_mask=np.asarray(next_action_masks[aid], dtype=np.bool_),
            )

    def train_step(self) -> dict[str, dict[str, float] | None]:
        out: dict[str, dict[str, float] | None] = {}
        for aid, agent in self.agents.items():
            out[aid] = agent.train_step()
        return out

    def save(self, path: Path) -> None:
        import torch

        payload = {
            "agent_ids": sorted(self.agents.keys()),
            "default_config": dict(self.default_config.__dict__),
            "agents": {aid: agent.state_dict() for aid, agent in self.agents.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load(self, path: Path, map_location: str = "cpu") -> None:
        import torch

        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        agents_state = payload.get("agents", {})
        for aid, state in agents_state.items():
            if aid in self.agents:
                self.agents[aid].load_state_dict(state)


def safe_agent_filename(agent_id: str) -> str:
    safe = agent_id.replace("/", "__").replace("\\", "__").replace(":", "_")
    return f"agent_{safe}.pt"
