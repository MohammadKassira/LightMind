"""Environment interface for SUMO traffic-light control.

This module provides a simple reinforcement learning environment wrapper
around the SUMO simulator using ``traci``.
"""

from typing import Any

import traci


class TrafficLightEnv:
    """SUMO environment for controlling multiple traffic lights."""

    WARMUP_STEPS = 600
    TARGET_ACTIVE_VEHICLES = 80

    def __init__(self, sumo_cmd: list[str]) -> None:
        """Start a SUMO simulation with the provided command."""
        self.sumo_cmd = sumo_cmd
        self.episode_metrics = self._empty_episode_metrics()
        traci.start(self.sumo_cmd)

    def reset(self) -> dict[str, list[float]]:
        """Restart SUMO and return the initial state of all traffic lights."""
        # Close any running simulation before starting a fresh episode.
        self._close_simulation()
        traci.start(self.sumo_cmd)

        # Route flows in SUMO spawn vehicles only after the simulation advances.
        # Warm up the simulation so each episode starts with active traffic.
        for _ in range(self.WARMUP_STEPS):
            if len(traci.vehicle.getIDList()) >= self.TARGET_ACTIVE_VEHICLES:
                break
            traci.simulationStep()

        print(f"Reset vehicles={len(traci.vehicle.getIDList())}")

        self.episode_metrics = self._empty_episode_metrics()
        tls_ids = self.get_all_tls_ids()

        return {
            tls_id: self.get_state(tls_id)
            for tls_id in tls_ids
        }

    def step(
        self,
        action_dict: dict[str, int],
    ) -> tuple[dict[str, list[float]], dict[str, float], bool, dict[str, Any]]:
        """Apply actions, advance the simulation, and collect transition data."""
        # Action 0 keeps the current phase, while action 1 advances to the next one.
        action_changed_by_tls: dict[str, bool] = {}
        for tls_id, action in action_dict.items():
            action_changed = False
            if action == 1:
                current_phase = traci.trafficlight.getPhase(tls_id)
                phase_count = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
                next_phase = (current_phase + 1) % phase_count
                if next_phase != current_phase:
                    traci.trafficlight.setPhase(tls_id, next_phase)
                    action_changed = True
            action_changed_by_tls[tls_id] = action_changed

        # Advance SUMO by one simulation step.
        traci.simulationStep()

        tls_ids = self.get_all_tls_ids()
        next_state = {tls_id: self.get_state(tls_id) for tls_id in tls_ids}
        step_metrics = self.get_step_metrics(tls_ids)
        num_tls = max(len(tls_ids), 1)
        rewards = {
            tls_id: self.compute_reward(step_metrics["per_tls"][tls_id], step_metrics, num_tls)
            for tls_id in tls_ids
        }
        for tls_id, action_changed in action_changed_by_tls.items():
            if action_changed:
                rewards[tls_id] -= 0.1
        self._update_episode_metrics(step_metrics)
        done = False
        info: dict[str, Any] = {
            "step_metrics": step_metrics,
            "episode_metrics": dict(self.episode_metrics),
        }

        return next_state, rewards, done, info

    def get_state(self, tls_id: str) -> list[float]:
        """Return a fixed-length numeric state vector for one traffic light."""
        lane_ids = traci.trafficlight.getControlledLanes(tls_id)

        # Remove duplicates while preserving order because SUMO may repeat lanes.
        unique_lane_ids = list(dict.fromkeys(lane_ids))

        # Neural networks expect a consistent, fixed-size numerical input vector.
        # We therefore keep the lane order stable and append the current phase.
        queue_lengths = [
            float(traci.lane.getLastStepHaltingNumber(lane_id))
            for lane_id in unique_lane_ids
        ]
        current_phase = float(traci.trafficlight.getPhase(tls_id))

        return queue_lengths + [current_phase]

    def compute_reward(
        self,
        tls_metrics: dict[str, float],
        step_metrics: dict[str, Any],
        num_tls: int,
    ) -> float:
        """Combine throughput and queue penalty terms into a dense reward."""
        cars_passed_per_tls = step_metrics["cars_passed"] / num_tls
        teleports_per_tls = step_metrics["teleports"] / num_tls
        reward = (
            cars_passed_per_tls * 1.0
            - (tls_metrics["waiting_time"] / 200.0)
            - (tls_metrics["stopped_cars"] / 1000.0)
            - teleports_per_tls * 20.0
        )
        return float(max(-1.0, min(1.0, reward)))

    def get_total_queue(self, tls_id: str) -> float:
        """Return the total halted vehicles across lanes for one traffic light."""
        lane_ids = traci.trafficlight.getControlledLanes(tls_id)
        unique_lane_ids = list(dict.fromkeys(lane_ids))

        return float(
            sum(
                traci.lane.getLastStepHaltingNumber(lane_id)
                for lane_id in unique_lane_ids
            )
        )

    def get_all_tls_ids(self) -> list[str]:
        """Return all traffic light IDs in the current SUMO simulation."""
        return list(traci.trafficlight.getIDList())

    def get_step_metrics(self, tls_ids: list[str]) -> dict[str, Any]:
        """Collect network-level diagnostics for one simulation step."""
        per_tls = {tls_id: self.get_tls_metrics(tls_id) for tls_id in tls_ids}
        waiting_time = sum(m["waiting_time"] for m in per_tls.values())
        stopped_cars = sum(m["stopped_cars"] for m in per_tls.values())
        avg_speed_values = [m["avg_speed"] for m in per_tls.values()]

        return {
            "teleports": float(traci.simulation.getStartingTeleportNumber()),
            "cars_passed": float(traci.simulation.getArrivedNumber()),
            "waiting_time": float(waiting_time),
            "stopped_cars": float(stopped_cars),
            "avg_speed": float(sum(avg_speed_values) / len(avg_speed_values))
            if avg_speed_values
            else 0.0,
            "per_tls": per_tls,
        }

    def get_tls_metrics(self, tls_id: str) -> dict[str, float]:
        """Collect reward features for a single traffic light."""
        lane_ids = traci.trafficlight.getControlledLanes(tls_id)
        unique_lane_ids = list(dict.fromkeys(lane_ids))
        step_length = float(traci.simulation.getDeltaT())
        waiting_time = sum(
            traci.lane.getLastStepHaltingNumber(lane_id) * step_length
            for lane_id in unique_lane_ids
        )
        stopped_cars = sum(
            traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in unique_lane_ids
        )
        speeds = [traci.lane.getLastStepMeanSpeed(lane_id) for lane_id in unique_lane_ids]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        return {
            "waiting_time": float(waiting_time),
            "stopped_cars": float(stopped_cars),
            "avg_speed": float(avg_speed),
        }

    def _update_episode_metrics(self, step_metrics: dict[str, Any]) -> None:
        """Accumulate episode-level diagnostics for logging."""
        self.episode_metrics["teleports"] += step_metrics["teleports"]
        self.episode_metrics["cars_passed"] += step_metrics["cars_passed"]
        self.episode_metrics["waiting_time"] += step_metrics["waiting_time"]
        self.episode_metrics["stopped_cars"] += step_metrics["stopped_cars"]
        self.episode_metrics["avg_speed_total"] += step_metrics["avg_speed"]
        self.episode_metrics["metric_steps"] += 1.0

    def _empty_episode_metrics(self) -> dict[str, float]:
        """Create a fresh metrics container for a new episode."""
        return {
            "teleports": 0.0,
            "cars_passed": 0.0,
            "waiting_time": 0.0,
            "stopped_cars": 0.0,
            "avg_speed_total": 0.0,
            "metric_steps": 0.0,
        }

    def _close_simulation(self) -> None:
        """Close the current SUMO connection if one is active."""
        if traci.isLoaded():
            traci.close()
