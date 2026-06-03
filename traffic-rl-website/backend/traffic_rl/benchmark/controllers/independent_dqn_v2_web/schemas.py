from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScenarioSpec:
    scenario_id: str
    level: str
    seed: int
    begin_s: int
    end_s: int
    period_s: float
    net_file: Path
    route_file: Path
    trips_file: Path
    sumocfg_file: Path
    additional_files: list[Path] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "level": self.level,
            "seed": int(self.seed),
            "begin_s": int(self.begin_s),
            "end_s": int(self.end_s),
            "period_s": float(self.period_s),
            "net_file": str(self.net_file),
            "route_file": str(self.route_file),
            "trips_file": str(self.trips_file),
            "sumocfg_file": str(self.sumocfg_file),
            "additional_files": [str(p) for p in self.additional_files],
        }


@dataclass
class WebJobConfig:
    job_id: str
    osm_path: Path
    output_root: Path
    network_name: str = "uploaded_map"
    begin_s: int = 0
    end_s: int = 3600
    train_episodes: int = 500
    train_max_steps_per_episode: int = 720
    train_every_steps: int = 8
    learning_starts_steps: int = 1000
    target_update_interval: int = 1000
    wall_clock_cap_minutes: float = 240.0
    train_seed: int = 17
    smoke_episodes: int = 2
    smoke_max_steps: int = 120
    smoke_seed: int = 11
    eval_seeds: list[int] = field(default_factory=lambda: [2, 3, 4])
    low_period_s: float = 4.0
    medium_period_s: float = 2.5
    high_period_s: float = 1.5

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["osm_path"] = str(self.osm_path)
        payload["output_root"] = str(self.output_root)
        return payload


@dataclass
class KPIResult:
    mean_waiting_time_completed_s: float | None
    throughput_completed_trips: int | None
    mean_total_queue_length_m: float | None
    phase_change_rate_per_tls_per_min: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "mean_waiting_time_completed_s": self.mean_waiting_time_completed_s,
            "throughput_completed_trips": self.throughput_completed_trips,
            "mean_total_queue_length_m": self.mean_total_queue_length_m,
            "phase_change_rate_per_tls_per_min": self.phase_change_rate_per_tls_per_min,
        }
