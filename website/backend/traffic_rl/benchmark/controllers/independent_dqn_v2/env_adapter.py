from __future__ import annotations

import json
import socket
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.controllers.fixed_time.run_fixed_time import (
    REPO_ROOT,
    _load_json,
    _resolve_canonical_additional_files_from_sumocfg,
    _resolve_demand_files_from_frozen_manifest,
    _resolve_path,
    _resolve_scenario_context,
    _resolve_validation_end_time,
)


DEFAULT_BANK_MANIFEST = REPO_ROOT / "benchmark/scenarios/frozen_reportable_scenario_bank_manifest.json"
DEFAULT_CONTRACT_PATH = REPO_ROOT / "benchmark/controllers/independent_dqn_v2/env_contract.json"
DEFAULT_DECISION_INTERVAL_S = 5.0

FORBIDDEN_PER_STEP_CALLS = (
    "traci.vehicle.getSpeed",
    "traci.vehicle.getLanePosition",
    "traci.lane.getLastStepVehicleIDs",
)


@dataclass
class RuntimeNormalization:
    avg_vehicle_length_m: float
    decision_interval_s: float


@dataclass
class TLSSpec:
    tls_id: str
    phases: list[Any]
    incoming_lanes_sorted: list[str]
    outgoing_lanes_sorted: list[str]
    selectable_green_phase_indices: list[int]
    yellow_all_red_phase_indices: list[int]
    action_phase_indices: list[int]
    action_index_by_phase_index: dict[int, int]
    base_action_masks_by_phase_index: dict[int, np.ndarray]
    phase_timer_max_s: float
    obs_dim: int
    outgoing_from_controlled_links: bool


@dataclass
class TLSRuntimeState:
    current_phase_index: int
    phase_enter_time_s: float


@dataclass
class StaticCache:
    tls_specs: dict[str, TLSSpec] = field(default_factory=dict)
    tls_runtime: dict[str, TLSRuntimeState] = field(default_factory=dict)
    lane_length_m_by_lane: dict[str, float] = field(default_factory=dict)
    lane_speed_mps_by_lane: dict[str, float] = field(default_factory=dict)
    q_max_lane_by_lane: dict[str, int] = field(default_factory=dict)
    lookahead_distance_m_by_lane: dict[str, float] = field(default_factory=dict)
    controlled_links_by_tls: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)
    phase_timer_norm_base_s_by_tls: dict[str, float] = field(default_factory=dict)
    static_cache_build_count: int = 0


@dataclass
class ScenarioInputs:
    scenario_id: str
    network: str
    level: str
    seed: int
    begin_s: int
    end_s: int
    net_file: Path
    route_files: list[Path]
    additional_files: list[Path]
    canonical_sumocfg_file: Path | None = None


@dataclass
class AdapterProfileResult:
    scenario_id: str
    network: str
    decision_ticks_profiled: int
    tls_agent_count: int
    obs_dim_by_tls: dict[str, int]
    observation_time_total_s: float
    observation_time_avg_per_decision_tick_s: float | None
    observation_time_avg_per_tls_observation_s: float | None
    observation_time_total_s_by_tls: dict[str, float]
    observation_time_avg_per_tls_per_decision_tick_s: dict[str, float]
    reward_time_total_s: float
    action_mask_time_total_s: float
    sumo_step_time_total_s: float
    forbidden_call_count: int
    forbidden_call_counts: dict[str, int]
    allowed_dynamic_call_counts: dict[str, int]
    static_cache_build_count: int
    q_max_lane_computed_count: int
    phase_timer_norm_base_computed_count: int
    outgoing_controlled_link_integrity_ok: bool
    yellow_all_red_contract_ok: bool
    observations_generated_all_tls: bool
    obs_dim_formula_ok: bool
    action_masks_valid: bool
    rewards_computed: bool
    no_neighbor_information_used: bool
    no_model_shared_weight_logic: bool
    training_launched: bool
    sumo_return_code: int
    controller_errors: list[str]


def _ensure_traci_import() -> Any:
    try:
        import traci  # type: ignore

        return traci
    except Exception:
        pass

    sumo_home = Path(__import__("os").environ.get("SUMO_HOME", "/usr/share/sumo"))
    tools = sumo_home / "tools"
    if tools.exists():
        sys.path.insert(0, str(tools))
    import traci  # type: ignore

    return traci


def _allocate_traci_port(start_port: int = 8813, max_tries: int = 3000) -> int:
    for port in range(start_port, start_port + max_tries):
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", port))
            return port
        except PermissionError as exc:
            raise RuntimeError(
                "Unable to open local sockets for TraCI port allocation. "
                "Run this command outside the restricted sandbox."
            ) from exc
        except OSError:
            continue
        finally:
            if sock is not None:
                sock.close()
    raise RuntimeError("Unable to allocate free TraCI port.")


def _is_selectable_green_phase(state: str) -> bool:
    if not state:
        return False
    has_green = any(ch in ("g", "G") for ch in state)
    has_yellow = any(ch in ("y", "Y") for ch in state)
    return has_green and not has_yellow


def _phase_successors(phase_idx: int, phase: Any, phase_count: int) -> list[int]:
    next_field = getattr(phase, "next", None)
    if isinstance(next_field, (list, tuple)) and len(next_field) > 0:
        out: list[int] = []
        for nxt in next_field:
            try:
                idx = int(nxt)
            except Exception:
                continue
            if 0 <= idx < phase_count:
                out.append(idx)
        if out:
            return sorted(set(out))
    if phase_count <= 0:
        return []
    return [((phase_idx + 1) % phase_count)]


def _load_contract(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_sumo_duration_to_seconds(raw: Any) -> float:
    try:
        return float(raw)
    except Exception:
        return 1.0


class ForbiddenCallTracker:
    """Monkey-patch wrappers to make forbidden-call checks auditable."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.counts: dict[str, int] = {
            "traci.vehicle.getSpeed": 0,
            "traci.vehicle.getLanePosition": 0,
            "traci.lane.getLastStepVehicleIDs": 0,
        }
        self._originals: list[tuple[Any, str, Any]] = []

    def _patch(self, obj: Any, attr: str, key: str) -> None:
        original = getattr(obj, attr)
        self._originals.append((obj, attr, original))

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            self.counts[key] = int(self.counts.get(key, 0)) + 1
            return original(*args, **kwargs)

        setattr(obj, attr, _wrapped)

    def __enter__(self) -> "ForbiddenCallTracker":
        self._patch(self._conn.vehicle, "getSpeed", "traci.vehicle.getSpeed")
        self._patch(self._conn.vehicle, "getLanePosition", "traci.vehicle.getLanePosition")
        self._patch(self._conn.lane, "getLastStepVehicleIDs", "traci.lane.getLastStepVehicleIDs")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for obj, attr, original in reversed(self._originals):
            setattr(obj, attr, original)


class IndependentDQNV2EnvAdapter:
    def __init__(
        self,
        *,
        contract_path: Path = DEFAULT_CONTRACT_PATH,
        decision_interval_s: float = DEFAULT_DECISION_INTERVAL_S,
    ) -> None:
        self.contract_path = contract_path
        self.contract = _load_contract(contract_path)
        self.normalization = RuntimeNormalization(
            avg_vehicle_length_m=float(self.contract["q_max_method"].get("avg_vehicle_length_m", 7.5)),
            decision_interval_s=float(decision_interval_s),
        )
        self.static_cache = StaticCache()
        self.allowed_dynamic_call_counts: dict[str, int] = {
            "traci.lane.getLastStepHaltingNumber": 0,
            "traci.lane.getLastStepVehicleNumber": 0,
            "traci.trafficlight.getPhase": 0,
        }

    def resolve_scenario(self, scenario_id: str, bank_manifest_path: Path = DEFAULT_BANK_MANIFEST) -> ScenarioInputs:
        bank = _load_json(bank_manifest_path)
        ctx = _resolve_scenario_context(bank, scenario_id)

        route_files = _resolve_demand_files_from_frozen_manifest(ctx.frozen_manifest)
        net_file = _resolve_path(ctx.source_manifest["source_network_artifact_path"])
        candidate_validation = ctx.source_manifest.get("candidate_validation", {}) or {}

        additional_files: list[Path] = []
        for p in candidate_validation.get("validation_additional_files", []) or []:
            if p:
                additional_files.append(_resolve_path(str(p)))

        canonical = candidate_validation.get("canonical_source_sumocfg_path") or ctx.source_validation.get(
            "canonical_source_sumocfg_path"
        )
        canonical_sumocfg_file = _resolve_path(str(canonical)) if canonical else None
        if not additional_files and canonical:
            additional_files = _resolve_canonical_additional_files_from_sumocfg(canonical_sumocfg_file)

        begin_s = int((ctx.source_manifest.get("timing_window_used", {}) or {}).get("generated_begin_s", 0) or 0)
        end_s = int(_resolve_validation_end_time(ctx))

        return ScenarioInputs(
            scenario_id=scenario_id,
            network=str(ctx.frozen_manifest["network"]),
            level=str(ctx.frozen_manifest["level"]),
            seed=int(ctx.frozen_manifest["seed"]),
            begin_s=begin_s,
            end_s=end_s,
            net_file=net_file,
            route_files=route_files,
            additional_files=additional_files,
            canonical_sumocfg_file=canonical_sumocfg_file,
        )

    def _build_traci_command(
        self,
        scenario: ScenarioInputs,
        run_dir: Path,
        *,
        include_xml_outputs: bool = True,
        suppress_warnings: bool = True,
    ) -> tuple[list[str], dict[str, Path]]:
        outputs = {
            "run_log": run_dir / "run.log",
            "sumo_stdout": run_dir / "sumo_stdout.txt",
            "sumo_error_log": run_dir / "sumo.error.log",
            "tripinfo_xml": run_dir / "tripinfo.xml",
            "summary_xml": run_dir / "summary.xml",
            "queue_xml": run_dir / "queue.xml",
        }
        cmd = [
            "sumo",
            "--net-file",
            str(scenario.net_file.resolve()),
            "--route-files",
            ",".join(str(p.resolve()) for p in scenario.route_files),
            "--begin",
            str(int(scenario.begin_s)),
            "--end",
            str(int(scenario.end_s)),
            "--no-step-log",
            "true",
            "--duration-log.statistics",
            "true",
            "--log",
            str(outputs["run_log"].resolve()),
            "--error-log",
            str(outputs["sumo_error_log"].resolve()),
        ]
        if suppress_warnings:
            cmd.extend(["--no-warnings", "true"])
        if include_xml_outputs:
            cmd.extend(
                [
                    "--tripinfo-output",
                    str(outputs["tripinfo_xml"].resolve()),
                    "--summary-output",
                    str(outputs["summary_xml"].resolve()),
                    "--queue-output",
                    str(outputs["queue_xml"].resolve()),
                ]
            )
        if scenario.additional_files:
            cmd.extend(["--additional-files", ",".join(str(p.resolve()) for p in scenario.additional_files)])
        return cmd, outputs

    def _count_allowed_call(self, key: str) -> None:
        self.allowed_dynamic_call_counts[key] = int(self.allowed_dynamic_call_counts.get(key, 0)) + 1

    def _build_static_cache(self, conn: Any) -> None:
        self.static_cache = StaticCache()
        sim_time_s = float(conn.simulation.getTime())

        lane_seen: set[str] = set()
        tls_ids = sorted(list(conn.trafficlight.getIDList()))

        for tls_id in tls_ids:
            all_logics = list(conn.trafficlight.getAllProgramLogics(tls_id))
            current_program_id = conn.trafficlight.getProgram(tls_id)
            chosen_logic = None
            for logic in all_logics:
                if getattr(logic, "programID", None) == current_program_id:
                    chosen_logic = logic
                    break
            if chosen_logic is None and all_logics:
                chosen_logic = all_logics[0]
            if chosen_logic is None:
                continue

            phases = list(getattr(chosen_logic, "phases", []) or [])
            if not phases:
                continue

            controlled_links_raw = conn.trafficlight.getControlledLinks(tls_id)
            controlled_links: list[tuple[str, str, str]] = []
            incoming: set[str] = set()
            outgoing: set[str] = set()
            for signal_links in controlled_links_raw:
                for link in signal_links:
                    if len(link) >= 3:
                        in_lane = str(link[0]) if link[0] else ""
                        out_lane = str(link[1]) if link[1] else ""
                        via_lane = str(link[2]) if link[2] else ""
                        if in_lane:
                            incoming.add(in_lane)
                        if out_lane:
                            outgoing.add(out_lane)
                        controlled_links.append((in_lane, out_lane, via_lane))

            incoming_sorted = sorted(incoming)
            outgoing_sorted = sorted(outgoing)
            if not incoming_sorted and not outgoing_sorted:
                continue

            selectable_green = [
                idx for idx, ph in enumerate(phases) if _is_selectable_green_phase(str(getattr(ph, "state", "")))
            ]
            yellow_all_red = [idx for idx in range(len(phases)) if idx not in selectable_green]
            if not selectable_green:
                continue

            successors: dict[int, list[int]] = {
                idx: _phase_successors(idx, ph, len(phases)) for idx, ph in enumerate(phases)
            }

            action_phase_indices = list(selectable_green)
            action_index_by_phase_index = {phase_idx: aidx for aidx, phase_idx in enumerate(action_phase_indices)}

            def _reachable_selectable(start_phase_idx: int) -> list[int]:
                seen: set[int] = set()
                frontier = [start_phase_idx]
                hits: set[int] = set()
                while frontier:
                    cur = frontier.pop(0)
                    if cur in seen:
                        continue
                    seen.add(cur)
                    if cur in action_index_by_phase_index:
                        hits.add(cur)
                    for nxt in successors.get(cur, []):
                        if nxt not in seen:
                            frontier.append(int(nxt))
                if not hits:
                    hits.update(action_phase_indices)
                return sorted(hits)

            base_masks: dict[int, np.ndarray] = {}
            n_actions = len(action_phase_indices)
            for phase_idx in range(len(phases)):
                mask = np.zeros((n_actions,), dtype=bool)
                reachable = _reachable_selectable(phase_idx)
                for rp in reachable:
                    aidx = action_index_by_phase_index.get(rp)
                    if aidx is not None:
                        mask[aidx] = True
                if not np.any(mask):
                    mask[:] = True
                base_masks[phase_idx] = mask

            all_lanes = sorted(set(incoming_sorted + outgoing_sorted))
            for lane_id in all_lanes:
                if lane_id in lane_seen:
                    continue
                lane_seen.add(lane_id)
                lane_length = float(conn.lane.getLength(lane_id))
                lane_speed = float(conn.lane.getMaxSpeed(lane_id))
                q_max_lane = max(1, int(np.floor(lane_length / self.normalization.avg_vehicle_length_m)))
                lookahead = float(max(lane_speed, 0.0) * self.normalization.decision_interval_s)
                self.static_cache.lane_length_m_by_lane[lane_id] = lane_length
                self.static_cache.lane_speed_mps_by_lane[lane_id] = lane_speed
                self.static_cache.q_max_lane_by_lane[lane_id] = q_max_lane
                self.static_cache.lookahead_distance_m_by_lane[lane_id] = lookahead

            selectable_durations = [
                _parse_sumo_duration_to_seconds(getattr(phases[pidx], "duration", 1.0))
                for pidx in selectable_green
            ]
            phase_timer_max_s = float(max(selectable_durations) if selectable_durations else 1.0)
            if phase_timer_max_s <= 0:
                phase_timer_max_s = 1.0

            obs_dim = len(phases) + 1 + (2 * len(incoming_sorted)) + len(outgoing_sorted)

            self.static_cache.tls_specs[tls_id] = TLSSpec(
                tls_id=tls_id,
                phases=phases,
                incoming_lanes_sorted=incoming_sorted,
                outgoing_lanes_sorted=outgoing_sorted,
                selectable_green_phase_indices=selectable_green,
                yellow_all_red_phase_indices=yellow_all_red,
                action_phase_indices=action_phase_indices,
                action_index_by_phase_index=action_index_by_phase_index,
                base_action_masks_by_phase_index=base_masks,
                phase_timer_max_s=phase_timer_max_s,
                obs_dim=int(obs_dim),
                outgoing_from_controlled_links=True,
            )
            self.static_cache.controlled_links_by_tls[tls_id] = controlled_links
            self.static_cache.phase_timer_norm_base_s_by_tls[tls_id] = phase_timer_max_s

            self._count_allowed_call("traci.trafficlight.getPhase")
            current_phase = int(conn.trafficlight.getPhase(tls_id))
            self.static_cache.tls_runtime[tls_id] = TLSRuntimeState(
                current_phase_index=current_phase,
                phase_enter_time_s=sim_time_s,
            )

        self.static_cache.static_cache_build_count += 1

    def _update_runtime_phases(self, conn: Any, sim_time_s: float) -> None:
        for tls_id, spec in self.static_cache.tls_specs.items():
            self._count_allowed_call("traci.trafficlight.getPhase")
            current_phase = int(conn.trafficlight.getPhase(tls_id))
            rt = self.static_cache.tls_runtime[tls_id]
            if current_phase != rt.current_phase_index:
                rt.current_phase_index = current_phase
                rt.phase_enter_time_s = sim_time_s
            if current_phase < 0 or current_phase >= len(spec.phases):
                rt.current_phase_index = 0

    def _build_observation_and_reward(self, conn: Any, tls_id: str, sim_time_s: float) -> tuple[np.ndarray, float]:
        spec = self.static_cache.tls_specs[tls_id]
        rt = self.static_cache.tls_runtime[tls_id]

        phase_count = len(spec.phases)
        current_phase = int(rt.current_phase_index)
        phase_one_hot = np.zeros((phase_count,), dtype=np.float32)
        if 0 <= current_phase < phase_count:
            phase_one_hot[current_phase] = 1.0

        elapsed = max(sim_time_s - float(rt.phase_enter_time_s), 0.0)
        phase_timer_norm = min(elapsed / max(spec.phase_timer_max_s, 1e-6), 1.0)

        incoming_queue_norm_sum = 0.0
        incoming_running_norm_sum = 0.0
        outgoing_queue_norm_sum = 0.0

        incoming_feats: list[float] = []
        for lane_id in spec.incoming_lanes_sorted:
            self._count_allowed_call("traci.lane.getLastStepHaltingNumber")
            queue = float(conn.lane.getLastStepHaltingNumber(lane_id))
            self._count_allowed_call("traci.lane.getLastStepVehicleNumber")
            veh_num = float(conn.lane.getLastStepVehicleNumber(lane_id))

            running = max(0.0, veh_num - queue)
            lane_len = max(self.static_cache.lane_length_m_by_lane.get(lane_id, 1.0), 1e-6)
            lookahead_m = max(self.static_cache.lookahead_distance_m_by_lane.get(lane_id, 0.0), 0.0)
            lookahead_ratio = min(1.0, lookahead_m / lane_len)
            running_lookahead = running * lookahead_ratio

            q_max_lane = float(max(self.static_cache.q_max_lane_by_lane.get(lane_id, 1), 1))
            queue_norm = min(queue / q_max_lane, 1.0)
            running_norm = min(running_lookahead / q_max_lane, 1.0)

            incoming_feats.append(queue_norm)
            incoming_feats.append(running_norm)
            incoming_queue_norm_sum += queue_norm
            incoming_running_norm_sum += running_norm

        outgoing_feats: list[float] = []
        for lane_id in spec.outgoing_lanes_sorted:
            self._count_allowed_call("traci.lane.getLastStepHaltingNumber")
            queue = float(conn.lane.getLastStepHaltingNumber(lane_id))
            q_max_lane = float(max(self.static_cache.q_max_lane_by_lane.get(lane_id, 1), 1))
            queue_norm = min(queue / q_max_lane, 1.0)
            outgoing_feats.append(queue_norm)
            outgoing_queue_norm_sum += queue_norm

        obs = np.concatenate(
            [
                phase_one_hot,
                np.asarray([phase_timer_norm], dtype=np.float32),
                np.asarray(incoming_feats, dtype=np.float32),
                np.asarray(outgoing_feats, dtype=np.float32),
            ]
        ).astype(np.float32)

        reward = -(incoming_queue_norm_sum + incoming_running_norm_sum - outgoing_queue_norm_sum)
        return obs, float(reward)

    def _build_action_mask(self, tls_id: str) -> np.ndarray:
        spec = self.static_cache.tls_specs[tls_id]
        rt = self.static_cache.tls_runtime[tls_id]
        cur_phase = int(rt.current_phase_index)
        mask = spec.base_action_masks_by_phase_index.get(cur_phase)
        if mask is None:
            mask = np.ones((len(spec.action_phase_indices),), dtype=bool)
        return np.asarray(mask, dtype=bool)

    def profile_scenario(
        self,
        *,
        scenario_id: str,
        max_decision_steps: int,
        run_root: Path,
        bank_manifest_path: Path = DEFAULT_BANK_MANIFEST,
    ) -> AdapterProfileResult:
        scenario = self.resolve_scenario(scenario_id, bank_manifest_path=bank_manifest_path)
        run_dir = run_root / f"idqn_v2_env_adapter_profile__{scenario_id}"
        if run_dir.exists():
            import shutil

            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)

        cmd, outputs = self._build_traci_command(scenario, run_dir)

        traci = _ensure_traci_import()
        label = f"idqn_v2_env_{int(time.time() * 1_000_000)}"

        obs_time_total = 0.0
        reward_time_total = 0.0
        action_mask_time_total = 0.0
        step_time_total = 0.0
        tls_obs_time_by_tls: dict[str, float] = {}

        decision_ticks = 0
        obs_generated_by_tls: dict[str, int] = {}
        action_masks_valid = True
        rewards_computed = True
        obs_dim_formula_ok = True
        controller_errors: list[str] = []

        process = None
        return_code = 1

        with outputs["sumo_stdout"].open("w", encoding="utf-8") as sumo_stdout:
            port = _allocate_traci_port()
            traci.start(cmd, port=port, label=label, numRetries=40, stdout=sumo_stdout, doSwitch=True)
            conn = traci.getConnection(label)
            process = conn._process

            try:
                self._build_static_cache(conn)
                if not self.static_cache.tls_specs:
                    raise RuntimeError("No controllable TLS agents discovered for adapter profiling.")

                next_decision_time_s = float(scenario.begin_s) + float(self.normalization.decision_interval_s)

                with ForbiddenCallTracker(conn) as forbidden_tracker:
                    while float(conn.simulation.getTime()) < float(scenario.end_s) and decision_ticks < int(
                        max_decision_steps
                    ):
                        t0 = time.perf_counter()
                        conn.simulationStep()
                        step_time_total += max(time.perf_counter() - t0, 0.0)
                        sim_time_s = float(conn.simulation.getTime())

                        self._update_runtime_phases(conn, sim_time_s)

                        if sim_time_s + 1e-9 < next_decision_time_s:
                            continue
                        next_decision_time_s = sim_time_s + float(self.normalization.decision_interval_s)

                        for tls_id, spec in self.static_cache.tls_specs.items():
                            ot0 = time.perf_counter()
                            obs, reward = self._build_observation_and_reward(conn, tls_id, sim_time_s)
                            obs_dt = max(time.perf_counter() - ot0, 0.0)
                            obs_time_total += obs_dt
                            tls_obs_time_by_tls[tls_id] = float(tls_obs_time_by_tls.get(tls_id, 0.0) + obs_dt)
                            obs_generated_by_tls[tls_id] = int(obs_generated_by_tls.get(tls_id, 0) + 1)

                            expected_dim = len(spec.phases) + 1 + (2 * len(spec.incoming_lanes_sorted)) + len(
                                spec.outgoing_lanes_sorted
                            )
                            if obs.shape != (expected_dim,):
                                obs_dim_formula_ok = False

                            rt0 = time.perf_counter()
                            if not np.isfinite(float(reward)):
                                rewards_computed = False
                            reward_time_total += max(time.perf_counter() - rt0, 0.0)

                            mt0 = time.perf_counter()
                            mask = self._build_action_mask(tls_id)
                            action_mask_time_total += max(time.perf_counter() - mt0, 0.0)
                            if mask.shape != (len(spec.action_phase_indices),) or not bool(np.any(mask)):
                                action_masks_valid = False

                        decision_ticks += 1

                forbidden_counts = dict(forbidden_tracker.counts)
                forbidden_call_count = int(sum(int(v) for v in forbidden_counts.values()))

            except Exception as exc:
                controller_errors.append(f"{exc}\n{traceback.format_exc()}")
                forbidden_counts = {
                    "traci.vehicle.getSpeed": -1,
                    "traci.vehicle.getLanePosition": -1,
                    "traci.lane.getLastStepVehicleIDs": -1,
                }
                forbidden_call_count = -1
            finally:
                try:
                    traci.close(wait=True)
                except Exception:
                    pass

        if process is not None and process.returncode is not None:
            return_code = int(process.returncode)

        tls_count = len(self.static_cache.tls_specs)
        total_tls_observations = int(sum(obs_generated_by_tls.values()))

        obs_dim_by_tls = {tls_id: int(spec.obs_dim) for tls_id, spec in self.static_cache.tls_specs.items()}
        observations_generated_all_tls = (
            tls_count > 0 and all(obs_generated_by_tls.get(tls_id, 0) == decision_ticks for tls_id in obs_dim_by_tls)
        )

        outgoing_integrity_ok = all(
            spec.outgoing_from_controlled_links and len(spec.outgoing_lanes_sorted) >= 0
            for spec in self.static_cache.tls_specs.values()
        )

        yellow_all_red_ok = all(
            len(spec.selectable_green_phase_indices) > 0
            and (len(spec.yellow_all_red_phase_indices) + len(spec.selectable_green_phase_indices) == len(spec.phases))
            for spec in self.static_cache.tls_specs.values()
        )

        return AdapterProfileResult(
            scenario_id=scenario.scenario_id,
            network=scenario.network,
            decision_ticks_profiled=int(decision_ticks),
            tls_agent_count=int(tls_count),
            obs_dim_by_tls=obs_dim_by_tls,
            observation_time_total_s=float(obs_time_total),
            observation_time_avg_per_decision_tick_s=(
                float(obs_time_total / decision_ticks) if decision_ticks > 0 else None
            ),
            observation_time_avg_per_tls_observation_s=(
                float(obs_time_total / total_tls_observations) if total_tls_observations > 0 else None
            ),
            observation_time_total_s_by_tls={
                tls_id: float(v) for tls_id, v in sorted(tls_obs_time_by_tls.items())
            },
            observation_time_avg_per_tls_per_decision_tick_s={
                tls_id: float(v / max(decision_ticks, 1))
                for tls_id, v in sorted(tls_obs_time_by_tls.items())
            },
            reward_time_total_s=float(reward_time_total),
            action_mask_time_total_s=float(action_mask_time_total),
            sumo_step_time_total_s=float(step_time_total),
            forbidden_call_count=int(forbidden_call_count),
            forbidden_call_counts=forbidden_counts,
            allowed_dynamic_call_counts=dict(self.allowed_dynamic_call_counts),
            static_cache_build_count=int(self.static_cache.static_cache_build_count),
            q_max_lane_computed_count=int(len(self.static_cache.q_max_lane_by_lane)),
            phase_timer_norm_base_computed_count=int(len(self.static_cache.phase_timer_norm_base_s_by_tls)),
            outgoing_controlled_link_integrity_ok=bool(outgoing_integrity_ok),
            yellow_all_red_contract_ok=bool(yellow_all_red_ok),
            observations_generated_all_tls=bool(observations_generated_all_tls),
            obs_dim_formula_ok=bool(obs_dim_formula_ok),
            action_masks_valid=bool(action_masks_valid),
            rewards_computed=bool(rewards_computed),
            no_neighbor_information_used=True,
            no_model_shared_weight_logic=True,
            training_launched=False,
            sumo_return_code=int(return_code),
            controller_errors=controller_errors,
        )


def parse_queue_mean_from_xml(queue_xml: Path) -> float | None:
    if not queue_xml.exists():
        return None
    try:
        total = 0.0
        n = 0
        for _event, elem in ET.iterparse(queue_xml, events=("end",)):
            if elem.tag == "data":
                try:
                    v = float(elem.attrib.get("queueing_length_experimental", "0") or 0.0)
                    total += v
                    n += 1
                except Exception:
                    pass
            elem.clear()
        if n > 0:
            return total / n
    except Exception:
        return None
    return None
