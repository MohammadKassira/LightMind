"""Max Pressure utilities and controller logic for SUMO + TraCI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import traci

LaneMovement = tuple[str, str]
PhaseMovements = list[LaneMovement]
LaneMapping = dict[str, dict[int, PhaseMovements]]

EXTERNAL_PHASE_HOLD_SECONDS = 1_000_000.0


def compute_pressure_for_phase(phase_movements: Sequence[LaneMovement]) -> int:
    """Compute the pressure of one phase as ``sum(q_in - q_out)``."""
    pressure = 0
    for incoming_lane, outgoing_lane in phase_movements:
        q_in = traci.lane.getLastStepVehicleNumber(incoming_lane)
        q_out = traci.lane.getLastStepVehicleNumber(outgoing_lane)
        pressure += q_in - q_out

    return int(pressure)


def select_phase(tls_id: str, lane_mapping: LaneMapping) -> int:
    """Select the configured phase with the highest pressure for one TLS."""
    phase_mapping = lane_mapping.get(tls_id)
    if not phase_mapping:
        raise KeyError(f"No lane mapping found for traffic light '{tls_id}'.")

    best_phase_id: int | None = None
    best_pressure: int | None = None

    for phase_id, phase_movements in sorted(phase_mapping.items()):
        pressure = compute_pressure_for_phase(phase_movements)
        if best_pressure is None or pressure > best_pressure:
            best_phase_id = phase_id
            best_pressure = pressure

    if best_phase_id is None:
        raise ValueError(f"Traffic light '{tls_id}' has no configured phases.")

    return best_phase_id


def load_lane_mapping(mapping_path: str | Path) -> LaneMapping:
    """Load and normalize a lane-mapping JSON file."""
    path = Path(mapping_path)
    with path.open("r", encoding="utf-8") as handle:
        raw_mapping = json.load(handle)

    return normalize_lane_mapping(raw_mapping)


def normalize_lane_mapping(raw_mapping: Mapping[str, Any]) -> LaneMapping:
    """Normalize ``{tls_id: {phase_id: movements}}`` into typed Python data."""
    normalized: LaneMapping = {}

    for tls_id, phase_mapping in raw_mapping.items():
        if not isinstance(phase_mapping, Mapping):
            raise TypeError(
                f"Lane mapping for traffic light '{tls_id}' must be a phase dictionary."
            )

        normalized[tls_id] = {}
        for raw_phase_id, raw_movements in phase_mapping.items():
            phase_id = _coerce_phase_id(raw_phase_id)
            movements = _normalize_phase_movements(tls_id, phase_id, raw_movements)
            normalized[tls_id][phase_id] = movements

    return normalized


def validate_lane_mapping_for_tls_ids(
    tls_ids: Sequence[str],
    lane_mapping: LaneMapping,
) -> None:
    """Validate that every controlled traffic light has mapping data."""
    missing_tls_ids = [tls_id for tls_id in tls_ids if tls_id not in lane_mapping]
    if missing_tls_ids:
        raise KeyError(
            "Missing lane mapping for traffic lights: "
            + ", ".join(sorted(missing_tls_ids))
        )


def validate_lane_mapping_against_sumo(
    tls_ids: Sequence[str],
    lane_mapping: LaneMapping,
) -> None:
    """Ensure every mapped phase ID exists in the active SUMO signal program."""
    validate_lane_mapping_for_tls_ids(tls_ids, lane_mapping)

    for tls_id in tls_ids:
        valid_phase_ids = get_valid_phase_ids(tls_id)
        configured_phase_ids = set(lane_mapping[tls_id].keys())
        invalid_phase_ids = sorted(configured_phase_ids - valid_phase_ids)

        if invalid_phase_ids:
            raise ValueError(
                f"Traffic light '{tls_id}' has invalid phase IDs {invalid_phase_ids}. "
                f"Valid phase IDs are {sorted(valid_phase_ids)}."
            )


def get_valid_phase_ids(tls_id: str) -> set[int]:
    """Return the valid phase IDs for the active SUMO program of one TLS."""
    logic = get_active_program_logic(tls_id)
    return set(range(len(logic.phases)))


def get_phase_state_string(tls_id: str, phase_id: int) -> str:
    """Return the signal-state string for one phase of the active TLS program."""
    logic = get_active_program_logic(tls_id)
    return str(logic.phases[phase_id].state)


def get_active_program_logic(tls_id: str) -> Any:
    """Return the active TraCI program logic for a traffic light."""
    program_logics = list(traci.trafficlight.getAllProgramLogics(tls_id))
    if not program_logics:
        raise RuntimeError(f"No program logics found for traffic light '{tls_id}'.")

    active_program_id = traci.trafficlight.getProgram(tls_id)
    for logic in program_logics:
        if getattr(logic, "programID", None) == active_program_id:
            return logic

    return program_logics[0]


@dataclass
class TrafficLightControlState:
    """Mutable control state tracked per traffic light."""

    current_green_phase: int
    target_phase: int | None
    is_in_yellow: bool
    yellow_timer: int
    last_switch_step: int


class MaxPressureController:
    """Owns phase selection, min-green, and optional yellow transitions."""

    def __init__(
        self,
        lane_mapping: LaneMapping,
        tls_ids: Sequence[str],
        action_interval: int,
        min_green_time: int,
        enable_yellow: bool = True,
        yellow_time: int = 3,
    ) -> None:
        if action_interval <= 0:
            raise ValueError("action_interval must be greater than zero.")
        if min_green_time < 0:
            raise ValueError("min_green_time must be zero or greater.")
        if yellow_time < 0:
            raise ValueError("yellow_time must be zero or greater.")

        self.lane_mapping = lane_mapping
        self.tls_ids = list(tls_ids)
        self.action_interval = action_interval
        self.min_green_time = min_green_time
        self.enable_yellow = enable_yellow
        self.yellow_time = yellow_time
        self.control_states: dict[str, TrafficLightControlState] = {}

        self.reset()

    def compute_action(self, tls_id: str) -> int:
        """Compute the highest-pressure target green phase for one TLS."""
        return select_phase(tls_id, self.lane_mapping)

    def update_state(self, tls_id: str, step: int) -> int:
        """Advance one TLS control state and apply changes only when needed."""
        control_state = self.control_states[tls_id]

        if control_state.is_in_yellow:
            self._advance_yellow_transition(tls_id, control_state, step)
            return self.get_current_phase_snapshot(tls_id)

        if step % self.action_interval != 0:
            return self.get_current_phase_snapshot(tls_id)

        selected_phase = self.compute_action(tls_id)
        if selected_phase == control_state.current_green_phase:
            return self.get_current_phase_snapshot(tls_id)

        green_elapsed = step - control_state.last_switch_step
        if green_elapsed < self.min_green_time:
            return self.get_current_phase_snapshot(tls_id)

        if self.enable_yellow and self.yellow_time > 0:
            self._start_yellow_transition(tls_id, control_state, selected_phase)
        else:
            self._apply_green_phase(tls_id, selected_phase)
            control_state.current_green_phase = selected_phase
            control_state.last_switch_step = step

        return self.get_current_phase_snapshot(tls_id)

    def update_all(self, step: int) -> dict[str, int]:
        """Update every traffic light for the current simulation step."""
        return {
            tls_id: self.update_state(tls_id, step)
            for tls_id in self.tls_ids
        }

    def reset(self) -> None:
        """Reinitialize controller state after the SUMO simulation resets."""
        validate_lane_mapping_against_sumo(self.tls_ids, self.lane_mapping)
        self.control_states.clear()
        self._initialize_control_states()

    def get_current_phase_snapshot(self, tls_id: str) -> int:
        """Return the controller's current green-phase snapshot for logging."""
        return self.control_states[tls_id].current_green_phase

    def _initialize_control_states(self) -> None:
        """Initialize controller state from the current SUMO simulation."""
        for tls_id in self.tls_ids:
            current_phase = int(traci.trafficlight.getPhase(tls_id))
            initial_phase = current_phase
            if initial_phase not in self.lane_mapping[tls_id]:
                initial_phase = select_phase(tls_id, self.lane_mapping)
                self._apply_green_phase(tls_id, initial_phase)

            self.control_states[tls_id] = TrafficLightControlState(
                current_green_phase=initial_phase,
                target_phase=None,
                is_in_yellow=False,
                yellow_timer=0,
                last_switch_step=-self.min_green_time,
            )

            self._hold_phase(tls_id)

    def _advance_yellow_transition(
        self,
        tls_id: str,
        control_state: TrafficLightControlState,
        step: int,
    ) -> None:
        """Count down yellow time and activate the target green when finished."""
        control_state.yellow_timer -= 1
        if control_state.yellow_timer > 0:
            return

        if control_state.target_phase is None:
            raise RuntimeError(
                f"Traffic light '{tls_id}' finished yellow without a target phase."
            )

        self._apply_green_phase(tls_id, control_state.target_phase)
        control_state.current_green_phase = control_state.target_phase
        control_state.target_phase = None
        control_state.is_in_yellow = False
        control_state.yellow_timer = 0
        control_state.last_switch_step = step

    def _start_yellow_transition(
        self,
        tls_id: str,
        control_state: TrafficLightControlState,
        target_phase: int,
    ) -> None:
        """Start a fixed-time yellow transition before a green-phase change."""
        yellow_state = self._build_yellow_state(tls_id)
        traci.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
        self._hold_phase(tls_id)
        control_state.target_phase = target_phase
        control_state.is_in_yellow = True
        control_state.yellow_timer = self.yellow_time

    def _apply_green_phase(self, tls_id: str, phase_id: int) -> None:
        """Apply a green phase only when it differs from the current phase."""
        current_phase = int(traci.trafficlight.getPhase(tls_id))
        if current_phase != phase_id:
            traci.trafficlight.setPhase(tls_id, phase_id)

        self._hold_phase(tls_id)

    def _hold_phase(self, tls_id: str) -> None:
        """Extend the active phase so SUMO does not auto-advance it."""
        traci.trafficlight.setPhaseDuration(tls_id, EXTERNAL_PHASE_HOLD_SECONDS)

    def _build_yellow_state(self, tls_id: str) -> str:
        """Derive a transient yellow signal state from the current signal state."""
        current_state = traci.trafficlight.getRedYellowGreenState(tls_id)
        yellow_state_chars: list[str] = []

        for signal_char in current_state:
            if signal_char in {"g", "G"}:
                yellow_state_chars.append("y")
            elif signal_char in {"r", "R"}:
                yellow_state_chars.append("r")
            elif signal_char in {"y", "Y"}:
                yellow_state_chars.append("y")
            else:
                yellow_state_chars.append(signal_char)

        return "".join(yellow_state_chars)


def _coerce_phase_id(raw_phase_id: Any) -> int:
    """Convert a JSON phase key to an integer phase ID."""
    if isinstance(raw_phase_id, int):
        return raw_phase_id
    if isinstance(raw_phase_id, str) and raw_phase_id.strip():
        return int(raw_phase_id)

    raise TypeError(f"Invalid phase ID '{raw_phase_id}'. Expected an integer-like value.")


def _normalize_phase_movements(
    tls_id: str,
    phase_id: int,
    raw_movements: Any,
) -> PhaseMovements:
    """Validate and normalize movement tuples for one phase."""
    if not isinstance(raw_movements, Sequence) or isinstance(raw_movements, (str, bytes)):
        raise TypeError(
            f"Phase '{phase_id}' for traffic light '{tls_id}' must be a list of movements."
        )

    movements: PhaseMovements = []
    for raw_movement in raw_movements:
        if (
            not isinstance(raw_movement, Sequence)
            or isinstance(raw_movement, (str, bytes))
            or len(raw_movement) != 2
        ):
            raise TypeError(
                f"Phase '{phase_id}' for traffic light '{tls_id}' contains "
                "an invalid movement. Expected [incoming_lane, outgoing_lane]."
            )

        incoming_lane, outgoing_lane = raw_movement
        if not isinstance(incoming_lane, str) or not isinstance(outgoing_lane, str):
            raise TypeError(
                f"Phase '{phase_id}' for traffic light '{tls_id}' must use string lane IDs."
            )

        movements.append((incoming_lane, outgoing_lane))

    return movements
