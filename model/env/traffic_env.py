"""Real SUMO/TraCI traffic environment matching the MockEnv §3.5 API.

Drop-in replacement for MockEnv: same reset()/step() signatures and
obs_dict schema (§3.2), so DQNTrainer and all downstream code work unchanged.
step() returns 0.0 placeholder rewards — the trainer owns all reward computation.

Requires SUMO_HOME to be set in the environment.
"""

import os
import random
import sys
import threading
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

# -- SUMO bootstrap ----------------------------------------------------------
_SUMO_HOME = os.environ.get("SUMO_HOME")
if _SUMO_HOME:
    sys.path.insert(0, os.path.join(_SUMO_HOME, "tools"))
else:
    raise RuntimeError(
        "SUMO_HOME is not set. "
        "Set it to your SUMO installation directory before importing traffic_env."
    )
import sumolib  # noqa: E402  (after sys.path update)
import traci    # noqa: E402

# -- Project imports ---------------------------------------------------------
from data.graph_builder import build_graph

_REPO_ROOT = Path(__file__).parent.parent
_NORM_CFG  = _REPO_ROOT / "configs" / "normalization.yaml"

_norm = yaml.safe_load(_NORM_CFG.read_text())
_Q_MAX         = float(_norm.get("q_max", 30))
_MAX_PHASE_TIME = float(_norm.get("max_phase_time", 90.0))

# Unique connection label counter (allows multiple TrafficEnv instances).
# Lock guards against duplicate labels when multiple envs are created concurrently
# (e.g. 5-episode parallel eval).
_LABEL_LOCK = threading.Lock()
_LABEL_COUNTER = 0


def _next_label() -> str:
    global _LABEL_COUNTER
    with _LABEL_LOCK:
        _LABEL_COUNTER += 1
        return f"trafficenv_{_LABEL_COUNTER}"


# ---------------------------------------------------------------------------
# Internal XML helpers (run once at __init__, no SUMO needed)
# ---------------------------------------------------------------------------

def _parse_net_xml(net_path: Path):
    """Extract TLS phase states, incoming lane IDs, and outgoing lane IDs from net.xml.

    Returns:
        green_states  {jid: [state_str, ...]}  — actionable (green) phase states
        yellow_dict   {jid: {(i,j): state_str}} — yellow transition states
        incoming_lanes {jid: [lane_id, ...]}   — canonical sorted order
        outgoing_lanes {jid: [lane_id, ...]}   — canonical sorted order
    """
    root = ET.parse(net_path).getroot()

    # Phase states per TLS (all phases, including yellow/red)
    tl_phases: Dict[str, List[str]] = {
        tl.get("id"): [p.get("state", "") for p in tl.findall("phase")]
        for tl in root.findall("tlLogic")
    }

    # Connection records per TLS: (from_edge, from_lane_idx, link_idx, to_edge, to_lane_idx)
    tl_conns: Dict[str, List[Tuple]] = defaultdict(list)
    for conn in root.findall("connection"):
        tl_id    = conn.get("tl")
        link_idx = conn.get("linkIndex")
        if tl_id is None or link_idx is None:
            continue
        tl_conns[tl_id].append((
            conn.get("from", ""),
            int(conn.get("fromLane", "0")),
            int(link_idx),
            conn.get("to", ""),
            int(conn.get("toLane", "0")),
        ))

    def _is_actionable(state: str) -> bool:
        return any(c in ("G", "g") for c in state) and "y" not in state

    green_states:   Dict[str, List[str]]           = {}
    yellow_dict:    Dict[str, Dict[Tuple, str]]    = {}
    incoming_lanes: Dict[str, List[str]]           = {}
    outgoing_lanes: Dict[str, List[str]]           = {}

    for jid, all_states in tl_phases.items():
        gs = [s for s in all_states if _is_actionable(s)]
        green_states[jid] = gs

        yd: Dict[Tuple, str] = {}
        for i, p1 in enumerate(gs):
            for j, p2 in enumerate(gs):
                if i == j:
                    continue
                yellow_str = ""
                for k in range(len(p1)):
                    if p1[k] in ("G", "g") and p2[k] in ("r", "s"):
                        yellow_str += "y"
                    else:
                        yellow_str += p1[k]
                yd[(i, j)] = yellow_str
        yellow_dict[jid] = yd

        conns = tl_conns.get(jid, [])
        # Incoming: sorted by (from_edge, from_lane_idx) — matches graph_builder ordering
        inc_pairs = sorted({(fe, fl) for fe, fl, _, _, _ in conns})
        incoming_lanes[jid] = [f"{fe}_{fl}" for fe, fl in inc_pairs]
        # Outgoing: sorted for determinism
        out_pairs = sorted({(te, tl_) for _, _, _, te, tl_ in conns})
        outgoing_lanes[jid] = [f"{te}_{tl_}" for te, tl_ in out_pairs]

    return green_states, yellow_dict, incoming_lanes, outgoing_lanes


# ---------------------------------------------------------------------------
# TrafficEnv
# ---------------------------------------------------------------------------

class TrafficEnv:
    """SUMO/TraCI environment implementing the §3.5 step API.

    Args:
        net_file:   Path to SUMO .net.xml
        route_file: Path to SUMO .rou.xml (must exist; use demand_generator if needed)
        max_steps:  Episode length in action steps (not simulation seconds)
        delta_time: Simulation seconds advanced per step() call
        yellow_time: Seconds to hold yellow phase on a phase transition
        min_green:   Minimum green-phase seconds before a change is allowed
        use_gui:    Open SUMO-GUI window (slow; useful for debugging)
        begin_time: SUMO simulation start time in seconds (0 = midnight; 25200 = 7 AM)
    """

    def __init__(
        self,
        net_file:             str | Path,
        route_file:           str | Path,
        max_steps:            int  = 200,
        delta_time:           int  = 5,
        yellow_time:          int  = 2,
        min_green:            int  = 5,
        use_gui:              bool = False,
        begin_time:           int  = 0,
        additional_files:     str | None = None,
        route_files:          list | None = None,
        override_tl_program:  bool = True,
        passive:              bool = False,
        step_delay_sec:       float = 0.0,
    ) -> None:
        self._net_file        = str(Path(net_file).resolve())
        self._route_file      = str(Path(route_file).resolve())
        self._route_files     = [str(Path(f).resolve()) for f in route_files] if route_files else None
        self._additional_files = str(Path(additional_files).resolve()) if additional_files else None
        self._max_steps  = max_steps
        self._delta_time = delta_time
        self._yellow_time = yellow_time
        self._min_green  = min_green
        self._use_gui    = use_gui
        self._begin_time = begin_time
        self._override_tl_program = override_tl_program
        self._passive    = passive
        self._step_delay_sec = step_delay_sec
        self._label      = _next_label()
        self._conn       = None   # traci connection; None until reset()
        self._step_count = 0

        # Graph (static; built from net.xml once)
        self._graph = build_graph(Path(net_file))

        # Lane ordering and phase tables (parsed from net.xml; no SUMO needed)
        (
            self._green_states,
            self._yellow_dict,
            self._incoming_lanes,
            self._outgoing_lanes,
        ) = _parse_net_xml(Path(net_file))

        # Per-node phase state (initialised in reset)
        self._phase_state: Dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # §3.5 API
    # -----------------------------------------------------------------------

    def reset(self, seed=None, network_cfg=None) -> tuple:
        """Start a new SUMO episode. Returns (obs_dict, graph).

        On the first call, starts a fresh SUMO process. On subsequent calls,
        reuses the existing process via traci.load() — no process kill/spawn,
        which eliminates the port-conflict crashes seen with parallel workers.
        """
        active_route = (
            random.choice(self._route_files) if self._route_files
            else self._route_file
        )
        self.active_route = active_route
        load_args = [
            "-n", self._net_file,
            "-r", active_route,
            "--no-warnings",
            "--no-step-log",
            "--time-to-teleport", "300",   # vehicles stuck >5 min teleport; -1 caused permanent gridlock
            "--max-depart-delay", "60",    # cancel vehicles that can't enter network within 60 s
            "--ignore-route-errors",
            "-b", str(self._begin_time),
        ]
        if self._additional_files:
            load_args += ["--additional-files", self._additional_files]
        if seed is not None:
            load_args += ["--seed", str(int(seed) & 0x7FFFFFFF)]
        else:
            load_args += ["--random"]

        if self._conn is None:
            # First episode: start the SUMO process
            binary = sumolib.checkBinary("sumo-gui" if self._use_gui else "sumo")
            if self._use_gui:
                load_args += [
                    "--start",                   # auto-start so TraCI can drive without pressing play
                    "--window-size", "1280,800", # fill the Xvfb virtual screen
                    "--window-pos", "0,20",
                ]
            traci.start([binary] + load_args, label=self._label)
            self._conn = traci.getConnection(self._label)
        else:
            # Subsequent episodes: reload simulation in the existing process
            self._conn.load(load_args)

        if self._override_tl_program:
            self._setup_tls()
        self._step_count = 0

        obs_dict = self._extract_obs()
        return obs_dict, self._graph

    def step(self, actions: dict) -> tuple:
        """Advance the simulation one action step.

        Args:
            actions: {node_id: phase_idx}

        Returns:
            (obs_dict, graph, reward_dict, done, info)
        """
        if self._passive:
            # Fixed-time baseline: SUMO drives its own TL phases from net.xml program.
            self._run_sim_steps()
            self._step_count += 1
            obs_dict    = self._extract_obs()
            reward_dict = {nid: 0.0 for nid in self._graph["node_ids"]}
            no_vehicles_left = self._conn.simulation.getMinExpectedNumber() == 0
            done = self._step_count >= self._max_steps or no_vehicles_left
            return obs_dict, self._graph, reward_dict, done, self._build_info()

        self._apply_actions(actions)
        self._run_sim_steps()
        self._step_count += 1
        obs_dict    = self._extract_obs()
        reward_dict = {nid: 0.0 for nid in self._graph["node_ids"]}
        # Natural termination: all vehicles have left and none are pending
        no_vehicles_left = self._conn.simulation.getMinExpectedNumber() == 0
        done        = self._step_count >= self._max_steps or no_vehicles_left
        return obs_dict, self._graph, reward_dict, done, self._build_info()

    def close(self) -> None:
        self._close_sumo()

    # -----------------------------------------------------------------------
    # SUMO lifecycle helpers
    # -----------------------------------------------------------------------

    def _build_info(self) -> dict:
        return {
            "sim_time":               self._conn.simulation.getTime(),
            "step_mean_waiting_time": self._mean_waiting_time(),
            "step_num_vehicles":      self._conn.vehicle.getIDCount(),
            "step_throughput":        self._conn.simulation.getArrivedNumber(),
            "step_queue_length":      self._total_queue_length(),
        }

    def _close_sumo(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _setup_tls(self) -> None:
        """Upload custom program logic to SUMO and initialise phase-state dicts."""
        for node_id in self._graph["node_ids"]:
            gs = self._green_states.get(node_id, [])
            if not gs:
                continue

            # Build Phase objects for our green phases + yellow transitions
            green_phase_objs = [
                self._conn.trafficlight.Phase(60, s) for s in gs
            ]
            all_phase_objs = green_phase_objs.copy()
            yd = self._yellow_dict.get(node_id, {})
            # Build yellow phase objects in the same (i,j) iteration order
            yellow_idx_map: Dict[Tuple, int] = {}
            for i in range(len(gs)):
                for j in range(len(gs)):
                    if i == j:
                        continue
                    yellow_idx_map[(i, j)] = len(all_phase_objs)
                    all_phase_objs.append(
                        self._conn.trafficlight.Phase(self._yellow_time, yd[(i, j)])
                    )

            # Upload modified program
            programs = self._conn.trafficlight.getAllProgramLogics(node_id)
            logic = programs[0]
            logic.type = 0
            logic.phases = all_phase_objs
            self._conn.trafficlight.setProgramLogic(node_id, logic)
            # Set SUMO to green phase 0
            self._conn.trafficlight.setRedYellowGreenState(node_id, gs[0])

            self._phase_state[node_id] = {
                "current_phase": 0,
                "time_in_phase": 0,
                "is_yellow":     False,
                "yellow_target": 0,
                "yellow_timer":  0,
                "yellow_idx_map": yellow_idx_map,  # (i,j) -> all_phases index (unused; we use state strings)
            }

    # -----------------------------------------------------------------------
    # Action application
    # -----------------------------------------------------------------------

    def _apply_actions(self, actions: dict) -> None:
        for node_id, new_phase in actions.items():
            ps = self._phase_state.get(node_id)
            if ps is None:
                continue
            gs  = self._green_states.get(node_id, [])
            cur = ps["current_phase"]

            if ps["is_yellow"]:
                # Mid-yellow: ignore action, let yellow finish
                continue

            # Min-green guard
            if new_phase == cur or ps["time_in_phase"] < self._min_green + self._yellow_time:
                # Hold current green state
                self._conn.trafficlight.setRedYellowGreenState(node_id, gs[cur])
            else:
                # Transition: insert yellow
                yellow_str = self._yellow_dict[node_id].get((cur, new_phase), gs[cur])
                self._conn.trafficlight.setRedYellowGreenState(node_id, yellow_str)
                ps["is_yellow"]     = True
                ps["yellow_target"] = new_phase
                ps["yellow_timer"]  = self._yellow_time
                ps["time_in_phase"] = 0

    def _run_sim_steps(self) -> None:
        """Advance SUMO by delta_time seconds, ticking phase-state timers."""
        import time as _time
        for _ in range(self._delta_time):
            self._conn.simulationStep()
            self._tick_phase_states()
            if self._step_delay_sec > 0:
                _time.sleep(self._step_delay_sec)

    def _tick_phase_states(self) -> None:
        for node_id, ps in self._phase_state.items():
            gs = self._green_states.get(node_id, [])
            if ps["is_yellow"]:
                ps["yellow_timer"] -= 1
                if ps["yellow_timer"] <= 0:
                    # Yellow over — switch to target green
                    ps["current_phase"] = ps["yellow_target"]
                    ps["is_yellow"]     = False
                    ps["time_in_phase"] = 0
                    self._conn.trafficlight.setRedYellowGreenState(
                        node_id, gs[ps["current_phase"]]
                    )
            else:
                ps["time_in_phase"] += 1

    # -----------------------------------------------------------------------
    # Vehicle metrics helpers
    # -----------------------------------------------------------------------

    def _mean_waiting_time(self) -> float:
        """Mean per-vehicle waiting time (seconds) for all vehicles in network."""
        veh_ids = self._conn.vehicle.getIDList()
        if not veh_ids:
            return 0.0
        return sum(self._conn.vehicle.getWaitingTime(v) for v in veh_ids) / len(veh_ids)

    def _total_queue_length(self) -> float:
        """Total halting vehicles across all incoming lanes of controlled nodes."""
        total = 0
        for node_id in self._graph["node_ids"]:
            for lid in self._incoming_lanes.get(node_id, []):
                try:
                    total += self._conn.lane.getLastStepHaltingNumber(lid)
                except Exception:
                    pass
        return float(total)

    # -----------------------------------------------------------------------
    # Observation extraction (§3.2 layout)
    # -----------------------------------------------------------------------

    def _extract_obs(self) -> dict:
        obs_dict = {}
        for node_id in self._graph["node_ids"]:
            obs_dict[node_id] = self._node_obs(node_id)
        return obs_dict

    def _node_obs(self, node_id: str) -> tuple:
        """Build (obs, validity) tensors for one node matching §3.2.

        Layout:
          phase_onehot     [num_phases]
          time_in_phase    [1]
          queue_in/q_max   [num_incoming]
          running_in/q_max [num_incoming]
          queue_out/q_max  [num_outgoing]

        validity is all-ones — real env returns clean data; apply_perception
        is applied by the trainer outside this env.
        """
        node_idx   = self._graph["node_to_idx"][node_id]
        num_phases = self._graph["node_meta"][node_idx]["num_phases"]
        ps         = self._phase_state.get(node_id)

        # Phase one-hot
        phase_onehot = [0.0] * num_phases
        if ps is not None and num_phases > 0:
            cur = ps["current_phase"]
            if 0 <= cur < num_phases:
                phase_onehot[cur] = 1.0

        # Time in phase (normalised)
        t_norm = 0.0
        if ps is not None:
            t_norm = min(ps["time_in_phase"] / _MAX_PHASE_TIME, 1.0)

        # Per incoming lane
        inc_lanes = self._incoming_lanes.get(node_id, [])
        queue_in  = []
        running_in = []
        for lid in inc_lanes:
            try:
                halting = self._conn.lane.getLastStepHaltingNumber(lid)
                total   = self._conn.lane.getLastStepVehicleNumber(lid)
            except traci.exceptions.TraCIException:
                halting, total = 0, 0
            queue_in.append(min(halting / _Q_MAX, 1.0))
            running_in.append(min(max(total - halting, 0) / _Q_MAX, 1.0))

        # Per outgoing lane
        out_lanes = self._outgoing_lanes.get(node_id, [])
        queue_out = []
        for lid in out_lanes:
            try:
                halting = self._conn.lane.getLastStepHaltingNumber(lid)
            except traci.exceptions.TraCIException:
                halting = 0
            queue_out.append(min(halting / _Q_MAX, 1.0))

        obs = torch.tensor(
            phase_onehot + [t_norm] + queue_in + running_in + queue_out,
            dtype=torch.float32,
        )
        validity = torch.ones_like(obs)
        return obs, validity

    # -----------------------------------------------------------------------
    # Context-manager support + destructor
    # -----------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        # Best-effort cleanup if close() was never called. Suppresses all
        # exceptions because __del__ must not raise (Python ignores them anyway).
        try:
            self._close_sumo()
        except Exception:
            pass
