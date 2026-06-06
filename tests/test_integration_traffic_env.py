"""Integration tests: verify TrafficEnv obs values match raw TraCI reads.
Also verifies that the model's phase actions actually reach SUMO signals.

These tests run real SUMO. They are slow (~30s each) and require SUMO_HOME.
Run with:  pytest tests/test_integration_traffic_env.py -v

What each test proves:
  test_num_incoming_graph_matches_obs_length
      graph["phase_features"][n][0] length == actual q_in slice length in obs.
      If this fails, reward.py reads the wrong indices entirely.

  test_q_in_obs_matches_traci_halting
      obs[q_in_start : q_in_start+n] == traci halting/Q_MAX per incoming lane.
      If this fails, obs is reading wrong lanes or wrong metric from SUMO.

  test_q_out_obs_matches_traci_halting
      obs[q_out_start:] == traci halting/Q_MAX per outgoing lane.
      Same — wrong lanes or wrong metric.

  test_phase_onehot_sums_to_one
      Exactly one phase is active per node.

  test_obs_values_in_range
      All queue values in [0, 1] after normalization.

  test_reward_nonpositive
      Mixed reward is always <= 0 on real obs.

  test_model_action_changes_sumo_signal
      Forces a phase switch by holding one phase long enough to satisfy min-green,
      then commanding a different phase. Verifies SUMO's traffic light state string
      changes — proving the model's action reaches the actual simulation.
"""

import math
import pytest
import torch

NET  = "networks/external/RESCO/grid4x4/grid4x4.net.xml"
ROU  = "networks/external/RESCO/grid4x4/grid4x4_dense.rou.xml"
Q_MAX = 30.0


@pytest.fixture(scope="module")
def env():
    from env.traffic_env import TrafficEnv
    e = TrafficEnv(net_file=NET, route_file=ROU, max_steps=200, begin_time=0)
    yield e
    e.close()


@pytest.fixture(scope="module")
def obs_after_steps(env):
    """Run 10 steps with all-zero actions; return (obs_dict, graph, env)."""
    obs_dict, graph = env.reset(seed=42)
    actions = {nid: 0 for nid in graph["node_ids"]}
    for _ in range(10):
        obs_dict, _, _, done, _ = env.step(actions)
        if done:
            obs_dict, graph = env.reset(seed=42)
    return obs_dict, graph


# ---------------------------------------------------------------------------
# 1. graph num_incoming == obs slice length
# ---------------------------------------------------------------------------

def test_num_incoming_graph_matches_obs_length(env, obs_after_steps):
    """Critical: reward.py derives num_incoming from graph['phase_features'].
    If that doesn't match the actual q_in slice in obs, every reward is wrong."""
    obs_dict, graph = obs_after_steps
    for node_id in graph["node_ids"]:
        node_idx     = graph["node_to_idx"][node_id]
        pf           = graph["phase_features"][node_idx]
        num_phases   = graph["node_meta"][node_idx]["num_phases"]
        num_incoming_graph = len(pf[0]) if pf else 0

        obs, _ = obs_dict[node_id]
        # From the obs layout: skip phase_onehot + t_norm, then num_incoming q_in values,
        # then num_incoming running_in values, then q_out to end.
        # We can infer num_incoming_obs from the actual incoming_lanes list.
        num_incoming_env = len(env._incoming_lanes.get(node_id, []))

        assert num_incoming_graph == num_incoming_env, (
            f"{node_id}: graph says num_incoming={num_incoming_graph} "
            f"but TrafficEnv._incoming_lanes has {num_incoming_env} lanes. "
            f"reward.py will read wrong indices."
        )


# ---------------------------------------------------------------------------
# 2. q_in in obs matches TraCI halting counts
# ---------------------------------------------------------------------------

def test_q_in_obs_matches_traci_halting(env, obs_after_steps):
    """obs[q_in_start + i] must equal traci.lane.getLastStepHaltingNumber(lane) / Q_MAX.
    This is the direct proof that SUMO data reaches the reward correctly."""
    obs_dict, graph = obs_after_steps
    conn = env._conn

    for node_id in graph["node_ids"]:
        node_idx   = graph["node_to_idx"][node_id]
        num_phases = graph["node_meta"][node_idx]["num_phases"]
        pf         = graph["phase_features"][node_idx]
        num_incoming = len(pf[0]) if pf else 0
        q_in_start = num_phases + 1

        obs, _ = obs_dict[node_id]
        inc_lanes = env._incoming_lanes.get(node_id, [])

        for i, lane_id in enumerate(inc_lanes):
            traci_halting = conn.lane.getLastStepHaltingNumber(lane_id)
            expected = min(traci_halting / Q_MAX, 1.0)
            actual   = obs[q_in_start + i].item()
            assert abs(actual - expected) < 1e-5, (
                f"{node_id} lane {lane_id} (idx {i}): "
                f"obs={actual:.4f} but traci={traci_halting}/{Q_MAX}={expected:.4f}. "
                f"Wrong lane or wrong metric being read."
            )


# ---------------------------------------------------------------------------
# 3. q_out in obs matches TraCI halting counts on outgoing lanes
# ---------------------------------------------------------------------------

def test_q_out_obs_matches_traci_halting(env, obs_after_steps):
    """obs[q_out_start + i] must equal traci halting / Q_MAX for outgoing lanes."""
    obs_dict, graph = obs_after_steps
    conn = env._conn

    for node_id in graph["node_ids"]:
        node_idx     = graph["node_to_idx"][node_id]
        num_phases   = graph["node_meta"][node_idx]["num_phases"]
        pf           = graph["phase_features"][node_idx]
        num_incoming = len(pf[0]) if pf else 0
        q_in_start   = num_phases + 1
        q_out_start  = q_in_start + 2 * num_incoming

        obs, _ = obs_dict[node_id]
        out_lanes = env._outgoing_lanes.get(node_id, [])

        assert obs.shape[0] - q_out_start == len(out_lanes), (
            f"{node_id}: obs has {obs.shape[0] - q_out_start} q_out values "
            f"but _outgoing_lanes has {len(out_lanes)} lanes."
        )

        for i, lane_id in enumerate(out_lanes):
            traci_halting = conn.lane.getLastStepHaltingNumber(lane_id)
            expected = min(traci_halting / Q_MAX, 1.0)
            actual   = obs[q_out_start + i].item()
            assert abs(actual - expected) < 1e-5, (
                f"{node_id} out-lane {lane_id} (idx {i}): "
                f"obs={actual:.4f} but traci={expected:.4f}"
            )


# ---------------------------------------------------------------------------
# 4. Phase onehot is exactly one-hot
# ---------------------------------------------------------------------------

def test_phase_onehot_sums_to_one(env, obs_after_steps):
    """Exactly one phase should be active per node at all times."""
    obs_dict, graph = obs_after_steps
    for node_id in graph["node_ids"]:
        node_idx   = graph["node_to_idx"][node_id]
        num_phases = graph["node_meta"][node_idx]["num_phases"]
        obs, _     = obs_dict[node_id]
        onehot_sum = obs[:num_phases].sum().item()
        assert abs(onehot_sum - 1.0) < 1e-5, (
            f"{node_id}: phase onehot sums to {onehot_sum}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# 5. All queue values in [0, 1]
# ---------------------------------------------------------------------------

def test_obs_queue_values_in_range(env, obs_after_steps):
    """All queue values must be in [0, 1] after normalization by Q_MAX=30."""
    obs_dict, graph = obs_after_steps
    for node_id in graph["node_ids"]:
        node_idx     = graph["node_to_idx"][node_id]
        num_phases   = graph["node_meta"][node_idx]["num_phases"]
        pf           = graph["phase_features"][node_idx]
        num_incoming = len(pf[0]) if pf else 0
        q_in_start   = num_phases + 1

        obs, _ = obs_dict[node_id]
        queue_slice = obs[q_in_start:]  # everything after phase+time
        assert (queue_slice >= 0.0).all() and (queue_slice <= 1.0).all(), (
            f"{node_id}: queue values out of [0,1] range: min={queue_slice.min():.3f} max={queue_slice.max():.3f}"
        )


# ---------------------------------------------------------------------------
# 6. Mixed reward is non-positive on real obs
# ---------------------------------------------------------------------------

def test_reward_nonpositive_on_real_obs(env, obs_after_steps):
    from training.reward import PressureReward
    obs_dict, graph = obs_after_steps
    pr = PressureReward(queue_weight=0.5, pressure_weight=0.5)
    reward = pr.compute(obs_dict, graph)
    for node_id, r in reward.items():
        assert r <= 1e-9, f"{node_id}: reward={r:.6f} is positive — formula broken"


# ---------------------------------------------------------------------------
# 7. Reward directly computed from TraCI matches reward computed from obs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 8. Model action actually changes the SUMO traffic light state
# ---------------------------------------------------------------------------

def test_model_action_changes_sumo_signal(env):
    """Prove the model's chosen phase reaches SUMO.

    Steps:
      1. Reset, hold phase 0 for enough steps to clear min-green guard.
      2. Read SUMO's current signal state string for one node.
      3. Command ALL nodes to a DIFFERENT phase (phase 1).
      4. Run steps until yellow clears.
      5. Assert the signal state string changed to phase 1's green string.
    """
    obs_dict, graph = env.reset(seed=0)
    node_id  = graph["node_ids"][0]
    conn     = env._conn
    gs       = env._green_states[node_id]   # list of green phase state strings

    assert len(gs) >= 2, f"{node_id} has only {len(gs)} phases — need at least 2"

    # Hold phase 0 for 4 steps (20 sim seconds >> min_green+yellow_time=7)
    actions_phase0 = {nid: 0 for nid in graph["node_ids"]}
    for _ in range(4):
        env.step(actions_phase0)

    state_before = conn.trafficlight.getRedYellowGreenState(node_id)

    # Command a switch to phase 1 for ALL nodes
    actions_phase1 = {nid: 1 for nid in graph["node_ids"]}

    # Run enough steps for yellow (2s) to complete — 2 steps (10 sim seconds)
    for _ in range(2):
        env.step(actions_phase1)

    state_after = conn.trafficlight.getRedYellowGreenState(node_id)

    assert state_after != state_before, (
        f"{node_id}: signal state did not change after commanding phase switch.\n"
        f"  before='{state_before}'\n"
        f"  after= '{state_after}'\n"
        f"  phase0 green='{gs[0]}'\n"
        f"  phase1 green='{gs[1]}'\n"
        "  The model's actions are NOT reaching SUMO."
    )
    # Final state should be phase 1's green string
    assert state_after == gs[1], (
        f"{node_id}: expected phase 1 state '{gs[1]}' but got '{state_after}'"
    )


def test_reward_from_obs_matches_direct_traci_computation(env):
    """The reward.py path and a direct TraCI computation must agree.
    This is the end-to-end proof that the reward function sees real SUMO data.
    Uses its own reset so obs_dict and TraCI read are from the same sim state."""
    from training.reward import PressureReward

    obs_dict, graph = env.reset(seed=99)
    actions = {nid: 0 for nid in graph["node_ids"]}
    # Run enough steps to get vehicles into the network
    for _ in range(15):
        obs_dict, _, _, done, _ = env.step(actions)
        if done:
            break

    conn = env._conn
    pr = PressureReward(queue_weight=0.5, pressure_weight=0.5)
    # Compute reward immediately — same SUMO step, no advance between obs and TraCI read
    reward_from_obs = pr.compute(obs_dict, graph)

    for node_id in graph["node_ids"]:
        node_idx     = graph["node_to_idx"][node_id]
        pf           = graph["phase_features"][node_idx]
        num_incoming = len(pf[0]) if pf else 0

        inc_lanes = env._incoming_lanes.get(node_id, [])
        out_lanes = env._outgoing_lanes.get(node_id, [])

        q_in_vals  = [min(conn.lane.getLastStepHaltingNumber(l) / Q_MAX, 1.0) for l in inc_lanes]
        q_out_vals = [min(conn.lane.getLastStepHaltingNumber(l) / Q_MAX, 1.0) for l in out_lanes]

        n = max(len(q_in_vals), 1)
        q_in_sum  = sum(q_in_vals)
        q_out_sum = sum(q_out_vals)
        expected  = -(0.5 * (q_in_sum / n) + 0.5 * abs(q_in_sum - q_out_sum) / n)

        actual = reward_from_obs[node_id]
        assert abs(actual - expected) < 1e-4, (
            f"{node_id}: reward from obs={actual:.5f}, direct TraCI={expected:.5f}. "
            f"q_in={q_in_vals}, q_out={q_out_vals}"
        )
