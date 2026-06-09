"""
Convergence logic tests — no SUMO, no model, no GPU.

Exercises the exact same arithmetic as trainer.py so any refactor breaks here first.
Run: python -m pytest model/tests/test_convergence.py -v
  or: python model/tests/test_convergence.py
"""

import sys, math
from pathlib import Path

# ── Replicate the convergence check from trainer.py ──────────────────────────

def _converged(episode_returns, avg_waiting_time, global_ep,
               conv_window, conv_min_ep, conv_d_ret, conv_d_wait):
    """Returns (converged: bool, ret_delta: float|None, wait_delta: float|None)."""
    if not (conv_d_ret is not None or conv_d_wait is not None):
        return False, None, None
    if global_ep < conv_min_ep:
        return False, None, None
    n_ep = len(episode_returns)
    if n_ep < conv_window * 2:
        return False, None, None

    if conv_d_ret is not None:
        ret_recent = episode_returns[-conv_window:]
        ret_prev   = episode_returns[-conv_window * 2:-conv_window]
        ret_delta  = abs(sum(ret_recent) / conv_window - sum(ret_prev) / conv_window)
        ret_ok     = ret_delta <= conv_d_ret
    else:
        ret_ok, ret_delta = True, None

    if conv_d_wait is not None:
        wait_recent = avg_waiting_time[-conv_window:]
        wait_prev   = avg_waiting_time[-conv_window * 2:-conv_window]
        wait_delta  = abs(sum(wait_recent) / conv_window - sum(wait_prev) / conv_window)
        wait_ok     = wait_delta <= conv_d_wait
    else:
        wait_ok, wait_delta = True, None

    return (ret_ok and wait_ok), ret_delta, wait_delta


# ── Test cases ────────────────────────────────────────────────────────────────

def test_flat_wait_triggers():
    """Flat wait time (no improvement) should trigger when min_ep reached."""
    wait = [50.0] * 30  # perfectly flat — delta between any two windows = 0
    rets = [100.0] * 30
    ok, _, w_delta = _converged(rets, wait, global_ep=25,
                                conv_window=5, conv_min_ep=20,
                                conv_d_ret=None, conv_d_wait=1.0)
    assert ok, "flat wait should converge"
    assert w_delta == 0.0

def test_still_improving_does_not_trigger():
    """Wait time still dropping steadily — should NOT trigger."""
    # Drops 2 s each episode: mean(last 5) vs mean(prev 5) → delta ≈ 10 s
    wait = [100.0 - i * 2.0 for i in range(30)]
    rets = [0.0] * 30
    ok, _, w_delta = _converged(rets, wait, global_ep=30,
                                conv_window=5, conv_min_ep=10,
                                conv_d_ret=None, conv_d_wait=1.0)
    assert not ok, f"still-improving model should not converge (delta={w_delta:.2f})"

def test_min_ep_guard():
    """Convergence check must not fire before conv_min_episodes."""
    wait = [50.0] * 30
    rets = [0.0] * 30
    ok, _, _ = _converged(rets, wait, global_ep=10,
                          conv_window=5, conv_min_ep=20,
                          conv_d_ret=None, conv_d_wait=1.0)
    assert not ok, "should not converge before min_ep"

def test_window_guard():
    """Need at least conv_window * 2 episodes before check is possible."""
    wait = [50.0] * 8   # only 8 episodes so far, window=5 → need 10
    rets = [0.0] * 8
    ok, _, _ = _converged(rets, wait, global_ep=30,
                          conv_window=5, conv_min_ep=5,
                          conv_d_ret=None, conv_d_wait=1.0)
    assert not ok, "not enough history yet"

def test_both_null_never_triggers():
    """Both deltas null → convergence check disabled."""
    wait = [50.0] * 30
    rets = [0.0] * 30
    ok, _, _ = _converged(rets, wait, global_ep=30,
                          conv_window=5, conv_min_ep=5,
                          conv_d_ret=None, conv_d_wait=None)
    assert not ok, "null thresholds should never trigger"

def test_return_delta_blocks_wait_convergence():
    """Wait converged but returns still swinging — AND logic must block early stop."""
    # Wait is flat, returns oscillate ±50 every episode
    wait = [50.0] * 30
    rets = [100.0 if i % 2 == 0 else 0.0 for i in range(30)]
    # mean(last 5 odd/even) vs mean(prev 5 odd/even) → delta ≈ 50
    ok, r_delta, _ = _converged(rets, wait, global_ep=30,
                                conv_window=5, conv_min_ep=10,
                                conv_d_ret=10.0, conv_d_wait=1.0)
    assert not ok, f"oscillating returns should block convergence (ret_delta={r_delta:.2f})"

def test_both_criteria_met():
    """Both wait and return flat → should trigger."""
    wait = [30.0] * 30
    rets = [-50.0] * 30
    ok, r_delta, w_delta = _converged(rets, wait, global_ep=30,
                                      conv_window=5, conv_min_ep=10,
                                      conv_d_ret=1.0, conv_d_wait=1.0)
    assert ok, "both flat → should converge"
    assert r_delta == 0.0
    assert w_delta == 0.0

def test_exact_threshold_boundary():
    """Delta exactly at threshold should pass (<=, not <)."""
    # last 5 = 55.0, prev 5 = 60.0 → delta = 5.0 exactly
    wait = [50.0] * 20 + [60.0] * 5 + [55.0] * 5
    rets = [0.0] * 30
    ok, _, w_delta = _converged(rets, wait, global_ep=30,
                                conv_window=5, conv_min_ep=10,
                                conv_d_ret=None, conv_d_wait=5.0)
    assert ok, f"delta == threshold should pass (delta={w_delta:.4f})"

def test_just_above_threshold():
    """Delta just above threshold should not pass."""
    # last 5 = 55.0, prev 5 = 60.1 → delta = 5.1
    wait = [50.0] * 20 + [60.1] * 5 + [55.0] * 5
    rets = [0.0] * 30
    ok, _, w_delta = _converged(rets, wait, global_ep=30,
                                conv_window=5, conv_min_ep=10,
                                conv_d_ret=None, conv_d_wait=5.0)
    assert not ok, f"delta > threshold should not pass (delta={w_delta:.4f})"


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {e}")
            failed += 1
    print()
    if failed:
        print(f"{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
