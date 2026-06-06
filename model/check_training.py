"""Quick training progress check — run any time to see current status.

Usage:
    python check_training.py                                    # defaults: r8 multi-network
    python check_training.py --log training_sync.log --ckpt checkpoints/grid4x4_sync
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from collections import Counter


def _tail(path, n=25):
    try:
        raw = Path(path).read_bytes()
        # Detect UTF-16 BOM written by PowerShell 5 Tee-Object
        if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
            text = raw.decode("utf-16", errors="replace")
        else:
            text = raw.decode("utf-8", errors="replace")
        return text.splitlines()[-n:]
    except FileNotFoundError:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",  default="training_r8.log",
                        help="Training log file to tail (default: training_r8.log)")
    parser.add_argument("--ckpt", default="checkpoints/r8_multi",
                        help="Checkpoint directory (default: checkpoints/r8_multi)")
    args = parser.parse_args()

    log      = Path(args.log)
    err_log  = Path(str(log).replace(".log", "_err.log"))
    ckpt_dir = Path(args.ckpt)

    # --- Log freshness (proxy for process alive) ---
    print("=== Log freshness ===")
    import time as _time
    if log.exists():
        age = _time.time() - log.stat().st_mtime
        status = "ACTIVE (updated <30s ago)" if age < 30 else f"last updated {int(age)}s ago"
        print(f"  training_r8.log : {status}")
    else:
        print("  training_r8.log : not found")
    try:
        out = subprocess.check_output(
            ["powershell", "-Command",
             "Get-Process python -ErrorAction SilentlyContinue | "
             "Select-Object Id, @{N='CPU_s';E={[int]$_.CPU}}, "
             "@{N='MB';E={[int]($_.WorkingSet/1MB)}}, StartTime | "
             "Format-Table -AutoSize"],
            text=True, stderr=subprocess.DEVNULL
        )
        procs = [l for l in out.splitlines() if l.strip() and not l.startswith((" ", "-", "I", "C"))]
        if procs:
            print("\n  Python processes:")
            for l in out.splitlines():
                print("  " + l)
    except Exception:
        pass

    # --- Last N log lines ---
    print("\n=== Last 25 log lines ===")
    for line in _tail(log, 25):
        print(line)

    # --- Errors? ---
    err_lines = _tail(err_log, 5)
    if any("Error" in l or "Traceback" in l for l in err_lines):
        print("\n=== Recent errors ===")
        for line in err_lines:
            print(line)

    # --- Checkpoint metrics ---
    metrics_path = ckpt_dir / "training_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        ep  = m.get("episode_returns", [])
        lss = m.get("losses", [])
        seq = m.get("network_sequence", [])
        print(f"\n=== Metrics (from {metrics_path}) ===")
        print(f"  Episodes completed : {len(ep)}")
        print(f"  Gradient steps     : {len(lss)}")
        if ep:
            last = ep[-50:]
            print(f"  Last-50 mean return: {sum(last)/len(last):+.3f}")
        if lss:
            last_l = lss[-200:]
            print(f"  Last-200 mean loss : {sum(last_l)/len(last_l):.5f}")
        if seq:
            print(f"  Network counts     : {dict(Counter(seq))}")
    else:
        print("\n(no training_metrics.json yet — checkpoint not saved)")

    # --- Checkpoint files ---
    if ckpt_dir.exists():
        pts = sorted(ckpt_dir.glob("*.pt"))
        print(f"\n=== Checkpoints ({len(pts)} files) ===")
        for p in pts[-5:]:
            print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
