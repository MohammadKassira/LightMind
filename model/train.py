"""Unified DQN training script for all rings.

Usage:
    python train.py --config configs/r6.yaml                      # R6 typed edges
    python train.py --config configs/r6.yaml --untyped            # override typed_edges=False
    python train.py --config configs/r4.yaml --zero-hop           # override zero_hop=True
    python train.py --config configs/r5.yaml --one-layer          # override num_layers=1
    python train.py --config configs/r6.yaml --episodes 10        # smoke run
    python train.py --config configs/r6.yaml --eval-episodes 5

Config is loaded from the YAML file; CLI flags override specific keys before the trainer
is constructed.  No ring-specific logic lives here — all ring behaviour comes from the
config file.
"""

import argparse
import json
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Unified DQN training")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (e.g. configs/r6.yaml)")
    parser.add_argument("--net",
                        default="networks/external/RESCO/cologne3/cologne3.net.xml",
                        help="SUMO .net.xml path")
    parser.add_argument("--rou",
                        default="networks/external/RESCO/cologne3/cologne3.rou.xml",
                        help="SUMO .rou.xml path")
    parser.add_argument("--begin-time", type=int, default=25200,
                        help="SUMO simulation start time in seconds (cologne3: 25200 = 7 AM)")
    # Config key overrides
    parser.add_argument("--untyped", action="store_true",
                        help="Override model.gat.typed_edges=False (untyped comparison run)")
    parser.add_argument("--zero-hop", action="store_true",
                        help="Override model.gat.zero_hop=True (0-hop ablation)")
    parser.add_argument("--one-layer", action="store_true",
                        help="Override model.gat.num_layers=1 (1-hop comparison)")
    parser.add_argument("--no-masking", action="store_true",
                        help="Override model.gat.neighbor_masking=False")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt checkpoint to resume or fine-tune from")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override trainer.num_episodes from config")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Greedy eval episodes after training (0 to skip)")
    parser.add_argument("--log-file", default=None,
                        help="Write all output to this file (UTF-8) in addition to stdout")
    parser.add_argument("--metrics-file", default=None,
                        help="Path to .jsonl file for per-episode KPI streaming (one JSON line per episode)")
    parser.add_argument("--epsilon-start", type=float, default=None,
                        help="Override trainer.epsilon_start and reset grad-step counter to 0 (for re-explore retrain)")
    args = parser.parse_args()

    # Python-level tee: write to both stdout and log file so no pipe is needed
    if args.log_file:
        import sys, io
        _log_fh = open(args.log_file, "w", encoding="utf-8", buffering=1)
        class _Tee:
            def __init__(self, primary, secondary):
                self._p = primary; self._s = secondary
            def write(self, data):
                self._p.write(data); self._p.flush()
                self._s.write(data); self._s.flush()
            def flush(self):
                self._p.flush(); self._s.flush()
            def __getattr__(self, name):
                return getattr(self._p, name)
        sys.stdout = _Tee(sys.__stdout__, _log_fh)
        sys.stderr = _Tee(sys.__stderr__, _log_fh)

    import torch
    from env.traffic_env import TrafficEnv
    from training.trainer import DQNTrainer
    from evaluation.eval_runner import evaluate, print_summary

    cfg = yaml.safe_load(Path(args.config).read_text())

    # Apply CLI overrides to config before constructing any objects
    cfg.setdefault("model", {}).setdefault("gat", {})
    if args.untyped:
        cfg["model"]["gat"]["typed_edges"] = False
    if args.zero_hop:
        cfg["model"]["gat"]["zero_hop"] = True
    if args.one_layer:
        cfg["model"]["gat"]["num_layers"] = 1
    if args.no_masking:
        cfg["model"]["gat"]["neighbor_masking"] = False
    if args.episodes is not None:
        cfg.setdefault("trainer", {})["num_episodes"] = args.episodes
    if args.epsilon_start is not None:
        cfg.setdefault("trainer", {})["epsilon_start"] = args.epsilon_start
    cfg.setdefault("reward", {})["use_pressure"] = True

    # --- Multi-network path (cfg["networks"] list present) ---
    network_list = cfg.get("networks", None)
    if network_list:
        from training.multi_network import MultiNetworkTrainer
        from env.traffic_env import TrafficEnv as _TE
        envs = [
            _TE(
                net_file         = n["net"],
                route_file       = n["rou"],
                begin_time       = n.get("begin_time", 0),
                max_steps        = n.get("max_steps", cfg.get("env", {}).get("max_steps", 200)),
                additional_files = n.get("additional"),
            )
            for n in network_list
        ]
        names = [
            n.get("name", n["net"].split("/")[-1].replace(".net.xml", ""))
            for n in network_list
        ]
        device  = torch.device(cfg.get("device", "cpu"))
        trainer = MultiNetworkTrainer(cfg, envs, device=device, network_names=names)
        if args.checkpoint:
            MultiNetworkTrainer.load_checkpoint(args.checkpoint, cfg, envs, device, network_names=names)

        num_episodes = cfg.get("trainer", {}).get("num_episodes", 300)
        ckpt_dir     = cfg.get("trainer", {}).get("checkpoint_dir", "checkpoints/r8_multi")
        print(f"\nTraining — multi-network shared-parameter GAT")
        print(f"  Config:    {args.config}")
        print(f"  Networks:  {', '.join(names)}")
        print(f"  Episodes:  {num_episodes}")
        print(f"  Output:    {ckpt_dir}/\n")

        start = time.time()
        try:
            metrics = trainer.train()
        finally:
            for env in envs:
                env.close()

        elapsed = int(time.time() - start)
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        print(f"\nTraining done in {h}h{m:02d}m{s:02d}s")

        ckpt_path = Path(ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        (ckpt_path / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
        trainer.save_checkpoint(str(ckpt_path / "final.pt"), step=0, metrics=metrics)

        returns = metrics.get("episode_returns", [])
        if returns:
            last100 = returns[-100:]
            mean_r  = sum(last100) / len(last100)
            losses  = metrics.get("losses", [])
            mean_l  = sum(losses[-500:]) / min(len(losses), 500) if losses else float("nan")
            print(f"  Last-100 mean return: {mean_r:+.3f}")
            print(f"  Last-500 mean loss:   {mean_l:.5f}")

        print(f"\nDone. Checkpoints in {ckpt_dir}/")
        return

    # --- Single-network path from config (network: key, num_workers=1 → DQNTrainer) ---
    network_cfg = cfg.get("network", None)
    if network_cfg and int(cfg.get("trainer", {}).get("num_workers", 1)) == 1:
        net_entry    = network_cfg
        env_kwargs   = dict(
            net_file   = net_entry["net"],
            route_file = net_entry["rou"],
            begin_time = net_entry.get("begin_time", 0),
            max_steps  = net_entry.get("max_steps", cfg.get("env", {}).get("max_steps", 200)),
        )
        if net_entry.get("additional"):
            env_kwargs["additional_files"] = net_entry["additional"]
        if net_entry.get("route_files"):
            env_kwargs["route_files"] = net_entry["route_files"]

        net_name     = net_entry.get("name", net_entry["net"].split("/")[-1].replace(".net.xml", ""))
        ckpt_dir     = cfg.get("trainer", {}).get("checkpoint_dir", "checkpoints/run")
        num_episodes = args.episodes if args.episodes is not None else cfg.get("trainer", {}).get("num_episodes", 500)
        num_layers   = cfg["model"]["gat"].get("num_layers", 1)
        typed_edges  = cfg["model"]["gat"].get("typed_edges", False)
        zero_hop     = cfg["model"]["gat"].get("zero_hop", False)
        label = (
            f"{num_layers}-layer GAT"
            + (" (typed)" if typed_edges else "")
            + (" (zero-hop)" if zero_hop else "")
        )

        print(f"\nTraining — {label}")
        print(f"  Config:    {args.config}")
        print(f"  Network:   {net_name}")
        print(f"  Episodes:  {num_episodes}")
        print(f"  Output:    {ckpt_dir}/\n")

        env     = TrafficEnv(**env_kwargs)
        device  = torch.device(cfg.get("device", "cpu"))
        trainer = DQNTrainer(cfg, env, device=device)
        if args.checkpoint:
            trainer = DQNTrainer.load_checkpoint(args.checkpoint, cfg, env, device)
            if args.epsilon_start is not None:
                trainer.start_grad_steps = 0

        start   = time.time()
        metrics = trainer.train(num_episodes=num_episodes, metrics_file=args.metrics_file)

        elapsed = int(time.time() - start)
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        print(f"\nTraining done in {h}h{m:02d}m{s:02d}s")

        ckpt_path = Path(ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        (ckpt_path / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
        trainer.save_checkpoint(str(ckpt_path / "final.pt"), step=0, metrics=metrics)

        returns = metrics.get("episode_returns", [])
        if returns:
            last100 = returns[-100:]
            mean_r  = sum(last100) / len(last100)
            losses  = metrics.get("losses", [])
            mean_l  = sum(losses[-500:]) / min(len(losses), 500) if losses else float("nan")
            waits   = metrics.get("avg_waiting_time", [])
            mean_w  = sum(waits[-100:]) / min(len(waits), 100) if waits else float("nan")
            print(f"  Last-100 mean return:       {mean_r:+.3f}")
            print(f"  Last-500 mean loss:         {mean_l:.5f}")
            print(f"  Last-100 mean waiting time: {mean_w:.1f}s")

        if args.eval_episodes > 0:
            print(f"\nRunning {args.eval_episodes} greedy eval episodes...")
            eval_metrics = evaluate(
                trainer, env,
                num_episodes        = args.eval_episodes,
                use_pressure        = True,
                perception_severity = cfg.get("perception", {}).get("severity", 0.0),
                sentinel            = cfg.get("perception", {}).get("sentinel_value", -1.0),
                seed                = 999,
            )
            print_summary(label, eval_metrics)
            eval_out = {k: v for k, v in eval_metrics.items() if k != "episode_records"}
            (ckpt_path / "eval_metrics.json").write_text(json.dumps(eval_out, indent=2))

        env.close()
        print(f"\nDone. Checkpoints in {ckpt_dir}/")
        return

    # --- Sync parallel path (cfg["network"] singular + num_workers > 1) ---
    if network_cfg:
        from training.sync_trainer import SyncParallelTrainer
        num_workers = int(cfg.get("trainer", {}).get("num_workers", 1))
        net_entry   = network_cfg
        env_kwargs  = dict(
            net_file   = net_entry["net"],
            route_file = net_entry["rou"],
            begin_time = net_entry.get("begin_time", 0),
            max_steps  = net_entry.get("max_steps", cfg.get("env", {}).get("max_steps", 200)),
        )
        if net_entry.get("additional"):
            env_kwargs["additional_files"] = net_entry["additional"]
        if net_entry.get("route_files"):
            env_kwargs["route_files"] = net_entry["route_files"]

        envs = [TrafficEnv(**env_kwargs) for _ in range(num_workers)]
        net_name  = net_entry.get("name", net_entry["net"].split("/")[-1].replace(".net.xml", ""))
        ckpt_dir  = cfg.get("trainer", {}).get("checkpoint_dir", "checkpoints/sync")
        num_episodes = args.episodes if args.episodes is not None else cfg.get("trainer", {}).get("num_episodes", 500)

        print(f"\nTraining — synchronous parallel GAT ({num_workers} workers)")
        print(f"  Config:    {args.config}")
        print(f"  Network:   {net_name}")
        print(f"  Episodes:  {num_episodes}")
        print(f"  Output:    {ckpt_dir}/\n")

        device  = torch.device(cfg.get("device", "cpu"))
        trainer = SyncParallelTrainer(cfg, envs, device=device, network_name=net_name)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device)
            trainer.encoder.load_state_dict(ckpt["encoder"])
            trainer.gat.load_state_dict(ckpt["gat"])
            trainer.head.load_state_dict(ckpt["head"])
            trainer.optimizer.load_state_dict(ckpt["optimizer"])

        start = time.time()
        try:
            metrics = trainer.train()
        finally:
            for env in envs:
                env.close()

        elapsed = int(time.time() - start)
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        print(f"\nTraining done in {h}h{m:02d}m{s:02d}s")

        ckpt_path = Path(ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        (ckpt_path / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
        trainer.save_checkpoint(str(ckpt_path / "final.pt"), step=0, metrics=metrics)

        returns = metrics.get("episode_returns", [])
        if returns:
            last100 = returns[-100:]
            mean_r  = sum(last100) / len(last100)
            losses  = metrics.get("losses", [])
            mean_l  = sum(losses[-500:]) / min(len(losses), 500) if losses else float("nan")
            waits   = metrics.get("avg_waiting_time", [])
            mean_w  = sum(waits[-100:]) / min(len(waits), 100) if waits else float("nan")
            print(f"  Last-100 mean return:       {mean_r:+.3f}")
            print(f"  Last-500 mean loss:         {mean_l:.5f}")
            print(f"  Last-100 mean waiting time: {mean_w:.1f}s")

        print(f"\nDone. Checkpoints in {ckpt_dir}/")
        return

    # --- Single-network path (unchanged DQNTrainer) ---
    ckpt_dir     = cfg.get("trainer", {}).get("checkpoint_dir", "checkpoints/run")
    num_episodes = cfg.get("trainer", {}).get("num_episodes", 300)
    num_layers   = cfg["model"]["gat"].get("num_layers", 1)
    typed_edges  = cfg["model"]["gat"].get("typed_edges", False)
    zero_hop     = cfg["model"]["gat"].get("zero_hop", False)
    label = (
        f"{num_layers}-layer GAT"
        + (" (typed)" if typed_edges else "")
        + (" (zero-hop)" if zero_hop else "")
    )

    print(f"\nTraining — {label}")
    print(f"  Config:    {args.config}")
    print(f"  Network:   {args.net}")
    print(f"  Episodes:  {num_episodes}")
    print(f"  Output:    {ckpt_dir}/\n")

    env = TrafficEnv(
        net_file   = args.net,
        route_file = args.rou,
        begin_time = args.begin_time,
        max_steps  = cfg.get("env", {}).get("max_steps", 200),
    )

    device  = torch.device(cfg.get("device", "cpu"))
    trainer = DQNTrainer(cfg, env, device=device)
    if args.checkpoint:
        trainer = DQNTrainer.load_checkpoint(args.checkpoint, cfg, env, device)

    start = time.time()
    metrics = trainer.train()

    elapsed = int(time.time() - start)
    h, rem  = divmod(elapsed, 3600)
    m, s    = divmod(rem, 60)
    print(f"\nTraining done in {h}h{m:02d}m{s:02d}s")

    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    (ckpt_path / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    trainer.save_checkpoint(str(ckpt_path / "final.pt"), step=0, metrics=metrics)

    returns = metrics["episode_returns"]
    if returns:
        last100 = returns[-100:]
        mean_r  = sum(last100) / len(last100)
        losses  = metrics["losses"]
        mean_l  = sum(losses[-500:]) / min(len(losses), 500) if losses else float("nan")
        print(f"  Last-100 mean return: {mean_r:+.3f}")
        print(f"  Last-500 mean loss:   {mean_l:.5f}")

    if args.eval_episodes > 0:
        print(f"\nRunning {args.eval_episodes} greedy eval episodes...")
        eval_metrics = evaluate(
            trainer,
            env,
            num_episodes        = args.eval_episodes,
            use_pressure        = True,
            perception_severity = cfg.get("perception", {}).get("severity", 0.0),
            sentinel            = cfg.get("perception", {}).get("sentinel_value", -1.0),
            seed                = 999,
        )
        print_summary(label, eval_metrics)

        eval_out = {k: v for k, v in eval_metrics.items() if k != "episode_records"}
        (ckpt_path / "eval_metrics.json").write_text(json.dumps(eval_out, indent=2))
        print(f"  Eval metrics → {ckpt_path / 'eval_metrics.json'}")

    env.close()
    print(f"\nDone. Checkpoints in {ckpt_dir}/")


if __name__ == "__main__":
    main()
