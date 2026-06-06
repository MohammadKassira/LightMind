"""Watch one episode in SUMO-GUI with the trained policy (or random if no checkpoint).

Usage:
    # Random policy — just see the traffic
    python watch.py --config configs/grid4x4_dense.yaml

    # Trained policy
    python watch.py --config configs/grid4x4_dense.yaml --checkpoint checkpoints/grid4x4_dense/checkpoint_ep50.pt
"""

import argparse
from pathlib import Path

import yaml
import torch

from env.traffic_env import TrafficEnv
from env.perception import apply_perception
from models.node_encoder import pad_obs_dict
from models.phase_head import pad_phase_features
from training.sync_trainer import SyncParallelTrainer
from training.reward import ObservationImputer, PressureReward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epsilon",    type=float, default=0.0,
                        help="Action randomness (0=greedy, 1=random)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    net_cfg = cfg["network"]
    device  = torch.device("cpu")

    env = TrafficEnv(
        net_file   = net_cfg["net"],
        route_file = net_cfg["rou"],
        begin_time = net_cfg.get("begin_time", 0),
        max_steps  = net_cfg.get("max_steps", cfg.get("env", {}).get("max_steps", 200)),
        use_gui    = True,
    )

    trainer = SyncParallelTrainer(cfg, [env], device=device,
                                  network_name=net_cfg.get("name", "watch"))

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        trainer.encoder.load_state_dict(ckpt["encoder"])
        trainer.gat.load_state_dict(ckpt["gat"])
        trainer.head.load_state_dict(ckpt["head"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint — running random policy (epsilon=1.0)")
        args.epsilon = 1.0

    severity  = cfg.get("perception", {}).get("severity", 0.0)
    sentinel  = cfg.get("perception", {}).get("sentinel_value", -1.0)
    pressure  = PressureReward()
    imputer   = ObservationImputer()
    neighbor_masking = cfg.get("model", {}).get("gat", {}).get("neighbor_masking", True)

    obs_dict, graph = env.reset()
    obs_dict = apply_perception(obs_dict, severity, sentinel)
    imputer.reset(); pressure.reset()
    obs_dict = imputer.impute(obs_dict)
    _, padded_obs = pad_obs_dict(obs_dict)
    _, padded_pf  = pad_phase_features(graph)

    total_return = 0.0
    step = 0
    done = False

    print(f"\nRunning episode (epsilon={args.epsilon:.2f}) — close SUMO-GUI window to stop.\n")

    while not done:
        actions = SyncParallelTrainer._select_actions_local(
            padded_obs, graph, padded_pf, args.epsilon,
            trainer.encoder, trainer.gat, trainer.head,
            device, neighbor_masking,
        )
        next_obs_dict, _, reward_dict, done, info = env.step(actions)
        next_obs_dict = apply_perception(next_obs_dict, severity, sentinel)
        reward_dict   = pressure.compute(next_obs_dict, graph)
        total_return += sum(reward_dict.values())

        next_obs_dict = imputer.impute(next_obs_dict)
        _, padded_obs = pad_obs_dict(next_obs_dict)
        step += 1

        wait = info.get("step_mean_waiting_time", float("nan"))
        tput = info.get("step_throughput", 0)
        print(f"\rstep={step:3d}  wait={wait:.1f}s  tput={tput:3d}  return={total_return:+.1f}", end="", flush=True)

    print(f"\n\nEpisode done — {step} steps, total return={total_return:+.2f}")
    env.close()


if __name__ == "__main__":
    main()
