from __future__ import annotations

import argparse
import json
from pathlib import Path

from .job import run_web_job
from .schemas import WebJobConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Website-ready Independent DQN v2 pipeline for uploaded OSM maps."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--osm-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("benchmark/web_jobs"))
    parser.add_argument("--network-name", default="uploaded_map")
    parser.add_argument("--begin-s", type=int, default=0)
    parser.add_argument("--end-s", type=int, default=3600)
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--train-max-steps-per-episode", type=int, default=720)
    parser.add_argument("--train-seed", type=int, default=17)
    parser.add_argument("--smoke-episodes", type=int, default=2)
    parser.add_argument("--smoke-max-steps", type=int, default=120)
    parser.add_argument("--smoke-seed", type=int, default=11)
    parser.add_argument("--wall-clock-cap-minutes", type=float, default=240.0)
    parser.add_argument("--low-period-s", type=float, default=4.0)
    parser.add_argument("--medium-period-s", type=float, default=2.5)
    parser.add_argument("--high-period-s", type=float, default=1.5)
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Held-out route generation seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = WebJobConfig(
        job_id=str(args.job_id),
        osm_path=Path(args.osm_path),
        output_root=Path(args.output_root),
        network_name=str(args.network_name),
        begin_s=int(args.begin_s),
        end_s=int(args.end_s),
        train_episodes=int(args.train_episodes),
        train_max_steps_per_episode=int(args.train_max_steps_per_episode),
        train_seed=int(args.train_seed),
        smoke_episodes=int(args.smoke_episodes),
        smoke_max_steps=int(args.smoke_max_steps),
        smoke_seed=int(args.smoke_seed),
        wall_clock_cap_minutes=float(args.wall_clock_cap_minutes),
        low_period_s=float(args.low_period_s),
        medium_period_s=float(args.medium_period_s),
        high_period_s=float(args.high_period_s),
        eval_seeds=[int(x) for x in args.eval_seeds],
    )
    result = run_web_job(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
