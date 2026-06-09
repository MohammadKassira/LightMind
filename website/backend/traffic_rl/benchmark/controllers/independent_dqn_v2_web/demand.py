from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .exceptions import WebIntegrationError
from .schemas import ScenarioSpec
from .utils import ensure_dir, now_iso, resolve_random_trips_path, run_command


LEVEL_PERIOD_KEY = {
    "low": "low_period_s",
    "medium": "medium_period_s",
    "high": "high_period_s",
}


def _write_sumocfg(
    *,
    path: Path,
    net_file: Path,
    route_file: Path,
    begin_s: int,
    end_s: int,
) -> None:
    root = ET.Element("configuration")
    input_elem = ET.SubElement(root, "input")
    ET.SubElement(input_elem, "net-file", value=str(net_file.resolve()))
    ET.SubElement(input_elem, "route-files", value=str(route_file.resolve()))

    time_elem = ET.SubElement(root, "time")
    ET.SubElement(time_elem, "begin", value=str(int(begin_s)))
    ET.SubElement(time_elem, "end", value=str(int(end_s)))

    report_elem = ET.SubElement(root, "report")
    ET.SubElement(report_elem, "no-step-log", value="true")
    ET.SubElement(report_elem, "duration-log.statistics", value="true")

    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def generate_random_route_scenario(
    *,
    scenario_id: str,
    level: str,
    seed: int,
    period_s: float,
    net_file: Path,
    begin_s: int,
    end_s: int,
    output_dir: Path,
) -> ScenarioSpec:
    stage = "random_route_generation"
    ensure_dir(output_dir)

    trips_file = output_dir / f"{scenario_id}.trips.xml"
    route_file = output_dir / f"{scenario_id}.rou.xml"
    sumocfg_file = output_dir / f"{scenario_id}.sumocfg"
    stdout_path = output_dir / "randomtrips_stdout.txt"
    stderr_path = output_dir / "randomtrips_stderr.txt"

    random_trips_path = resolve_random_trips_path()
    cmd = [
        "python3",
        str(random_trips_path),
        "--net-file",
        str(net_file),
        "--output-trip-file",
        str(trips_file),
        "--route-file",
        str(route_file),
        "--seed",
        str(int(seed)),
        "--begin",
        str(int(begin_s)),
        "--end",
        str(int(end_s)),
        "--period",
        str(float(period_s)),
        "--fringe-factor",
        "5",
        "--validate",
        "--remove-loops",
    ]
    run_command(cmd, stage=stage, stdout_path=stdout_path, stderr_path=stderr_path)

    if not route_file.exists():
        raise WebIntegrationError(stage, f"Route generation failed for scenario={scenario_id}: missing {route_file}")

    _write_sumocfg(
        path=sumocfg_file,
        net_file=net_file,
        route_file=route_file,
        begin_s=begin_s,
        end_s=end_s,
    )

    return ScenarioSpec(
        scenario_id=scenario_id,
        level=level,
        seed=int(seed),
        begin_s=int(begin_s),
        end_s=int(end_s),
        period_s=float(period_s),
        net_file=net_file,
        route_file=route_file,
        trips_file=trips_file,
        sumocfg_file=sumocfg_file,
        additional_files=[],
    )


def generate_train_and_eval_scenarios(
    *,
    net_file: Path,
    begin_s: int,
    end_s: int,
    low_period_s: float,
    medium_period_s: float,
    high_period_s: float,
    eval_seeds: list[int],
    scenario_root: Path,
    train_seed: int,
) -> dict[str, Any]:
    ensure_dir(scenario_root)

    level_period = {
        "low": float(low_period_s),
        "medium": float(medium_period_s),
        "high": float(high_period_s),
    }

    train_scenario_id = "uploaded_map__medium__seed_001"
    train_scenario = generate_random_route_scenario(
        scenario_id=train_scenario_id,
        level="medium",
        seed=int(train_seed),
        period_s=level_period["medium"],
        net_file=net_file,
        begin_s=begin_s,
        end_s=end_s,
        output_dir=scenario_root / "medium" / "seed_001",
    )

    heldout_eval_seeds = [int(s) for s in eval_seeds if int(s) != int(train_seed)]
    if not heldout_eval_seeds:
        raise WebIntegrationError(
            "random_route_generation",
            "Held-out evaluation seeds are empty after excluding the training seed.",
        )

    eval_scenarios: list[ScenarioSpec] = []
    for level in ["low", "medium", "high"]:
        for idx, seed in enumerate(heldout_eval_seeds, start=1):
            scenario_id = f"uploaded_map__{level}__seed_{int(seed):03d}"
            eval_spec = generate_random_route_scenario(
                scenario_id=scenario_id,
                level=level,
                seed=int(seed),
                period_s=level_period[level],
                net_file=net_file,
                begin_s=begin_s,
                end_s=end_s,
                output_dir=scenario_root / level / f"seed_{int(seed):03d}",
            )
            eval_scenarios.append(eval_spec)

    return {
        "created_at_utc": now_iso(),
        "train": train_scenario,
        "eval": eval_scenarios,
        "level_period_s": level_period,
    }
