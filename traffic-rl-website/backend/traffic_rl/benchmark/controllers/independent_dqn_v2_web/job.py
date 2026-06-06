from __future__ import annotations

from pathlib import Path
from typing import Any

from .demand import generate_train_and_eval_scenarios
from .evaluation import evaluate_on_heldout_scenarios
from .exceptions import WebIntegrationError
from .kpi import extract_kpis_from_evaluation_outputs
from .osm_pipeline import (
    convert_osm_to_sumo_net,
    detect_traffic_lights,
    validate_osm_input,
    validate_sample_sumo_boot,
    write_conversion_report,
)
from .packaging import package_job_outputs
from .scenario_manifest import create_generated_scenario_manifest
from .schemas import WebJobConfig
from .training import train_independent_dqn_from_scratch
from .utils import ensure_dir, json_dump, now_iso


STAGES = [
    "osm_validation",
    "osm_to_sumo_conversion",
    "traffic_light_detection",
    "random_route_generation",
    "scenario_manifest_creation",
    "independent_dqn_training",
    "evaluation",
    "kpi_extraction",
    "output_packaging",
]


def _status_payload(*, config: WebJobConfig, stage: str, status: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "updated_at_utc": now_iso(),
        "job_id": config.job_id,
        "stage": stage,
        "status": status,
        "details": details or {},
    }


def run_web_job(config: WebJobConfig) -> dict[str, Any]:
    job_dir = ensure_dir(config.output_root / config.job_id)
    reports_dir = ensure_dir(job_dir / "reports")
    status_path = reports_dir / "job_status.json"

    def write_status(stage: str, status: str, details: dict[str, Any] | None = None) -> None:
        json_dump(status_path, _status_payload(config=config, stage=stage, status=status, details=details))

    results: dict[str, Any] = {
        "created_at_utc": now_iso(),
        "job_id": config.job_id,
        "config": config.to_json(),
        "stages": {},
    }
    current_stage = "initialization"

    write_status("initialization", "running")

    try:
        # 1) OSM validation
        stage = "osm_validation"
        current_stage = stage
        write_status(stage, "running")
        osm_validation = validate_osm_input(config.osm_path)
        results["stages"][stage] = osm_validation
        write_status(stage, "passed", osm_validation)

        # 2) OSM -> SUMO net conversion
        stage = "osm_to_sumo_conversion"
        current_stage = stage
        write_status(stage, "running")
        conversion_dir = ensure_dir(job_dir / "converted")
        conversion = convert_osm_to_sumo_net(config.osm_path, conversion_dir, config.network_name)
        results["stages"][stage] = conversion
        write_status(stage, "passed", conversion)

        net_file = Path(conversion["net_file"])

        # 2b) sample conversion boot validation requirement
        sample_boot = validate_sample_sumo_boot(net_file, ensure_dir(conversion_dir / "sample_boot"))

        # 3) traffic light detection
        stage = "traffic_light_detection"
        current_stage = stage
        write_status(stage, "running")
        tls_detection = detect_traffic_lights(net_file)
        results["stages"][stage] = tls_detection
        write_status(stage, "passed", tls_detection)

        write_conversion_report(
            output_path=reports_dir / "osm_conversion_report.json",
            validation=osm_validation,
            conversion=conversion,
            tls_detection=tls_detection,
            sample_boot=sample_boot,
        )

        # 4) random route generation (train+heldout)
        stage = "random_route_generation"
        current_stage = stage
        write_status(stage, "running")
        demand_bundle = generate_train_and_eval_scenarios(
            net_file=net_file,
            begin_s=int(config.begin_s),
            end_s=int(config.end_s),
            low_period_s=float(config.low_period_s),
            medium_period_s=float(config.medium_period_s),
            high_period_s=float(config.high_period_s),
            eval_seeds=list(config.eval_seeds),
            scenario_root=ensure_dir(job_dir / "scenarios"),
            train_seed=int(config.train_seed),
        )
        results["stages"][stage] = {
            "train_scenario_id": demand_bundle["train"].scenario_id,
            "eval_scenario_count": len(demand_bundle["eval"]),
            "demand_level_period_s": demand_bundle["level_period_s"],
        }
        write_status(stage, "passed", results["stages"][stage])

        # 5) generated scenario manifest creation
        stage = "scenario_manifest_creation"
        current_stage = stage
        write_status(stage, "running")
        generated_manifest = create_generated_scenario_manifest(
            output_path=reports_dir / "generated_scenario_manifest.json",
            job_id=config.job_id,
            network_name=config.network_name,
            net_file=net_file,
            train_scenario=demand_bundle["train"],
            eval_scenarios=demand_bundle["eval"],
            level_period_s=demand_bundle["level_period_s"],
        )
        results["stages"][stage] = {
            "manifest_path": str(reports_dir / "generated_scenario_manifest.json"),
            "evaluation_count": generated_manifest["evaluation_count"],
        }
        write_status(stage, "passed", results["stages"][stage])

        # 6) Independent DQN training on uploaded map (smoke + full)
        stage = "independent_dqn_training"
        current_stage = stage
        write_status(stage, "running")

        smoke_out = ensure_dir(job_dir / "training" / "smoke")
        smoke_training = train_independent_dqn_from_scratch(
            train_scenario=demand_bundle["train"],
            output_dir=smoke_out,
            episodes=int(config.smoke_episodes),
            max_steps_per_episode=int(config.smoke_max_steps),
            seed=int(config.smoke_seed),
            train_every_steps=int(config.train_every_steps),
            learning_starts_steps=int(config.learning_starts_steps),
            target_update_interval=int(config.target_update_interval),
            wall_clock_cap_minutes=min(float(config.wall_clock_cap_minutes), 30.0),
            training_label="web_smoke",
        )

        full_out = ensure_dir(job_dir / "training" / "full")
        full_training = train_independent_dqn_from_scratch(
            train_scenario=demand_bundle["train"],
            output_dir=full_out,
            episodes=int(config.train_episodes),
            max_steps_per_episode=int(config.train_max_steps_per_episode),
            seed=int(config.train_seed),
            train_every_steps=int(config.train_every_steps),
            learning_starts_steps=int(config.learning_starts_steps),
            target_update_interval=int(config.target_update_interval),
            wall_clock_cap_minutes=float(config.wall_clock_cap_minutes),
            training_label="web_full",
        )

        results["stages"][stage] = {
            "smoke_summary_path": smoke_training["summary_path"],
            "smoke_checks": smoke_training["checks"],
            "full_summary_path": full_training["summary_path"],
            "full_checks": full_training["checks"],
            "controller_checkpoint": full_training["controller_checkpoint"],
        }
        write_status(stage, "passed", results["stages"][stage])

        # 7) evaluation on generated held-out scenarios
        stage = "evaluation"
        current_stage = stage
        write_status(stage, "running")
        evaluation = evaluate_on_heldout_scenarios(
            checkpoint_path=Path(full_training["controller_checkpoint"]),
            scenarios=demand_bundle["eval"],
            output_root=ensure_dir(job_dir / "evaluation" / "results"),
            run_root=ensure_dir(job_dir / "evaluation" / "runs"),
            seed=int(config.train_seed),
        )
        results["stages"][stage] = {
            "evaluation_summary_path": evaluation["summary_path"],
            "checks": evaluation["checks"],
        }
        write_status(stage, "passed", results["stages"][stage])

        # 8) KPI extraction
        stage = "kpi_extraction"
        current_stage = stage
        write_status(stage, "running")
        kpi_payload = extract_kpis_from_evaluation_outputs(
            evaluation_payload=evaluation["payload"],
            output_path=reports_dir / "kpi_extraction_report.json",
        )
        results["stages"][stage] = {
            "kpi_report_path": str(reports_dir / "kpi_extraction_report.json"),
            "kpi_definitions": kpi_payload["kpi_definitions"],
        }
        write_status(stage, "passed", results["stages"][stage])

        # 9) output packaging
        stage = "output_packaging"
        current_stage = stage
        write_status(stage, "running")
        packaging = package_job_outputs(
            job_dir=job_dir,
            package_dir=ensure_dir(config.output_root / "packages"),
            package_name=f"{config.job_id}_independent_dqn_v2_web_bundle",
        )
        results["stages"][stage] = packaging
        write_status(stage, "passed", results["stages"][stage])

        results["status"] = "passed"
        results["finished_at_utc"] = now_iso()
        json_dump(reports_dir / "job_result.json", results)
        write_status("complete", "passed", {"job_result": str(reports_dir / "job_result.json")})
        return results

    except WebIntegrationError as exc:
        failure = {
            "status": "failed",
            "failed_at_utc": now_iso(),
            "error_type": "WebIntegrationError",
            "error_stage": exc.stage,
            "error_message": str(exc),
            "partial_results": results,
        }
        json_dump(reports_dir / "job_result.json", failure)
        write_status(exc.stage, "failed", {"error": str(exc)})
        return failure

    except Exception as exc:  # pragma: no cover
        failure = {
            "status": "failed",
            "failed_at_utc": now_iso(),
            "error_type": type(exc).__name__,
            "error_stage": current_stage,
            "error_message": str(exc),
            "partial_results": results,
        }
        json_dump(reports_dir / "job_result.json", failure)
        write_status(current_stage, "failed", {"error": str(exc)})
        return failure
