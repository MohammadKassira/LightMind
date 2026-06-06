# Independent DQN v2 Web Integration Handoff

## Scope
This package adds a website-ready Python backend pipeline for uploaded `.osm` maps:
1. Validate uploaded OSM
2. Convert OSM to SUMO net
3. Detect traffic lights
4. Generate random traffic demand (no route upload required)
5. Build generated-scenario manifest
6. Train Independent DQN v2 from scratch on uploaded map
7. Evaluate on generated held-out scenarios
8. Extract KPI outputs (same 4 KPI definitions)
9. Package job outputs for download

## Hard Constraints Satisfied
- Uses `models/independent_dqn_v2.py`
- Uses `benchmark/controllers/independent_dqn_v2/env_adapter.py`
- Uses `benchmark/controllers/independent_dqn_v2/env_contract.json`
- Uses `aggregate_local_efficient_pressure_v2` reward contract
- Uses same 4 KPI definitions:
  - `mean_waiting_time_completed_s`
  - `throughput_completed_trips`
  - `mean_total_queue_length_m`
  - `phase_change_rate_per_tls_per_min`
- No trained benchmark checkpoint dependency
- No zero-shot transfer
- No modification to benchmark final report outputs

## New Package
Directory:
- `benchmark/controllers/independent_dqn_v2_web/`

Modules:
- `osm_pipeline.py`
- `demand.py`
- `scenario_manifest.py`
- `training.py`
- `evaluation.py`
- `kpi.py`
- `packaging.py`
- `job.py`
- `cli.py`
- `schemas.py`
- `exceptions.py`
- `utils.py`

## Backend Entry Point
Use programmatically:
- `from benchmark.controllers.independent_dqn_v2_web import run_web_job, WebJobConfig`

Or CLI:
```bash
python3 -m benchmark.controllers.independent_dqn_v2_web.cli \
  --job-id job_20260601_001 \
  --osm-path /abs/path/uploaded_map.osm \
  --output-root benchmark/web_jobs \
  --train-episodes 500 \
  --train-max-steps-per-episode 720
```

## Input Contract
- Required input: uploaded `.osm` file path
- No uploaded route files expected
- SUMO must be installed (`sumo`, `netconvert`, `randomTrips.py`)

## Output Layout
Each job writes under:
- `benchmark/web_jobs/<job_id>/`

Key outputs:
- `reports/job_status.json`
- `reports/job_result.json`
- `reports/osm_conversion_report.json`
- `reports/generated_scenario_manifest.json`
- `reports/kpi_extraction_report.json`
- `training/smoke/training_summary.json`
- `training/full/training_summary.json`
- `training/full/controller.pt`
- `evaluation/results/kpi_per_run.csv`
- `evaluation/results/kpi_per_run.json`
- `evaluation/results/kpi_summary_by_level.csv`
- `evaluation/results/kpi_summary_by_level.json`
- `evaluation/results/evaluation_summary.json`
- Packaged zip: `benchmark/web_jobs/packages/<job_id>_independent_dqn_v2_web_bundle.zip`

## Error Handling
- Structured failures are returned via `WebIntegrationError(stage=..., message=...)`
- `job_result.json` includes:
  - `status: failed`
  - `error_stage`
  - `error_message`
  - `partial_results`
- `job_status.json` updates per stage with `running/passed/failed`

## Validation Checklist Coverage
Implemented checks cover:
- No benchmark checkpoint required (`checkpoint_source = none` for training)
- Sample OSM conversion works (short SUMO boot check on converted net)
- Generated demand works (`randomTrips.py` output required)
- Smoke training works (dedicated smoke training stage)
- Checkpoint saved (`controller.pt` + per-agent checkpoints)
- Evaluation outputs KPI CSV/JSON
- Clean stage-scoped error handling

## Website Integration Suggestion
Backend flow:
1. Save upload to disk
2. Build `WebJobConfig`
3. Call `run_web_job(config)` in background worker
4. Poll `reports/job_status.json`
5. On completion return:
   - `reports/job_result.json`
   - KPI files
   - packaged zip path

## Notes
- This package is isolated from benchmark final report generation paths.
- Training is from scratch on generated map demand only.
- Evaluation is held-out generated scenarios, with no controller comparison mode.
