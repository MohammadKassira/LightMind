#!/usr/bin/env bash
# Run uvicorn with reload directories restricted so torch/venv files don't trigger restarts.
cd "$(dirname "$0")"
source venv/bin/activate

exec uvicorn main:app --reload \
  --reload-dir routers \
  --reload-dir services \
  --reload-include "*.py" \
  --reload-exclude "venv*" \
  --reload-exclude "traffic_rl*" \
  --reload-exclude "data*" \
  --reload-exclude "*.pt" \
  --reload-exclude "*.zip" \
  --reload-exclude "*.json"
