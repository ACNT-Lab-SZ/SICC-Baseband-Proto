#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -z "${CONDA_BIN:-}" ]]; then
  if [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
  elif [[ -x "/home/heqing/anaconda3/bin/conda" ]]; then
    CONDA_BIN="/home/heqing/anaconda3/bin/conda"
  else
    CONDA_BIN="conda"
  fi
fi
CONDA_ENV="${CONDA_ENV:-djscc}"
PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)

SAT_ROOT="${SAT_ROOT:-logs/task_update_sim/satellite}"
GROUND_ROOT="${GROUND_ROOT:-logs/task_update_sim/ground}"
CAR_MODEL="${CAR_MODEL:-models/viso_yolov8n_car_best.pt}"
EMERGENCY_MODEL="${EMERGENCY_MODEL:-models/emergency_sen1floods11_flood_best.pt}"
TASK_ID="${TASK_ID:-satellite_flood_assessment}"
TASK_VERSION="${TASK_VERSION:-sen1floods11_flood_v1}"
SCENE="${SCENE:-flood_rescue}"
RESOURCE_SCHEME="${RESOURCE_SCHEME:-roi-layered}"
MAX_FRAMES="${MAX_FRAMES:-120}"

cd "$PROJECT_ROOT"
mkdir -p "$SAT_ROOT" "$GROUND_ROOT"

if [[ ! -f "$CAR_MODEL" ]]; then
  echo "[RS-AI] car model not found: $CAR_MODEL" >&2
  exit 2
fi
if [[ ! -f "$EMERGENCY_MODEL" ]]; then
  echo "[RS-AI] emergency model not found: $EMERGENCY_MODEL" >&2
  echo "[RS-AI] Train it first with scripts/rs_ai_train_sen1floods11_yolo.sh or set EMERGENCY_MODEL." >&2
  exit 2
fi

"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py init \
  --sat-root "$SAT_ROOT" \
  --task-id "viso_car_monitor" \
  --version "car_v1" \
  --scene "land_urban_road" \
  --classes "car" \
  --model "$CAR_MODEL" \
  --resource-scheme "$RESOURCE_SCHEME"

UPDATE_ZIP="$GROUND_ROOT/${TASK_VERSION}.zip"
"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py make-update \
  --task-id "$TASK_ID" \
  --version "$TASK_VERSION" \
  --scene "$SCENE" \
  --classes "flood_water" \
  --model "$EMERGENCY_MODEL" \
  --resource-scheme "$RESOURCE_SCHEME" \
  --output "$UPDATE_ZIP" \
  --notes "Satellite task switched from VISO car monitoring to Sen1Floods11 flood rescue assessment."

"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py uplink \
  --package "$UPDATE_ZIP" \
  --sat-root "$SAT_ROOT"

"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py apply \
  --package "$(basename "$UPDATE_ZIP")" \
  --sat-root "$SAT_ROOT"

"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py status --sat-root "$SAT_ROOT"

echo "[RS-AI] suggested_gpu_direct_command:"
"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py run-command \
  --sat-root "$SAT_ROOT" \
  --video "sample_data/sen1floods11_demo/flood_demo.mp4" \
  --max-frames "$MAX_FRAMES" \
  --conda-env "$CONDA_ENV"

