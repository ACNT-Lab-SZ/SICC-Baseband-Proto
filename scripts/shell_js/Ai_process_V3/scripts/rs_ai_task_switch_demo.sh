#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${CONDA_BIN:-}" ]]; then
  if [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
  elif [[ -x "/home/heqing/anaconda3/bin/conda" ]]; then
    CONDA_BIN="/home/heqing/anaconda3/bin/conda"
  else
    CONDA_BIN="conda"
  fi
fi
CONDA_ENV="${CONDA_ENV:-}"

SAT_ROOT="${SAT_ROOT:-logs/task_update_sim/satellite}"
GROUND_ROOT="${GROUND_ROOT:-logs/task_update_sim/ground}"
CAR_MODEL="${CAR_MODEL:-models/viso_yolov8n_car_best.pt}"
SHIP_MODEL="${SHIP_MODEL:-models/vhr10_yolov8n_ship_demo.pt}"
SHIP_VERSION="${SHIP_VERSION:-ship_ocean_v1}"
MAX_FRAMES="${MAX_FRAMES:-120}"

cd "$PROJECT_ROOT"
mkdir -p "$SAT_ROOT" "$GROUND_ROOT"

if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=("$PYTHON_BIN")
fi

if [[ ! -f "$CAR_MODEL" ]]; then
  echo "[RS-AI] car model not found: $CAR_MODEL" >&2
  exit 2
fi
if [[ ! -f "$SHIP_MODEL" ]]; then
  echo "[RS-AI] ship model not found: $SHIP_MODEL" >&2
  echo "[RS-AI] Set SHIP_MODEL to a ship detector .pt file." >&2
  exit 2
fi

"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py init \
  --sat-root "$SAT_ROOT" \
  --task-id "viso_car_monitor" \
  --version "car_v1" \
  --scene "land_urban_road" \
  --classes "car" \
  --model "$CAR_MODEL" \
  --resource-scheme "roi-layered"

UPDATE_ZIP="$GROUND_ROOT/${SHIP_VERSION}.zip"
"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py make-update \
  --task-id "ocean_ship_monitor" \
  --version "$SHIP_VERSION" \
  --scene "ocean_maritime" \
  --classes "ship" \
  --model "$SHIP_MODEL" \
  --resource-scheme "roi-layered" \
  --output "$UPDATE_ZIP" \
  --notes "Satellite moved from VISO car monitoring to ocean ship monitoring."

"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py uplink \
  --package "$UPDATE_ZIP" \
  --sat-root "$SAT_ROOT"

"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py apply \
  --package "$(basename "$UPDATE_ZIP")" \
  --sat-root "$SAT_ROOT"

"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py status --sat-root "$SAT_ROOT"

echo "[RS-AI] active_model=$("${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py active-model --sat-root "$SAT_ROOT")"
echo "[RS-AI] suggested_gpu_direct_command:"
"${PYTHON_CMD[@]}" scripts/rs_ai_task_model_update.py run-command \
  --sat-root "$SAT_ROOT" \
  --video "sample_data/ship_demo/ship_demo.mp4" \
  --max-frames "$MAX_FRAMES" \
  --conda-env "${CONDA_ENV:-djscc}"

