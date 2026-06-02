#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/satburn_validation}"
LOG_FILE="$LOG_DIR/full_validation.log"
PID_FILE="$LOG_DIR/full_validation.pid"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "[RS-AI] validation already running: pid=$OLD_PID"
    echo "[RS-AI] log=$LOG_FILE"
    exit 0
  fi
fi

nohup bash -lc "
  set -euo pipefail
  cd \"$PROJECT_ROOT\"
  CONDA_ENV=\"${CONDA_ENV:-djscc}\" \
  SATBURN_ROOT=\"${SATBURN_ROOT:-/mnt/sda/heqing/SatelliteBurnedArea/extracted}\" \
  DATASET_ROOT=\"${DATASET_ROOT:-$PROJECT_ROOT/open_data/SatelliteBurnedArea/satburn_yolo_burned_area}\" \
  BASE_MODEL=\"${BASE_MODEL:-yolov8n.pt}\" \
  CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES:-0}\" \
  SENSOR=\"${SENSOR:-s2}\" \
  EPOCHS=\"${EPOCHS:-50}\" \
  BATCH=\"${BATCH:-16}\" \
  DEVICE=\"${DEVICE:-0}\" \
  MAX_FRAMES=\"${MAX_FRAMES:-40}\" \
  MODEL_OUT=\"${MODEL_OUT:-$PROJECT_ROOT/models/emergency_satburn_burned_area_best.pt}\" \
  ./scripts/rs_ai_satburn_full_validation.sh
" > "$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"
echo "[RS-AI] started pid=$(cat "$PID_FILE")"
echo "[RS-AI] log=$LOG_FILE"
