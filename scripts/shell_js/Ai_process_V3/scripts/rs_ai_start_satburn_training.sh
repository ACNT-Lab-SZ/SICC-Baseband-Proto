#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/satburn_training}"
LOG_FILE="$LOG_DIR/train.log"
PID_FILE="$LOG_DIR/train.pid"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "[RS-AI] training already running: pid=$OLD_PID"
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
  IMGSZ=\"${IMGSZ:-640}\" \
  BATCH=\"${BATCH:-16}\" \
  DEVICE=\"${DEVICE:-0}\" \
  WORKERS=\"${WORKERS:-8}\" \
  VAL=\"${VAL:-true}\" \
  AMP=\"${AMP:-false}\" \
  MODEL_OUT=\"${MODEL_OUT:-$PROJECT_ROOT/models/emergency_satburn_burned_area_best.pt}\" \
  ./scripts/rs_ai_train_satburn_yolo.sh
" > "$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"
echo "[RS-AI] started pid=$(cat "$PID_FILE")"
echo "[RS-AI] log=$LOG_FILE"
