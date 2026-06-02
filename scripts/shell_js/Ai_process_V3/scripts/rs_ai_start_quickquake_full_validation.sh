#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/quickquake_validation}"
LOG_FILE="$LOG_DIR/full_validation.log"
PID_FILE="$LOG_DIR/full_validation.pid"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "[RS-AI] QuickQuake validation already running: pid=$OLD_PID"
    echo "[RS-AI] log=$LOG_FILE"
    exit 0
  fi
fi

nohup bash -lc "
  set -euo pipefail
  cd \"$PROJECT_ROOT\"
  CONDA_ENV=\"${CONDA_ENV:-djscc}\" \
  QQB_ROOT=\"${QQB_ROOT:-/mnt/sda/heqing/QuickQuakeBuildings/earthquake_building_dataset}\" \
  MODE=\"${MODE:-opt}\" \
  EPOCHS=\"${EPOCHS:-50}\" \
  BATCH=\"${BATCH:-16}\" \
  DEVICE=\"${DEVICE:-0}\" \
  ./scripts/rs_ai_quickquake_full_validation.sh
" > "$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"
echo "[RS-AI] started pid=$(cat "$PID_FILE")"
echo "[RS-AI] log=$LOG_FILE"
