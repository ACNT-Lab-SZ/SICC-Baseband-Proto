#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/satburn_download}"
LOG_FILE="$LOG_DIR/download.log"
PID_FILE="$LOG_DIR/download.pid"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "[RS-AI] download already running: pid=$OLD_PID"
    echo "[RS-AI] log=$LOG_FILE"
    exit 0
  fi
fi

nohup bash -lc "
  set -euo pipefail
  cd \"$PROJECT_ROOT\"
  CONDA_ENV=\"${CONDA_ENV:-djscc}\" \
  DEST=\"${DEST:-/mnt/sda/heqing/SatelliteBurnedArea}\" \
  EXTRACT=\"${EXTRACT:-1}\" \
  ./scripts/rs_ai_download_satburn.sh
" > "$LOG_FILE" 2>&1 &

echo "$!" > "$PID_FILE"
echo "[RS-AI] started pid=$(cat "$PID_FILE")"
echo "[RS-AI] log=$LOG_FILE"
