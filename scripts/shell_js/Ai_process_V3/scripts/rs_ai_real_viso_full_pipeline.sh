#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-djscc}"
VISO_DATA_ROOT="${VISO_DATA_ROOT:-/mnt/sda/guocheng/VISO dataset/extracted/data/data}"
MODEL="${MODEL:-$PROJECT_ROOT/models/viso_yolov8n_car_best.pt}"
FALLBACK_MODEL="${FALLBACK_MODEL:-$PROJECT_ROOT/open_data/vhr10_demo/models/vhr10_yolov8n_best.pt}"
MAX_FRAMES="${MAX_FRAMES:-120}"

cd "$PROJECT_ROOT"
if [[ ! -f "$MODEL" ]]; then
  echo "[RS-AI] VISO model not found, using fallback: $FALLBACK_MODEL"
  MODEL="$FALLBACK_MODEL"
fi

VIDEO_FILE="${VIDEO_FILE:-$(find "$VISO_DATA_ROOT" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' \) | sort | head -n 1)}"
FRAMES_DIR="${FRAMES_DIR:-}"
if [[ -z "$VIDEO_FILE" && -z "$FRAMES_DIR" ]]; then
  FRAMES_DIR="$(find "$VISO_DATA_ROOT" -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' \) -printf '%h\n' | sort | uniq | head -n 1)"
fi

if [[ -n "$VIDEO_FILE" ]]; then
  echo "[RS-AI] running real VISO video: $VIDEO_FILE"
  CONDA_ENV="$CONDA_ENV" VIDEO_FILE="$VIDEO_FILE" MODEL="$MODEL" MAX_FRAMES="$MAX_FRAMES" ./scripts/rs_ai_viso_sequence_demo.sh
elif [[ -n "$FRAMES_DIR" ]]; then
  echo "[RS-AI] running real VISO frames: $FRAMES_DIR"
  CONDA_ENV="$CONDA_ENV" FRAMES_DIR="$FRAMES_DIR" MODEL="$MODEL" MAX_FRAMES="$MAX_FRAMES" ./scripts/rs_ai_viso_sequence_demo.sh
else
  echo "[RS-AI] no VISO video or frame directory found under $VISO_DATA_ROOT" >&2
  exit 2
fi

