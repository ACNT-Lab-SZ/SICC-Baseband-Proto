#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VISO_RAR_DIR="${VISO_RAR_DIR:-/mnt/sda/guocheng/VISO dataset}"
EXTRACT_DIR="${EXTRACT_DIR:-$VISO_RAR_DIR/extracted}"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-djscc}"
MODEL="${MODEL:-$PROJECT_ROOT/open_data/vhr10_demo/models/vhr10_yolov8n_best.pt}"
MAX_FRAMES="${MAX_FRAMES:-120}"
PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)

cd "$PROJECT_ROOT"
mkdir -p logs "$EXTRACT_DIR"/{data,detection_coco,mot}

extract_if_needed() {
  local archive="$1"
  local outdir="$2"
  local marker="$outdir/.extract_done"
  if [[ -f "$marker" ]]; then
    echo "[RS-AI] already_extracted=$archive"
    return
  fi
  if [[ ! -f "$archive" ]]; then
    echo "[RS-AI] missing_archive=$archive" >&2
    return
  fi
  echo "[RS-AI] extracting=$archive -> $outdir"
  unrar x -o+ "$archive" "$outdir/"
  touch "$marker"
}

extract_if_needed "$VISO_RAR_DIR/data.rar" "$EXTRACT_DIR/data"
extract_if_needed "$VISO_RAR_DIR/Detection_coco_format.rar" "$EXTRACT_DIR/detection_coco"
extract_if_needed "$VISO_RAR_DIR/mot.rar" "$EXTRACT_DIR/mot"

PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link scan-viso \
  --root "$EXTRACT_DIR" \
  --output-json logs/viso_real_scan.json

VIDEO_FILE="$(find "$EXTRACT_DIR/data" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' \) | sort | head -n 1 || true)"
FRAMES_DIR=""
if [[ -z "$VIDEO_FILE" ]]; then
  FRAMES_DIR="$(find "$EXTRACT_DIR/data" -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' \) -printf '%h\n' | sort | uniq | head -n 1 || true)"
fi
ANNOTATION="$(find "$EXTRACT_DIR/mot" "$EXTRACT_DIR/detection_coco" -type f \( -iname '*.txt' -o -iname '*.json' -o -iname '*.xml' \) | sort | head -n 1 || true)"

echo "[RS-AI] selected_video=$VIDEO_FILE"
echo "[RS-AI] selected_frames=$FRAMES_DIR"
echo "[RS-AI] selected_annotation=$ANNOTATION"

if [[ -n "$VIDEO_FILE" ]]; then
  CONDA_ENV="$CONDA_ENV" \
  MODEL="$MODEL" \
  VIDEO_FILE="$VIDEO_FILE" \
  ANNOTATION="$ANNOTATION" \
  MAX_FRAMES="$MAX_FRAMES" \
  ./scripts/rs_ai_viso_sequence_demo.sh
elif [[ -n "$FRAMES_DIR" ]]; then
  CONDA_ENV="$CONDA_ENV" \
  MODEL="$MODEL" \
  FRAMES_DIR="$FRAMES_DIR" \
  ANNOTATION="$ANNOTATION" \
  MAX_FRAMES="$MAX_FRAMES" \
  ./scripts/rs_ai_viso_sequence_demo.sh
else
  echo "[RS-AI] no video or frame sequence found under $EXTRACT_DIR/data" >&2
  exit 2
fi

echo "[RS-AI] done"
echo "[RS-AI] scan=logs/viso_real_scan.json"
echo "[RS-AI] metrics=logs/viso_ui_assets/ui_metrics.json"
echo "[RS-AI] video=logs/viso_ui_assets/annotated.mp4"

