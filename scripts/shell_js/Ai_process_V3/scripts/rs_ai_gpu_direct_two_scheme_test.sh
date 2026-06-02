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
MODEL="${MODEL:-models/viso_yolov8n_car_best.pt}"
VIDEO_FILE="${VIDEO_FILE:-}"
FRAMES_DIR="${FRAMES_DIR:-}"
PATTERN="${PATTERN:-*.jpg}"
MAX_FRAMES="${MAX_FRAMES:-120}"
SPLIT_LAYER="${SPLIT_LAYER:-22}"
DEVICE="${DEVICE:-cuda:0}"
IMGSZ="${IMGSZ:-640}"
TENSOR_CODEC="${TENSOR_CODEC:-int8}"
CONF="${CONF:-0.20}"
IOU="${IOU:-0.45}"
ROI_CONF="${ROI_CONF:-0.20}"
ROI_MARGIN="${ROI_MARGIN:-0.10}"
BG_DOWNSAMPLE="${BG_DOWNSAMPLE:-4}"
OUT_DIR="${OUT_DIR:-logs/gpu_direct_two_scheme}"

cd "$PROJECT_ROOT"
mkdir -p "$OUT_DIR"

SOURCE_ENV=()
if [[ -n "$VIDEO_FILE" ]]; then
  SOURCE_ENV+=(VIDEO_FILE="$VIDEO_FILE")
elif [[ -n "$FRAMES_DIR" ]]; then
  SOURCE_ENV+=(FRAMES_DIR="$FRAMES_DIR" PATTERN="$PATTERN")
else
  echo "Set VIDEO_FILE or FRAMES_DIR before running rs_ai_gpu_direct_two_scheme_test.sh" >&2
  exit 2
fi

for scheme in uniform roi-layered; do
  env \
    CONDA_BIN="$CONDA_BIN" \
    CONDA_ENV="$CONDA_ENV" \
    MODEL="$MODEL" \
    "${SOURCE_ENV[@]}" \
    SCHEME="$scheme" \
    MAX_FRAMES="$MAX_FRAMES" \
    SPLIT_LAYER="$SPLIT_LAYER" \
    DEVICE="$DEVICE" \
    IMGSZ="$IMGSZ" \
    TENSOR_CODEC="$TENSOR_CODEC" \
    CONF="$CONF" \
    IOU="$IOU" \
    ROI_CONF="$ROI_CONF" \
    ROI_MARGIN="$ROI_MARGIN" \
    BG_DOWNSAMPLE="$BG_DOWNSAMPLE" \
    OUTPUT_JSON="$OUT_DIR/${scheme}_roundtrip.json" \
    MANIFEST_JSON="$OUT_DIR/${scheme}_manifest.json" \
    ./scripts/rs_ai_gpu_direct_viso_demo.sh
done

"$CONDA_BIN" run -n "$CONDA_ENV" python scripts/rs_ai_collect_gpu_direct_summary.py --root "$OUT_DIR"

echo "[RS-AI] gpu_direct_two_scheme_summary=$OUT_DIR/summary.json"

