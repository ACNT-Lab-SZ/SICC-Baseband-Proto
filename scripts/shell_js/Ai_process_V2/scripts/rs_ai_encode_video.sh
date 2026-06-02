#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-}"
VIDEO_FILE=""
MODEL_PATH=""
OUTPUT_BITSTREAM="$PROJECT_ROOT/logs/rs_ai_tx_features.rsbf"
SPLIT_LAYER=22
IMAGE_SIZE=640
DEVICE="cuda:0"
MAX_FRAMES=0
STRIDE=1
TENSOR_CODEC="int8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO_FILE="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --output) OUTPUT_BITSTREAM="$2"; shift 2 ;;
    --split-layer) SPLIT_LAYER="$2"; shift 2 ;;
    --imgsz) IMAGE_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --tensor-codec) TENSOR_CODEC="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$VIDEO_FILE" || -z "$MODEL_PATH" ]]; then
  echo "usage: $0 --video <video> --model <weights-or-yaml> [--output <file>]" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_BITSTREAM")"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=("$PYTHON_BIN")
fi

"${PYTHON_CMD[@]}" -m rs_ai_link encode-video \
  --video "$VIDEO_FILE" \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_BITSTREAM" \
  --split-layer "$SPLIT_LAYER" \
  --device "$DEVICE" \
  --imgsz "$IMAGE_SIZE" \
  --max-frames "$MAX_FRAMES" \
  --stride "$STRIDE" \
  --tensor-codec "$TENSOR_CODEC"
