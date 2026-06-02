#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-}"
BITSTREAM_FILE=""
MODEL_PATH=""
OUTPUT_JSON="$PROJECT_ROOT/logs/rs_ai_detections.json"
SPLIT_LAYER=0
IMAGE_SIZE=640
DEVICE="cuda:0"
CONF=0.25
IOU=0.45
MAX_FRAMES=0
TENSOR_CODEC="int8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) BITSTREAM_FILE="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --split-layer) SPLIT_LAYER="$2"; shift 2 ;;
    --imgsz) IMAGE_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --conf) CONF="$2"; shift 2 ;;
    --iou) IOU="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --tensor-codec) TENSOR_CODEC="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BITSTREAM_FILE" || -z "$MODEL_PATH" ]]; then
  echo "usage: $0 --input <rsbf> --model <weights-or-yaml> [--output-json <file>]" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_JSON")"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=("$PYTHON_BIN")
fi

"${PYTHON_CMD[@]}" -m rs_ai_link decode-bitstream \
  --input "$BITSTREAM_FILE" \
  --model "$MODEL_PATH" \
  --output-json "$OUTPUT_JSON" \
  --split-layer "$SPLIT_LAYER" \
  --device "$DEVICE" \
  --imgsz "$IMAGE_SIZE" \
  --conf "$CONF" \
  --iou "$IOU" \
  --max-frames "$MAX_FRAMES" \
  --tensor-codec "$TENSOR_CODEC"
