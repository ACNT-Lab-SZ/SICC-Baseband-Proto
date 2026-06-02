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
SCHEME="${SCHEME:-uniform}"
MAX_FRAMES="${MAX_FRAMES:-120}"
SPLIT_LAYER="${SPLIT_LAYER:-22}"
DEVICE="${DEVICE:-cuda:0}"
TORCH_DEVICE="$DEVICE"
if [[ "$TORCH_DEVICE" =~ ^[0-9]+$ ]]; then
  TORCH_DEVICE="cuda:$TORCH_DEVICE"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IMGSZ="${IMGSZ:-640}"
TENSOR_CODEC="${TENSOR_CODEC:-int8}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.45}"
ROI_CONF="${ROI_CONF:-0.20}"
ROI_MARGIN="${ROI_MARGIN:-0.10}"
BG_DOWNSAMPLE="${BG_DOWNSAMPLE:-4}"
OUTPUT_JSON="${OUTPUT_JSON:-logs/gpu_direct/viso_gpu_direct_roundtrip.json}"
MANIFEST_JSON="${MANIFEST_JSON:-logs/gpu_direct/viso_gpu_direct_manifest.json}"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$OUTPUT_JSON")" "$(dirname "$MANIFEST_JSON")"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)
ARGS=(-m rs_ai_link gpu-direct-roundtrip --model "$MODEL" --scheme "$SCHEME" --split-layer "$SPLIT_LAYER" --device "$TORCH_DEVICE" --imgsz "$IMGSZ" --tensor-codec "$TENSOR_CODEC" --max-frames "$MAX_FRAMES" --conf "$CONF" --iou "$IOU" --roi-conf "$ROI_CONF" --roi-margin "$ROI_MARGIN" --bg-downsample "$BG_DOWNSAMPLE" --output-json "$OUTPUT_JSON" --manifest-json "$MANIFEST_JSON")
if [[ -n "$VIDEO_FILE" ]]; then
  ARGS+=(--video "$VIDEO_FILE")
elif [[ -n "$FRAMES_DIR" ]]; then
  ARGS+=(--frames-dir "$FRAMES_DIR" --pattern "$PATTERN")
else
  echo "Set VIDEO_FILE or FRAMES_DIR before running rs_ai_gpu_direct_viso_demo.sh" >&2
  exit 2
fi

"${PYTHON_CMD[@]}" "${ARGS[@]}"

echo "[RS-AI] gpu_direct_json=$OUTPUT_JSON"
echo "[RS-AI] gpu_direct_manifest=$MANIFEST_JSON"
