#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-djscc}"
PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)
IMAGES="${IMAGES:-3}"
SPLIT_LAYER="${SPLIT_LAYER:-22}"
DEVICE="${DEVICE:-cuda:0}"
TENSOR_CODEC="${TENSOR_CODEC:-int8}"

cd "$PROJECT_ROOT"
mkdir -p logs open_data

PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link bootstrap-vhr10-demo \
  --output-dir open_data/vhr10_demo \
  --images "$IMAGES"

MODEL="open_data/vhr10_demo/models/vhr10_yolov8n_best.pt"
IMAGE_DIR="open_data/vhr10_demo/images"
TX_BITS="logs/vhr10_tx_features.rsbf"
RX_BITS="logs/vhr10_rx_features.rsbf"
JSON_OUT="logs/vhr10_detections.json"

PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link encode-images \
  --image-dir "$IMAGE_DIR" \
  --pattern '*.jpg' \
  --model "$MODEL" \
  --output "$TX_BITS" \
  --split-layer "$SPLIT_LAYER" \
  --device "$DEVICE" \
  --tensor-codec "$TENSOR_CODEC" \
  --max-images "$IMAGES"

# In the full USRP workflow, TX_BITS is sent by uhd_ldpc_ofdm_link --traffic file
# and RX_BITS is the decoded output file. The remote AI-only demo uses a lossless
# copy so the satellite detector can be validated without USRP hardware.
cp "$TX_BITS" "$RX_BITS"

PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link decode-bitstream \
  --input "$RX_BITS" \
  --model "$MODEL" \
  --output-json "$JSON_OUT" \
  --split-layer "$SPLIT_LAYER" \
  --device "$DEVICE" \
  --tensor-codec "$TENSOR_CODEC" \
  --conf 0.20 \
  --iou 0.45

PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link render-detections \
  --detections-json "$JSON_OUT" \
  --image-dir "$IMAGE_DIR" \
  --pattern '*.jpg' \
  --output-dir logs/ui_assets/annotated \
  --output-video logs/ui_assets/vhr10_annotated.mp4 \
  --metrics-json logs/ui_assets/ui_metrics.json \
  --tx-bitstream "$TX_BITS" \
  --rx-bitstream "$RX_BITS" \
  --fps 2

echo "[RS-AI] open_dataset_demo_json=$JSON_OUT"
echo "[RS-AI] ui_metrics=logs/ui_assets/ui_metrics.json"
echo "[RS-AI] annotated_video=logs/ui_assets/vhr10_annotated.mp4"
