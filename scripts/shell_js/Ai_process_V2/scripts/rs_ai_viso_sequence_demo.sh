#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-djscc}"
PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)

MODEL="${MODEL:-open_data/vhr10_demo/models/vhr10_yolov8n_best.pt}"
FRAMES_DIR="${FRAMES_DIR:-open_data/vhr10_demo/images}"
VIDEO_FILE="${VIDEO_FILE:-}"
ANNOTATION="${ANNOTATION:-}"
MAX_FRAMES="${MAX_FRAMES:-0}"
SPLIT_LAYER="${SPLIT_LAYER:-22}"
DEVICE="${DEVICE:-cuda:0}"
TENSOR_CODEC="${TENSOR_CODEC:-int8}"
TX_BITS="${TX_BITS:-logs/viso_tx_features.rsbf}"
RX_BITS="${RX_BITS:-logs/viso_rx_features.rsbf}"
JSON_OUT="${JSON_OUT:-logs/viso_detections.json}"
UI_DIR="${UI_DIR:-logs/viso_ui_assets}"
EMBED_VISUAL_PREVIEW="${EMBED_VISUAL_PREVIEW:-1}"
VISUAL_DOWNSAMPLE="${VISUAL_DOWNSAMPLE:-1}"
VISUAL_QUALITY="${VISUAL_QUALITY:-85}"

cd "$PROJECT_ROOT"
mkdir -p logs "$UI_DIR"

SOURCE_ARGS=()
RENDER_SOURCE_ARGS=()
if [[ -n "$VIDEO_FILE" ]]; then
  SOURCE_ARGS=(--video "$VIDEO_FILE")
  RENDER_SOURCE_ARGS=(--video "$VIDEO_FILE")
else
  SOURCE_ARGS=(--frames-dir "$FRAMES_DIR")
  RENDER_SOURCE_ARGS=(--image-dir "$FRAMES_DIR" --pattern '*.jpg')
fi
if [[ -n "$ANNOTATION" ]]; then
  SOURCE_ARGS+=(--annotation "$ANNOTATION")
fi

PREVIEW_ARGS=()
if [[ "$EMBED_VISUAL_PREVIEW" != "0" ]]; then
  PREVIEW_ARGS=(--embed-visual-preview --visual-downsample "$VISUAL_DOWNSAMPLE" --visual-quality "$VISUAL_QUALITY")
fi

PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link encode-viso \
  "${SOURCE_ARGS[@]}" \
  --model "$MODEL" \
  --output "$TX_BITS" \
  --manifest-json logs/viso_manifest.json \
  --split-layer "$SPLIT_LAYER" \
  --device "$DEVICE" \
  --tensor-codec "$TENSOR_CODEC" \
  --max-frames "$MAX_FRAMES" \
  "${PREVIEW_ARGS[@]}"

# Replace this copy with uhd_ldpc_ofdm_link --traffic file for the real radio path.
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

RENDER_ARGS=()
if [[ -n "$ANNOTATION" ]]; then
  RENDER_ARGS+=(--gt-annotation "$ANNOTATION")
fi
if [[ "$EMBED_VISUAL_PREVIEW" != "0" ]]; then
  RENDER_ARGS+=(--source-bitstream "$RX_BITS")
else
  RENDER_ARGS+=("${RENDER_SOURCE_ARGS[@]}")
fi

PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link render-detections \
  --detections-json "$JSON_OUT" \
  --output-dir "$UI_DIR/annotated" \
  --output-video "$UI_DIR/annotated.mp4" \
  --metrics-json "$UI_DIR/ui_metrics.json" \
  --tx-bitstream "$TX_BITS" \
  --rx-bitstream "$RX_BITS" \
  --track \
  "${RENDER_ARGS[@]}" \
  --fps 8

echo "[RS-AI] viso_detections=$JSON_OUT"
echo "[RS-AI] viso_ui_metrics=$UI_DIR/ui_metrics.json"
echo "[RS-AI] viso_annotated_video=$UI_DIR/annotated.mp4"
