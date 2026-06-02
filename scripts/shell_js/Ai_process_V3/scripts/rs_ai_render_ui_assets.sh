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
PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)

DETECTIONS_JSON="${DETECTIONS_JSON:-logs/vhr10_detections.json}"
IMAGE_DIR="${IMAGE_DIR:-open_data/vhr10_demo/images}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/ui_assets/annotated}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-logs/ui_assets/annotated.mp4}"
METRICS_JSON="${METRICS_JSON:-logs/ui_assets/ui_metrics.json}"
TX_BITSTREAM="${TX_BITSTREAM:-logs/vhr10_tx_features.rsbf}"
RX_BITSTREAM="${RX_BITSTREAM:-logs/vhr10_rx_features.rsbf}"
FPS="${FPS:-2}"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$METRICS_JSON")" "$OUTPUT_DIR"

PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link render-detections \
  --detections-json "$DETECTIONS_JSON" \
  --image-dir "$IMAGE_DIR" \
  --pattern '*.jpg' \
  --output-dir "$OUTPUT_DIR" \
  --output-video "$OUTPUT_VIDEO" \
  --metrics-json "$METRICS_JSON" \
  --tx-bitstream "$TX_BITSTREAM" \
  --rx-bitstream "$RX_BITSTREAM" \
  --fps "$FPS"


