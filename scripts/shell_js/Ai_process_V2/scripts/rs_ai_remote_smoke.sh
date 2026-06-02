#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-djscc}"
PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)

cd "$PROJECT_ROOT"
PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link mock-bitstream --output logs/mock_features.rsbf --frames 4
PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link inspect-bitstream --input logs/mock_features.rsbf

if [[ -f logs/test_video1.mp4 ]]; then
  PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link encode-video \
    --video logs/test_video1.mp4 \
    --model yolov8n.yaml \
    --output logs/yolov8n_yaml_features_1f.rsbf \
    --split-layer 22 \
    --device cuda:0 \
    --max-frames 1 \
    --tensor-codec int8
  PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link decode-bitstream \
    --input logs/yolov8n_yaml_features_1f.rsbf \
    --model yolov8n.yaml \
    --output-json logs/yolov8n_yaml_detections_1f.json \
    --split-layer 22 \
    --device cuda:0 \
    --tensor-codec int8
fi
