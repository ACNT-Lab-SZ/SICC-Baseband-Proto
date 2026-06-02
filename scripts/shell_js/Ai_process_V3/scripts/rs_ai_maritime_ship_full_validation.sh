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

SHIP_ROOT="${SHIP_ROOT:-/mnt/sda/heqing/MaritimeShipDataset}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/open_data/MaritimeSecurity/maritime_ship_yolo}"
MODEL_OUT="${MODEL_OUT:-$PROJECT_ROOT/models/security_maritime_ship_best.pt}"
DATASET_FORMAT="${DATASET_FORMAT:-auto}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONF="${CONF:-0.20}"
MAX_FRAMES="${MAX_FRAMES:-40}"
SCHEME="${SCHEME:-roi-layered}"
VIDEO_FPS="${VIDEO_FPS:-2}"

cd "$PROJECT_ROOT"
mkdir -p logs/maritime_ship_validation logs/maritime_ship_ui_videos

if [[ ! -d "$SHIP_ROOT" ]]; then
  echo "[RS-AI] maritime ship dataset root not found: $SHIP_ROOT" >&2
  echo "[RS-AI] Put HRSID/SSDD/xView3-export data there or run with SHIP_ROOT=/path/to/dataset." >&2
  exit 2
fi

echo "[RS-AI] converting and training maritime ship detector"
CONDA_ENV="$CONDA_ENV" \
SHIP_ROOT="$SHIP_ROOT" \
DATASET_ROOT="$DATASET_ROOT" \
DATASET_FORMAT="$DATASET_FORMAT" \
MODEL_OUT="$MODEL_OUT" \
EPOCHS="$EPOCHS" \
BATCH="$BATCH" \
DEVICE="$DEVICE" \
./scripts/rs_ai_train_maritime_ship_yolo.sh

DATA_YAML="$DATASET_ROOT/maritime_ship.yaml"

echo "[RS-AI] validating on test split"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$CONDA_BIN" run -n "$CONDA_ENV" yolo detect val \
  model="$MODEL_OUT" \
  data="$DATA_YAML" \
  split=test \
  imgsz=640 \
  device="$DEVICE" \
  project="$PROJECT_ROOT/runs/maritime_ship" \
  name=maritime_ship_yolov8n_test \
  exist_ok=True

echo "[RS-AI] rendering prediction images"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$CONDA_BIN" run -n "$CONDA_ENV" yolo detect predict \
  model="$MODEL_OUT" \
  source="$DATASET_ROOT/images/test" \
  imgsz=640 \
  conf="$CONF" \
  device="$DEVICE" \
  project="$PROJECT_ROOT/logs/maritime_ship_predict" \
  name=test_images \
  save=True \
  exist_ok=True

echo "[RS-AI] rendering UI mosaic videos"
"${PYTHON_CMD[@]}" scripts/rs_ai_render_mosaic_video.py \
  --input-dir "$DATASET_ROOT/images/test" \
  --pattern "*" \
  --output logs/maritime_ship_ui_videos/original_maritime_ship_mosaic.mp4 \
  --fps "$VIDEO_FPS" \
  --rows 3 \
  --cols 4 \
  --cell 150 \
  --title Maritime_security_original

"${PYTHON_CMD[@]}" scripts/rs_ai_render_mosaic_video.py \
  --input-dir logs/maritime_ship_predict/test_images \
  --pattern "*" \
  --output logs/maritime_ship_ui_videos/receiver_detected_ship_mosaic.mp4 \
  --fps "$VIDEO_FPS" \
  --rows 3 \
  --cols 4 \
  --cell 150 \
  --title Receiver_detected_ship

echo "[RS-AI] gpu-direct roundtrip"
CONDA_ENV="$CONDA_ENV" \
MODEL="$MODEL_OUT" \
FRAMES_DIR="$DATASET_ROOT/images/test" \
PATTERN="*" \
SCHEME="$SCHEME" \
MAX_FRAMES="$MAX_FRAMES" \
CONF="$CONF" \
OUTPUT_JSON=logs/maritime_ship_validation/gpu_direct_roundtrip.json \
MANIFEST_JSON=logs/maritime_ship_validation/gpu_direct_manifest.json \
./scripts/rs_ai_gpu_direct_viso_demo.sh

echo "[RS-AI] outputs:"
echo "[RS-AI] model=$MODEL_OUT"
echo "[RS-AI] data_yaml=$DATA_YAML"
echo "[RS-AI] predicted_images=logs/maritime_ship_predict/test_images"
echo "[RS-AI] original_mosaic=logs/maritime_ship_ui_videos/original_maritime_ship_mosaic.mp4"
echo "[RS-AI] detected_mosaic=logs/maritime_ship_ui_videos/receiver_detected_ship_mosaic.mp4"
echo "[RS-AI] gpu_direct_json=logs/maritime_ship_validation/gpu_direct_roundtrip.json"
