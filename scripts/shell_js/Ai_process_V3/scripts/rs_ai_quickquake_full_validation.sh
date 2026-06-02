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

QQB_ROOT="${QQB_ROOT:-/mnt/sda/heqing/QuickQuakeBuildings}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/open_data/QuickQuakeBuildings/quickquake_yolo_damage}"
MODEL_OUT="${MODEL_OUT:-$PROJECT_ROOT/models/emergency_quickquake_damage_best.pt}"
MODE="${MODE:-opt}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONF="${CONF:-0.20}"
MAX_FRAMES="${MAX_FRAMES:-40}"
SCHEME="${SCHEME:-roi-layered}"
VIDEO_FPS="${VIDEO_FPS:-2}"

cd "$PROJECT_ROOT"
mkdir -p logs/quickquake_validation logs/quickquake_ui_videos

if [[ ! -d "$QQB_ROOT" ]]; then
  echo "[RS-AI] QuickQuakeBuildings root not found: $QQB_ROOT" >&2
  echo "[RS-AI] Put the extracted real dataset at QQB_ROOT or run with QQB_ROOT=/path/to/dataset." >&2
  exit 2
fi

echo "[RS-AI] converting and training QuickQuakeBuildings"
CONDA_ENV="$CONDA_ENV" \
QQB_ROOT="$QQB_ROOT" \
DATASET_ROOT="$DATASET_ROOT" \
MODEL_OUT="$MODEL_OUT" \
MODE="$MODE" \
EPOCHS="$EPOCHS" \
BATCH="$BATCH" \
DEVICE="$DEVICE" \
./scripts/rs_ai_train_quickquakebuildings_yolo.sh

DATA_YAML="$DATASET_ROOT/quickquakebuildings_damage.yaml"

echo "[RS-AI] validating on test split"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$CONDA_BIN" run -n "$CONDA_ENV" yolo detect val \
  model="$MODEL_OUT" \
  data="$DATA_YAML" \
  split=test \
  imgsz=640 \
  device="$DEVICE" \
  project="$PROJECT_ROOT/runs/quickquake" \
  name=quickquake_yolov8n_damage_test \
  exist_ok=True

echo "[RS-AI] rendering prediction images"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$CONDA_BIN" run -n "$CONDA_ENV" yolo detect predict \
  model="$MODEL_OUT" \
  source="$DATASET_ROOT/images/test" \
  imgsz=640 \
  conf="$CONF" \
  device="$DEVICE" \
  project="$PROJECT_ROOT/logs/quickquake_predict" \
  name=test_images \
  save=True \
  exist_ok=True

echo "[RS-AI] rendering UI videos"
"${PYTHON_CMD[@]}" scripts/rs_ai_render_frame_folder_video.py \
  --input-dir "$DATASET_ROOT/images/test" \
  --pattern "*.png" \
  --output logs/quickquake_ui_videos/original_earthquake_building_sequence.mp4 \
  --fps "$VIDEO_FPS" \
  --width 1280 \
  --height 720 \
  --title QuickQuake_original

"${PYTHON_CMD[@]}" scripts/rs_ai_render_frame_folder_video.py \
  --input-dir logs/quickquake_predict/test_images \
  --pattern "*.jpg" \
  --output logs/quickquake_ui_videos/receiver_detected_building_damage_sequence.mp4 \
  --fps "$VIDEO_FPS" \
  --width 1280 \
  --height 720 \
  --title Receiver_detected_building_damage

echo "[RS-AI] gpu-direct roundtrip"
CONDA_ENV="$CONDA_ENV" \
MODEL="$MODEL_OUT" \
FRAMES_DIR="$DATASET_ROOT/images/test" \
PATTERN="*.png" \
SCHEME="$SCHEME" \
MAX_FRAMES="$MAX_FRAMES" \
CONF="$CONF" \
OUTPUT_JSON=logs/quickquake_validation/gpu_direct_roundtrip.json \
MANIFEST_JSON=logs/quickquake_validation/gpu_direct_manifest.json \
./scripts/rs_ai_gpu_direct_viso_demo.sh

echo "[RS-AI] outputs:"
echo "[RS-AI] model=$MODEL_OUT"
echo "[RS-AI] data_yaml=$DATA_YAML"
echo "[RS-AI] predicted_images=logs/quickquake_predict/test_images"
echo "[RS-AI] original_video=logs/quickquake_ui_videos/original_earthquake_building_sequence.mp4"
echo "[RS-AI] detected_video=logs/quickquake_ui_videos/receiver_detected_building_damage_sequence.mp4"
echo "[RS-AI] gpu_direct_json=logs/quickquake_validation/gpu_direct_roundtrip.json"
