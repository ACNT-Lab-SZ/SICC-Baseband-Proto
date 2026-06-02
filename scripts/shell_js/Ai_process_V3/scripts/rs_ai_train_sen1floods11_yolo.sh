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

SEN1_ROOT="${SEN1_ROOT:-/mnt/sda/guocheng/Sen1Floods11}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/open_data/Sen1Floods11/sen1floods11_yolo_flood}"
BASE_MODEL="${BASE_MODEL:-yolov8n.pt}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORKERS="${WORKERS:-8}"
RUN_NAME="${RUN_NAME:-sen1floods11_yolov8n_flood}"
FRACTION="${FRACTION:-1.0}"
VAL="${VAL:-true}"
AMP="${AMP:-false}"
SENSOR="${SENSOR:-s2}"
INCLUDE_WEAK="${INCLUDE_WEAK:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MIN_AREA="${MIN_AREA:-64}"
COPY_IMAGES="${COPY_IMAGES:-0}"
MODEL_OUT="${MODEL_OUT:-$PROJECT_ROOT/models/emergency_sen1floods11_flood_best.pt}"

cd "$PROJECT_ROOT"
mkdir -p open_data/Sen1Floods11 logs/sen1floods11_training models

if [[ ! -d "$SEN1_ROOT" ]]; then
  echo "[RS-AI] Sen1Floods11 root not found: $SEN1_ROOT" >&2
  echo "[RS-AI] Download first with scripts/rs_ai_download_sen1floods11.sh or set SEN1_ROOT." >&2
  exit 2
fi

CONVERT_ARGS=()
case "${COPY_IMAGES,,}" in
  1|true|yes|on) CONVERT_ARGS+=(--copy-images) ;;
esac
case "${INCLUDE_WEAK,,}" in
  1|true|yes|on) CONVERT_ARGS+=(--include-weak) ;;
esac
if [[ "$MAX_SAMPLES" != "0" ]]; then
  CONVERT_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

PYTHONPATH=. "${PYTHON_CMD[@]}" scripts/rs_ai_convert_sen1floods11_to_yolo.py \
  --sen1-root "$SEN1_ROOT" \
  --output-root "$DATASET_ROOT" \
  --sensor "$SENSOR" \
  --min-area "$MIN_AREA" \
  "${CONVERT_ARGS[@]}"

DATA_YAML="$DATASET_ROOT/sen1floods11_flood.yaml"
if [[ ! -f "$DATA_YAML" ]]; then
  echo "[RS-AI] Sen1Floods11 YOLO yaml not found: $DATA_YAML" >&2
  exit 3
fi

VAL_ARGS=()
case "${VAL,,}" in
  1|true|yes|on) VAL_ARGS=(--val) ;;
esac
AMP_ARGS=()
case "${AMP,,}" in
  1|true|yes|on) AMP_ARGS=(--amp) ;;
esac

"${PYTHON_CMD[@]}" scripts/rs_ai_train_yolo.py \
  --model "$BASE_MODEL" \
  --data "$DATA_YAML" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --project "$PROJECT_ROOT/runs/sen1floods11" \
  --name "$RUN_NAME" \
  --fraction "$FRACTION" \
  "${VAL_ARGS[@]}" \
  "${AMP_ARGS[@]}"

BEST="$(find "$PROJECT_ROOT/runs" -type f -path "*/$RUN_NAME/weights/best.pt" | sort | tail -n 1 || true)"
if [[ -z "$BEST" ]]; then
  echo "[RS-AI] training finished but best.pt was not found under $PROJECT_ROOT/runs" >&2
  exit 4
fi

cp "$BEST" "$MODEL_OUT"
echo "[RS-AI] data_yaml=$DATA_YAML"
echo "[RS-AI] roi_masks=$DATASET_ROOT/masks"
echo "[RS-AI] best_source=$BEST"
echo "[RS-AI] emergency_model=$MODEL_OUT"

