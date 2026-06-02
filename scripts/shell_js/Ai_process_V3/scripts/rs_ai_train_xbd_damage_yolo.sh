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

XBD_ROOT="${XBD_ROOT:-/mnt/sda/guocheng/xBD}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/open_data/xBD/xbd_yolo_damage}"
BASE_MODEL="${BASE_MODEL:-yolov8n.pt}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
RUN_NAME="${RUN_NAME:-xbd_yolov8n_damage}"
FRACTION="${FRACTION:-1.0}"
VAL="${VAL:-true}"
AMP="${AMP:-false}"
COPY_IMAGES="${COPY_IMAGES:-0}"
MODEL_OUT="${MODEL_OUT:-$PROJECT_ROOT/models/emergency_xbd_damage_best.pt}"

cd "$PROJECT_ROOT"
mkdir -p open_data/xBD logs/xbd_training models

if [[ ! -d "$XBD_ROOT" ]]; then
  echo "[RS-AI] xBD root not found: $XBD_ROOT" >&2
  echo "[RS-AI] Set XBD_ROOT to the extracted xBD/xView2 dataset root." >&2
  exit 2
fi

COPY_ARGS=()
case "${COPY_IMAGES,,}" in
  1|true|yes|on) COPY_ARGS=(--copy-images) ;;
esac

PYTHONPATH=. "${PYTHON_CMD[@]}" scripts/rs_ai_convert_xbd_to_yolo.py \
  --xbd-root "$XBD_ROOT" \
  --output-root "$DATASET_ROOT" \
  "${COPY_ARGS[@]}"

DATA_YAML="$DATASET_ROOT/xbd_damage.yaml"
if [[ ! -f "$DATA_YAML" ]]; then
  echo "[RS-AI] xBD YOLO yaml not found: $DATA_YAML" >&2
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
  --project "$PROJECT_ROOT/runs/xbd" \
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
echo "[RS-AI] best_source=$BEST"
echo "[RS-AI] emergency_model=$MODEL_OUT"

