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
DATASET_FORMAT="${DATASET_FORMAT:-auto}"
BASE_MODEL="${BASE_MODEL:-yolov8n.pt}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORKERS="${WORKERS:-8}"
RUN_NAME="${RUN_NAME:-maritime_ship_yolov8n}"
FRACTION="${FRACTION:-1.0}"
VAL="${VAL:-true}"
AMP="${AMP:-false}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
COPY_IMAGES="${COPY_IMAGES:-0}"
MODEL_OUT="${MODEL_OUT:-$PROJECT_ROOT/models/security_maritime_ship_best.pt}"

cd "$PROJECT_ROOT"
mkdir -p open_data/MaritimeSecurity logs/maritime_ship_training models

if [[ ! -d "$SHIP_ROOT" ]]; then
  echo "[RS-AI] maritime ship dataset root not found: $SHIP_ROOT" >&2
  echo "[RS-AI] Put HRSID/SSDD/xView3-export data there or set SHIP_ROOT." >&2
  exit 2
fi

CONVERT_ARGS=()
case "${COPY_IMAGES,,}" in
  1|true|yes|on) CONVERT_ARGS+=(--copy-images) ;;
esac
if [[ "$MAX_SAMPLES" != "0" ]]; then
  CONVERT_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

PYTHONPATH=. "${PYTHON_CMD[@]}" scripts/rs_ai_convert_maritime_ship_to_yolo.py \
  --ship-root "$SHIP_ROOT" \
  --output-root "$DATASET_ROOT" \
  --format "$DATASET_FORMAT" \
  "${CONVERT_ARGS[@]}"

DATA_YAML="$DATASET_ROOT/maritime_ship.yaml"
if [[ ! -f "$DATA_YAML" ]]; then
  echo "[RS-AI] maritime ship YOLO yaml not found: $DATA_YAML" >&2
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
  --project "$PROJECT_ROOT/runs/maritime_ship" \
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
echo "[RS-AI] security_model=$MODEL_OUT"
