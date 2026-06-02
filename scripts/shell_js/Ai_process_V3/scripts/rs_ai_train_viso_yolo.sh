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

VISO_COCO_ROOT="${VISO_COCO_ROOT:-/mnt/sda/guocheng/VISO dataset/extracted/detection_coco/coco/car}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/open_data/VISO/viso_yolo_car}"
DATA_YAML="${DATA_YAML:-}"
DEFAULT_BASE_MODEL="$PROJECT_ROOT/open_data/vhr10_demo/models/vhr10_yolov8n_best.pt"
if [[ -z "${BASE_MODEL:-}" ]]; then
  if [[ -f "$DEFAULT_BASE_MODEL" ]]; then
    BASE_MODEL="$DEFAULT_BASE_MODEL"
  else
    BASE_MODEL="yolov8n.pt"
  fi
else
  BASE_MODEL="${BASE_MODEL}"
fi
EPOCHS="${EPOCHS:-30}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
RUN_NAME="${RUN_NAME:-viso_yolov8n_car}"
FRACTION="${FRACTION:-1.0}"
VAL="${VAL:-true}"
AMP="${AMP:-false}"

cd "$PROJECT_ROOT"
mkdir -p open_data/VISO logs/viso_training

if [[ ! -f "$BASE_MODEL" && "$BASE_MODEL" != "yolov8n.pt" ]]; then
  echo "[RS-AI] base model not found: $BASE_MODEL" >&2
  exit 2
fi
echo "[RS-AI] base_model=$BASE_MODEL"

PYTHONPATH=. "${PYTHON_CMD[@]}" -m rs_ai_link convert-viso-coco \
  --coco-root "$VISO_COCO_ROOT" \
  --output-root "$DATASET_ROOT" \
  --class-name car

if [[ -z "$DATA_YAML" ]]; then
  if [[ -f "$DATASET_ROOT/viso_car.yaml" ]]; then
    DATA_YAML="$DATASET_ROOT/viso_car.yaml"
  else
    DATA_YAML="$(find "$DATASET_ROOT" -maxdepth 1 -type f -name '*.yaml' | sort | head -n 1 || true)"
  fi
fi
if [[ -z "$DATA_YAML" || ! -f "$DATA_YAML" ]]; then
  echo "[RS-AI] dataset yaml not found under $DATASET_ROOT" >&2
  exit 2
fi
echo "[RS-AI] data_yaml=$DATA_YAML"

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
  --project "$PROJECT_ROOT/runs/viso" \
  --name "$RUN_NAME" \
  --fraction "$FRACTION" \
  "${VAL_ARGS[@]}" \
  "${AMP_ARGS[@]}"

BEST_CANDIDATES=(
  "$PROJECT_ROOT/runs/viso/$RUN_NAME/weights/best.pt"
  "$PROJECT_ROOT/runs/detect/runs/viso/$RUN_NAME/weights/best.pt"
)
BEST=""
for candidate in "${BEST_CANDIDATES[@]}"; do
  if [[ -f "$candidate" ]]; then
    BEST="$candidate"
    break
  fi
done
if [[ -z "$BEST" ]]; then
  BEST="$(find "$PROJECT_ROOT/runs" -type f -path "*/$RUN_NAME/weights/best.pt" | sort | tail -n 1 || true)"
fi
if [[ -z "$BEST" ]]; then
  echo "[RS-AI] training finished but best.pt was not found under $PROJECT_ROOT/runs" >&2
  exit 3
fi

mkdir -p "$PROJECT_ROOT/models"
cp "$BEST" "$PROJECT_ROOT/models/viso_yolov8n_car_best.pt"
echo "[RS-AI] best_source=$BEST"
echo "[RS-AI] viso_model=$PROJECT_ROOT/models/viso_yolov8n_car_best.pt"

