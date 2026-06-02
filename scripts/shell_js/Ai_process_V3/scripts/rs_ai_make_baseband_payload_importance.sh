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
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODE="${MODE:-video}"
VIDEO_FILE="${VIDEO_FILE:-}"
IMAGE_DIR="${IMAGE_DIR:-}"
FRAMES_DIR="${FRAMES_DIR:-}"
VISO_SEQUENCE="${VISO_SEQUENCE:-}"
ANNOTATION="${ANNOTATION:-}"
MODEL="${MODEL:-models/viso_yolov8n_car_best.pt}"
OUTPUT="${OUTPUT:-logs/baseband/rs_ai_tx_features_importance.rsbf}"
INSPECT_JSON="${INSPECT_JSON:-logs/baseband/rs_ai_tx_features_importance.inspect.json}"
MANIFEST_JSON="${MANIFEST_JSON:-logs/baseband/rs_ai_source_manifest_importance.json}"
PATTERN="${PATTERN:-*.jpg}"
SPLIT_LAYER="${SPLIT_LAYER:-22}"
DEVICE="${DEVICE:-cuda:0}"
IMGSZ="${IMGSZ:-640}"
MAX_FRAMES="${MAX_FRAMES:-0}"
STRIDE="${STRIDE:-1}"
TENSOR_CODEC="${TENSOR_CODEC:-int8}"
ZLIB_LEVEL="${ZLIB_LEVEL:-1}"
ROI_CONF="${ROI_CONF:-0.20}"
ROI_MARGIN="${ROI_MARGIN:-0.10}"
BG_KEEP_RATIO="${BG_KEEP_RATIO:-0.15}"
BG_SCALE_BOOST="${BG_SCALE_BOOST:-4.0}"

usage() {
  cat >&2 <<'USAGE'
usage:
  rs_ai_make_baseband_payload_importance.sh --mode video --video <video> --model <weights.pt> --output <tx.rsbf>
  rs_ai_make_baseband_payload_importance.sh --mode images --image-dir <dir> --model <weights.pt> --output <tx.rsbf>
  rs_ai_make_baseband_payload_importance.sh --mode viso --video <video> --model <weights.pt> --output <tx.rsbf>
  rs_ai_make_baseband_payload_importance.sh --mode viso --frames-dir <dir> --model <weights.pt> --output <tx.rsbf>

optional:
  --roi-conf 0.20 --roi-margin 0.10 --bg-keep-ratio 0.15 --bg-scale-boost 4.0
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --video) VIDEO_FILE="$2"; shift 2 ;;
    --image-dir) IMAGE_DIR="$2"; shift 2 ;;
    --frames-dir) FRAMES_DIR="$2"; shift 2 ;;
    --sequence) VISO_SEQUENCE="$2"; shift 2 ;;
    --annotation) ANNOTATION="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --inspect-json) INSPECT_JSON="$2"; shift 2 ;;
    --manifest-json) MANIFEST_JSON="$2"; shift 2 ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    --split-layer) SPLIT_LAYER="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --tensor-codec) TENSOR_CODEC="$2"; shift 2 ;;
    --zlib-level) ZLIB_LEVEL="$2"; shift 2 ;;
    --roi-conf) ROI_CONF="$2"; shift 2 ;;
    --roi-margin) ROI_MARGIN="$2"; shift 2 ;;
    --bg-keep-ratio) BG_KEEP_RATIO="$2"; shift 2 ;;
    --bg-scale-boost) BG_SCALE_BOOST="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$INSPECT_JSON")" "$(dirname "$MANIFEST_JSON")"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=("$PYTHON_BIN")
fi

COMMON_ARGS=(
  --model "$MODEL"
  --output "$OUTPUT"
  --split-layer "$SPLIT_LAYER"
  --device "$DEVICE"
  --imgsz "$IMGSZ"
  --tensor-codec "$TENSOR_CODEC"
  --zlib-level "$ZLIB_LEVEL"
  --roi-conf "$ROI_CONF"
  --roi-margin "$ROI_MARGIN"
  --bg-keep-ratio "$BG_KEEP_RATIO"
  --bg-scale-boost "$BG_SCALE_BOOST"
)

case "$MODE" in
  video)
    [[ -n "$VIDEO_FILE" ]] || { usage; exit 2; }
    "${PYTHON_CMD[@]}" -m rs_ai_link encode-video-importance \
      --video "$VIDEO_FILE" \
      --max-frames "$MAX_FRAMES" \
      --stride "$STRIDE" \
      "${COMMON_ARGS[@]}"
    ;;
  images)
    [[ -n "$IMAGE_DIR" ]] || { usage; exit 2; }
    "${PYTHON_CMD[@]}" -m rs_ai_link encode-images-importance \
      --image-dir "$IMAGE_DIR" \
      --pattern "$PATTERN" \
      --max-images "$MAX_FRAMES" \
      "${COMMON_ARGS[@]}"
    ;;
  viso)
    SOURCE_ARGS=()
    [[ -n "$VISO_SEQUENCE" ]] && SOURCE_ARGS+=(--sequence "$VISO_SEQUENCE")
    [[ -n "$VIDEO_FILE" ]] && SOURCE_ARGS+=(--video "$VIDEO_FILE")
    [[ -n "$FRAMES_DIR" ]] && SOURCE_ARGS+=(--frames-dir "$FRAMES_DIR" --pattern "$PATTERN")
    [[ -n "$ANNOTATION" ]] && SOURCE_ARGS+=(--annotation "$ANNOTATION")
    [[ ${#SOURCE_ARGS[@]} -gt 0 ]] || { usage; exit 2; }
    "${PYTHON_CMD[@]}" -m rs_ai_link encode-viso-importance \
      "${SOURCE_ARGS[@]}" \
      --manifest-json "$MANIFEST_JSON" \
      --max-frames "$MAX_FRAMES" \
      "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    usage
    exit 2
    ;;
esac

"${PYTHON_CMD[@]}" -m rs_ai_link inspect-bitstream --input "$OUTPUT" --max-frames 1 > "$INSPECT_JSON"

echo "[RS-AI] baseband_payload_importance=$OUTPUT"
echo "[RS-AI] inspect_json=$INSPECT_JSON"
echo "[RS-AI] resource_mode=importance_aware_v1"

