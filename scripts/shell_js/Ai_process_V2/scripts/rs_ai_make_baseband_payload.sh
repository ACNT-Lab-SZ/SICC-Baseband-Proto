#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-djscc}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODE="${MODE:-video}"
VIDEO_FILE="${VIDEO_FILE:-}"
IMAGE_DIR="${IMAGE_DIR:-}"
FRAMES_DIR="${FRAMES_DIR:-}"
VISO_SEQUENCE="${VISO_SEQUENCE:-}"
ANNOTATION="${ANNOTATION:-}"
MODEL="${MODEL:-models/viso_yolov8n_car_best.pt}"
OUTPUT="${OUTPUT:-logs/baseband/rs_ai_tx_features.rsbf}"
INSPECT_JSON="${INSPECT_JSON:-logs/baseband/rs_ai_tx_features.inspect.json}"
MANIFEST_JSON="${MANIFEST_JSON:-logs/baseband/rs_ai_source_manifest.json}"
PATTERN="${PATTERN:-*.jpg}"
SPLIT_LAYER="${SPLIT_LAYER:-22}"
DEVICE="${DEVICE:-cuda:0}"
IMGSZ="${IMGSZ:-640}"
MAX_FRAMES="${MAX_FRAMES:-0}"
STRIDE="${STRIDE:-1}"
TENSOR_CODEC="${TENSOR_CODEC:-int8}"
ZLIB_LEVEL="${ZLIB_LEVEL:-1}"
EMBED_VISUAL_PREVIEW="${EMBED_VISUAL_PREVIEW:-0}"
VISUAL_DOWNSAMPLE="${VISUAL_DOWNSAMPLE:-1}"
VISUAL_QUALITY="${VISUAL_QUALITY:-85}"

usage() {
  cat >&2 <<'USAGE'
usage:
  rs_ai_make_baseband_payload.sh --mode video --video <video> --model <weights.pt> --output <tx.rsbf>
  rs_ai_make_baseband_payload.sh --mode images --image-dir <dir> --model <weights.pt> --output <tx.rsbf>
  rs_ai_make_baseband_payload.sh --mode viso --video <video> --model <weights.pt> --output <tx.rsbf>
  rs_ai_make_baseband_payload.sh --mode viso --frames-dir <dir> --model <weights.pt> --output <tx.rsbf>

optional:
  --annotation <gt.txt> --split-layer 22 --device cuda:0 --imgsz 640
  --max-frames 120 --stride 1 --tensor-codec int8 --inspect-json <inspect.json>
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

PREVIEW_ARGS=()
if [[ "$EMBED_VISUAL_PREVIEW" != "0" ]]; then
  PREVIEW_ARGS=(--embed-visual-preview --visual-downsample "$VISUAL_DOWNSAMPLE" --visual-quality "$VISUAL_QUALITY")
fi

case "$MODE" in
  video)
    if [[ -z "$VIDEO_FILE" || -z "$MODEL" ]]; then
      usage
      exit 2
    fi
    "${PYTHON_CMD[@]}" -m rs_ai_link encode-video \
      --video "$VIDEO_FILE" \
      --model "$MODEL" \
      --output "$OUTPUT" \
      --split-layer "$SPLIT_LAYER" \
      --device "$DEVICE" \
      --imgsz "$IMGSZ" \
      --max-frames "$MAX_FRAMES" \
      --stride "$STRIDE" \
      --tensor-codec "$TENSOR_CODEC" \
      --zlib-level "$ZLIB_LEVEL" \
      "${PREVIEW_ARGS[@]}"
    ;;
  images)
    if [[ -z "$IMAGE_DIR" || -z "$MODEL" ]]; then
      usage
      exit 2
    fi
    "${PYTHON_CMD[@]}" -m rs_ai_link encode-images \
      --image-dir "$IMAGE_DIR" \
      --pattern "$PATTERN" \
      --model "$MODEL" \
      --output "$OUTPUT" \
      --split-layer "$SPLIT_LAYER" \
      --device "$DEVICE" \
      --imgsz "$IMGSZ" \
      --max-images "$MAX_FRAMES" \
      --tensor-codec "$TENSOR_CODEC" \
      --zlib-level "$ZLIB_LEVEL" \
      "${PREVIEW_ARGS[@]}"
    ;;
  viso)
    SOURCE_ARGS=()
    if [[ -n "$VISO_SEQUENCE" ]]; then
      SOURCE_ARGS+=(--sequence "$VISO_SEQUENCE")
    fi
    if [[ -n "$VIDEO_FILE" ]]; then
      SOURCE_ARGS+=(--video "$VIDEO_FILE")
    fi
    if [[ -n "$FRAMES_DIR" ]]; then
      SOURCE_ARGS+=(--frames-dir "$FRAMES_DIR" --pattern "$PATTERN")
    fi
    if [[ -n "$ANNOTATION" ]]; then
      SOURCE_ARGS+=(--annotation "$ANNOTATION")
    fi
    if [[ ${#SOURCE_ARGS[@]} -eq 0 || -z "$MODEL" ]]; then
      usage
      exit 2
    fi
    "${PYTHON_CMD[@]}" -m rs_ai_link encode-viso \
      "${SOURCE_ARGS[@]}" \
      --model "$MODEL" \
      --output "$OUTPUT" \
      --manifest-json "$MANIFEST_JSON" \
      --split-layer "$SPLIT_LAYER" \
      --device "$DEVICE" \
      --imgsz "$IMGSZ" \
      --max-frames "$MAX_FRAMES" \
      --tensor-codec "$TENSOR_CODEC" \
      --zlib-level "$ZLIB_LEVEL" \
      "${PREVIEW_ARGS[@]}"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    usage
    exit 2
    ;;
esac

"${PYTHON_CMD[@]}" -m rs_ai_link inspect-bitstream --input "$OUTPUT" --max-frames 1 > "$INSPECT_JSON"

echo "[RS-AI] baseband_payload=$OUTPUT"
echo "[RS-AI] inspect_json=$INSPECT_JSON"
echo "[RS-AI] give the .rsbf file to uhd_ldpc_ofdm_link --traffic file --input"
