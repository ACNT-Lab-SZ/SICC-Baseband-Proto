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

MODE="${MODE:-viso}"
LINK_MODE="${LINK_MODE:-copy}"
VIDEO_FILE="${VIDEO_FILE:-}"
IMAGE_DIR="${IMAGE_DIR:-}"
FRAMES_DIR="${FRAMES_DIR:-}"
VISO_SEQUENCE="${VISO_SEQUENCE:-}"
ANNOTATION="${ANNOTATION:-}"
MODEL="${MODEL:-models/viso_yolov8n_car_best.pt}"
BASEBAND_EXE="${BASEBAND_EXE:-$PROJECT_ROOT/bin/uhd_ldpc_ofdm_link}"
TX_BITS="${TX_BITS:-$PROJECT_ROOT/logs/baseband/rs_ai_tx_features.rsbf}"
RX_BITS="${RX_BITS:-$PROJECT_ROOT/logs/baseband/rs_ai_rx_features.rsbf}"
JSON_OUT="${JSON_OUT:-$PROJECT_ROOT/logs/baseband/rs_ai_detections.json}"
MANIFEST_JSON="${MANIFEST_JSON:-$PROJECT_ROOT/logs/baseband/rs_ai_source_manifest.json}"
INSPECT_JSON="${INSPECT_JSON:-$PROJECT_ROOT/logs/baseband/rs_ai_tx_features.inspect.json}"
UI_DIR="${UI_DIR:-$PROJECT_ROOT/logs/baseband/ui_assets}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-$PROJECT_ROOT/logs/baseband/ui_assets/annotated.mp4}"
METRICS_JSON="${METRICS_JSON:-$PROJECT_ROOT/logs/baseband/ui_assets/ui_metrics.json}"
PATTERN="${PATTERN:-*.jpg}"
SPLIT_LAYER="${SPLIT_LAYER:-22}"
DEVICE="${DEVICE:-cuda:0}"
IMGSZ="${IMGSZ:-640}"
MAX_FRAMES="${MAX_FRAMES:-120}"
STRIDE="${STRIDE:-1}"
TENSOR_CODEC="${TENSOR_CODEC:-int8}"
ZLIB_LEVEL="${ZLIB_LEVEL:-1}"
SNR_DB="${SNR_DB:-8}"
DECODER="${DECODER:-cuda-osd}"
ALIST="${ALIST:-$PROJECT_ROOT/matrices/LDPC/CCSDS_ldpc_n128_k64.alist}"
RATE="${RATE:-12.5e6}"
FREQ="${FREQ:-5e9}"
ACTIVE_SC="${ACTIVE_SC:-768}"
NUM_SYMBOLS="${NUM_SYMBOLS:-96}"
PILOT_PERIOD="${PILOT_PERIOD:-4}"
MODULATION="${MODULATION:-qpsk}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.45}"
RENDER="${RENDER:-1}"

usage() {
  cat >&2 <<'USAGE'
usage:
  rs_ai_cpp_file_roundtrip.sh --mode viso --video <video> --model <weights.pt> [--link-mode copy|baseband]

examples:
  LINK_MODE=copy CONDA_ENV=djscc ./scripts/rs_ai_cpp_file_roundtrip.sh \
    --mode viso --video /path/to/002.avi --annotation /path/to/gt2.txt \
    --model models/viso_yolov8n_car_best.pt --max-frames 120

  LINK_MODE=baseband ./scripts/rs_ai_cpp_file_roundtrip.sh \
    --mode video --video /path/to/input.mp4 --model models/demo.pt \
    --baseband-exe ./bin/uhd_ldpc_ofdm_link
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --link-mode) LINK_MODE="$2"; shift 2 ;;
    --video) VIDEO_FILE="$2"; shift 2 ;;
    --image-dir) IMAGE_DIR="$2"; shift 2 ;;
    --frames-dir) FRAMES_DIR="$2"; shift 2 ;;
    --sequence) VISO_SEQUENCE="$2"; shift 2 ;;
    --annotation) ANNOTATION="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --baseband-exe) BASEBAND_EXE="$2"; shift 2 ;;
    --tx-bitstream) TX_BITS="$2"; shift 2 ;;
    --rx-bitstream) RX_BITS="$2"; shift 2 ;;
    --output-json) JSON_OUT="$2"; shift 2 ;;
    --manifest-json) MANIFEST_JSON="$2"; shift 2 ;;
    --inspect-json) INSPECT_JSON="$2"; shift 2 ;;
    --ui-dir) UI_DIR="$2"; shift 2 ;;
    --output-video) OUTPUT_VIDEO="$2"; shift 2 ;;
    --metrics-json) METRICS_JSON="$2"; shift 2 ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    --split-layer) SPLIT_LAYER="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --tensor-codec) TENSOR_CODEC="$2"; shift 2 ;;
    --zlib-level) ZLIB_LEVEL="$2"; shift 2 ;;
    --sim-snr-db) SNR_DB="$2"; shift 2 ;;
    --decoder) DECODER="$2"; shift 2 ;;
    --alist) ALIST="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --freq) FREQ="$2"; shift 2 ;;
    --active-sc) ACTIVE_SC="$2"; shift 2 ;;
    --num-symbols) NUM_SYMBOLS="$2"; shift 2 ;;
    --pilot-period) PILOT_PERIOD="$2"; shift 2 ;;
    --modulation) MODULATION="$2"; shift 2 ;;
    --conf) CONF="$2"; shift 2 ;;
    --iou) IOU="$2"; shift 2 ;;
    --render) RENDER="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$TX_BITS")" "$(dirname "$JSON_OUT")" "$UI_DIR"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [[ -n "$CONDA_ENV" ]]; then
  PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)
else
  PYTHON_CMD=("$PYTHON_BIN")
fi

PAYLOAD_ARGS=(--mode "$MODE" --model "$MODEL" --output "$TX_BITS" --inspect-json "$INSPECT_JSON" --manifest-json "$MANIFEST_JSON" --pattern "$PATTERN" --split-layer "$SPLIT_LAYER" --device "$DEVICE" --imgsz "$IMGSZ" --max-frames "$MAX_FRAMES" --stride "$STRIDE" --tensor-codec "$TENSOR_CODEC" --zlib-level "$ZLIB_LEVEL")
[[ -n "$VIDEO_FILE" ]] && PAYLOAD_ARGS+=(--video "$VIDEO_FILE")
[[ -n "$IMAGE_DIR" ]] && PAYLOAD_ARGS+=(--image-dir "$IMAGE_DIR")
[[ -n "$FRAMES_DIR" ]] && PAYLOAD_ARGS+=(--frames-dir "$FRAMES_DIR")
[[ -n "$VISO_SEQUENCE" ]] && PAYLOAD_ARGS+=(--sequence "$VISO_SEQUENCE")
[[ -n "$ANNOTATION" ]] && PAYLOAD_ARGS+=(--annotation "$ANNOTATION")

"$PROJECT_ROOT/scripts/rs_ai_make_baseband_payload.sh" "${PAYLOAD_ARGS[@]}"

case "$LINK_MODE" in
  copy)
    cp "$TX_BITS" "$RX_BITS"
    ;;
  baseband)
    if [[ ! -x "$BASEBAND_EXE" ]]; then
      echo "[RS-AI] missing baseband executable: $BASEBAND_EXE" >&2
      echo "[RS-AI] use --link-mode copy on the 4090 server, or point --baseband-exe to a built uhd_ldpc_ofdm_link binary." >&2
      exit 3
    fi
    "$BASEBAND_EXE" \
      --mode sim \
      --traffic file \
      --input "$TX_BITS" \
      --output "$RX_BITS" \
      --alist "$ALIST" \
      --systematic-front-info \
      --rate "$RATE" \
      --freq "$FREQ" \
      --active-sc "$ACTIVE_SC" \
      --num-symbols "$NUM_SYMBOLS" \
      --pilot-period "$PILOT_PERIOD" \
      --modulation "$MODULATION" \
      --decoder "$DECODER" \
      --sim-snr-db "$SNR_DB"
    ;;
  *)
    echo "unknown link mode: $LINK_MODE" >&2
    exit 2
    ;;
esac

"$PROJECT_ROOT/scripts/rs_ai_decode_detection.sh" \
  --input "$RX_BITS" \
  --model "$MODEL" \
  --output-json "$JSON_OUT" \
  --split-layer "$SPLIT_LAYER" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  --conf "$CONF" \
  --iou "$IOU" \
  --tensor-codec "$TENSOR_CODEC"

if [[ "$RENDER" != "0" ]]; then
  RENDER_ARGS=(--detections-json "$JSON_OUT" --output-dir "$UI_DIR/annotated" --output-video "$OUTPUT_VIDEO" --metrics-json "$METRICS_JSON" --tx-bitstream "$TX_BITS" --rx-bitstream "$RX_BITS" --track)
  case "$MODE" in
    video)
      [[ -n "$VIDEO_FILE" ]] || { echo "Mode=video requires --video for rendering" >&2; exit 2; }
      RENDER_ARGS+=(--video "$VIDEO_FILE")
      ;;
    images)
      [[ -n "$IMAGE_DIR" ]] || { echo "Mode=images requires --image-dir for rendering" >&2; exit 2; }
      RENDER_ARGS+=(--image-dir "$IMAGE_DIR" --pattern "$PATTERN")
      ;;
    viso)
      if [[ -n "$VIDEO_FILE" ]]; then
        RENDER_ARGS+=(--video "$VIDEO_FILE")
      elif [[ -n "$FRAMES_DIR" ]]; then
        RENDER_ARGS+=(--image-dir "$FRAMES_DIR" --pattern "$PATTERN")
      else
        echo "Mode=viso requires --video or --frames-dir for rendering" >&2
        exit 2
      fi
      [[ -n "$ANNOTATION" ]] && RENDER_ARGS+=(--gt-annotation "$ANNOTATION")
      RENDER_ARGS+=(--fps 8)
      ;;
  esac
  "${PYTHON_CMD[@]}" -m rs_ai_link render-detections "${RENDER_ARGS[@]}"
fi

echo "[RS-AI] tx_bitstream=$TX_BITS"
echo "[RS-AI] rx_bitstream=$RX_BITS"
echo "[RS-AI] detections_json=$JSON_OUT"
if [[ "$RENDER" != "0" ]]; then
  echo "[RS-AI] annotated_video=$OUTPUT_VIDEO"
  echo "[RS-AI] ui_metrics=$METRICS_JSON"
fi

