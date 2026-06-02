#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-djscc}"
MODEL="${MODEL:-models/viso_yolov8n_car_best.pt}"
VIDEO_FILE="${VIDEO_FILE:-}"
ANNOTATION="${ANNOTATION:-}"
MAX_FRAMES="${MAX_FRAMES:-120}"
BG_DOWNSAMPLE="${BG_DOWNSAMPLE:-4}"
ROI_CONF="${ROI_CONF:-0.20}"
ROI_MARGIN="${ROI_MARGIN:-0.10}"

cd "$PROJECT_ROOT"
mkdir -p logs/roi_layered_eval/baseline logs/roi_layered_eval/layered
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
PYTHON_CMD=("$CONDA_BIN" run -n "$CONDA_ENV" python)

[[ -n "$VIDEO_FILE" ]] || { echo "Set VIDEO_FILE before running" >&2; exit 2; }

CONDA_ENV="$CONDA_ENV" \
MODEL="$MODEL" \
VIDEO_FILE="$VIDEO_FILE" \
ANNOTATION="$ANNOTATION" \
MAX_FRAMES="$MAX_FRAMES" \
TX_BITS="logs/roi_layered_eval/baseline/tx.rsbf" \
RX_BITS="logs/roi_layered_eval/baseline/rx.rsbf" \
JSON_OUT="logs/roi_layered_eval/baseline/detections.json" \
UI_DIR="logs/roi_layered_eval/baseline/ui" \
./scripts/rs_ai_viso_sequence_demo.sh

if [[ -n "$ANNOTATION" ]]; then
  CONDA_ENV="$CONDA_ENV" ./scripts/rs_ai_make_baseband_payload_roi_layered.sh \
    --mode viso \
    --video "$VIDEO_FILE" \
    --annotation "$ANNOTATION" \
    --model "$MODEL" \
    --output logs/roi_layered_eval/layered/tx.rsbf \
    --manifest-json logs/roi_layered_eval/layered/manifest.json \
    --inspect-json logs/roi_layered_eval/layered/inspect.json \
    --max-frames "$MAX_FRAMES" \
    --roi-conf "$ROI_CONF" \
    --roi-margin "$ROI_MARGIN" \
    --bg-downsample "$BG_DOWNSAMPLE"
else
  CONDA_ENV="$CONDA_ENV" ./scripts/rs_ai_make_baseband_payload_roi_layered.sh \
    --mode viso \
    --video "$VIDEO_FILE" \
    --model "$MODEL" \
    --output logs/roi_layered_eval/layered/tx.rsbf \
    --manifest-json logs/roi_layered_eval/layered/manifest.json \
    --inspect-json logs/roi_layered_eval/layered/inspect.json \
    --max-frames "$MAX_FRAMES" \
    --roi-conf "$ROI_CONF" \
    --roi-margin "$ROI_MARGIN" \
    --bg-downsample "$BG_DOWNSAMPLE"
fi

cp logs/roi_layered_eval/layered/tx.rsbf logs/roi_layered_eval/layered/rx.rsbf

CONDA_ENV="$CONDA_ENV" ./scripts/rs_ai_decode_detection.sh \
  --input logs/roi_layered_eval/layered/rx.rsbf \
  --model "$MODEL" \
  --output-json logs/roi_layered_eval/layered/detections.json \
  --split-layer 22 \
  --device cuda:0 \
  --conf 0.20 \
  --iou 0.45

RENDER_ARGS=(
  -m rs_ai_link render-detections
  --detections-json logs/roi_layered_eval/layered/detections.json
  --source-bitstream logs/roi_layered_eval/layered/rx.rsbf
  --output-dir logs/roi_layered_eval/layered/ui/annotated
  --output-video logs/roi_layered_eval/layered/ui/annotated.mp4
  --metrics-json logs/roi_layered_eval/layered/ui/ui_metrics.json
  --tx-bitstream logs/roi_layered_eval/layered/tx.rsbf
  --rx-bitstream logs/roi_layered_eval/layered/rx.rsbf
  --track
  --fps 8
)
if [[ -n "$ANNOTATION" ]]; then
  RENDER_ARGS+=(--gt-annotation "$ANNOTATION")
fi
"${PYTHON_CMD[@]}" "${RENDER_ARGS[@]}"

"${PYTHON_CMD[@]}" scripts/rs_ai_eval_detections.py \
  --detections-json logs/roi_layered_eval/baseline/detections.json \
  --annotation "$ANNOTATION" \
  --output-json logs/roi_layered_eval/baseline/eval.json

"${PYTHON_CMD[@]}" scripts/rs_ai_eval_detections.py \
  --detections-json logs/roi_layered_eval/layered/detections.json \
  --annotation "$ANNOTATION" \
  --output-json logs/roi_layered_eval/layered/eval.json

"${PYTHON_CMD[@]}" - <<'PY'
import json, os
from pathlib import Path

root = Path("logs/roi_layered_eval")
baseline_det = json.loads((root / "baseline" / "detections.json").read_text(encoding="utf-8"))
layered_det = json.loads((root / "layered" / "detections.json").read_text(encoding="utf-8"))
baseline_eval = json.loads((root / "baseline" / "eval.json").read_text(encoding="utf-8"))["metrics"]
layered_eval = json.loads((root / "layered" / "eval.json").read_text(encoding="utf-8"))["metrics"]
summary = {
    "baseline": {
        "frames": baseline_det["summary"]["frames"],
        "detections": baseline_det["summary"]["detections"],
        "fps": baseline_det["summary"]["fps"],
        "tx_bytes": os.path.getsize(root / "baseline" / "tx.rsbf"),
        "precision": baseline_eval.get("precision"),
        "recall": baseline_eval.get("recall"),
        "f1": baseline_eval.get("f1"),
    },
    "roi_layered": {
        "frames": layered_det["summary"]["frames"],
        "detections": layered_det["summary"]["detections"],
        "fps": layered_det["summary"]["fps"],
        "tx_bytes": os.path.getsize(root / "layered" / "tx.rsbf"),
        "precision": layered_eval.get("precision"),
        "recall": layered_eval.get("recall"),
        "f1": layered_eval.get("f1"),
    },
}
summary["delta"] = {
    "tx_bytes_ratio": (summary["roi_layered"]["tx_bytes"] / summary["baseline"]["tx_bytes"]) if summary["baseline"]["tx_bytes"] else None,
    "precision_delta": None if summary["baseline"]["precision"] is None or summary["roi_layered"]["precision"] is None else summary["roi_layered"]["precision"] - summary["baseline"]["precision"],
    "recall_delta": None if summary["baseline"]["recall"] is None or summary["roi_layered"]["recall"] is None else summary["roi_layered"]["recall"] - summary["baseline"]["recall"],
    "f1_delta": None if summary["baseline"]["f1"] is None or summary["roi_layered"]["f1"] is None else summary["roi_layered"]["f1"] - summary["baseline"]["f1"],
}
out = root / "summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
