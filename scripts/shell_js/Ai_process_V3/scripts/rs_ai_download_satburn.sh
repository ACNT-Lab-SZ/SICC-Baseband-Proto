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
DEST="${DEST:-/mnt/sda/heqing/SatelliteBurnedArea}"
EXTRACT="${EXTRACT:-1}"
LIST_ONLY="${LIST_ONLY:-0}"
ZENODO_API="${ZENODO_API:-https://zenodo.org/api/records/6597139}"

cd "$PROJECT_ROOT"
mkdir -p "$DEST"

ARGS=(scripts/rs_ai_download_satburn.py --dest "$DEST" --api-url "$ZENODO_API")
case "${LIST_ONLY,,}" in
  1|true|yes|on) ARGS+=(--list-only) ;;
esac
case "${EXTRACT,,}" in
  1|true|yes|on) ARGS+=(--extract) ;;
esac

echo "[RS-AI] downloading Satellite Burned Area Dataset"
echo "[RS-AI] dest=$DEST"
"$CONDA_BIN" run -n "$CONDA_ENV" python "${ARGS[@]}"

echo "[RS-AI] download complete"
find "$DEST" -maxdepth 3 -type f | head -40
