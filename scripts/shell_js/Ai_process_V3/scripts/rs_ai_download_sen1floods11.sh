#!/usr/bin/env bash
set -euo pipefail

DEST="${DEST:-/mnt/sda/guocheng/Sen1Floods11}"
BUCKET="${BUCKET:-gs://sen1floods11}"
SOURCE="${SOURCE:-gcs}"
ZENODO_URL="${ZENODO_URL:-https://zenodo.org/records/8298840/files/sen1_floods11.7z?download=1}"

mkdir -p "$DEST"

echo "[RS-AI] downloading Sen1Floods11"
echo "[RS-AI] source=$SOURCE"
echo "[RS-AI] dest=$DEST"

if [[ "$SOURCE" == "zenodo" ]]; then
  ARCHIVE="$DEST/sen1_floods11.7z"
  if command -v wget >/dev/null 2>&1; then
    wget -c "$ZENODO_URL" -O "$ARCHIVE"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -C - "$ZENODO_URL" -o "$ARCHIVE"
  else
    echo "[RS-AI] wget/curl was not found." >&2
    exit 2
  fi
  if command -v 7z >/dev/null 2>&1; then
    7z x -y "$ARCHIVE" -o"$DEST/extracted"
  elif python -c "import py7zr" >/dev/null 2>&1; then
    python scripts/rs_ai_extract_sen1floods11.py --archive "$ARCHIVE" --output "$DEST/extracted"
  else
    echo "[RS-AI] 7z was not found; archive downloaded but not extracted: $ARCHIVE" >&2
    echo "[RS-AI] Install py7zr in the active environment and run:" >&2
    echo "        python scripts/rs_ai_extract_sen1floods11.py --archive $ARCHIVE --output $DEST/extracted" >&2
  fi
  echo "[RS-AI] download complete"
  find "$DEST" -maxdepth 3 -type f | head -20
  exit 0
fi

if command -v gsutil >/dev/null 2>&1; then
  gsutil -m rsync -r "$BUCKET" "$DEST"
elif command -v gcloud >/dev/null 2>&1; then
  gcloud storage rsync --recursive "$BUCKET" "$DEST"
else
  echo "[RS-AI] gsutil/gcloud was not found." >&2
  echo "[RS-AI] Install Google Cloud SDK or run:" >&2
  echo "        python -m pip install gsutil" >&2
  echo "        gsutil -m rsync -r $BUCKET $DEST" >&2
  exit 2
fi

echo "[RS-AI] download complete"
find "$DEST" -maxdepth 3 -type f | head -20
