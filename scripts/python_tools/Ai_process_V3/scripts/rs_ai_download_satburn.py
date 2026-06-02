from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


ZENODO_API = "https://zenodo.org/api/records/6597139"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Satellite Burned Area Dataset files from Zenodo.")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--api-url", default=ZENODO_API)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--list-only", action="store_true", help="Only print the Zenodo file list; do not download.")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    dest = Path(args.dest)
    archive_dir = dest / "archives"
    extract_dir = dest / "extracted"
    archive_dir.mkdir(parents=True, exist_ok=True)
    if args.extract:
        extract_dir.mkdir(parents=True, exist_ok=True)

    record = _read_json(args.api_url)
    files = record.get("files", [])
    if not files:
        raise RuntimeError(f"No files returned by Zenodo API: {args.api_url}")

    if args.list_only:
        listing = []
        for item in files:
            key = item.get("key") or item.get("filename") or "download.bin"
            listing.append({"key": key, "size": item.get("size"), "checksum": item.get("checksum")})
        print(json.dumps({"api": args.api_url, "files": listing}, ensure_ascii=False, indent=2))
        return 0

    downloaded: list[str] = []
    for item in files:
        key = item.get("key") or item.get("filename") or "download.bin"
        links = item.get("links") or {}
        url = links.get("self") or links.get("download")
        if not url:
            continue
        out = archive_dir / Path(key).name
        if out.exists() and args.skip_existing and out.stat().st_size > 0:
            print(f"[RS-AI] exists {out}")
        else:
            print(f"[RS-AI] download {key}")
            _download(url, out)
        downloaded.append(str(out))
        if args.extract and out.suffix.lower() == ".zip":
            print(f"[RS-AI] extract {out.name}")
            with zipfile.ZipFile(out) as zf:
                zf.extractall(extract_dir)

    summary = {
        "dataset": record.get("metadata", {}).get("title", "Satellite Burned Area Dataset"),
        "api": args.api_url,
        "archives": str(archive_dir),
        "extracted": str(extract_dir) if args.extract else None,
        "files": downloaded,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, out: Path) -> None:
    tmp = out.with_suffix(out.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "rs-ai-satburn-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1024 * 1024)
    tmp.replace(out)


if __name__ == "__main__":
    raise SystemExit(main())
