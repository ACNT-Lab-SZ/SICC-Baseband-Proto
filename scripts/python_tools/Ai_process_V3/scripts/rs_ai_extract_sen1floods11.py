#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract the Sen1Floods11 Zenodo .7z archive with py7zr.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    archive = Path(args.archive)
    output = Path(args.output)
    if not archive.exists():
        raise FileNotFoundError(archive)
    output.mkdir(parents=True, exist_ok=True)

    import py7zr

    with py7zr.SevenZipFile(archive, mode="r") as zf:
        zf.extractall(path=output)
    print(f"[RS-AI] extracted={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
