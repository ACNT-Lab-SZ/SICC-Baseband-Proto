#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    root = Path("/tmp/rs_ai_sen1_smoke")
    out = Path("/tmp/rs_ai_sen1_yolo_smoke")
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    (root / "LabelHand").mkdir(parents=True)
    (root / "S2Hand").mkdir(parents=True)

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 12:36] = 255
    Image.fromarray(mask).save(root / "LabelHand" / "demo.png")

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 1] = 80
    image[10:30, 12:36] = [40, 80, 200]
    Image.fromarray(image).save(root / "S2Hand" / "demo.png")

    cmd = [
        sys.executable,
        "scripts/rs_ai_convert_sen1floods11_to_yolo.py",
        "--sen1-root",
        str(root),
        "--output-root",
        str(out),
        "--sensor",
        "s2",
        "--min-area",
        "10",
    ]
    subprocess.run(cmd, check=True)

    expected = [
        out / "sen1floods11_flood.yaml",
        out / "images" / "train" / "demo.png",
        out / "labels" / "train" / "demo.txt",
        out / "masks" / "train" / "demo.png",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    label = (out / "labels" / "train" / "demo.txt").read_text(encoding="utf-8").strip()
    if not label.startswith("0 "):
        raise AssertionError(f"unexpected label: {label}")
    print("[RS-AI] Sen1Floods11 converter smoke test passed")
    print(label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
