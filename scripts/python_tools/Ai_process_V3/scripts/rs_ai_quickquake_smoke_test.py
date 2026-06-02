#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    root = Path("/tmp/rs_ai_quickquake_smoke")
    out = Path("/tmp/rs_ai_quickquake_yolo_smoke")
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    (root / "intact").mkdir(parents=True)
    (root / "damaged").mkdir(parents=True)

    for idx in range(4):
        image = np.full((96, 96, 3), 70 + idx * 10, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (76, 76), (110, 110, 150), -1)
        cv2.imwrite(str(root / "intact" / f"intact_{idx}_opt.png"), image)
    for idx in range(4):
        image = np.full((96, 96, 3), 80 + idx * 10, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (76, 76), (80, 80, 130), -1)
        cv2.line(image, (24, 24), (72, 72), (210, 210, 230), 4)
        cv2.imwrite(str(root / "damaged" / f"damaged_{idx}_opt.png"), image)

    cmd = [
        sys.executable,
        "scripts/rs_ai_convert_quickquakebuildings_to_yolo.py",
        "--qqb-root",
        str(root),
        "--output-root",
        str(out),
        "--mode",
        "opt",
    ]
    subprocess.run(cmd, check=True)

    yaml_path = out / "quickquakebuildings_damage.yaml"
    labels = sorted((out / "labels").rglob("*.txt"))
    if not yaml_path.exists() or len(labels) != 8:
        raise AssertionError(f"unexpected conversion output yaml={yaml_path.exists()} labels={len(labels)}")
    classes = {line.split()[0] for label in labels for line in label.read_text(encoding="utf-8").splitlines() if line}
    if classes != {"0", "1"}:
        raise AssertionError(f"unexpected classes: {classes}")
    print("[RS-AI] QuickQuakeBuildings converter smoke test passed")
    print(f"[RS-AI] yaml={yaml_path}")
    print(f"[RS-AI] labels={len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
