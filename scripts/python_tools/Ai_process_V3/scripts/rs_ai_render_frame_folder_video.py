#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a sorted image folder into a fixed-size MP4 video.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    frames = sorted(p for p in input_dir.glob(args.pattern) if p.suffix.lower() in IMAGE_EXTS)
    if not frames:
        raise FileNotFoundError(f"No image frames found under {input_dir} with pattern {args.pattern}")
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output}")

    for idx, frame_path in enumerate(frames):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {frame_path}")
        canvas = _letterbox(image, args.width, args.height)
        label = f"{idx + 1:03d}/{len(frames):03d}  {frame_path.stem}"
        _draw_caption(canvas, label, args.title)
        writer.write(canvas)
    writer.release()
    print(f"[RS-AI] video={output}")
    print(f"[RS-AI] frames={len(frames)} fps={args.fps} size={args.width}x{args.height}")
    return 0


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    ih, iw = image.shape[:2]
    scale = min(width / iw, height / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _draw_caption(canvas: np.ndarray, label: str, title: str) -> None:
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 56), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0.0, dst=canvas)
    text = f"{title}  {label}" if title else label
    cv2.putText(canvas, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
