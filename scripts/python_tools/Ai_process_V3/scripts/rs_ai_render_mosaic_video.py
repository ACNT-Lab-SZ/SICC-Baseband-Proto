#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render sorted image patches into a mosaic MP4 video.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--cell", type=int, default=128)
    parser.add_argument("--gap", type=int, default=10)
    parser.add_argument("--title", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    frames = sorted(p for p in input_dir.glob(args.pattern) if p.suffix.lower() in IMAGE_EXTS)
    if not frames:
        raise FileNotFoundError(f"No image frames found under {input_dir} with pattern {args.pattern}")

    per_page = max(1, int(args.rows) * int(args.cols))
    pages = [frames[i : i + per_page] for i in range(0, len(frames), per_page)]
    if args.max_frames > 0:
        pages = pages[: args.max_frames]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (int(args.width), int(args.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output}")

    for page_idx, page in enumerate(pages):
        canvas = np.full((args.height, args.width, 3), 24, dtype=np.uint8)
        _draw_header(canvas, args.title, page_idx + 1, len(pages), len(frames))
        _draw_grid(canvas, page, args.rows, args.cols, args.cell, args.gap)
        writer.write(canvas)

    writer.release()
    print(f"[RS-AI] mosaic_video={output}")
    print(f"[RS-AI] source_images={len(frames)} pages={len(pages)} fps={args.fps} size={args.width}x{args.height}")
    return 0


def _draw_header(canvas: np.ndarray, title: str, page: int, pages: int, image_count: int) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (12, 12, 12), -1)
    label = f"{title}  page {page:03d}/{pages:03d}  patches={image_count}" if title else f"page {page:03d}/{pages:03d}  patches={image_count}"
    cv2.putText(canvas, label, (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)


def _draw_grid(canvas: np.ndarray, paths: list[Path], rows: int, cols: int, cell: int, gap: int) -> None:
    grid_w = cols * cell + (cols - 1) * gap
    grid_h = rows * cell + (rows - 1) * gap
    x_start = max(0, (canvas.shape[1] - grid_w) // 2)
    y_start = max(74, 74 + (canvas.shape[0] - 74 - grid_h) // 2)

    for idx, path in enumerate(paths):
        row = idx // cols
        col = idx % cols
        x = x_start + col * (cell + gap)
        y = y_start + row * (cell + gap)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        tile = _fit_tile(image, cell)
        canvas[y : y + cell, x : x + cell] = tile
        cv2.rectangle(canvas, (x, y), (x + cell - 1, y + cell - 1), (95, 95, 95), 1)
        _draw_tile_label(canvas, path.stem, x, y, cell)


def _fit_tile(image: np.ndarray, cell: int) -> np.ndarray:
    ih, iw = image.shape[:2]
    scale = min(cell / max(1, iw), cell / max(1, ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    interp = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    tile = np.full((cell, cell, 3), 36, dtype=np.uint8)
    x0 = (cell - nw) // 2
    y0 = (cell - nh) // 2
    tile[y0 : y0 + nh, x0 : x0 + nw] = resized
    return tile


def _draw_tile_label(canvas: np.ndarray, label: str, x: int, y: int, cell: int) -> None:
    short = label[:24]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y + cell - 23), (x + cell, y + cell), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
    cv2.putText(canvas, short, (x + 5, y + cell - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
