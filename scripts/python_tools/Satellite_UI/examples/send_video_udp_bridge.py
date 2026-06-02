#!/usr/bin/env python3
"""Stream mp4/avi frames to Satellite_UI video panels over UDP.

Satellite_UI currently expects raw planar RGB datagrams on UDP 127.0.0.1:65434:
  pair mode:   TX 84x112 RGB planes + RX 84x112 RGB planes = 56448 bytes
  single mode: RX 120x160 RGB planes                         = 57600 bytes
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")


PAIR_W = 112
PAIR_H = 84
SINGLE_W = 160
SINGLE_H = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send UI-ready raw RGB video frames to UDP 127.0.0.1:65434."
    )
    parser.add_argument("--video", default="", help="Use the same video for TX and RX panels.")
    parser.add_argument("--tx-video", default="", help="Video shown in the TX panel.")
    parser.add_argument("--rx-video", default="", help="Video shown in the RX panel.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65434)
    parser.add_argument("--mode", choices=("pair", "single"), default="pair")
    parser.add_argument("--fps", type=float, default=0.0, help="Override playback FPS. 0 uses source FPS.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means no limit.")
    parser.add_argument("--loop", action="store_true", help="Loop input video when it reaches EOF.")
    parser.add_argument("--progress-interval", type=int, default=50)
    return parser.parse_args()


def require_video_path(path_text: str, name: str) -> Path:
    if not path_text:
        raise SystemExit(f"{name} is required. Use --video or --{name}.")
    path = Path(path_text)
    if not path.exists():
        raise SystemExit(f"{name} not found: {path}")
    return path


def open_capture(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"failed to open video: {path}")
    return cap


def read_rgb_frame(cap, path: Path, *, loop: bool, size: tuple[int, int]) -> np.ndarray | None:
    import cv2

    ok, frame_bgr = cap.read()
    if not ok:
        if not loop:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame_bgr = cap.read()
        if not ok:
            return None

    frame_bgr = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(frame_rgb, dtype=np.uint8)


def planar_rgb_bytes(frame_rgb: np.ndarray) -> bytes:
    return np.concatenate(
        (
            frame_rgb[:, :, 0].reshape(-1),
            frame_rgb[:, :, 1].reshape(-1),
            frame_rgb[:, :, 2].reshape(-1),
        )
    ).tobytes()


def source_fps(cap) -> float:
    import cv2

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        return 20.0
    return min(max(fps, 1.0), 60.0)


def main() -> int:
    args = parse_args()

    try:
        import cv2
        try:
            cv2.setLogLevel(0)
        except Exception:
            pass
    except Exception as exc:
        raise SystemExit(
            "OpenCV is required for video decoding. Install opencv-python in the Python environment "
            f"used to run this script. Original error: {exc}"
        ) from exc

    if args.video:
        args.tx_video = args.tx_video or args.video
        args.rx_video = args.rx_video or args.video

    rx_path = require_video_path(args.rx_video, "rx-video")
    rx_cap = open_capture(rx_path)
    tx_cap = None
    tx_path = None
    if args.mode == "pair":
        tx_path = require_video_path(args.tx_video, "tx-video")
        tx_cap = open_capture(tx_path)

    fps = args.fps if args.fps > 0 else source_fps(rx_cap)
    frame_period = 1.0 / max(fps, 1e-6)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, int(args.port))

    print(
        f"[UI-VIDEO] sending {args.mode} video to {args.host}:{args.port}, "
        f"fps={fps:.2f}, loop={args.loop}"
    )
    if tx_path:
        print(f"[UI-VIDEO] tx={tx_path}")
    print(f"[UI-VIDEO] rx={rx_path}")

    sent = 0
    start = time.perf_counter()
    next_deadline = start
    size = (PAIR_W, PAIR_H) if args.mode == "pair" else (SINGLE_W, SINGLE_H)

    while args.max_frames <= 0 or sent < args.max_frames:
        rx_frame = read_rgb_frame(rx_cap, rx_path, loop=args.loop, size=size)
        if rx_frame is None:
            break

        if args.mode == "pair":
            assert tx_cap is not None and tx_path is not None
            tx_frame = read_rgb_frame(tx_cap, tx_path, loop=args.loop, size=size)
            if tx_frame is None:
                break
            payload = planar_rgb_bytes(tx_frame) + planar_rgb_bytes(rx_frame)
        else:
            payload = planar_rgb_bytes(rx_frame)

        sock.sendto(payload, target)
        sent += 1

        if args.progress_interval > 0 and sent % args.progress_interval == 0:
            elapsed = max(time.perf_counter() - start, 1e-9)
            print(f"[UI-VIDEO] sent={sent} rate={sent / elapsed:.2f} fps bytes={len(payload)}")

        next_deadline += frame_period
        delay = next_deadline - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -frame_period:
            next_deadline = time.perf_counter()

    elapsed = max(time.perf_counter() - start, 1e-9)
    print(f"[UI-VIDEO] done sent={sent} elapsed={elapsed:.2f}s avg_fps={sent / elapsed:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
