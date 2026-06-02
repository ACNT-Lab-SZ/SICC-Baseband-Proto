import json
import math
import socket
import time

import numpy as np


HOST = "127.0.0.1"
PORT = 65436


def make_dd_grid(frame: int, rows: int = 32, cols: int = 32):
    y, x = np.mgrid[0:rows, 0:cols]
    phase = frame * 0.08
    grid = 0.08 * np.sin(0.75 * x + phase) + 0.06 * np.cos(0.65 * y - phase)
    centers = [
        (9 + 2.5 * math.sin(phase), 11 + 2.0 * math.cos(phase * 0.7), 0.85),
        (21 + 2.0 * math.cos(phase * 0.6), 18 + 2.5 * math.sin(phase), 0.55),
        (15 + 1.5 * math.sin(phase * 1.3), 25, 0.38),
    ]
    for cy, cx, amp in centers:
        grid += amp * np.exp(-(((x - cx) ** 2) + ((y - cy) ** 2)) / 12.0)
    grid = grid - grid.min()
    grid = grid / (grid.max() + 1e-12)
    return grid.tolist()


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    total_frames = 1000
    input_window = 96
    target_window = 24
    stride = 12

    print(f"[demo] sending metrics to UDP {HOST}:{PORT}. Press Ctrl+C to stop.")
    frame = 1
    while True:
        snr = 12.0 + 4.0 * math.sin(frame * 0.035)
        mcs = max(1, min(8, int(round((snr - 3.0) / 2.0))))
        throughput = max(0.5, 4.8 * mcs + 1.2 * math.sin(frame * 0.10))
        sample_count = max(0, math.floor((frame - input_window - target_window) / stride) + 1)

        clean = np.array([0.78, 0.43, 0.26]) * (1.0 + 0.12 * np.sin(frame * 0.05 + np.arange(3)))
        noisy = clean + 0.08 * np.sin(frame * 0.13 + np.arange(3) * 1.7)
        noisy = np.maximum(noisy, 0.02)

        payload = {
            "type": "ntn_channel_subplots",
            "profile": "NTN-TDL-C",
            "frame": frame,
            "totalFrames": total_frames,
            "ddGrid": make_dd_grid(frame),
            "cleanPathMagnitude": clean.tolist(),
            "noisyPathMagnitude": noisy.tolist(),
            "inputWindow": input_window,
            "targetWindow": target_window,
            "stride": stride,
            "sampleCount": sample_count,
            "throughputMbps": throughput,
            "mcsIndex": mcs,
        }
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        sock.sendto(data, (HOST, PORT))
        if frame % 25 == 1:
            print(f"[demo] frame={frame:04d} throughput={throughput:.2f} Mbps mcs={mcs}")
        frame = 1 if frame >= total_frames else frame + 1
        time.sleep(0.08)


if __name__ == "__main__":
    main()

