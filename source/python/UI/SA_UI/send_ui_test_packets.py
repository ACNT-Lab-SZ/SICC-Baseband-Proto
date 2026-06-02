import argparse
import math
import os
import socket
import sys
import time

import numpy as np


HOST = "127.0.0.1"
IQ_PORT = 65432
SPECTRUM_PORT = 65433
VIDEO_PORT = 65434
METRIC_PORT = 65435


def make_constellation(rng, modulation, count, snr_db):
    if modulation == "qpsk":
        idx = rng.integers(0, 4, count)
        phase = idx * (math.pi / 2.0) + math.pi / 4.0
        tx = np.exp(1j * phase)
    else:
        order = 16 if modulation == "16qam" else 64
        side = int(math.sqrt(order))
        levels = np.arange(-(side - 1), side + 1, 2, dtype=np.float32)
        i = rng.choice(levels, count)
        q = rng.choice(levels, count)
        tx = i + 1j * q
        tx = tx / np.sqrt(np.mean(np.abs(tx) ** 2))

    noise_power = 10.0 ** (-snr_db / 10.0)
    noise = (rng.normal(size=count) + 1j * rng.normal(size=count)) * math.sqrt(noise_power / 2.0)
    rx = tx + noise
    return np.concatenate((tx.real, tx.imag, rx.real, rx.imag)).astype("<f4")


def make_spectrum(rng, frame, bins=512, rate=12.5e6, nfft=1024, active_sc=768, snr_db=5.0):
    span_hz = rate
    freqs = np.linspace(-span_hz / 2.0, span_hz / 2.0, bins, dtype=np.float32)
    occupied_bw = rate * (active_sc / max(nfft, 1))
    edge = occupied_bw / 2.0
    transition = max(rate / max(nfft, 1) * 4.0, 1.0)
    flat_top = 0.5 * (1.0 - np.tanh((np.abs(freqs) - edge) / transition))
    ripple = 1.2 * np.cos(2.0 * np.pi * freqs / max(occupied_bw, 1.0) * 7.0 + frame * 0.08)
    noise_floor = -42.0 + rng.normal(0.0, 0.7, bins)
    signal_level = noise_floor + snr_db + 12.0 + ripple
    powers = noise_floor * (1.0 - flat_top) + signal_level * flat_top
    powers += rng.normal(0.0, 0.8, bins)
    return np.concatenate((freqs, powers.astype(np.float32))).astype("<f4")


def planar_rgb(frame):
    h, w = 84, 112
    y, x = np.mgrid[0:h, 0:w]
    tx = np.zeros((h, w, 3), dtype=np.uint8)
    tx[..., 0] = (x * 2 + frame * 3) % 256
    tx[..., 1] = (y * 3 + 80) % 256
    tx[..., 2] = ((x + y) * 2 + 40) % 256

    rx = tx.copy()
    bar = (frame * 3) % w
    rx[:, max(0, bar - 2):min(w, bar + 2), :] = np.array([0, 229, 255], dtype=np.uint8)
    rx = np.clip(rx.astype(np.int16) + 8 * np.sin((x + frame) / 8.0)[..., None], 0, 255).astype(np.uint8)

    def pack(img):
        return np.concatenate((img[..., 0].ravel(), img[..., 1].ravel(), img[..., 2].ravel())).astype(np.uint8)

    return bytes(np.concatenate((pack(tx), pack(rx))))


def pack_rgb_pair(tx, rx):
    def pack(img):
        return np.concatenate((img[..., 0].ravel(), img[..., 1].ravel(), img[..., 2].ravel())).astype(np.uint8)

    return bytes(np.concatenate((pack(tx), pack(rx))))


def open_video_reader(path):
    deps = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".deps"))
    if os.path.isdir(deps) and deps not in sys.path:
        sys.path.insert(0, deps)
    import imageio.v2 as imageio

    return imageio.get_reader(path)


def make_video_file_payload(reader, frame_index):
    from PIL import Image

    try:
        frame = reader.get_data(frame_index)
    except IndexError:
        frame = reader.get_data(0)

    if frame.ndim == 2:
        frame = np.stack((frame, frame, frame), axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]

    img = Image.fromarray(frame.astype(np.uint8), "RGB").resize((112, 84), Image.Resampling.BILINEAR)
    tx = np.asarray(img, dtype=np.uint8)
    rx = tx.copy()
    return pack_rgb_pair(tx, rx)


def main():
    parser = argparse.ArgumentParser(description="Send synthetic UDP data to satellite_UI_demo1.py")
    parser.add_argument("--duration-sec", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--points", type=int, default=1200)
    parser.add_argument("--video-file", default="", help="Optional MP4/image video source for the UI video panes.")
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=12.5e6)
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--active-sc", type=int, default=768)
    args = parser.parse_args()

    rng = np.random.default_rng(20260510)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    profiles = (("qpsk", 12.0, 3.0e-3), ("16qam", 18.0, 6.0e-4), ("64qam", 24.0, 1.2e-4))
    period = 1.0 / max(args.fps, 1.0)
    end_time = time.perf_counter() + args.duration_sec
    frame = 0
    video_reader = open_video_reader(args.video_file) if args.video_file else None

    while time.perf_counter() < end_time:
        modulation, snr_db, base_ber = profiles[(frame // int(args.fps * 2)) % len(profiles)]
        sock.sendto(make_constellation(rng, modulation, args.points, snr_db).tobytes(), (HOST, IQ_PORT))
        sock.sendto(make_spectrum(rng, frame, rate=args.rate, nfft=args.nfft, active_sc=args.active_sc, snr_db=args.snr_db).tobytes(), (HOST, SPECTRUM_PORT))
        if video_reader is not None:
            sock.sendto(make_video_file_payload(video_reader, frame), (HOST, VIDEO_PORT))
        else:
            sock.sendto(planar_rgb(frame), (HOST, VIDEO_PORT))
        ber = max(1e-8, base_ber * (0.65 + 0.35 * math.exp(-frame / 80.0)) * (1.0 + 0.08 * math.sin(frame / 7.0)))
        metric = np.array([[frame, ber]], dtype="<f8")
        sock.sendto(metric.tobytes(), (HOST, METRIC_PORT))
        frame += 1
        time.sleep(period)

    if video_reader is not None:
        video_reader.close()
    print(f"sent_frames={frame} ports={IQ_PORT},{SPECTRUM_PORT},{VIDEO_PORT},{METRIC_PORT}")


if __name__ == "__main__":
    main()
