#!/usr/bin/env python3
"""Capture constellation/spectrum/metrics UDP packets from uhd_ldpc_ofdm_link."""

import argparse
import csv
import socket
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--constellation-port", type=int, default=65432)
    p.add_argument("--spectrum-port", type=int, default=65433)
    p.add_argument("--metrics-port", type=int, default=65434)
    return p.parse_args()


def recv_loop(host, port, stop, packets):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.settimeout(0.2)
    while not stop.is_set():
        try:
            data, _ = sock.recvfrom(1048576)
        except socket.timeout:
            continue
        packets.append((time.time(), data))
    sock.close()


def save_constellation(path, packets):
    if not packets:
        return None
    _, data = packets[-1]
    arr = np.frombuffer(data, dtype=np.float32)
    if arr.size == 0 or arr.size % 4 != 0:
        return None
    n = arr.size // 4
    tx_i, tx_q = arr[:n], arr[n : 2 * n]
    rx_i, rx_q = arr[2 * n : 3 * n], arr[3 * n : 4 * n]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    axes[0].scatter(tx_i, tx_q, s=4, c="#f59e0b", alpha=0.75)
    axes[1].scatter(rx_i, rx_q, s=4, c="#06b6d4", alpha=0.75)
    for ax, title in zip(axes, ["TX constellation", "RX equalized constellation"]):
        ax.set_title(title)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
    fig.suptitle(f"Last received constellation frame, points={n}")
    out = path / "constellation_last.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def save_spectrum(path, packets):
    if not packets:
        return None
    _, data = packets[-1]
    arr = np.frombuffer(data, dtype=np.float32)
    if arr.size == 0 or arr.size % 2 != 0:
        return None
    n = arr.size // 2
    freq = arr[:n] / 1e6
    psd = arr[n:]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(freq, psd, color="#2563eb", linewidth=1.2)
    ax.set_title("RF/baseband Welch PSD")
    ax.set_xlabel("Frequency offset (MHz)")
    ax.set_ylabel("Relative power (dB)")
    ax.grid(True, alpha=0.25)
    out = path / "spectrum_last.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def save_metrics(path, packets):
    out = path / "metrics_udp.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "frames", "ber", "fer", "preber", "snr_db", "goodput_mbps"])
        for ts, data in packets:
            arr = np.frombuffer(data, dtype=np.float64)
            if arr.size >= 6:
                writer.writerow([ts, *arr[:6]])
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    constellation = []
    spectrum = []
    metrics = []

    threads = [
        threading.Thread(target=recv_loop, args=(args.host, args.constellation_port, stop, constellation), daemon=True),
        threading.Thread(target=recv_loop, args=(args.host, args.spectrum_port, stop, spectrum), daemon=True),
        threading.Thread(target=recv_loop, args=(args.host, args.metrics_port, stop, metrics), daemon=True),
    ]
    for t in threads:
        t.start()

    time.sleep(args.duration)
    stop.set()
    for t in threads:
        t.join(timeout=1.0)

    const_path = save_constellation(out_dir, constellation)
    spec_path = save_spectrum(out_dir, spectrum)
    metrics_path = save_metrics(out_dir, metrics)

    summary = out_dir / "capture_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"constellation_packets={len(constellation)}",
                f"spectrum_packets={len(spectrum)}",
                f"metrics_packets={len(metrics)}",
                f"constellation_png={const_path or ''}",
                f"spectrum_png={spec_path or ''}",
                f"metrics_csv={metrics_path}",
            ]
        ),
        encoding="utf-8",
    )
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
