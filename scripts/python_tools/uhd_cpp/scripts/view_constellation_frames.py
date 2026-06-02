#!/usr/bin/env python3
"""Frame-by-frame constellation viewer for uhd_ldpc_ofdm_link UDP packets."""

import argparse
import socket
import time

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Display TX/RX constellation snapshots, one UDP packet per frame.")
    parser.add_argument("--host", default="127.0.0.1", help="UDP bind host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=65432, help="UDP bind port. Default: 65432")
    parser.add_argument("--axis", type=float, default=2.0, help="Plot axis half-width. Default: 2.0")
    parser.add_argument("--max-points", type=int, default=4000, help="Max displayed points per frame. Default: 4000")
    parser.add_argument("--hold-ms", type=float, default=1.0, help="Matplotlib pause after each packet. Default: 1 ms")
    parser.add_argument("--save-prefix", default="", help="Optional image prefix, e.g. frames/constellation")
    parser.add_argument("--save-every", type=int, default=0, help="Save every Nth received frame when --save-prefix is set")
    return parser.parse_args()


def main():
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.settimeout(0.5)
    print(f"[CONST] listening on {args.host}:{args.port}")
    print("[CONST] expected payload: float32[tx_i, tx_q, rx_i, rx_q]")

    plt.ion()
    fig, (ax_tx, ax_rx) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    tx_scatter = ax_tx.scatter([], [], s=4, c="#fbbf24", alpha=0.75)
    rx_scatter = ax_rx.scatter([], [], s=4, c="#00e5ff", alpha=0.75)

    for ax, title in ((ax_tx, "TX constellation"), (ax_rx, "RX equalized constellation")):
        ax.set_title(title)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_xlim(-args.axis, args.axis)
        ax.set_ylim(-args.axis, args.axis)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    frame = 0
    t0 = time.time()
    while True:
        try:
            data, _ = sock.recvfrom(262144)
        except socket.timeout:
            plt.pause(0.01)
            continue

        arr = np.frombuffer(data, dtype=np.float32)
        if arr.size == 0 or arr.size % 4 != 0:
            print(f"[CONST] drop malformed packet: {len(data)} bytes")
            continue

        n = arr.size // 4
        if n > args.max_points:
            step = int(np.ceil(n / args.max_points))
            idx = np.arange(0, n, step)[: args.max_points]
            tx_i = arr[0:n][idx]
            tx_q = arr[n : 2 * n][idx]
            rx_i = arr[2 * n : 3 * n][idx]
            rx_q = arr[3 * n : 4 * n][idx]
            shown = idx.size
        else:
            tx_i = arr[0:n]
            tx_q = arr[n : 2 * n]
            rx_i = arr[2 * n : 3 * n]
            rx_q = arr[3 * n : 4 * n]
            shown = n

        tx_scatter.set_offsets(np.column_stack((tx_i, tx_q)))
        rx_scatter.set_offsets(np.column_stack((rx_i, rx_q)))
        elapsed = max(time.time() - t0, 1.0e-6)
        fig.suptitle(f"Constellation frame {frame} | points {shown}/{n} | UI fps {frame / elapsed:.1f}")
        fig.canvas.draw_idle()

        if args.save_prefix and args.save_every > 0 and frame % args.save_every == 0:
            fig.savefig(f"{args.save_prefix}_{frame:06d}.png", dpi=140)

        frame += 1
        plt.pause(max(args.hold_ms, 0.0) / 1000.0)


if __name__ == "__main__":
    main()
