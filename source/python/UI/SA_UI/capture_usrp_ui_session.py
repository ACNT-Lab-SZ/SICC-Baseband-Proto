import argparse
import os
import socket
import sys
import threading
import time

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from satellite_UI_demo1 import SatelliteSystemUI


VIDEO_PORT = 65434


def pack_rgb_pair(tx, rx):
    def pack(img):
        return np.concatenate((img[..., 0].ravel(), img[..., 1].ravel(), img[..., 2].ravel()))

    return bytes(np.concatenate((pack(tx), pack(rx))).astype(np.uint8))


def open_video_reader(path):
    deps = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".deps"))
    if os.path.isdir(deps) and deps not in sys.path:
        sys.path.insert(0, deps)
    import imageio.v2 as imageio

    return imageio.get_reader(path)


def make_video_file_payload(reader, frame_idx):
    from PIL import Image

    try:
        frame = reader.get_data(frame_idx)
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


def make_video_payload(frame_idx):
    h, w = 84, 112
    y, x = np.mgrid[0:h, 0:w]
    tx = np.zeros((h, w, 3), dtype=np.uint8)
    tx[..., 0] = (x * 2 + frame_idx * 4) % 256
    tx[..., 1] = (y * 3 + 60) % 256
    tx[..., 2] = ((x + y) * 2 + 30) % 256

    rx = tx.copy()
    edge = (frame_idx * 2) % w
    rx[:, max(0, edge - 1):min(w, edge + 2), :] = np.array([0, 229, 255], dtype=np.uint8)
    rx = np.clip(rx.astype(np.int16) + 6 * np.cos((x + frame_idx) / 7.0)[..., None], 0, 255).astype(np.uint8)

    return pack_rgb_pair(tx, rx)


def video_preview(stop_event, fps, video_file):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / max(fps, 1.0)
    frame = 0
    reader = open_video_reader(video_file) if video_file else None
    while not stop_event.is_set():
        if reader is not None:
            sock.sendto(make_video_file_payload(reader, frame), ("127.0.0.1", VIDEO_PORT))
        else:
            sock.sendto(make_video_payload(frame), ("127.0.0.1", VIDEO_PORT))
        frame += 1
        time.sleep(period)
    if reader is not None:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description="Run the satellite UI during a real USRP session and capture it.")
    parser.add_argument("--screenshot", required=True)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--video-file", default="")
    args = parser.parse_args()

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = SatelliteSystemUI()
    screen = QApplication.primaryScreen().geometry()
    w = int(screen.width() * 0.8)
    h = int(screen.height() * 0.85)
    x = (screen.width() - w) // 2
    y = max(40, (screen.height() - h) // 2)
    window.setGeometry(x, y, w, h)
    window.show()

    stop_event = threading.Event()
    thread = threading.Thread(target=video_preview, args=(stop_event, args.video_fps, args.video_file), daemon=True)
    thread.start()

    def capture_and_exit():
        stop_event.set()
        pixmap = window.grab()
        pixmap.save(args.screenshot, "PNG")
        window.close()
        app.quit()

    QTimer.singleShot(int(args.duration_sec * 1000), capture_and_exit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
