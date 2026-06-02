import argparse
import sys
import threading

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

import send_ui_test_packets
from satellite_UI_demo1 import SatelliteSystemUI


def main():
    parser = argparse.ArgumentParser(description="Run the satellite UI, feed UDP demo data, and capture the UI widget.")
    parser.add_argument("--screenshot", required=True)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--points", type=int, default=1200)
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

    def feed_packets():
        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "send_ui_test_packets.py",
                "--duration-sec", str(args.duration_sec),
                "--fps", str(args.fps),
                "--points", str(args.points),
            ]
            send_ui_test_packets.main()
        finally:
            sys.argv = old_argv

    def capture_and_exit():
        pixmap = window.grab()
        pixmap.save(args.screenshot, "PNG")
        window.close()
        app.quit()

    QTimer.singleShot(1000, lambda: threading.Thread(target=feed_packets, daemon=True).start())
    QTimer.singleShot(int((args.duration_sec + 2.5) * 1000), capture_and_exit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
