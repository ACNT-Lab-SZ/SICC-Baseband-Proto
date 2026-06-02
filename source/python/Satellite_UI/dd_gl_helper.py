import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QVector3D
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

import pyqtgraph.opengl as gl


APP_DIR = Path(__file__).resolve().parent


class DdRenderWidget(QWidget):
    def __init__(self, output_path: Path, port: int = 65437):
        super().__init__()
        self.output_path = output_path
        self.port = port
        self.pending_grid = None
        self.last_save = 0.0
        self.dirty = True
        self.current_shape = (32, 32)
        self.center_x = 0.0
        self.center_y = 0.0
        self.center_z = 2.8

        self.setWindowTitle("DD GL Helper")
        self.setWindowFlag(Qt.Tool, True)
        self.setStyleSheet("background-color: #080B14;")
        self.resize(560, 360)
        self.move(-2200, 120)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel("3D Delay-Doppler Signal Intensity")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#38BDF8; font: 700 14px 'Microsoft YaHei'; background:#080B14;")
        layout.addWidget(title)

        legend = QLabel("X Doppler index    Y Delay index    Z Normalized intensity")
        legend.setAlignment(Qt.AlignCenter)
        legend.setStyleSheet("color:#CBD5E1; font: 600 11px Consolas; background:#080B14;")
        layout.addWidget(legend)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("#080B14")
        self.view.opts["distance"] = 92
        self.view.opts["elevation"] = 18
        self.view.opts["azimuth"] = -38
        self.view.opts["fov"] = 34
        self.view.opts["center"] = QVector3D(self.center_x, self.center_y, self.center_z)
        layout.addWidget(self.view, stretch=1)

        self.grid_item = gl.GLGridItem()
        self.grid_item.setSize(x=32, y=32, z=1)
        self.grid_item.setSpacing(x=4, y=4, z=1)
        self.grid_item.translate(0, 0, -0.08)
        self.view.addItem(self.grid_item)

        self.surface = None
        self.axis_items = []
        self.rebuild_scene(32, 32)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", self.port))
        self.sock.setblocking(False)

        self.recv_timer = QTimer(self)
        self.recv_timer.timeout.connect(self.poll_socket)
        self.recv_timer.start(25)

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.flush_surface)
        self.render_timer.start(65)

        self.save_timer = QTimer(self)
        self.save_timer.timeout.connect(self.save_frame)
        self.save_timer.start(90)

        self.set_surface(self.demo_grid())

    def surface_colors(self, z_values):
        stops = np.array([
            [0.04, 0.05, 0.26, 1.0],
            [0.00, 0.43, 0.72, 1.0],
            [0.00, 0.82, 0.70, 1.0],
            [0.96, 0.86, 0.18, 1.0],
            [1.00, 0.23, 0.12, 1.0],
        ], dtype=float)
        positions = np.array([0.0, 0.28, 0.58, 0.82, 1.0], dtype=float)
        values = np.clip(z_values, 0.0, 1.0)
        colors = np.empty(values.shape + (4,), dtype=float)
        for channel in range(4):
            colors[..., channel] = np.interp(values, positions, stops[:, channel])
        return colors

    def demo_grid(self):
        rows = cols = 32
        phase = time.time() * 0.9
        grid = np.zeros((rows, cols), dtype=float)
        for r in range(rows):
            for c in range(cols):
                p1 = np.exp(-((r - 11 - np.sin(phase) * 2.8) ** 2 + (c - 16) ** 2) / 9.0)
                p2 = np.exp(-((r - 23) ** 2 + (c - 8 - np.cos(phase) * 2.8) ** 2) / 16.0)
                grid[r, c] = min(1.0, 0.10 + 0.56 * p1 + 0.34 * p2 + 0.04 * np.sin(r * 0.8 + c * 0.45))
        return grid

    def poll_socket(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(262144)
            except BlockingIOError:
                break
            except Exception:
                break
            try:
                payload = json.loads(data.decode("utf-8", errors="ignore"))
                payload_type = payload.get("type", "grid")
                if payload_type == "view":
                    self.apply_view(payload)
                else:
                    grid = np.asarray(payload.get("grid", []), dtype=float)
                    if grid.ndim == 2 and grid.size:
                        self.set_surface(grid)
            except Exception:
                continue

    def apply_view(self, payload):
        distance = float(payload.get("distance", self.view.opts["distance"]))
        elevation = float(payload.get("elevation", self.view.opts["elevation"]))
        azimuth = float(payload.get("azimuth", self.view.opts["azimuth"]))
        self.view.opts["distance"] = max(24.0, min(120.0, distance))
        self.view.opts["elevation"] = max(5.0, min(60.0, elevation))
        self.view.opts["azimuth"] = azimuth
        center_x = float(payload.get("center_x", self.center_x))
        center_y = float(payload.get("center_y", self.center_y))
        center_z = float(payload.get("center_z", self.center_z))
        self.center_x = max(-10.0, min(10.0, center_x))
        self.center_y = max(-10.0, min(10.0, center_y))
        self.center_z = max(-2.0, min(10.0, center_z))
        self.view.opts["center"] = QVector3D(self.center_x, self.center_y, self.center_z)
        self.view.update()
        self.dirty = True

    def set_surface(self, grid):
        arr = np.asarray(grid, dtype=float)
        if arr.ndim != 2 or not arr.size:
            return
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        lo, hi = np.percentile(arr, [5, 99.5])
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        self.pending_grid = np.clip(arr, 0.0, 1.0)
        self.dirty = True

    def rebuild_scene(self, rows, cols):
        if self.surface is not None:
            self.view.removeItem(self.surface)
        for item in self.axis_items:
            self.view.removeItem(item)
        self.axis_items = []

        x = np.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols)
        y = np.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows)
        z = np.zeros((cols, rows), dtype=float)
        self.surface = gl.GLSurfacePlotItem(
            x=x,
            y=y,
            z=z,
            colors=self.surface_colors(z),
            shader="shaded",
            smooth=True,
            computeNormals=True,
        )
        self.surface.setGLOptions("opaque")
        self.view.addItem(self.surface)
        self.add_axes(rows, cols)
        self.current_shape = (rows, cols)

    def add_axes(self, rows, cols):
        x0, x1 = -(cols - 1) / 2, (cols - 1) / 2
        y0, y1 = -(rows - 1) / 2, (rows - 1) / 2
        z1 = 11.0
        origin = np.asarray([[x0, y0, 0], [x1, y0, 0]], dtype=float)
        x_line = gl.GLLinePlotItem(pos=origin, color=(0.22, 0.74, 0.97, 1.0), width=2.0, antialias=True)
        self.view.addItem(x_line)
        self.axis_items.append(x_line)
        origin = np.asarray([[x0, y0, 0], [x0, y1, 0]], dtype=float)
        y_line = gl.GLLinePlotItem(pos=origin, color=(0.29, 0.87, 0.50, 1.0), width=2.0, antialias=True)
        self.view.addItem(y_line)
        self.axis_items.append(y_line)
        origin = np.asarray([[x0, y0, 0], [x0, y0, z1]], dtype=float)
        z_line = gl.GLLinePlotItem(pos=origin, color=(0.98, 0.75, 0.14, 1.0), width=2.0, antialias=True)
        self.view.addItem(z_line)
        self.axis_items.append(z_line)

    def flush_surface(self):
        if self.pending_grid is None:
            return
        grid = self.pending_grid
        self.pending_grid = None
        rows, cols = grid.shape
        if (rows, cols) != self.current_shape:
            self.rebuild_scene(rows, cols)
        z = grid.T
        self.surface.setData(z=z * 6.6, colors=self.surface_colors(z))
        self.dirty = True

    def save_frame(self):
        if not self.dirty or time.monotonic() - self.last_save < 0.08:
            return
        try:
            img = self.view.grabFramebuffer()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(self.output_path))
            self.last_save = time.monotonic()
            self.dirty = False
        except Exception:
            pass


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else APP_DIR / "runtime" / "dd_surface_frame.png"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 65437
    app = QApplication(sys.argv)
    widget = DdRenderWidget(output, port)
    widget.show()
    sys.exit(app.exec())
