import sys
import os
import socket
import json
import time
import math
import subprocess
from pathlib import Path
import numpy as np
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--ignore-gpu-blocklist --enable-webgl --use-angle=d3d11",
)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QGroupBox, QSplitter,
                               QLabel, QFormLayout, QComboBox, QLineEdit, QFrame, QTextEdit,
                               QSizePolicy, QStackedWidget, QGridLayout)
from PySide6.QtWebEngineWidgets import QWebEngineView
try:
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
except Exception:
    QWebEnginePage = None
    QWebEngineSettings = None
from PySide6.QtCore import QUrl, Qt, QTimer, QThread, Signal, QRectF, QPointF
from PySide6.QtGui import QFont, QPixmap, QPainter, QColor, QPen, QPainterPath, QImage, QPolygonF
import pyqtgraph as pg
gl = None
HAS_PG_OPENGL = False
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB_3D = True
except Exception:
    FigureCanvas = None
    Figure = None
    HAS_MATPLOTLIB_3D = False

APP_DIR = Path(__file__).resolve().parent


if QWebEnginePage is not None:
    class ConsoleWebEnginePage(QWebEnginePage):
        console_message = Signal(str)

        def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
            self.console_message.emit(f"[Constellation/Web] {message} (line {lineNumber})")
else:
    ConsoleWebEnginePage = None

# ==================== 科研风 QSS 全局样式表 ====================
ACADEMIC_STYLE = """
QMainWindow {
    background-color: #0F111A;  
}
/* 统一 Header 的背景为高级暗色，消除割裂感 */
#HeaderFrame {
    background-color: #151B2B;
    border-bottom: 1px solid #1E293B;
    min-height: 80px; 
}
#MainTitle {
    color: #FFFFFF;
    font-size: 26px;
    font-weight: 900; 
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    letter-spacing: 2px; 
}
#SubTitle {
    color: #A0AEC0;
    font-size: 14px;
    font-weight: bold;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}
QGroupBox {
    color: #63B3ED;
    font-weight: bold;
    font-size: 15px;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    border: 1px solid #3A4A69; 
    border-radius: 6px;
    margin-top: 22px; 
    background-color: #1A202C;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    top: 0px; 
    padding: 2px 8px;
    background-color: #0F111A; 
    border-radius: 4px;
}
QLineEdit, QComboBox {
    background-color: #2D3748;
    border: 1px solid #4A5568;
    border-radius: 4px;
    padding: 6px;
    color: #E2E8F0;
    font-family: "Consolas", "Roboto Mono", monospace; 
    font-size: 14px;
    font-weight: bold; 
}
QComboBox QAbstractItemView {
    background-color: #1A202C;
    color: #E2E8F0;
    selection-background-color: #2B6CB0;
    selection-color: #FFFFFF;
    border: 1px solid #4A5568;
    outline: none;
    font-weight: bold;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}
QComboBox QAbstractItemView::item {
    min-height: 32px;
    padding: 4px 10px;
}
QPushButton {
    background-color: #3182CE;
    color: white;
    font-weight: bold;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    border-radius: 5px;
    padding: 10px;
    font-size: 14px;
    letter-spacing: 1px;
}
QPushButton:hover { background-color: #4299E1; }
QPushButton:pressed { background-color: #2B6CB0; }

/* === 优化伸缩条(Splitter)，增加鼠标悬停提示 === */
QSplitter::handle {
    background-color: #2D3748;
    margin: 2px;
    border-radius: 2px;
}
QSplitter::handle:hover {
    background-color: #63B3ED;
}
"""


# ==================== 线程 1：UDP IQ 星座图接收 ====================
class IqDataReceiverThread(QThread):
    iq_data_received = Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    log_msg = Signal(str)

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 65432))
        except Exception:
            return
        while True:
            try:
                data, _ = sock.recvfrom(65536)
                if len(data) % 16 != 0 or len(data) == 0: continue
                arr = np.frombuffer(data, dtype=np.float32)
                num_points = len(arr) // 4
                tx_i, tx_q = arr[0:num_points], arr[num_points:2 * num_points]
                rx_i, rx_q = arr[2 * num_points:3 * num_points], arr[3 * num_points:4 * num_points]
                self.iq_data_received.emit(tx_i, tx_q, rx_i, rx_q)
            except Exception:
                break


# ==================== 线程 2：UDP 频谱数据接收 ====================
class SpectrumDataReceiverThread(QThread):
    spec_data_received = Signal(np.ndarray, np.ndarray)
    log_msg = Signal(str)

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 65433))
        except Exception:
            return
        while True:
            try:
                data, _ = sock.recvfrom(65536)
                if len(data) % 8 != 0 or len(data) == 0: continue
                arr = np.frombuffer(data, dtype=np.float32)
                num_points = len(arr) // 2
                freqs, powers = arr[0:num_points], arr[num_points:2 * num_points]
                self.spec_data_received.emit(freqs, powers)
            except Exception:
                break


# ==================== 线程 3：UDP 【双路】视频流接收 ====================
class VideoDataReceiverThread(QThread):
    video_frame_received = Signal(np.ndarray, np.ndarray)
    log_msg = Signal(str)

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 65434))
        except Exception:
            return
        while True:
            try:
                data, _ = sock.recvfrom(65536)
                if len(data) == 56448:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    frame_size = 84 * 112 * 3

                    tx_data = arr[0:frame_size]
                    R_tx = tx_data[0:9408].reshape(84, 112)
                    G_tx = tx_data[9408:18816].reshape(84, 112)
                    B_tx = tx_data[18816:28224].reshape(84, 112)
                    tx_rgb = np.stack((R_tx, G_tx, B_tx), axis=-1)

                    rx_data = arr[frame_size:]
                    R_rx = rx_data[0:9408].reshape(84, 112)
                    G_rx = rx_data[9408:18816].reshape(84, 112)
                    B_rx = rx_data[18816:28224].reshape(84, 112)
                    rx_rgb = np.stack((R_rx, G_rx, B_rx), axis=-1)

                    self.video_frame_received.emit(tx_rgb, rx_rgb)

                elif len(data) == 57600:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    R = arr[0:19200].reshape(120, 160)
                    G = arr[19200:38400].reshape(120, 160)
                    B = arr[38400:57600].reshape(120, 160)
                    rx_rgb = np.stack((R, G, B), axis=-1)
                    tx_dummy = np.zeros_like(rx_rgb)
                    self.video_frame_received.emit(tx_dummy, rx_rgb)
            except Exception:
                break


# ==================== 极简阵列流水灯特效 ====================
class ConstellationStateReceiverThread(QThread):
    state_received = Signal(dict)
    log_msg = Signal(str)

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 65435))
            self.log_msg.emit("[Constellation] Listening for MATLAB state on UDP 65435...")
        except Exception as exc:
            self.log_msg.emit(f"[Constellation] UDP 65435 bind failed: {exc}")
            return

        while True:
            try:
                data, _ = sock.recvfrom(262144)
                if not data:
                    continue
                text = data.decode("utf-8", errors="ignore").strip()
                start = text.find("{")
                if start < 0:
                    continue
                decoder = json.JSONDecoder()
                state, _ = decoder.raw_decode(text[start:])
                self.state_received.emit(state)
            except Exception as exc:
                self.bad_packet_count = getattr(self, "bad_packet_count", 0) + 1
                if self.bad_packet_count <= 3 or self.bad_packet_count % 20 == 0:
                    self.log_msg.emit(f"[Constellation] Ignored malformed state packet: {exc}")


class ChannelStateReceiverThread(QThread):
    channel_state_received = Signal(dict)
    log_msg = Signal(str)

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 65436))
            self.log_msg.emit("[Channel] Listening for MATLAB NTN channel data on UDP 65436...")
        except Exception as exc:
            self.log_msg.emit(f"[Channel] UDP 65436 bind failed: {exc}")
            return

        decoder = json.JSONDecoder()
        while True:
            try:
                data, _ = sock.recvfrom(262144)
                if not data:
                    continue
                text = data.decode("utf-8", errors="ignore").strip()
                start = text.find("{")
                if start < 0:
                    continue
                state, _ = decoder.raw_decode(text[start:])
                if state.get("type") in ("ntn_channel_subplots", "ntn_channel_state", "ntn_channel_vis", "ntn_tsne_frame"):
                    self.channel_state_received.emit(state)
            except Exception as exc:
                self.bad_packet_count = getattr(self, "bad_packet_count", 0) + 1
                if self.bad_packet_count <= 3 or self.bad_packet_count % 20 == 0:
                    self.log_msg.emit(f"[Channel] Ignored malformed channel packet: {exc}")


class CyberDataFlowWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(36)
        self.frame = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(150)

    def animate(self):
        self.frame += 1;
        self.update()

    def draw_chevron(self, painter, x, y, size, alpha):
        pen = QPen(QColor(74, 222, 128, alpha), 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen);
        painter.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(x, y - size / 2);
        path.lineTo(x + size / 2, y);
        path.lineTo(x, y + size / 2)
        painter.drawPath(path)

    def paintEvent(self, event):
        painter = QPainter(self);
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        y_offsets = [h * 0.35, h * 0.5, h * 0.65]
        start_x = (w - 18 - 4) / 2
        for i in range(3):
            x = start_x + i * 9
            offset = (self.frame - i) % 4
            alpha = 255 if offset == 0 else 120 if offset == 1 else 40 if offset == 2 else 10
            for y in y_offsets: self.draw_chevron(painter, x, y, 8, alpha)


# ==================== 主界面 ====================
class QtConstellationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = None
        self.rotation = -0.65
        self.pitch = -0.12
        self.earth_texture = QImage(str(APP_DIR / "web" / "assets" / "earth-blue-marble-2048.jpg"))
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background-color: #020617; border: 1px solid #334155; border-radius: 6px;")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(33)

    def set_state(self, state):
        self.state = state if isinstance(state, dict) else None
        self.update()

    def animate(self):
        self.rotation += 0.010
        self.update()

    def preview_state(self):
        phase = time.time() * 0.55
        sats = []
        for plane in range(6):
            inc = math.radians(88.9)
            raan = math.radians(plane * 30)
            for slot in range(10):
                a = phase + math.radians(slot * 36 + plane * 9)
                x0 = math.cos(a)
                y0 = math.sin(a) * math.cos(inc)
                z0 = math.sin(a) * math.sin(inc)
                x = x0 * math.cos(raan) - y0 * math.sin(raan)
                z = x0 * math.sin(raan) + y0 * math.cos(raan)
                sats.append({
                    "name": f"QF-P{plane + 1}-{slot + 1}",
                    "lat": math.degrees(math.asin(max(-1.0, min(1.0, z0)))),
                    "lon": math.degrees(math.atan2(z, x)),
                    "alt_m": 1100000,
                    "coverage_radius_deg": 20,
                })
        return {
            "type": "preview",
            "satellites": sats,
            "groundStations": [{"name": "Ground Station", "latitude": 35.6762, "longitude": 139.6503}],
            "bestLinks": [{"satellite": s["name"], "groundStation": "Ground Station", "visible": True} for s in sats[:3]],
        }

    def geo_to_xyz(self, lat, lon, radius=1.0):
        lat_r = math.radians(float(lat))
        lon_r = math.radians(float(lon)) + self.rotation
        x = radius * math.cos(lat_r) * math.cos(lon_r)
        y = radius * math.sin(lat_r)
        z = radius * math.cos(lat_r) * math.sin(lon_r)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return x, y * cp - z * sp, y * sp + z * cp

    def project(self, lat, lon, radius, cx, cy, scale):
        x, y, z = self.geo_to_xyz(lat, lon, radius)
        pscale = scale / max(0.2, 2.65 - z)
        return QPointF(cx + x * pscale, cy - y * pscale), z

    def normalize_sat(self, sat):
        return {
            "name": str(sat.get("name") or sat.get("Name") or "SAT"),
            "lat": float(sat.get("lat", sat.get("latitude", 0)) or 0),
            "lon": float(sat.get("lon", sat.get("longitude", 0)) or 0),
            "alt_m": float(sat.get("alt_m", sat.get("altitude_m", 1100000)) or 1100000),
            "coverage_radius_deg": float(sat.get("coverage_radius_deg", 18) or 18),
        }

    def active_link_sets(self, state):
        links = state.get("servingLinks") or state.get("bestLinks") or []
        if isinstance(links, dict):
            links = [links]
        active_sats, visible_links = set(), []
        for link in links:
            if not isinstance(link, dict) or not link.get("visible", False):
                continue
            sat_name = str(link.get("satellite") or link.get("satelliteName") or "")
            gs_name = str(link.get("groundStation") or link.get("groundStationName") or "")
            if sat_name:
                active_sats.add(sat_name)
            visible_links.append((sat_name, gs_name))
        return active_sats, visible_links

    def draw_earth(self, painter, cx, cy, r):
        earth_rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        painter.save()
        path = QPainterPath()
        path.addEllipse(earth_rect)
        painter.setClipPath(path)
        if not self.earth_texture.isNull():
            painter.drawImage(earth_rect, self.earth_texture)
        else:
            painter.fillPath(path, QColor("#0f3b6e"))
        painter.fillPath(path, QColor(14, 165, 233, 24))
        painter.restore()
        painter.setPen(QPen(QColor(56, 189, 248, 135), 1.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(earth_rect)
        painter.setPen(QPen(QColor(148, 163, 184, 45), 0.8))
        for frac in (-0.55, -0.25, 0.0, 0.25, 0.55):
            f = math.sqrt(max(0.0, 1 - frac * frac))
            painter.drawEllipse(QRectF(cx - r, cy - r * f, 2 * r, 2 * r * f))
            painter.drawEllipse(QRectF(cx - r * f, cy - r, 2 * r * f, 2 * r))

    def draw_orbits(self, painter, cx, cy, scale):
        painter.setPen(QPen(QColor(56, 189, 248, 70), 1.0))
        for plane in range(6):
            points = []
            inc = math.radians(88.9)
            raan = math.radians(plane * 30)
            for step in range(145):
                a = math.radians(step * 2.5)
                x0 = math.cos(a)
                y0 = math.sin(a) * math.cos(inc)
                z0 = math.sin(a) * math.sin(inc)
                x = x0 * math.cos(raan) - y0 * math.sin(raan)
                z = x0 * math.sin(raan) + y0 * math.cos(raan)
                lat = math.degrees(math.asin(max(-1.0, min(1.0, z0))))
                lon = math.degrees(math.atan2(z, x))
                point, _ = self.project(lat, lon, 1.23, cx, cy, scale)
                points.append(point)
            painter.drawPolyline(QPolygonF(points))

    def draw_coverage(self, painter, sat, cx, cy, scale):
        center, z = self.project(sat["lat"], sat["lon"], 1.006, cx, cy, scale)
        if z < -0.72:
            return
        radius_deg = max(4.0, min(35.0, sat["coverage_radius_deg"]))
        edge, _ = self.project(sat["lat"], sat["lon"] + radius_deg, 1.006, cx, cy, scale)
        rr = max(8.0, min(80.0, abs(edge.x() - center.x()) * 1.25))
        painter.setPen(QPen(QColor(134, 239, 172, 165), 1.2))
        painter.setBrush(QColor(34, 197, 94, 38))
        painter.drawEllipse(center, rr, rr * 0.55)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.fillRect(rect, QColor("#020617"))
        painter.setPen(Qt.NoPen)
        for i in range(70):
            painter.setBrush(QColor(125, 211, 252, 45 + (i * 29) % 105))
            painter.drawEllipse(QPointF(rect.left() + ((i * 73) % max(1, rect.width())),
                                        rect.top() + ((i * 41) % max(1, rect.height()))), 0.8, 0.8)

        state = self.state if isinstance(self.state, dict) and self.state.get("satellites") else self.preview_state()
        sats = [self.normalize_sat(s) for s in state.get("satellites", []) if isinstance(s, dict)]
        active_sats, visible_links = self.active_link_sets(state)
        if state.get("type") == "preview":
            active_sats = set(s["name"] for s in sats[:3])

        cx, cy = rect.center().x(), rect.center().y() + rect.height() * 0.06
        earth_r = min(rect.width(), rect.height()) * 0.33
        scale = earth_r * 2.55
        self.draw_orbits(painter, cx, cy, scale)
        self.draw_earth(painter, cx, cy, earth_r)

        sat_positions = {}
        for sat in sats:
            radius = 1.16 + min(0.16, max(0.0, sat["alt_m"]) / 9000000.0)
            point, z = self.project(sat["lat"], sat["lon"], radius, cx, cy, scale)
            sat_positions[sat["name"]] = (point, z, sat)

        for name in active_sats:
            if name in sat_positions:
                self.draw_coverage(painter, sat_positions[name][2], cx, cy, scale)

        stations = {}
        for gs in state.get("groundStations", []):
            if not isinstance(gs, dict):
                continue
            name = str(gs.get("name") or gs.get("Name") or "GS")
            lat = float(gs.get("latitude", gs.get("lat", 0)) or 0)
            lon = float(gs.get("longitude", gs.get("lon", 0)) or 0)
            point, z = self.project(lat, lon, 1.015, cx, cy, scale)
            stations[name] = (point, z)
            if z > -0.58:
                painter.setBrush(QColor("#fb7185"))
                painter.setPen(QPen(QColor("#fecdd3"), 1.0))
                painter.drawPolygon(QPolygonF([
                    QPointF(point.x(), point.y() - 7),
                    QPointF(point.x() - 7, point.y() + 7),
                    QPointF(point.x() + 7, point.y() + 7),
                ]))

        painter.setPen(QPen(QColor(34, 211, 238, 190), 1.4))
        for sat_name, gs_name in visible_links:
            if sat_name in sat_positions and gs_name in stations:
                sp, sz, _ = sat_positions[sat_name]
                gp, gz = stations[gs_name]
                if sz > -0.7 and gz > -0.7:
                    painter.drawLine(sp, gp)

        for point, z, sat in sorted(sat_positions.values(), key=lambda item: item[1]):
            if z < -0.74:
                continue
            active = sat["name"] in active_sats
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(134, 239, 172, 245) if active else QColor(34, 211, 238, 220))
            size = 5.5 if active else 3.8
            painter.drawEllipse(point, size, size)
            if active:
                painter.setPen(QPen(QColor("#d9fff2"), 1.0))
                painter.drawText(QPointF(point.x() + 7, point.y() - 7), sat["name"])

        painter.setPen(QPen(QColor("#67e8f9"), 1.0))
        label = "Qianfan NTN Native 3D Preview" if state.get("type") == "preview" else "Qianfan NTN Native 3D Live"
        painter.drawText(rect.adjusted(10, 8, -10, -10), Qt.AlignTop | Qt.AlignLeft, label)
        painter.setPen(QPen(QColor("#aebbd1"), 1.0))
        painter.drawText(rect.adjusted(10, 26, -10, -10), Qt.AlignTop | Qt.AlignLeft,
                         f"SAT {len(sats)}   LINK {len(visible_links)}")


class DdHeatmapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_draw_time = 0.0
        self.draw_interval_s = 0.04
        self.setMinimumHeight(220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plot = pg.PlotWidget(
            title="<span style='color: #38BDF8; font-size: 10pt; font-family: Microsoft YaHei;'>Delay-Doppler Signal Intensity</span>")
        self.plot.setBackground('#080B14')
        self.plot.showGrid(x=True, y=True, alpha=0.16)
        self.plot.setLabel('bottom', 'Doppler index', **{'font-family': 'Consolas'})
        self.plot.setLabel('left', 'Delay index', **{'font-family': 'Consolas'})
        self.plot.getAxis('left').setWidth(50)
        self.plot.getAxis('bottom').setHeight(38)
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        try:
            self.image.setColorMap(pg.colormap.get("viridis"))
        except Exception:
            pass
        layout.addWidget(self.plot)

    def set_surface(self, grid):
        now = time.monotonic()
        if now - self.last_draw_time < self.draw_interval_s:
            return
        arr = np.asarray(grid, dtype=float)
        if arr.ndim != 2 or not arr.size:
            return
        arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        rows, cols = arr.shape
        self.image.setImage(arr.T, autoLevels=False, levels=(0, 1))
        self.image.setRect(QRectF(0.5, 0.5, cols, rows))
        self.plot.setXRange(0.5, cols + 0.5, padding=0.02)
        self.plot.setYRange(0.5, rows + 0.5, padding=0.02)
        self.last_draw_time = now


class PyqtgraphDdSurfaceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pending_grid = None
        self.current_shape = (32, 32)
        self.setMinimumHeight(220)
        self.setStyleSheet("background-color: #080B14;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        title = QLabel(
            "<span style='color: #38BDF8; font-size: 10pt; font-family: Microsoft YaHei; font-weight: bold;'>3D Delay-Doppler Signal Intensity</span>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        legend = QLabel(
            "<span style='color:#38BDF8;font-family:Consolas;font-weight:bold;'>X: Doppler index</span>"
            "&nbsp;&nbsp; <span style='color:#4ADE80;font-family:Consolas;font-weight:bold;'>Y: Delay index</span>"
            "&nbsp;&nbsp; <span style='color:#FBBF24;font-family:Consolas;font-weight:bold;'>Z: Relative power</span>"
        )
        legend.setAlignment(Qt.AlignCenter)
        legend.setStyleSheet("background-color: #080B14; font-size: 9pt;")
        layout.addWidget(legend)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('#080B14')
        self.view.opts['distance'] = 34
        self.view.opts['elevation'] = 30
        self.view.opts['azimuth'] = -48
        self.view.opts['fov'] = 42
        layout.addWidget(self.view, stretch=1)
        self.axis_overlay_labels = []
        for text, color in (
            ("X Doppler", "#38BDF8"),
            ("Y Delay", "#4ADE80"),
            ("Z Relative power", "#FBBF24"),
        ):
            label = QLabel(text, self.view)
            label.setStyleSheet(
                f"color:{color}; background:rgba(2,6,23,145); "
                "font-family:Consolas; font-weight:bold; font-size:10px; padding:1px 4px; "
                "border:1px solid rgba(148,163,184,70); border-radius:3px;"
            )
            label.adjustSize()
            label.show()
            self.axis_overlay_labels.append(label)

        self.grid_item = gl.GLGridItem()
        self.grid_item.setSize(x=32, y=32, z=1)
        self.grid_item.setSpacing(x=4, y=4, z=1)
        self.grid_item.translate(0, 0, -0.08)
        self.view.addItem(self.grid_item)

        self.surface = None
        self.axis_items = []
        self.text_items = []
        self.rebuild_scene(32, 32)

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.flush_surface)
        self.render_timer.start(66)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, "axis_overlay_labels") or len(self.axis_overlay_labels) < 3:
            return
        vw = self.view.width()
        vh = self.view.height()
        self.axis_overlay_labels[0].move(int(vw * 0.66), max(4, int(vh * 0.78)))
        self.axis_overlay_labels[1].move(max(4, int(vw * 0.12)), max(4, int(vh * 0.82)))
        self.axis_overlay_labels[2].move(max(4, int(vw * 0.16)), max(4, int(vh * 0.16)))

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

    def set_surface(self, grid):
        arr = np.asarray(grid, dtype=float)
        if arr.ndim == 2 and arr.size:
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            self.pending_grid = np.clip(arr, 0.0, 1.0)

    def rebuild_scene(self, rows, cols):
        if self.surface is not None:
            self.view.removeItem(self.surface)
        for item in self.axis_items + self.text_items:
            self.view.removeItem(item)
        self.axis_items = []
        self.text_items = []

        x = np.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols)
        y = np.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows)
        z = np.zeros((cols, rows), dtype=float)
        colors = self.surface_colors(z)
        self.surface = gl.GLSurfacePlotItem(
            x=x, y=y, z=z, colors=colors, shader='shaded', smooth=True, computeNormals=True)
        self.surface.setGLOptions('opaque')
        self.view.addItem(self.surface)
        self.add_axes(rows, cols)
        self.current_shape = (rows, cols)

    def add_line(self, points, color):
        item = gl.GLLinePlotItem(pos=np.asarray(points, dtype=float), color=color, width=2.0, antialias=True)
        self.view.addItem(item)
        self.axis_items.append(item)

    def add_text(self, text, pos, color):
        if not hasattr(gl, "GLTextItem"):
            return
        item = gl.GLTextItem(pos=pos, text=text, color=color)
        self.view.addItem(item)
        self.text_items.append(item)

    def add_axes(self, rows, cols):
        x0, x1 = -(cols - 1) / 2, (cols - 1) / 2
        y0, y1 = -(rows - 1) / 2, (rows - 1) / 2
        z1 = 11.0
        origin = (x0, y0, 0)
        self.add_line([origin, (x1, y0, 0)], (0.22, 0.74, 0.97, 1.0))
        self.add_line([origin, (x0, y1, 0)], (0.29, 0.87, 0.50, 1.0))
        self.add_line([origin, (x0, y0, z1)], (0.98, 0.75, 0.14, 1.0))
        self.add_text("Doppler", (x1 + 1.2, y0, 0), (0.22, 0.74, 0.97, 1.0))
        self.add_text("Delay", (x0, y1 + 1.2, 0), (0.29, 0.87, 0.50, 1.0))
        self.add_text("Intensity", (x0, y0, z1 + 0.8), (0.98, 0.75, 0.14, 1.0))

    def flush_surface(self):
        if self.pending_grid is None:
            return
        grid = self.pending_grid
        self.pending_grid = None
        rows, cols = grid.shape
        if (rows, cols) != self.current_shape:
            self.rebuild_scene(rows, cols)
        z = grid.T
        self.surface.setData(z=z * 11.0, colors=self.surface_colors(z))


class MatplotlibDdSurfaceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_draw_time = 0.0
        self.draw_interval_s = 0.16
        self.setMinimumHeight(220)
        self.setStyleSheet("background-color: #080B14;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.figure = Figure(figsize=(5.2, 3.2), dpi=100, facecolor="#080B14")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #080B14;")
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.draw_surface(np.zeros((32, 32), dtype=float), force=True)

    def style_axes(self, rows, cols):
        self.ax.set_facecolor("#080B14")
        self.ax.set_title("3D Delay-Doppler Signal Intensity", color="#38BDF8", fontsize=10, pad=8, fontweight="bold")
        self.ax.set_xlabel("Doppler index", color="#38BDF8", labelpad=6)
        self.ax.set_ylabel("Delay index", color="#4ADE80", labelpad=6)
        self.ax.set_zlabel("Relative power", color="#FBBF24", labelpad=6)
        self.ax.set_xlim(0, max(1, cols - 1))
        self.ax.set_ylim(0, max(1, rows - 1))
        self.ax.set_zlim(-0.08, 1.05)
        self.ax.view_init(elev=31, azim=-54)
        try:
            self.ax.set_proj_type("persp", focal_length=0.88)
            self.ax.set_box_aspect((1.45, 1.0, 0.52))
        except Exception:
            pass
        self.ax.tick_params(colors="#CBD5E1", labelsize=7, pad=0)
        self.ax.set_xticks(np.linspace(0, max(1, cols - 1), 4))
        self.ax.set_yticks(np.linspace(0, max(1, rows - 1), 4))
        self.ax.set_zticks([0.0, 0.5, 1.0])
        self.ax.grid(True)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.line.set_color("#64748B")
            axis.pane.set_facecolor((0.02, 0.04, 0.08, 0.30))
            axis.pane.set_edgecolor("#1E3A5F")
            axis._axinfo["grid"]["color"] = (0.28, 0.42, 0.56, 0.22)
            axis._axinfo["grid"]["linewidth"] = 0.5

    def set_surface(self, grid):
        now = time.monotonic()
        if now - self.last_draw_time < self.draw_interval_s:
            return
        self.draw_surface(grid)

    def draw_surface(self, grid, force=False):
        arr = np.asarray(grid, dtype=float)
        if arr.ndim != 2 or not arr.size:
            return
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        rows, cols = arr.shape
        step = max(1, max(rows, cols) // 46)
        z = np.power(arr[::step, ::step], 0.72)
        y = np.arange(0, rows, step)
        x = np.arange(0, cols, step)
        x_grid, y_grid = np.meshgrid(x, y)
        cmap = "turbo"

        self.ax.clear()
        self.style_axes(rows, cols)
        surface = self.ax.plot_surface(
            x_grid, y_grid, z,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidth=0.12,
            edgecolor=(0.72, 0.88, 1.0, 0.22),
            antialiased=True,
            rstride=1,
            cstride=1,
            shade=True,
        )
        self.ax.contourf(
            x_grid, y_grid, z,
            zdir="z",
            offset=-0.075,
            levels=np.linspace(0, 1, 14),
            cmap=cmap,
            alpha=0.70,
        )
        if z.size:
            peak_idx = np.unravel_index(int(np.argmax(z)), z.shape)
            peak_x = x_grid[peak_idx]
            peak_y = y_grid[peak_idx]
            peak_z = z[peak_idx]
            self.ax.scatter([peak_x], [peak_y], [peak_z + 0.025], s=20, c="#FBBF24", depthshade=False)
        self.figure.subplots_adjust(left=-0.06, right=1.03, bottom=-0.08, top=0.94)
        self.canvas.draw_idle()
        self.last_draw_time = time.monotonic()


class DdImageLabel(QLabel):
    zoom_requested = Signal(int)
    drag_requested = Signal(float, float)
    pan_requested = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_mode = None
        self._last_pos = None
        self.setCursor(Qt.OpenHandCursor)
        self.setMouseTracking(True)
        self.setToolTip("滚轮缩放，左键拖动旋转，右键拖动平移视角")

    def wheelEvent(self, event):
        self.zoom_requested.emit(event.angleDelta().y())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_mode = "rotate"
            self._last_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.RightButton:
            self._drag_mode = "pan"
            self._last_pos = event.position()
            self.setCursor(Qt.SizeAllCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_mode and self._last_pos is not None:
            delta = event.position() - self._last_pos
            self._last_pos = event.position()
            if self._drag_mode == "rotate":
                self.drag_requested.emit(delta.x(), delta.y())
            else:
                self.pan_requested.emit(delta.x(), delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._drag_mode = None
            self._last_pos = None
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ExternalDdSurfaceWidget(QWidget):
    def __init__(self, log_callback=None, parent=None):
        super().__init__(parent)
        self.log_callback = log_callback
        self.last_send_time = 0.0
        self.draw_interval_s = 0.08
        self.output_path = APP_DIR / "runtime" / "dd_surface_frame_v2.png"
        self.helper_port = 65438
        self.helper_proc = None
        self.last_image_mtime = 0.0
        self.last_demo_update = 0.0
        self.base_pixmap = QPixmap()
        self.view_distance = 78.0
        self.view_elevation = 20.0
        self.view_azimuth = -36.0
        self.view_center_x = 0.0
        self.view_center_y = 0.0
        self.view_center_z = 2.8
        self.view_push_pending = False
        self.surface_rows = 32
        self.surface_cols = 32
        self.surface_peak = 0.0
        self.surface_peak_row = 0
        self.surface_peak_col = 0
        self.anomaly_enabled = False
        self.anomaly_attenuation_db = 0.0
        self.setMinimumHeight(220)
        self.setStyleSheet("background-color: #080B14;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.image_label = DdImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #080B14; color: #94A3B8; font-family: Consolas;")
        self.image_label.zoom_requested.connect(self.on_zoom_requested)
        self.image_label.drag_requested.connect(self.on_drag_requested)
        self.image_label.pan_requested.connect(self.on_pan_requested)
        layout.addWidget(self.image_label)

        self.start_helper()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_image)
        self.refresh_timer.start(80)
        self.view_timer = QTimer(self)
        self.view_timer.setSingleShot(True)
        self.view_timer.timeout.connect(self.flush_view_state)
        QTimer.singleShot(250, self.push_view_state)

    def start_helper(self):
        helper_script = APP_DIR / "dd_gl_helper.py"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.helper_proc = subprocess.Popen(
            [sys.executable, str(helper_script), str(self.output_path), str(self.helper_port)],
            cwd=str(APP_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if self.log_callback:
            self.log_callback("[Channel] External DD OpenGL helper started.")

    def send_helper_payload(self, payload_dict):
        try:
            payload = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(payload, ("127.0.0.1", self.helper_port))
            sock.close()
            return True
        except Exception as exc:
            if self.log_callback:
                self.log_callback(f"[Channel] External DD helper send failed: {exc}")
            return False

    def push_view_state(self):
        self.view_push_pending = True
        self.view_timer.start(45)

    def flush_view_state(self):
        self.view_push_pending = False
        self.send_helper_payload({
            "type": "view",
            "distance": self.view_distance,
            "elevation": self.view_elevation,
            "azimuth": self.view_azimuth,
            "center_x": self.view_center_x,
            "center_y": self.view_center_y,
            "center_z": self.view_center_z,
        })

    def set_surface(self, grid):
        arr = np.asarray(grid, dtype=float)
        if arr.ndim != 2 or not arr.size:
            return
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        self.surface_rows, self.surface_cols = arr.shape
        peak_index = np.unravel_index(int(np.argmax(arr)), arr.shape)
        self.surface_peak_row = int(peak_index[0])
        self.surface_peak_col = int(peak_index[1])
        self.surface_peak = float(arr[peak_index])
        now = time.monotonic()
        if now - self.last_send_time < self.draw_interval_s:
            return
        if self.send_helper_payload({"type": "grid", "grid": arr.tolist()}):
            self.last_send_time = now

    def set_channel_state(self, anomaly_enabled=False, anomaly_attenuation_db=0.0):
        self.anomaly_enabled = bool(anomaly_enabled)
        try:
            self.anomaly_attenuation_db = float(anomaly_attenuation_db)
        except Exception:
            self.anomaly_attenuation_db = 0.0
        self.update_scaled_pixmap()

    def refresh_image(self):
        if not self.output_path.exists():
            if time.monotonic() - self.last_demo_update > 1.0:
                self.image_label.setText("Waiting for external DD 3D renderer...")
                self.last_demo_update = time.monotonic()
            return
        mtime = self.output_path.stat().st_mtime
        if mtime <= self.last_image_mtime:
            return
        self.last_image_mtime = mtime
        pix = QPixmap(str(self.output_path))
        if pix.isNull():
            return
        self.base_pixmap = pix
        self.update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_scaled_pixmap()

    def on_zoom_requested(self, delta):
        if delta == 0:
            return
        factor = 0.90 if delta > 0 else 1.11
        self.view_distance = max(24.0, min(110.0, self.view_distance * factor))
        self.push_view_state()

    def on_drag_requested(self, dx, dy):
        self.view_azimuth += dx * 0.35
        self.view_elevation = max(5.0, min(55.0, self.view_elevation - dy * 0.18))
        self.push_view_state()

    def on_pan_requested(self, dx, dy):
        self.view_center_x = max(-10.0, min(10.0, self.view_center_x - dx * 0.08))
        self.view_center_y = max(-10.0, min(10.0, self.view_center_y + dy * 0.08))
        self.push_view_state()

    def update_scaled_pixmap(self):
        if self.base_pixmap.isNull():
            return
        target = self.image_label.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self.base_pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        canvas = QPixmap(scaled.size())
        canvas.fill(QColor("#080B14"))
        painter = QPainter(canvas)
        painter.drawPixmap(0, 0, scaled)
        painter.setRenderHint(QPainter.Antialiasing, True)

        legend_rect = QRectF(canvas.width() - 170, 10, 158, 66)
        painter.setPen(QPen(QColor(38, 189, 248, 90), 1))
        painter.setBrush(QColor(8, 11, 20, 185))
        painter.drawRoundedRect(legend_rect, 10, 10)
        painter.setFont(QFont("Consolas", 9))
        painter.setPen(QColor("#38BDF8"))
        painter.drawText(legend_rect.adjusted(10, 8, -10, -40), Qt.AlignRight | Qt.AlignVCenter, "X Doppler index")
        painter.setPen(QColor("#4ADE80"))
        painter.drawText(legend_rect.adjusted(10, 24, -10, -24), Qt.AlignRight | Qt.AlignVCenter, "Y Delay index")
        painter.setPen(QColor("#FACC15"))
        painter.drawText(legend_rect.adjusted(10, 40, -10, -8), Qt.AlignRight | Qt.AlignVCenter, "Z Relative power")

        painter.setFont(QFont("Consolas", 9))
        painter.setPen(QColor("#CBD5E1"))
        status = (
            f"DD grid {self.surface_rows} x {self.surface_cols} | "
            f"peak={self.surface_peak:.2f} at delay {self.surface_peak_row + 1}, "
            f"doppler {self.surface_peak_col + 1}"
        )
        if self.anomaly_enabled:
            status += f" | manual attenuation {self.anomaly_attenuation_db:.0f} dB"
        painter.drawText(QRectF(12, canvas.height() - 28, canvas.width() - 24, 20), Qt.AlignLeft | Qt.AlignVCenter, status)

        if self.anomaly_enabled:
            badge_rect = QRectF(12, 10, 184, 28)
            painter.setPen(QPen(QColor(251, 113, 133, 170), 1))
            painter.setBrush(QColor(127, 29, 29, 205))
            painter.drawRoundedRect(badge_rect, 6, 6)
            painter.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
            painter.setPen(QColor("#FFE4E6"))
            painter.drawText(
                badge_rect.adjusted(8, 0, -8, 0),
                Qt.AlignLeft | Qt.AlignVCenter,
                f"Manual anomaly: -{self.anomaly_attenuation_db:.0f} dB",
            )

        painter.end()
        self.image_label.setPixmap(canvas)

    def close(self):
        try:
            if self.helper_proc and self.helper_proc.poll() is None:
                self.helper_proc.terminate()
        except Exception:
            pass
        return super().close()


class WebDdSurfaceWidget(QWidget):
    def __init__(self, log_callback=None, parent=None):
        super().__init__(parent)
        self.log_callback = log_callback
        self.page_ready = False
        self.pending_grid = None
        self.last_send_time = 0.0
        self.draw_interval_s = 0.08
        self.anomaly_enabled = False
        self.anomaly_attenuation_db = 0.0
        self.setMinimumHeight(220)
        self.setStyleSheet("background-color: #080B14;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.browser = QWebEngineView()
        if ConsoleWebEnginePage is not None:
            page = ConsoleWebEnginePage(self.browser)
            if self.log_callback:
                page.console_message.connect(self.log_callback)
            self.browser.setPage(page)
        if QWebEngineSettings is not None:
            settings = self.browser.settings()
            for attr_name in (
                "WebGLEnabled",
                "Accelerated2dCanvasEnabled",
                "LocalContentCanAccessFileUrls",
                "LocalContentCanAccessRemoteUrls",
            ):
                attr = getattr(QWebEngineSettings.WebAttribute, attr_name, None)
                if attr is not None:
                    settings.setAttribute(attr, True)
        try:
            self.browser.page().setBackgroundColor(QColor("#080B14"))
        except Exception:
            pass
        self.browser.setStyleSheet("background-color: #080B14;")
        self.browser.loadFinished.connect(self.on_loaded)
        dd_html = APP_DIR / "web" / "dd_surface.html"
        dd_base_url = QUrl.fromLocalFile(str((APP_DIR / "web").resolve()) + "/")
        self.browser.setHtml(dd_html.read_text(encoding="utf-8"), dd_base_url)
        layout.addWidget(self.browser)

    def on_loaded(self, ok):
        self.page_ready = ok
        if self.log_callback:
            self.log_callback("[Channel] Three.js DD surface loaded." if ok else "[Channel] Three.js DD surface failed to load.")
        if ok and self.pending_grid is not None:
            self.send_grid(self.pending_grid, force=True)

    def set_surface(self, grid):
        arr = np.asarray(grid, dtype=float)
        if arr.ndim != 2 or not arr.size:
            return
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        max_dim = max(arr.shape)
        if max_dim > 64:
            step = int(np.ceil(max_dim / 64))
            arr = arr[::step, ::step]
        self.pending_grid = arr
        self.send_grid(arr)

    def send_grid(self, arr, force=False):
        now = time.monotonic()
        if not force and now - self.last_send_time < self.draw_interval_s:
            return
        if not self.page_ready:
            return
        payload = json.dumps({
            "grid": arr.tolist(),
            "anomalyEnabled": self.anomaly_enabled,
            "anomalyAttenuationDb": self.anomaly_attenuation_db,
        }, ensure_ascii=False)
        self.browser.page().runJavaScript(f"window.updateDdSurface({payload});")
        self.last_send_time = now

    def set_channel_state(self, anomaly_enabled=False, anomaly_attenuation_db=0.0):
        self.anomaly_enabled = bool(anomaly_enabled)
        try:
            self.anomaly_attenuation_db = float(anomaly_attenuation_db)
        except Exception:
            self.anomaly_attenuation_db = 0.0


class SoftwareDdSurfaceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = np.zeros((32, 32), dtype=float)
        self.setMinimumHeight(220)
        self.setStyleSheet("background-color: #080B14;")

    def set_surface(self, grid):
        arr = np.asarray(grid, dtype=float)
        if arr.ndim == 2 and arr.size:
            self.grid = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            self.update()

    def surface_color(self, value):
        stops = [
            (0.00, QColor(58, 28, 113)),
            (0.32, QColor(35, 104, 176)),
            (0.68, QColor(35, 196, 177)),
            (1.00, QColor(252, 214, 66)),
        ]
        value = max(0.0, min(1.0, float(value)))
        for idx in range(len(stops) - 1):
            p0, c0 = stops[idx]
            p1, c1 = stops[idx + 1]
            if value <= p1:
                t = (value - p0) / max(1e-9, p1 - p0)
                return QColor(
                    int(c0.red() + (c1.red() - c0.red()) * t),
                    int(c0.green() + (c1.green() - c0.green()) * t),
                    int(c0.blue() + (c1.blue() - c0.blue()) * t),
                    235,
                )
        return QColor(252, 214, 66, 235)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#080B14"))

        w, h = self.width(), self.height()
        painter.setPen(QColor("#38BDF8"))
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        painter.drawText(0, 6, w, 24, Qt.AlignCenter, "3D Delay-Doppler Signal Intensity")

        z = self.grid
        rows, cols = z.shape
        if rows < 2 or cols < 2:
            return

        step = 1 if max(rows, cols) <= 36 else 2
        ii = np.arange(0, rows, step)
        jj = np.arange(0, cols, step)
        if ii[-1] != rows - 1:
            ii = np.append(ii, rows - 1)
        if jj[-1] != cols - 1:
            jj = np.append(jj, cols - 1)

        xs = (jj / max(1, cols - 1) - 0.5) * 2.0
        ys = (ii / max(1, rows - 1) - 0.5) * 2.0
        xx, yy = np.meshgrid(xs, ys)
        zz = z[np.ix_(ii, jj)]

        raw_x = (xx - yy) * 0.92
        raw_y = (xx + yy) * 0.34 - zz * 1.10
        min_x, max_x = float(raw_x.min()), float(raw_x.max())
        min_y, max_y = float(raw_y.min()), float(raw_y.max())
        plot_w = max(10, w - 52)
        plot_h = max(10, h - 74)
        scale = min(plot_w / max(1e-9, max_x - min_x), plot_h / max(1e-9, max_y - min_y))
        offset_x = (w - (max_x - min_x) * scale) * 0.5
        offset_y = 38 + (plot_h - (max_y - min_y) * scale) * 0.5
        sx = offset_x + (raw_x - min_x) * scale
        sy = offset_y + (raw_y - min_y) * scale

        painter.setPen(QPen(QColor(120, 190, 220, 70), 0.8))
        cells = []
        for r in range(len(ii) - 1):
            for c in range(len(jj) - 1):
                depth = float(raw_y[r:r + 2, c:c + 2].mean())
                cells.append((depth, r, c))
        cells.sort()

        for _, r, c in cells:
            avg_z = float(zz[r:r + 2, c:c + 2].mean())
            poly = QPolygonF([
                QPointF(float(sx[r, c]), float(sy[r, c])),
                QPointF(float(sx[r, c + 1]), float(sy[r, c + 1])),
                QPointF(float(sx[r + 1, c + 1]), float(sy[r + 1, c + 1])),
                QPointF(float(sx[r + 1, c]), float(sy[r + 1, c])),
            ])
            painter.setBrush(self.surface_color(avg_z))
            painter.drawPolygon(poly)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(226, 232, 240, 110), 1.0))
        floor_poly = QPolygonF([
            QPointF(float(sx[0, 0]), float(sy[0, 0])),
            QPointF(float(sx[0, -1]), float(sy[0, -1])),
            QPointF(float(sx[-1, -1]), float(sy[-1, -1])),
            QPointF(float(sx[-1, 0]), float(sy[-1, 0])),
        ])
        painter.drawPolygon(floor_poly)

        base_sy = offset_y + (((xx + yy) * 0.34) - min_y) * scale
        origin = QPointF(float(sx[0, 0]), float(base_sy[0, 0]))
        doppler_end = QPointF(float(sx[0, -1]), float(base_sy[0, -1]))
        delay_end = QPointF(float(sx[-1, 0]), float(base_sy[-1, 0]))
        z_end = QPointF(float(origin.x()), float(origin.y() - 0.92 * scale))

        def draw_axis(start, end, color, label, label_offset):
            painter.setPen(QPen(color, 2.2))
            painter.drawLine(start, end)
            dx = end.x() - start.x()
            dy = end.y() - start.y()
            length = max(1.0, (dx * dx + dy * dy) ** 0.5)
            ux, uy = dx / length, dy / length
            nx, ny = -uy, ux
            arrow_len = 10
            arrow_w = 5
            arrow = QPolygonF([
                end,
                QPointF(end.x() - ux * arrow_len + nx * arrow_w, end.y() - uy * arrow_len + ny * arrow_w),
                QPointF(end.x() - ux * arrow_len - nx * arrow_w, end.y() - uy * arrow_len - ny * arrow_w),
            ])
            painter.setBrush(color)
            painter.drawPolygon(arrow)
            painter.setBrush(Qt.NoBrush)
            painter.setFont(QFont("Consolas", 9, QFont.Bold))
            painter.drawText(
                QRectF(end.x() + label_offset.x(), end.y() + label_offset.y(), 118, 20),
                Qt.AlignLeft | Qt.AlignVCenter,
                label,
            )

        def draw_ticks(start, end, values, normal_shift):
            painter.setPen(QPen(QColor("#CBD5E1"), 1.0))
            for value, label in values:
                px = start.x() + (end.x() - start.x()) * value
                py = start.y() + (end.y() - start.y()) * value
                painter.drawLine(
                    QPointF(px - normal_shift.x(), py - normal_shift.y()),
                    QPointF(px + normal_shift.x(), py + normal_shift.y()),
                )
                painter.setFont(QFont("Consolas", 8))
                painter.drawText(QRectF(px + normal_shift.x() + 2, py + normal_shift.y() - 8, 36, 16), label)

        axis_blue = QColor("#38BDF8")
        axis_green = QColor("#4ADE80")
        axis_yellow = QColor("#FBBF24")
        draw_axis(origin, doppler_end, axis_blue, "Doppler", QPointF(4, -2))
        draw_axis(origin, delay_end, axis_green, "Delay", QPointF(4, -2))
        draw_axis(origin, z_end, axis_yellow, "Intensity", QPointF(4, -18))
        draw_ticks(origin, doppler_end, [(0, "0"), (0.5, str((cols - 1) // 2)), (1, str(cols - 1))], QPointF(0, 4))
        draw_ticks(origin, delay_end, [(0, "0"), (0.5, str((rows - 1) // 2)), (1, str(rows - 1))], QPointF(4, 0))
        draw_ticks(origin, z_end, [(0, "0"), (0.5, "0.5"), (1, "1")], QPointF(4, 0))

        painter.setPen(QColor("#CBD5E1"))
        painter.setFont(QFont("Consolas", 9))
        painter.drawText(12, h - 28, "X: Doppler index")
        painter.drawText(w - 118, h - 28, "Y: Delay index")
        painter.drawText(12, 42, "Z: Relative power")


class SatelliteSystemUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("面向星载智能的通算一体基带传输原型系统")
        self.resize(1600, 950)
        self.setStyleSheet(ACADEMIC_STYLE)
        self.channel_anomaly_control_path = APP_DIR / "runtime" / "channel_anomaly_control.json"

        # ===== Channel power plot history and real-time accumulated x-axis =====
        # Keep all received channel-power samples in Python UI. The visible x-axis
        # starts from frame 1 and expands with the current frame; no samples are discarded.
        self.power_history_x = np.array([], dtype=float)
        self.power_history_y = np.array([], dtype=float)
        self.power_history_mask = np.array([], dtype=bool)
        self.power_x_view_mode = "accumulate"
        self.power_x_window_frames = 240
        self.power_last_total_frames = 1000
        self.power_last_current_frame = 1

        central_widget = QWidget();
        self.setCentralWidget(central_widget)
        total_layout = QVBoxLayout(central_widget)
        total_layout.setContentsMargins(0, 0, 0, 0);
        total_layout.setSpacing(0)
        self.create_header(total_layout)

        body_container = QWidget()
        body_layout = QHBoxLayout(body_container)
        body_layout.setContentsMargins(15, 15, 15, 15)
        total_layout.addWidget(body_container)

        self.main_splitter = QSplitter(Qt.Horizontal);
        body_layout.addWidget(self.main_splitter)
        left_widget = QWidget();
        left_layout = QVBoxLayout(left_widget);
        left_layout.setContentsMargins(0, 0, 10, 0)
        self.left_vertical_splitter = QSplitter(Qt.Vertical);
        left_layout.addWidget(self.left_vertical_splitter)

        self.init_control_panel()
        self.init_log_panel()
        self.init_cyber_physical_panel()
        self.left_vertical_splitter.setSizes([260, 150, 390])

        right_widget = QWidget();
        right_layout = QVBoxLayout(right_widget);
        right_layout.setContentsMargins(10, 0, 0, 0)
        self.init_dashboard_panel(right_layout)

        self.main_splitter.addWidget(left_widget);
        self.main_splitter.addWidget(right_widget)
        self.main_splitter.setSizes([600, 1000])

        self.mode_selector.currentIndexChanged.connect(self.config_stack.setCurrentIndex)
        self.mode_selector.currentIndexChanged.connect(self.dashboard_stack.setCurrentIndex)

        self.iq_thread = IqDataReceiverThread();
        self.iq_thread.iq_data_received.connect(self.update_iq_plot);
        self.iq_thread.log_msg.connect(self.append_log);
        self.iq_thread.start()
        self.spec_thread = SpectrumDataReceiverThread();
        self.spec_thread.spec_data_received.connect(self.update_spec_plot);
        self.spec_thread.log_msg.connect(self.append_log);
        self.spec_thread.start()
        self.video_thread = VideoDataReceiverThread();
        self.video_thread.video_frame_received.connect(self.update_video_frame);
        self.video_thread.log_msg.connect(self.append_log);
        self.video_thread.start()
        self.constellation_thread = ConstellationStateReceiverThread();
        self.constellation_thread.state_received.connect(self.update_constellation_view);
        self.constellation_thread.log_msg.connect(self.append_log);
        self.constellation_thread.start()
        self.channel_thread = ChannelStateReceiverThread();
        self.channel_thread.channel_state_received.connect(self.update_channel_state_view);
        self.channel_thread.log_msg.connect(self.append_log);
        self.channel_thread.start()
        self.write_channel_anomaly_control(silent=True)

    def create_header(self, parent_layout):
        header_frame = QFrame();
        header_frame.setObjectName("HeaderFrame");
        header_layout = QVBoxLayout(header_frame)
        header_layout.setSpacing(6)

        # === 修复 1：精简为两行，竞赛名与组名放一行 ===
        title_label = QLabel("面向星载智能的通算一体基带传输原型系统")
        title_label.setObjectName("MainTitle")
        title_label.setAlignment(Qt.AlignCenter)

        # 将赛事名称与组名用竖线分隔，并仅对组名做高亮
        sub_title = QLabel(
            "第二十一届中国研究生电子设计竞赛 &nbsp;|&nbsp; 团队名：<span style='color: #38BDF8;'>【星派队】</span>")
        sub_title.setObjectName("SubTitle")
        sub_title.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(title_label)
        header_layout.addWidget(sub_title)

        parent_layout.addWidget(header_frame)

    def create_label(self, text):
        lbl = QLabel(text);
        lbl.setStyleSheet("color: #E2E8F0; font-family: 'Microsoft YaHei';");
        return lbl

    def init_control_panel(self):
        group = QGroupBox("系统参数配置 (System Config)");
        main_layout = QVBoxLayout();
        main_layout.setContentsMargins(15, 20, 15, 15);
        main_layout.setSpacing(15);
        group.setLayout(main_layout)

        self.mode_selector = QComboBox();
        self.mode_selector.addItems(["✦  通用译码器配置模式", "✦  链路自适应配置模式"])
        # === 修复 2：彻底统一模式切换框背景颜色，消除刺眼感 ===
        self.mode_selector.setStyleSheet(
            """
            QComboBox { background-color: #1E293B; color: #38BDF8; font-size: 15px; font-weight: bold; font-family: 'Microsoft YaHei'; padding: 10px 15px; border: 1px solid #38BDF8; border-radius: 6px; } 
            QComboBox::drop-down { border: none; padding-right: 15px; } 
            QComboBox:hover { background-color: #283548; }
            QComboBox QAbstractItemView { background-color: #0B0F19; color: #F8FAFC; selection-background-color: #2563EB; selection-color: #FFFFFF; border: 1px solid #38BDF8; border-radius: 4px; outline: none; } 
            QComboBox QAbstractItemView::item { min-height: 40px; padding-left: 10px; }
            """
        )
        main_layout.addWidget(self.mode_selector)

        self.config_stack = QStackedWidget()

        panel_decoder = QWidget();
        form_decoder = QFormLayout(panel_decoder);
        form_decoder.setContentsMargins(5, 10, 5, 0);
        form_decoder.setVerticalSpacing(15)
        self.coding_cb = QComboBox();
        self.coding_cb.addItems([
            "CCSDS LDPC n128/k64",
            "CCSDS LDPC n256/k128",
            "CCSDS LDPC n512/k256",
            "DVB-S2 LDPC N16200 R1/4",
            "DVB-S2 LDPC N16200 R1/2",
            "DVB-S2 LDPC N16200 R5/6",
            "DVB-S2 LDPC N64800 R1/2",
            "Polar N128/K64",
            "Polar N256/K192",
        ])
        self.platform_cb = QComboBox();
        self.platform_cb.addItems([
            "自动选择",
            "CUDA BP",
            "CUDA BP-OSD",
            "CPU BP/NMS",
        ])
        self.standard_cb = QComboBox();
        self.standard_cb.addItems(["BPSK", "QPSK", "16QAM", "64QAM"])
        form_decoder.addRow(self.create_label("编码矩阵/FEC:"), self.coding_cb);
        form_decoder.addRow(self.create_label("译码后端:"), self.platform_cb);
        form_decoder.addRow(self.create_label("调制方式:"), self.standard_cb)

        panel_amc = QWidget();
        form_amc = QFormLayout(panel_amc);
        form_amc.setContentsMargins(5, 10, 5, 0);
        form_amc.setVerticalSpacing(15)
        self.ce_data_cb = QComboBox();
        self.ce_data_cb.addItems(["启用", "禁用"])
        self.cp_method_cb = QComboBox();
        self.cp_method_cb.addItems(["启用", "禁用"])
        self.amc_mode_cb = QComboBox();
        self.amc_mode_cb.addItems(["启用", "禁用"])
        form_amc.addRow(self.create_label("信道状态信息采集:"), self.ce_data_cb);
        form_amc.addRow(self.create_label("信道状态信息预测:"), self.cp_method_cb);
        form_amc.addRow(self.create_label("基于AMC链路传输:"), self.amc_mode_cb)

        self.channel_anomaly_cb = QComboBox();
        self.channel_anomaly_cb.addItems(["禁用", "启用"])
        self.channel_anomaly_cb.currentIndexChanged.connect(self.write_channel_anomaly_control)
        form_amc.addRow(self.create_label("信道异常注入:"), self.channel_anomaly_cb)

        self.config_stack.addWidget(panel_decoder);
        self.config_stack.addWidget(panel_amc);
        main_layout.addWidget(self.config_stack)
        main_layout.addStretch();
        self.btn_run = QPushButton("▶ 下发配置参数并启动测试");
        main_layout.addWidget(self.btn_run);
        self.left_vertical_splitter.addWidget(group)

    def init_log_panel(self):
        group = QGroupBox("系统状态与运行日志");
        layout = QVBoxLayout();
        layout.setContentsMargins(10, 15, 10, 10)
        self.log_text = QTextEdit();
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "background-color: #0b0e14; color: #4ade80; font-family: Consolas, 'Courier New', monospace; font-size: 13px; border: 1px solid #2D3748; padding: 5px;")
        self.log_text.append("[System] Initializing Hardware USRP SDR...")
        layout.addWidget(self.log_text);
        group.setLayout(layout);
        self.left_vertical_splitter.addWidget(group)

    def init_cyber_physical_panel(self):
        group = QGroupBox("软硬件协同的虚实交互验证");
        web_layout = QHBoxLayout();
        web_layout.setContentsMargins(5, 20, 5, 10);
        web_layout.setSpacing(5)
        self.browser = QWebEngineView()
        if ConsoleWebEnginePage is not None:
            web_page = ConsoleWebEnginePage(self.browser)
            web_page.console_message.connect(self.append_log)
            self.browser.setPage(web_page)
        if QWebEngineSettings is not None:
            web_settings = self.browser.settings()
            for attr_name in (
                "WebGLEnabled",
                "Accelerated2dCanvasEnabled",
                "LocalContentCanAccessFileUrls",
                "LocalContentCanAccessRemoteUrls",
            ):
                attr = getattr(QWebEngineSettings.WebAttribute, attr_name, None)
                if attr is not None:
                    web_settings.setAttribute(attr, True)
        try:
            self.browser.page().setBackgroundColor(QColor("#020617"))
        except Exception:
            pass
        self.constellation_page_ready = False
        self.latest_constellation_state = None
        constellation_html = APP_DIR / "web" / "constellation_map.html"
        self.browser.loadFinished.connect(self.on_constellation_page_loaded)
        constellation_url = QUrl.fromLocalFile(str(constellation_html))
        constellation_url.setQuery(f"v={int(constellation_html.stat().st_mtime)}")
        self.browser.setUrl(constellation_url)
        self.browser.setStyleSheet("background-color: #020617;")
        self.browser.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.append_log("[Constellation] Three.js WebEngine constellation renderer loading.")
        self.flow_widget = CyberDataFlowWidget()

        self.usrp_pic_label = QLabel();
        self.usrp_pic_label.setAlignment(Qt.AlignCenter)
        self.usrp_pic_label.setStyleSheet("background-color: #050505; border: 2px solid #3A4A69; border-radius: 6px;")
        self.usrp_pic_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        pix = QPixmap("usrp_real.jpg")
        if not pix.isNull():
            self.usrp_pic_label.setPixmap(pix.scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.usrp_pic_label.setScaledContents(True)
        else:
            self.usrp_pic_label.setText("[ 硬件实物接入端 ]\n缺少 usrp_real.jpg")
            self.usrp_pic_label.setStyleSheet("color: #A0AEC0; font-family: Consolas;")

        web_layout.addWidget(self.browser, stretch=10);
        web_layout.addWidget(self.flow_widget, stretch=0);
        web_layout.addWidget(self.usrp_pic_label, stretch=9)
        group.setLayout(web_layout);
        self.left_vertical_splitter.addWidget(group)

    def init_dashboard_panel(self, layout):
        group = QGroupBox("实时性能监测面板 (Real-time Dashboard)");
        main_dash_layout = QVBoxLayout();
        main_dash_layout.setContentsMargins(10, 20, 10, 10)
        pg.setConfigOption('background', '#1A202C');
        pg.setConfigOption('foreground', '#A0AEC0');
        pg.setConfigOptions(antialias=True);
        axis_font = QFont("Consolas", 10)
        self.dashboard_stack = QStackedWidget()

        def style_plot(plot):
            plot.setBackground('#080B14')
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.getAxis('bottom').setTickFont(axis_font)
            plot.getAxis('left').setTickFont(axis_font)
            plot.getAxis('left').setWidth(50)
            plot.getAxis('bottom').setHeight(40)

        # ==================== 视图 A：译码器布局 ====================
        decoder_view = QWidget();
        decoder_layout = QVBoxLayout(decoder_view);
        decoder_layout.setContentsMargins(0, 0, 0, 0);

        decoder_v_splitter = QSplitter(Qt.Vertical)

        # 第一行：双雷达星座图
        row1_h_splitter = QSplitter(Qt.Horizontal)
        self.tx_plot = pg.PlotWidget(
            title="<span style='color: #FBBF24; font-size: 11pt; font-family: Microsoft YaHei;'>▶ 发送端星座图 (USRP Tx)</span>")
        style_plot(self.tx_plot)
        self.tx_plot.setXRange(-1.8, 1.8);
        self.tx_plot.setYRange(-1.8, 1.8)
        self.tx_plot.addItem(pg.InfiniteLine(angle=90, movable=False,
                                             pen=pg.mkPen(color=(255, 255, 255, 40), width=1, style=Qt.DashLine)))
        self.tx_plot.addItem(pg.InfiniteLine(angle=0, movable=False,
                                             pen=pg.mkPen(color=(255, 255, 255, 40), width=1, style=Qt.DashLine)))
        self.tx_scatter = pg.ScatterPlotItem(symbol='+', size=12,
                                             pen=pg.mkPen(color=QColor(251, 191, 36, 255), width=2), brush=None)
        self.tx_plot.addItem(self.tx_scatter)

        self.rx_plot = pg.PlotWidget(
            title="<span style='color: #00E5FF; font-size: 11pt; font-family: Microsoft YaHei;'>▶ 接收端星座图 (USRP Rx)</span>")
        style_plot(self.rx_plot)
        self.rx_plot.setXRange(-1.8, 1.8);
        self.rx_plot.setYRange(-1.8, 1.8)
        self.rx_plot.addItem(
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(0, 229, 255, 40), width=1, style=Qt.DashLine)))
        self.rx_plot.addItem(
            pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color=(0, 229, 255, 40), width=1, style=Qt.DashLine)))
        self.rx_scatter = pg.ScatterPlotItem(symbol='o', size=6, pen=None, brush=pg.mkBrush(QColor(0, 229, 255, 120)))
        self.rx_plot.addItem(self.rx_scatter)
        row1_h_splitter.addWidget(self.tx_plot)
        row1_h_splitter.addWidget(self.rx_plot)

        # 第二行：频谱和BER图表
        row2_h_splitter = QSplitter(Qt.Horizontal)

        # === 修复 3：统一图表文字大小与颜色风格 ===
        self.spec_plot = pg.PlotWidget(
            title="<span style='color: #38BDF8; font-size: 11pt; font-family: Microsoft YaHei;'>▶ 实时频谱瀑布分析 (PSD)</span>")
        style_plot(self.spec_plot)
        self.spec_plot.setLabel('bottom', 'Frequency (Hz)', units='Hz', **{'font-family': 'Consolas'});
        self.spec_plot.setLabel('left', 'Power', units='dB', **{'font-family': 'Consolas'})
        self.spec_plot.setYRange(-40, 20)
        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen(color='#9F7AEA', width=2), fillLevel=-40,
                                              fillBrush=(159, 122, 234, 60), name="Live Spectrum")

        self.ber_plot = pg.PlotWidget(
            title="<span style='color: #818CF8; font-size: 11pt; font-family: Microsoft YaHei;'>▶ 实时译码性能曲线 (BER / FER)</span>")
        style_plot(self.ber_plot)
        self.ber_plot.setLogMode(y=True);
        self.ber_plot.setLabel('bottom', 'Decoded frame', **{'font-family': 'Consolas'})
        self.ber_plot.setLabel('left', 'Error rate', **{'font-family': 'Consolas'})
        self.ber_plot.addLegend(offset=(-10, 10));
        self.ber_curve = self.ber_plot.plot(pen=pg.mkPen(color='#63B3ED', width=2), name="BER")
        self.fer_curve = self.ber_plot.plot(pen=pg.mkPen(color='#F97316', width=2), name="FER")
        self.ber_history_x = []
        self.ber_history_y = []
        self.fer_history_x = []
        self.fer_history_y = []
        row2_h_splitter.addWidget(self.spec_plot)
        row2_h_splitter.addWidget(self.ber_plot)

        # 第三行：视频窗口
        row3_h_splitter = QSplitter(Qt.Horizontal)

        tx_vid_widget = QWidget();
        tx_vid_layout = QVBoxLayout(tx_vid_widget);
        tx_vid_layout.setContentsMargins(0, 5, 0, 0);
        tx_vid_layout.setSpacing(5)
        tx_vid_title = QLabel(
            "<span style='color: #FBBF24; font-family: Microsoft YaHei; font-weight: bold;'>▶ 发送端原始数据（图片/视频） (USRP Tx)</span>")
        tx_vid_title.setAlignment(Qt.AlignCenter)
        self.tx_video_label = QLabel()
        self.tx_video_label.setAlignment(Qt.AlignCenter)
        self.tx_video_label.setStyleSheet("background-color: #080B14; border: 1px solid #3A4A69; border-radius: 4px;")
        self.tx_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        tx_vid_layout.addWidget(tx_vid_title)
        tx_vid_layout.addWidget(self.tx_video_label, stretch=1)

        rx_vid_widget = QWidget();
        rx_vid_layout = QVBoxLayout(rx_vid_widget);
        rx_vid_layout.setContentsMargins(0, 5, 0, 0);
        rx_vid_layout.setSpacing(5)
        rx_vid_title = QLabel(
            "<span style='color: #00E5FF; font-family: Microsoft YaHei; font-weight: bold;'>▶ 接收端恢复数据（图片/视频） (USRP Rx)</span>")
        rx_vid_title.setAlignment(Qt.AlignCenter)
        self.rx_video_label = QLabel()
        self.rx_video_label.setAlignment(Qt.AlignCenter)
        self.rx_video_label.setStyleSheet("background-color: #080B14; border: 1px solid #3A4A69; border-radius: 4px;")
        self.rx_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        rx_vid_layout.addWidget(rx_vid_title)
        rx_vid_layout.addWidget(self.rx_video_label, stretch=1)

        row3_h_splitter.addWidget(tx_vid_widget)
        row3_h_splitter.addWidget(rx_vid_widget)

        decoder_v_splitter.addWidget(row1_h_splitter)
        decoder_v_splitter.addWidget(row2_h_splitter)
        decoder_v_splitter.addWidget(row3_h_splitter)
        decoder_v_splitter.setSizes([300, 300, 200])

        decoder_layout.addWidget(decoder_v_splitter)

        # ==================== 视图 B：链路自适应(AMC) ====================
        amc_view = QWidget();
        amc_layout = QVBoxLayout(amc_view);
        amc_layout.setContentsMargins(0, 0, 0, 0);

        amc_v_splitter = QSplitter(Qt.Vertical)

        self.csi_plot = pg.PlotWidget(
            title="<span style='color: #4ADE80; font-size: 11pt; font-family: Microsoft YaHei;'>▶ 实时信道状态响应 (Channel State Information)</span>")
        style_plot(self.csi_plot)
        self.csi_plot.setLabel('bottom', 'Frame', **{'font-family': 'Consolas'})
        self.csi_plot.setLabel('left', 'CSI', units='dB', **{'font-family': 'Consolas'})
        self.csi_curve = self.csi_plot.plot(pen=pg.mkPen(color='#4ADE80', width=2), fillLevel=0,
                                            fillBrush=(74, 222, 128, 50))

        self.csi_panel = QWidget()
        csi_panel_layout = QVBoxLayout(self.csi_panel)
        csi_panel_layout.setContentsMargins(0, 0, 0, 0)
        csi_panel_layout.setSpacing(4)
        csi_title = QLabel(
            "<span style='color: #4ADE80; font-size: 11pt; font-family: Microsoft YaHei; font-weight: bold;'>▶ 实时信道状态响应 (Channel State Information)</span>")
        csi_title.setAlignment(Qt.AlignCenter)
        csi_panel_layout.addWidget(csi_title)

        # Power plot uses real-time accumulated x-axis automatically.
        # No visible mode selector is kept here: the plot starts from frame 1
        # and expands with the current frame, while all historical samples remain cached.

        csi_subplots_layout = QHBoxLayout()
        csi_subplots_layout.setContentsMargins(0, 0, 0, 0)
        csi_subplots_layout.setSpacing(8)
        csi_panel_layout.addLayout(csi_subplots_layout, stretch=1)

        self.dd_surface_item = None
        dd_backend_error = None
        try:
            self.dd_widget = WebDdSurfaceWidget(log_callback=self.append_log)
            self.dd_backend_name = "Three.js WebEngine 3D surface"
        except Exception as exc:
            self.dd_widget = None
            dd_backend_error = f"Three.js DD init failed: {exc}"
        if self.dd_widget is None:
            try:
                self.dd_widget = MatplotlibDdSurfaceWidget()
                self.dd_backend_name = "matplotlib 3D surface"
            except Exception as exc:
                dd_backend_error = (dd_backend_error + " | " if dd_backend_error else "") + f"Matplotlib DD init failed: {exc}"
                self.dd_widget = DdHeatmapWidget()
                self.dd_backend_name = "pyqtgraph 2D heatmap fallback"
        if dd_backend_error:
            self.append_log(f"[Channel] {dd_backend_error}")
        self.dd_widget.setMinimumWidth(320)
        self.dd_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.append_log(f"[Channel] DD renderer: {self.dd_backend_name}")
        QTimer.singleShot(500, self.seed_dd_surface_demo)

        self.feature_buffer_plot = pg.PlotWidget(
            title="<span style='color: #FBBF24; font-size: 10pt; font-family: Microsoft YaHei;'>Relative Total Channel Power</span>")
        style_plot(self.feature_buffer_plot)
        self.feature_buffer_plot.setLabel('bottom', 'Collected frame', **{'font-family': 'Consolas'})
        self.feature_buffer_plot.setLabel('left', 'Relative power (dB)', **{'font-family': 'Consolas'})
        self.feature_buffer_plot.showGrid(x=True, y=True, alpha=0.18)
        self.feature_buffer_plot.getAxis('left').setWidth(52)
        self.feature_buffer_plot.getAxis('bottom').setHeight(38)
        self.feature_buffer_image = pg.ImageItem()
        self.feature_buffer_plot.addItem(self.feature_buffer_image)
        self.feature_buffer_image.setVisible(False)
        try:
            self.feature_buffer_image.setColorMap(pg.colormap.get("viridis"))
        except Exception:
            pass
        self.channel_power_region = pg.LinearRegionItem(
            values=[0, 0],
            orientation="vertical",
            movable=False,
            brush=pg.mkBrush(248, 113, 113, 42),
        )
        self.channel_power_region.setZValue(-10)
        self.channel_power_region.setVisible(False)
        self.feature_buffer_plot.addItem(self.channel_power_region)
        self.channel_power_baseline = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=pg.mkPen("#94A3B8", width=1.0, style=getattr(Qt, "DotLine", Qt.PenStyle.DotLine)),
        )
        self.channel_power_baseline.setZValue(-5)
        self.feature_buffer_plot.addItem(self.channel_power_baseline)
        self.channel_power_curve = self.feature_buffer_plot.plot(
            [],
            [],
            pen=pg.mkPen("#60A5FA", width=2.0),
            symbol="o",
            symbolSize=4,
            symbolBrush=pg.mkBrush("#93C5FD"),
            symbolPen=pg.mkPen("#BFDBFE", width=0.5),
        )
        self.channel_power_current = pg.ScatterPlotItem(
            size=8,
            brush=pg.mkBrush("#FBBF24"),
            pen=pg.mkPen("#FEF3C7", width=1.0),
        )
        self.feature_buffer_plot.addItem(self.channel_power_current)
        self.feature_buffer_status = pg.TextItem(
            text="Waiting for channel power trace...",
            color="#E5E7EB",
            fill=QColor(8, 11, 20, 205),
            border=pg.mkPen(QColor(148, 163, 184, 90)),
            anchor=(0, 0),
        )
        self.feature_buffer_plot.addItem(self.feature_buffer_status)
        self.feature_buffer_status.setPos(0.5, 1.0)

        self.tsne_plot = pg.PlotWidget(
            title="<span style='color: #F472B6; font-size: 10pt; font-family: Microsoft YaHei;'>Normal 20 dB t-SNE</span>")
        style_plot(self.tsne_plot)
        self.tsne_plot.setLabel('bottom', 'Dim 1', **{'font-family': 'Consolas'})
        self.tsne_plot.setLabel('left', 'Dim 2', **{'font-family': 'Consolas'})
        self.tsne_plot.showGrid(x=True, y=True, alpha=0.22)
        self.tsne_fixed_x_range = (-32.0, 32.0)
        self.tsne_fixed_y_range = (-32.0, 32.0)
        self.tsne_plot.setXRange(*self.tsne_fixed_x_range, padding=0.0)
        self.tsne_plot.setYRange(*self.tsne_fixed_y_range, padding=0.0)
        self.tsne_legend = self.tsne_plot.addLegend(
            size=(76, 38),
            offset=(-8, 8),
            horSpacing=4,
            verSpacing=-5,
            labelTextSize="7pt",
            labelTextColor="#E5E7EB",
        )
        try:
            self.tsne_legend.setBrush(pg.mkBrush(8, 11, 20, 220))
            self.tsne_legend.setPen(pg.mkPen(229, 231, 235, 110))
            self.tsne_legend.layout.setContentsMargins(4, 1, 4, 1)
        except Exception:
            pass
        self.tsne_true_scatter = pg.ScatterPlotItem(
            size=4,
            brush=pg.mkBrush(255, 127, 80, 180),
            pen=None,
            name="True"
        )
        self.tsne_pred_scatter = pg.ScatterPlotItem(
            size=4,
            brush=pg.mkBrush(56, 189, 248, 180),
            pen=None,
            name="Pred"
        )
        self.tsne_plot.addItem(self.tsne_true_scatter)
        self.tsne_plot.addItem(self.tsne_pred_scatter)
        self.tsne_status_text = pg.TextItem(
            text="Waiting for 20 dB t-SNE data...",
            color="#E5E7EB",
            fill=pg.mkBrush(8, 11, 20, 205),
            border=pg.mkPen(229, 231, 235, 90),
            anchor=(0, 0),
        )
        self.tsne_plot.addItem(self.tsne_status_text)
        self.tsne_status_anchor = (
            self.tsne_fixed_x_range[0] + 2.0,
            self.tsne_fixed_y_range[1] - 5.0,
        )
        self.tsne_status_text.setPos(*self.tsne_status_anchor)

        csi_subplots_layout.addWidget(self.dd_widget, stretch=1)
        csi_subplots_layout.addWidget(self.feature_buffer_plot, stretch=1)
        csi_subplots_layout.addWidget(self.tsne_plot, stretch=1)

        self.throughput_plot = pg.PlotWidget(
            title="<span style='color: #A78BFA; font-size: 11pt; font-family: Microsoft YaHei;'>▶ 系统动态吞吐量 (System Throughput)</span>")
        style_plot(self.throughput_plot)
        self.throughput_plot.setLabel('bottom', 'Frame', **{'font-family': 'Consolas'})
        self.throughput_plot.setLabel('left', 'Rate', units='Mbps', **{'font-family': 'Consolas'})
        self.throughput_curve = self.throughput_plot.plot(pen=pg.mkPen(color='#A78BFA', width=2), fillLevel=0,
                                                          fillBrush=(167, 139, 250, 50))
        self.throughput_history_x = []
        self.throughput_history_y = []

        self.amc_state_plot = pg.PlotWidget(
            title="<span style='color: #F472B6; font-size: 11pt; font-family: Microsoft YaHei;'>▶ MCS 阶数自适应切换轨迹 (AMC State Tracker)</span>")
        style_plot(self.amc_state_plot)
        self.amc_state_plot.setLabel('bottom', 'Frame', **{'font-family': 'Consolas'})
        self.amc_state_plot.setLabel('left', 'MCS index', **{'font-family': 'Consolas'})
        self.amc_state_curve = self.amc_state_plot.plot(pen=pg.mkPen(color='#F472B6', width=2), symbol='s',
                                                        symbolSize=8, symbolBrush='#F472B6')
        self.mcs_history_x = []
        self.mcs_history_y = []

        amc_v_splitter.addWidget(self.csi_panel)
        amc_v_splitter.addWidget(self.throughput_plot)
        amc_v_splitter.addWidget(self.amc_state_plot)
        amc_v_splitter.setSizes([430, 240, 240])

        amc_layout.addWidget(amc_v_splitter)

        self.dashboard_stack.addWidget(decoder_view);
        self.dashboard_stack.addWidget(amc_view)
        main_dash_layout.addWidget(self.dashboard_stack);
        group.setLayout(main_dash_layout);
        layout.addWidget(group)

    def seed_dd_surface_demo(self):
        if not hasattr(self, "dd_widget"):
            return
        try:
            rows, cols = 32, 32
            y, x = np.mgrid[0:rows, 0:cols]
            grid = (
                0.95 * np.exp(-(((x - 10) ** 2) + ((y - 11) ** 2)) / 26.0) +
                0.62 * np.exp(-(((x - 22) ** 2) + ((y - 19) ** 2)) / 34.0) +
                0.30 * np.exp(-(((x - 17) ** 2) + ((y - 26) ** 2)) / 18.0)
            )
            grid = grid / (np.max(grid) + 1e-12)
            self.dd_widget.set_surface(grid)
        except Exception as exc:
            self.append_log(f"[Channel] DD demo surface init failed: {exc}")

    # ---------------- 核心槽函数区 ----------------
    def update_iq_plot(self, tx_i, tx_q, rx_i, rx_q):
        if self.dashboard_stack.currentIndex() == 0:
            self.tx_scatter.setData(tx_i, tx_q)
            self.rx_scatter.setData(rx_i, rx_q)

    def update_spec_plot(self, freqs, powers):
        if self.dashboard_stack.currentIndex() == 0:
            self.spec_curve.setData(freqs, powers)

    def update_video_frame(self, tx_rgb, rx_rgb):
        if self.dashboard_stack.currentIndex() != 0: return

        def array_to_pixmap(rgb_arr):
            h, w, _ = rgb_arr.shape
            if np.all(rgb_arr == 0): return QPixmap()
            qImg = QImage(rgb_arr.data, w, h, 3 * w, QImage.Format_RGB888)
            return QPixmap.fromImage(qImg)

        tx_pix = array_to_pixmap(tx_rgb)
        rx_pix = array_to_pixmap(rx_rgb)

        if self.tx_video_label.width() > 0 and not tx_pix.isNull():
            self.tx_video_label.setPixmap(
                tx_pix.scaled(self.tx_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.rx_video_label.width() > 0 and not rx_pix.isNull():
            self.rx_video_label.setPixmap(
                rx_pix.scaled(self.rx_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_constellation_page_loaded(self, ok):
        self.constellation_page_ready = ok
        if ok:
            self.append_log("[Constellation] Local MATLAB-driven constellation view loaded.")
            QTimer.singleShot(900, self.probe_constellation_page)
            if self.latest_constellation_state:
                self.update_constellation_view(self.latest_constellation_state)
        else:
            self.append_log("[Constellation] Local constellation view failed to load.")
            self.browser.setHtml(
                "<html><body style='margin:0;background:#020617;color:#38bdf8;"
                "font-family:Consolas,Microsoft YaHei;display:flex;align-items:center;"
                "justify-content:center;height:100vh;text-align:center;'>"
                "<div>Constellation view failed to load<br/>Please restart UI or check web assets.</div>"
                "</body></html>"
            )

    def probe_constellation_page(self):
        if not getattr(self, "constellation_page_ready", False):
            return
        script = """
        (() => {
          const canvas = document.getElementById('scene');
          const bodyStyle = window.getComputedStyle(document.body);
          return {
            title: document.title,
            bg: bodyStyle.backgroundColor,
            three: typeof window.THREE,
            updater: typeof window.updateConstellation,
            canvas: canvas ? `${canvas.width}x${canvas.height}` : 'missing',
            text: (document.body && document.body.innerText || '').slice(0, 80)
          };
        })();
        """
        self.browser.page().runJavaScript(script, 0, self.on_constellation_probe_result)

    def on_constellation_probe_result(self, info):
        if not isinstance(info, dict):
            self.append_log("[Constellation] WebGL probe returned no page information.")
            return
        self.append_log(
            "[Constellation] WebGL probe: title={title} | THREE={three} | updater={updater} | canvas={canvas} | bg={bg}".format(
                title=info.get("title", "--"),
                three=info.get("three", "--"),
                updater=info.get("updater", "--"),
                canvas=info.get("canvas", "--"),
                bg=info.get("bg", "--"),
            )
        )

    def update_constellation_view(self, state):
        self.latest_constellation_state = state
        if hasattr(self, "constellation_widget"):
            self.constellation_widget.set_state(state)
        elif getattr(self, "constellation_page_ready", False) and hasattr(self, "browser"):
            payload = json.dumps(state, ensure_ascii=False)
            self.browser.page().runJavaScript(f"window.updateConstellation({payload});")

        best_links = state.get("servingLinks") or state.get("bestLinks", [])
        if isinstance(best_links, dict):
            best_links = [best_links]

        active_links = [link for link in best_links if link.get("visible")]
        self.constellation_log_counter = getattr(self, "constellation_log_counter", 0) + 1
        if active_links and self.constellation_log_counter % 5 == 1:
            link = active_links[0]
            self.append_log(
                "[NTN] {gs} -> {sat} | El={el:.1f} deg | Loss={loss:.1f} dB | Doppler={dop:.0f} Hz | {profile}".format(
                    gs=link.get("groundStation", "GS"),
                    sat=link.get("satellite", "SAT"),
                    el=float(link.get("elevation_deg", 0) or 0),
                    loss=float(link.get("pathLoss_dB", 0) or 0),
                    dop=float(link.get("satelliteDoppler_Hz", 0) or 0),
                    profile=link.get("delayProfile", "NTN-TDL")
                )
            )

    def update_tsne_view(self, state):
        true_xy = np.asarray(state.get("trueXY", state.get("true_xy", [])), dtype=float)
        pred_xy = np.asarray(state.get("predXY", state.get("pred_xy", [])), dtype=float)

        if true_xy.ndim == 1 and true_xy.size == 2:
            true_xy = true_xy.reshape(1, 2)
        if pred_xy.ndim == 1 and pred_xy.size == 2:
            pred_xy = pred_xy.reshape(1, 2)

        has_true = true_xy.ndim == 2 and true_xy.shape[1] >= 2 and true_xy.size > 0
        has_pred = pred_xy.ndim == 2 and pred_xy.shape[1] >= 2 and pred_xy.size > 0
        if not has_true and not has_pred:
            return

        if has_true:
            true_xy = np.nan_to_num(true_xy[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
            self.tsne_true_scatter.setData(x=true_xy[:, 0], y=true_xy[:, 1])
        else:
            self.tsne_true_scatter.setData([], [])

        if has_pred:
            pred_xy = np.nan_to_num(pred_xy[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
            self.tsne_pred_scatter.setData(x=pred_xy[:, 0], y=pred_xy[:, 1])
        else:
            self.tsne_pred_scatter.setData([], [])

        all_xy = []
        if has_true:
            all_xy.append(true_xy)
        if has_pred:
            all_xy.append(pred_xy)
        all_xy = np.vstack(all_xy)

        fixed_x = getattr(self, "tsne_fixed_x_range", (-32.0, 32.0))
        fixed_y = getattr(self, "tsne_fixed_y_range", (-32.0, 32.0))
        x_lim = np.asarray(state.get("xLim", state.get("x_lim", [])), dtype=float).reshape(-1)
        y_lim = np.asarray(state.get("yLim", state.get("y_lim", [])), dtype=float).reshape(-1)
        if x_lim.size >= 2 and np.all(np.isfinite(x_lim[:2])) and x_lim[1] > x_lim[0]:
            fixed_x = (float(x_lim[0]), float(x_lim[1]))
        if y_lim.size >= 2 and np.all(np.isfinite(y_lim[:2])) and y_lim[1] > y_lim[0]:
            fixed_y = (float(y_lim[0]), float(y_lim[1]))

        if fixed_x[1] > fixed_x[0]:
            self.tsne_plot.setXRange(fixed_x[0], fixed_x[1], padding=0.0)
        else:
            x_min, x_max = float(np.min(all_xy[:, 0])), float(np.max(all_xy[:, 0]))
            pad = max(1e-3, 0.08 * (x_max - x_min + 1e-9))
            self.tsne_plot.setXRange(x_min - pad, x_max + pad, padding=0.0)

        if fixed_y[1] > fixed_y[0]:
            self.tsne_plot.setYRange(fixed_y[0], fixed_y[1], padding=0.0)
        else:
            y_min, y_max = float(np.min(all_xy[:, 1])), float(np.max(all_xy[:, 1]))
            pad = max(1e-3, 0.08 * (y_max - y_min + 1e-9))
            self.tsne_plot.setYRange(y_min - pad, y_max + pad, padding=0.0)

        condition = str(state.get("condition", "normal")).replace("_", " ")
        snr = state.get("snrDb", state.get("snr_dB", 20))
        frame = int(state.get("frame", 1) or 1)
        total_frames = int(state.get("totalFrames", state.get("total_frames", max(1, frame))) or max(1, frame))
        source_t = state.get("sourceTime", state.get("source_t", frame))
        samples = state.get("samples", state.get("selectedSamples", 0))
        feature_dim = state.get("featureDim", state.get("feature_dim", 0))

        try:
            snr_text = f"{float(snr):.0f} dB"
        except Exception:
            snr_text = f"{snr} dB"

        self.tsne_plot.setTitle(
            f"<span style='color: #F472B6; font-size: 10pt; font-family: Microsoft YaHei;'>"
            f"{condition.title()} {snr_text} t-SNE | frame {frame}/{total_frames}</span>"
        )
        self.tsne_status_text.setText(
            f"t={source_t} | samples={samples} | feature dim={feature_dim}"
        )
        self.tsne_status_text.setPos(
            fixed_x[0] + 2.0,
            fixed_y[1] - 5.0,
        )

        self.tsne_log_counter = getattr(self, "tsne_log_counter", 0) + 1
        if self.tsne_log_counter % 12 == 1:
            self.append_log(
                f"[t-SNE] {condition} | SNR={snr_text} | frame={frame}/{total_frames} | "
                f"true={true_xy.shape[0] if has_true else 0} | pred={pred_xy.shape[0] if has_pred else 0}"
            )

    def on_power_x_window_changed(self, index):
        """Update the visible x-axis mode of the channel-power plot.

        The stored power_history_x/y arrays are not cleared here.
        """
        try:
            mode, win = self.power_x_window_cb.itemData(index)
        except Exception:
            # The selector may be hidden/removed in real-time display mode.
            mode, win = "accumulate", 240
        self.power_x_view_mode = str(mode)
        try:
            self.power_x_window_frames = int(win) if int(win) > 0 else self.power_x_window_frames
        except Exception:
            pass
        try:
            self.apply_channel_power_x_range(
                getattr(self, "power_last_total_frames", 1000),
                getattr(self, "power_last_current_frame", 1),
            )
        except Exception:
            pass

    def merge_channel_power_history(self, x_vals, y_vals, anomaly_mask):
        """Merge incoming channel-power data into a full-history cache.

        MATLAB may send either accumulated 1:t data or a sliding window. This method
        preserves previously received frames in either case.
        """
        x_vals = np.asarray(x_vals, dtype=float).reshape(-1)
        y_vals = np.asarray(y_vals, dtype=float).reshape(-1)
        if x_vals.size == 0 or y_vals.size == 0:
            return

        n = min(x_vals.size, y_vals.size)
        x_vals = x_vals[:n]
        y_vals = y_vals[:n]

        if anomaly_mask is None:
            anomaly_mask = np.zeros(n, dtype=bool)
        else:
            anomaly_mask = np.asarray(anomaly_mask, dtype=bool).reshape(-1)
            if anomaly_mask.size != n:
                anomaly_mask = np.zeros(n, dtype=bool)

        # Accumulated payload from MATLAB: replace directly with the complete trace.
        if int(round(x_vals[0])) == 1 and x_vals.size >= self.power_history_x.size:
            self.power_history_x = x_vals
            self.power_history_y = y_vals
            self.power_history_mask = anomaly_mask
            return

        # Sliding-window payload: merge by frame index without discarding old frames.
        history = {}
        if self.power_history_x.size == self.power_history_y.size == self.power_history_mask.size:
            for x, y, m in zip(self.power_history_x, self.power_history_y, self.power_history_mask):
                history[int(round(x))] = (float(y), bool(m))

        for x, y, m in zip(x_vals, y_vals, anomaly_mask):
            history[int(round(x))] = (float(y), bool(m))

        keys = sorted(history.keys())
        self.power_history_x = np.asarray(keys, dtype=float)
        self.power_history_y = np.asarray([history[k][0] for k in keys], dtype=float)
        self.power_history_mask = np.asarray([history[k][1] for k in keys], dtype=bool)

    def apply_channel_power_x_range(self, total_frames, current_frame):
        """Apply adjustable x-axis range while keeping all historical data.

        Modes:
        - accumulate: MATLAB-like accumulated view. The visible x-axis starts
          from frame 1 and expands with the current frame, instead of being
          fixed to 1~totalFrames at the beginning.
        - recent: show recent N frames, while power_history_x/y still keep all
          received samples.
        - full: show 1~totalFrames. This is useful after enough frames have
          arrived, but it can compress early traces if selected at startup.
        - manual: do not auto-adjust x range, allowing pyqtgraph pan/zoom.
        """
        if not hasattr(self, "feature_buffer_plot") or self.power_history_x.size == 0:
            return

        try:
            current_frame = int(round(float(current_frame)))
        except Exception:
            current_frame = int(self.power_history_x[-1]) if self.power_history_x.size else 1
        try:
            total_frames = int(round(float(total_frames)))
        except Exception:
            total_frames = max(1000, current_frame)
        total_frames = max(total_frames, current_frame, 1)

        mode = getattr(self, "power_x_view_mode", "accumulate")
        win = int(getattr(self, "power_x_window_frames", 240) or 240)
        win = max(20, win)

        if mode == "manual":
            # Leave the current pyqtgraph view as-is so the user can pan/zoom.
            return

        if mode == "recent":
            # Recent-window view: do not discard data; only move the visible window.
            if current_frame <= win:
                x_left = 1
                x_right = win
            else:
                x_left = current_frame - win + 1
                x_right = current_frame

        elif mode == "full":
            # Full-length view: 1~totalFrames. Not recommended for early frames,
            # but kept as an optional mode.
            x_left = 1
            x_right = total_frames

        else:
            # Accumulate view: MATLAB-like x-axis accumulation.
            # Early frames are shown in a compact 1~25 frame window, and then
            # the right boundary grows with current_frame. No history is lost.
            min_show = 25
            margin = 4
            x_left = 1
            x_right = max(min_show, current_frame + margin)

        self.feature_buffer_plot.setXRange(x_left - 0.5, x_right + 0.5, padding=0.01)

    def update_channel_state_view(self, state):
        def as_float_array(value):
            arr = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
            return arr[np.isfinite(arr)]

        payload_type = state.get("type")
        if payload_type == "ntn_tsne_frame":
            self.update_tsne_view(state)
            return
        if payload_type != "ntn_channel_subplots":
            return

        n_paths = 0
        try:
            n_paths = int(state.get("numPaths", 0) or 0)
        except Exception:
            n_paths = 0
        if n_paths <= 0:
            for key in ("paths", "pathMagnitudes", "path_magnitudes", "magnitudes"):
                value = state.get(key)
                if value is None:
                    continue
                try:
                    n_paths = int(len(value))
                    break
                except Exception:
                    pass

        dd_grid = np.asarray(state.get("ddGrid") if state.get("ddGrid") is not None else [], dtype=float)
        if dd_grid.ndim == 2 and dd_grid.size:
            dd_grid = np.nan_to_num(dd_grid, nan=0.0, posinf=1.0, neginf=0.0)
            dd_grid = np.clip(dd_grid, 0.0, 1.0)
            rows, cols = dd_grid.shape
            anomaly_enabled = bool(state.get("anomalyEnabled", state.get("anomaly_enabled", False)))
            try:
                anomaly_attenuation_db = float(state.get("anomalyAttenuationDb", 0.0) or 0.0)
            except Exception:
                anomaly_attenuation_db = 0.0
            if hasattr(self.dd_widget, "set_channel_state"):
                self.dd_widget.set_channel_state(anomaly_enabled, anomaly_attenuation_db)
            if hasattr(self.dd_widget, "set_surface"):
                self.dd_widget.set_surface(dd_grid)

        channel_power_db = np.asarray(state.get("channelPowerDb", []), dtype=float)
        if channel_power_db.ndim == 1 and channel_power_db.size:
            channel_power_db = np.nan_to_num(channel_power_db, nan=0.0, posinf=0.0, neginf=0.0)
            time_start = int(state.get("channelPowerTimeStart", 1) or 1)
            time_end = int(state.get("channelPowerTimeEnd", time_start + channel_power_db.size - 1) or (time_start + channel_power_db.size - 1))
            if time_end < time_start:
                time_end = time_start + channel_power_db.size - 1
            x_vals = np.arange(time_start, time_start + channel_power_db.size, dtype=float)

            anomaly_mask = np.asarray(state.get("anomalyMask", []), dtype=bool)
            if anomaly_mask.size != channel_power_db.size:
                anomaly_mask = np.zeros(channel_power_db.size, dtype=bool)

            # Keep all previous samples. The x-axis window below only controls the view.
            self.merge_channel_power_history(x_vals, channel_power_db, anomaly_mask)
            x_all = self.power_history_x
            y_all = self.power_history_y
            mask_all = self.power_history_mask

            self.feature_buffer_image.setVisible(False)
            self.channel_power_curve.setVisible(True)
            self.channel_power_current.setVisible(True)
            self.channel_power_curve.setData(x_all, y_all)

            current_x = float(x_vals[-1])
            current_y = float(channel_power_db[-1])
            self.channel_power_current.setData([current_x], [current_y])

            y_lim = state.get("channelPowerYLim", [-22.0, 8.0])
            try:
                y_min, y_max = float(y_lim[0]), float(y_lim[1])
            except Exception:
                y_min, y_max = -22.0, 8.0
            if y_max <= y_min:
                y_min, y_max = -22.0, 8.0
            self.feature_buffer_plot.setYRange(y_min, y_max, padding=0.0)

            try:
                total_frames = int(state.get("totalFrames", max(1000, int(current_x))) or max(1000, int(current_x)))
            except Exception:
                total_frames = max(1000, int(current_x))
            self.power_last_total_frames = total_frames
            self.power_last_current_frame = int(current_x)
            self.apply_channel_power_x_range(total_frames, current_x)

            if mask_all.size == x_all.size and np.any(mask_all):
                idx = np.flatnonzero(mask_all)
                self.channel_power_region.setRegion([x_all[idx[0]] - 0.5, x_all[idx[-1]] + 0.5])
                self.channel_power_region.setVisible(True)
            else:
                self.channel_power_region.setVisible(False)

            # Keep only one compact status line to avoid covering the curve.
            anomaly_enabled = bool(state.get("anomalyEnabled", state.get("anomaly_enabled", False)))
            if anomaly_enabled:
                try:
                    attenuation_db = float(state.get("anomalyAttenuationDb", 0) or 0)
                    power_status = f"Anomaly: -{attenuation_db:.0f} dB"
                except Exception:
                    power_status = "Anomaly: ON"
                self.feature_buffer_status.setText(power_status)
                self.feature_buffer_status.setVisible(True)
                try:
                    x_range = self.feature_buffer_plot.viewRange()[0]
                    x_left, x_right = float(x_range[0]), float(x_range[1])
                    x_pad = 0.02 * max(1.0, x_right - x_left)
                    self.feature_buffer_status.setPos(x_left + x_pad, y_max - 1.2)
                except Exception:
                    self.feature_buffer_status.setPos(float(x_all[0]) if x_all.size else 1.0, y_max - 1.2)
            else:
                # Hide the overlay in normal state. The curve itself already shows the power value.
                self.feature_buffer_status.setText("")
                self.feature_buffer_status.setVisible(False)

        feature_buffer = np.asarray(
            state.get("noisyFeatureBuffer", state.get("featureBuffer", [])),
            dtype=float,
        )
        if channel_power_db.size == 0 and feature_buffer.ndim == 2 and feature_buffer.size:
            feature_buffer = np.nan_to_num(feature_buffer, nan=0.0, posinf=1.0, neginf=0.0)
            feature_buffer = np.clip(feature_buffer, 0.0, 1.0)

            feature_rows, feature_cols = feature_buffer.shape
            time_start = int(state.get("featureTimeStart", 1) or 1)
            time_end = int(state.get("featureTimeEnd", time_start + feature_cols - 1) or (time_start + feature_cols - 1))
            if time_end < time_start:
                time_end = time_start + feature_cols - 1

            self.channel_power_region.setVisible(False)
            self.channel_power_curve.setVisible(False)
            self.channel_power_current.setVisible(False)
            self.feature_buffer_image.setVisible(True)

            # ImageItem uses image.T so that X is collected frame and Y is feature index.
            self.feature_buffer_image.setImage(
                feature_buffer.T,
                autoLevels=False,
                levels=(0.0, 1.0),
            )
            self.feature_buffer_image.setRect(QRectF(time_start - 0.5, 0.5, feature_cols, feature_rows))
            self.feature_buffer_plot.setXRange(time_start - 0.5, time_start + feature_cols - 0.5, padding=0.02)
            self.feature_buffer_plot.setYRange(0.5, feature_rows + 0.5, padding=0.02)
            # Keep the feature-buffer annotation to one line as well.
            if state.get("anomalyEnabled", state.get("anomaly_enabled", False)):
                try:
                    feature_status = f"Anomaly: -{float(state.get('anomalyAttenuationDb', 0)):.0f} dB"
                except Exception:
                    feature_status = "Anomaly: ON"
            else:
                feature_status = f"Feature dim: 1-{feature_rows}"
            self.feature_buffer_status.setText(feature_status)
            self.feature_buffer_status.setVisible(True)
            self.feature_buffer_status.setPos(time_start, max(1.0, feature_rows - 1.0))

        frame = int(state.get("frame", 1) or 1)
        total_frames = int(state.get("totalFrames", max(1000, frame)) or max(1000, frame))
        sample_count = int(state.get("sampleCount", 0) or 0)

        throughput_value = state.get("throughputMbps", state.get("throughput_Mbps", state.get("throughput")))
        if throughput_value is not None:
            try:
                throughput_value = float(throughput_value)
                if np.isfinite(throughput_value):
                    if self.throughput_history_x and frame < self.throughput_history_x[-1]:
                        self.throughput_history_x.clear()
                        self.throughput_history_y.clear()
                    self.throughput_history_x.append(frame)
                    self.throughput_history_y.append(throughput_value)
                    self.throughput_curve.setData(self.throughput_history_x, self.throughput_history_y)
                    self.throughput_plot.setXRange(1, max(10, total_frames, frame), padding=0.02)
                    ymax = max(1.0, max(self.throughput_history_y) * 1.20)
                    self.throughput_plot.setYRange(0.0, ymax, padding=0.0)
            except Exception:
                pass

        fer_value = state.get("fer", state.get("FER"))
        if fer_value is not None:
            try:
                fer_value = float(fer_value)
                if np.isfinite(fer_value):
                    self.fer_history_x.append(frame)
                    self.fer_history_y.append(max(fer_value, 1e-7))
                    self.fer_history_x = self.fer_history_x[-500:]
                    self.fer_history_y = self.fer_history_y[-500:]
                    self.fer_curve.setData(self.fer_history_x, self.fer_history_y)
            except Exception:
                pass

        ber_value = state.get("ber", state.get("BER"))
        if state.get("berValid", True) and ber_value is not None:
            try:
                ber_value = float(ber_value)
                if np.isfinite(ber_value):
                    self.ber_history_x.append(frame)
                    self.ber_history_y.append(max(ber_value, 1e-7))
                    self.ber_history_x = self.ber_history_x[-500:]
                    self.ber_history_y = self.ber_history_y[-500:]
                    self.ber_curve.setData(self.ber_history_x, self.ber_history_y)
            except Exception:
                pass

        mcs_value = state.get("mcsIndex", state.get("mcs_index", state.get("mcs")))
        if mcs_value is not None:
            try:
                mcs_value = float(mcs_value)
                if np.isfinite(mcs_value):
                    if self.mcs_history_x and frame < self.mcs_history_x[-1]:
                        self.mcs_history_x.clear()
                        self.mcs_history_y.clear()
                    self.mcs_history_x.append(frame)
                    self.mcs_history_y.append(mcs_value)
                    self.amc_state_curve.setData(self.mcs_history_x, self.mcs_history_y)
                    self.amc_state_plot.setXRange(1, max(10, total_frames, frame), padding=0.02)
                    ymax = max(8.0, max(self.mcs_history_y) + 1.0)
                    self.amc_state_plot.setYRange(0.0, ymax, padding=0.0)
            except Exception:
                pass

        self.channel_log_counter = getattr(self, "channel_log_counter", 0) + 1
        if self.channel_log_counter % 12 == 1:
            profile = state.get("profile", "NTN-TDL")
            extra = ""
            if throughput_value is not None:
                try:
                    extra += f" | rate={float(throughput_value):.2f}Mbps"
                except Exception:
                    pass
            if mcs_value is not None:
                try:
                    extra += f" | mcs={float(mcs_value):.0f}"
                except Exception:
                    pass
            if state.get("anomalyEnabled", state.get("anomaly_enabled", False)):
                try:
                    extra += f" | anomaly={float(state.get('anomalyAttenuationDb', 0)):.0f}dB"
                except Exception:
                    extra += " | anomaly=on"
            self.append_log(f"[Channel] {profile} frame={frame}/{total_frames} | paths={n_paths} | samples={sample_count}{extra}")

    def write_channel_anomaly_control(self, *args, silent=False):
        enabled = bool(getattr(self, "channel_anomaly_cb", None) and self.channel_anomaly_cb.currentIndex() == 1)
        attenuation_db = 12.0
        payload = {
            "type": "channel_anomaly_control",
            "enabled": enabled,
            "mode": "power_attenuation",
            "attenuationDb": attenuation_db,
            "updatedAt": time.time(),
        }
        try:
            self.channel_anomaly_control_path.parent.mkdir(parents=True, exist_ok=True)
            self.channel_anomaly_control_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            if not silent and hasattr(self, "log_text"):
                state_text = "enabled" if enabled else "disabled"
                self.append_log(f"[Channel] Manual anomaly injection {state_text} ({attenuation_db:.0f} dB attenuation).")
        except Exception as exc:
            if hasattr(self, "log_text"):
                self.append_log(f"[Channel] Failed to write anomaly control: {exc}")

    def append_log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def closeEvent(self, event):
        try:
            if hasattr(self, "dd_widget") and hasattr(self.dd_widget, "close"):
                self.dd_widget.close()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
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
    sys.exit(app.exec())
