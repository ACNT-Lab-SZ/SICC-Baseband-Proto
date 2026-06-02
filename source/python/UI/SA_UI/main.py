import sys
import socket
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QGroupBox, QSplitter,
                               QLabel, QFormLayout, QComboBox, QLineEdit, QFrame, QTextEdit,
                               QSizePolicy, QStackedWidget)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl, Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QPixmap, QPainter, QColor, QPen, QPainterPath
import pyqtgraph as pg

# ==================== 科研风 QSS 全局样式表 ====================
ACADEMIC_STYLE = """
QMainWindow {
    background-color: #0F111A;  
}
#HeaderFrame {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1A1C2C, stop:1 #4A569D);
    border-bottom: 2px solid #2B5797;
    min-height: 75px; 
}
#MainTitle {
    color: #FFFFFF;
    font-size: 28px;
    font-weight: 900; 
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    letter-spacing: 2px; 
}
#SubTitle {
    color: #A0AEC0;
    font-size: 14px;
    font-weight: normal;
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
QSplitter::handle {
    background-color: #2D3748;
    margin: 2px;
    border-radius: 2px;
}
"""


# ==================== UDP 实时数据接收线程 ====================
class IqDataReceiverThread(QThread):
    iq_data_received = Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    log_msg = Signal(str)

    def run(self):
        UDP_IP = "127.0.0.1"
        UDP_PORT = 65432

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((UDP_IP, UDP_PORT))
            self.log_msg.emit(f"[UDP 监听] 成功绑定 {UDP_IP}:{UDP_PORT}，支持动态变长数据包...")
        except Exception as e:
            self.log_msg.emit(f"[UDP 致命错误] 端口绑定失败: {e}")
            return

        packet_count = 0
        while True:
            try:
                data, addr = sock.recvfrom(65536)
                packet_count += 1

                if packet_count == 1:
                    self.log_msg.emit(f"[UDP 通信] 成功接收首帧数据 ({addr[0]}:{addr[1]})！")

                if len(data) % 16 != 0 or len(data) == 0:
                    if packet_count % 20 == 0:
                        self.log_msg.emit(f"[UDP 警告] 收到不规则残缺包，大小: {len(data)} 字节，已丢弃。")
                    continue

                arr = np.frombuffer(data, dtype=np.float32)
                num_points = len(arr) // 4

                tx_i, tx_q = arr[0:num_points], arr[num_points:2 * num_points]
                rx_i, rx_q = arr[2 * num_points:3 * num_points], arr[3 * num_points:4 * num_points]

                self.iq_data_received.emit(tx_i, tx_q, rx_i, rx_q)

            except Exception as e:
                self.log_msg.emit(f"[UDP 运行错误] {e}")
                break


class SpectrumDataReceiverThread(QThread):
    spectrum_received = Signal(np.ndarray, np.ndarray)
    log_msg = Signal(str)

    def run(self):
        UDP_IP = "127.0.0.1"
        UDP_PORT = 65433

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((UDP_IP, UDP_PORT))
            self.log_msg.emit(f"[UDP 频谱] 成功绑定 {UDP_IP}:{UDP_PORT}")
        except Exception as e:
            self.log_msg.emit(f"[UDP 频谱错误] 端口绑定失败: {e}")
            return

        packet_count = 0
        while True:
            try:
                data, addr = sock.recvfrom(65536)
                packet_count += 1
                if len(data) == 0 or len(data) % 8 != 0:
                    continue
                arr = np.frombuffer(data, dtype=np.float32)
                n = len(arr) // 2
                if n <= 0:
                    continue
                if packet_count == 1:
                    self.log_msg.emit(f"[UDP 频谱] 成功接收首帧数据 ({addr[0]}:{addr[1]})")
                self.spectrum_received.emit(arr[:n], arr[n:2 * n])
            except Exception as e:
                self.log_msg.emit(f"[UDP 频谱运行错误] {e}")
                break


class MetricsDataReceiverThread(QThread):
    metrics_received = Signal(float, float, float, float, float, float)
    log_msg = Signal(str)

    def run(self):
        UDP_IP = "127.0.0.1"
        UDP_PORT = 65434

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((UDP_IP, UDP_PORT))
            self.log_msg.emit(f"[UDP 指标] 成功绑定 {UDP_IP}:{UDP_PORT}")
        except Exception as e:
            self.log_msg.emit(f"[UDP 指标错误] 端口绑定失败: {e}")
            return

        packet_count = 0
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                packet_count += 1
                if len(data) != 48:
                    continue
                arr = np.frombuffer(data, dtype=np.float64)
                if packet_count == 1:
                    self.log_msg.emit(f"[UDP 指标] 成功接收首帧数据 ({addr[0]}:{addr[1]})")
                self.metrics_received.emit(float(arr[0]), float(arr[1]), float(arr[2]),
                                           float(arr[3]), float(arr[4]), float(arr[5]))
            except Exception as e:
                self.log_msg.emit(f"[UDP 指标运行错误] {e}")
                break


# ==================== 极简阵列流水灯特效控件 ====================
class CyberDataFlowWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(36)
        self.frame = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(150)

    def animate(self):
        self.frame += 1
        self.update()

    def draw_chevron(self, painter, x, y, size, alpha):
        pen = QPen(QColor(74, 222, 128, alpha), 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(x, y - size / 2)
        path.lineTo(x + size / 2, y)
        path.lineTo(x, y + size / 2)
        painter.drawPath(path)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        y_offsets = [h * 0.35, h * 0.5, h * 0.65]
        spacing = 9
        start_x = (w - (2 * spacing) - 4) / 2

        for i in range(3):
            x = start_x + i * spacing
            offset = (self.frame - i) % 4
            if offset == 0:
                alpha = 255
            elif offset == 1:
                alpha = 120
            elif offset == 2:
                alpha = 40
            else:
                alpha = 10
            for y in y_offsets:
                self.draw_chevron(painter, x, y, 8, alpha)


# ==================== 主界面 ====================
class SatelliteSystemUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("卫星通用译码器与链路自适应传输系统")
        self.resize(1600, 950)
        self.setStyleSheet(ACADEMIC_STYLE)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        total_layout = QVBoxLayout(central_widget)
        total_layout.setContentsMargins(0, 0, 0, 0)
        total_layout.setSpacing(0)

        self.create_header(total_layout)

        body_container = QWidget()
        body_layout = QHBoxLayout(body_container)
        body_layout.setContentsMargins(15, 15, 15, 15)
        total_layout.addWidget(body_container)

        self.main_splitter = QSplitter(Qt.Horizontal)
        body_layout.addWidget(self.main_splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)

        self.left_vertical_splitter = QSplitter(Qt.Vertical)
        left_layout.addWidget(self.left_vertical_splitter)

        self.init_control_panel()
        self.init_log_panel()
        self.init_cyber_physical_panel()
        self.left_vertical_splitter.setSizes([260, 150, 390])

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)

        self.init_dashboard_panel(right_layout)

        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(right_widget)
        self.main_splitter.setSizes([600, 1000])

        self.udp_thread = IqDataReceiverThread()
        self.udp_thread.iq_data_received.connect(self.update_iq_plot)
        self.udp_thread.log_msg.connect(self.append_log)
        self.udp_thread.start()

        self.spectrum_thread = SpectrumDataReceiverThread()
        self.spectrum_thread.spectrum_received.connect(self.update_spectrum_plot)
        self.spectrum_thread.log_msg.connect(self.append_log)
        self.spectrum_thread.start()

        self.metrics_thread = MetricsDataReceiverThread()
        self.metrics_thread.metrics_received.connect(self.update_metrics_plot)
        self.metrics_thread.log_msg.connect(self.append_log)
        self.metrics_thread.start()

    def create_header(self, parent_layout):
        header_frame = QFrame()
        header_frame.setObjectName("HeaderFrame")
        header_layout = QVBoxLayout(header_frame)
        title_label = QLabel("卫星通用译码器与链路自适应传输系统")
        title_label.setObjectName("MainTitle")
        title_label.setAlignment(Qt.AlignCenter)
        sub_title = QLabel("哈尔滨工业大学(深圳) 广东省空天通信重点实验室")
        sub_title.setObjectName("SubTitle")
        sub_title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)
        header_layout.addWidget(sub_title)
        parent_layout.addWidget(header_frame)

    def create_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #E2E8F0; font-family: 'Microsoft YaHei';")
        return lbl

    def init_control_panel(self):
        group = QGroupBox("系统参数配置 (System Config)")
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 20, 15, 15)
        main_layout.setSpacing(15)
        group.setLayout(main_layout)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["✦  通用译码器配置模式", "✦  链路自适应配置模式"])
        self.mode_selector.setStyleSheet("""
            QComboBox {
                background-color: #1E3A8A; 
                color: #FFFFFF;
                font-size: 15px;
                font-weight: bold;
                font-family: 'Microsoft YaHei';
                padding: 8px 15px;
                border: 2px solid #3B82F6; 
                border-radius: 6px;
            }
            QComboBox::drop-down { border: none; padding-right: 15px; }
            QComboBox QAbstractItemView {
                background-color: #0F111A;
                color: #93C5FD;
                selection-background-color: #2563EB;
                selection-color: #FFFFFF;
                border: 2px solid #3B82F6;
                border-radius: 4px;
                outline: none;
            }
            QComboBox QAbstractItemView::item { min-height: 40px; padding-left: 10px; }
        """)
        main_layout.addWidget(self.mode_selector)

        self.config_stack = QStackedWidget()

        # 面板 A
        panel_decoder = QWidget()
        form_decoder = QFormLayout(panel_decoder)
        form_decoder.setContentsMargins(5, 10, 5, 0)
        form_decoder.setVerticalSpacing(15)
        self.coding_cb = QComboBox()
        self.coding_cb.addItems(["Polar码", "LDPC码"])
        self.platform_cb = QComboBox()
        self.platform_cb.addItems(["启用", "禁用"])
        self.standard_cb = QComboBox()
        self.standard_cb.addItems(["3GPP", "DVB-S2X", "CCSDS"])
        form_decoder.addRow(self.create_label("编码方式:"), self.coding_cb)
        form_decoder.addRow(self.create_label("GPU译码:"), self.platform_cb)
        form_decoder.addRow(self.create_label("通信体制:"), self.standard_cb)

        # 面板 B
        panel_amc = QWidget()
        form_amc = QFormLayout(panel_amc)
        form_amc.setContentsMargins(5, 10, 5, 0)
        form_amc.setVerticalSpacing(15)
        self.ce_data_cb = QComboBox()
        self.ce_data_cb.addItems(["启用", "禁用"])
        self.cp_method_cb = QComboBox()
        self.cp_method_cb.addItems(["启用", "禁用"])
        self.amc_mode_cb = QComboBox()
        self.amc_mode_cb.addItems(["启用", "禁用"])
        form_amc.addRow(self.create_label("信道状态信息采集:"), self.ce_data_cb)
        form_amc.addRow(self.create_label("信道状态信息预测:"), self.cp_method_cb)
        form_amc.addRow(self.create_label("基于AMC链路传输:"), self.amc_mode_cb)

        self.config_stack.addWidget(panel_decoder)
        self.config_stack.addWidget(panel_amc)
        main_layout.addWidget(self.config_stack)
        self.mode_selector.currentIndexChanged.connect(self.config_stack.setCurrentIndex)

        main_layout.addStretch()
        self.btn_run = QPushButton("▶ 下发配置参数并启动测试")
        main_layout.addWidget(self.btn_run)
        self.left_vertical_splitter.addWidget(group)

    def init_log_panel(self):
        group = QGroupBox("系统状态与运行日志")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 15, 10, 10)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            background-color: #0b0e14; 
            color: #4ade80; 
            font-family: Consolas, 'Courier New', monospace; 
            font-size: 13px;
            border: 1px solid #2D3748;
            padding: 5px;
        """)
        self.log_text.append("[System] Initializing Hardware USRP SDR...")
        self.log_text.append("[Model] AI Engine ready for task-driven inference.")
        layout.addWidget(self.log_text)
        group.setLayout(layout)
        self.left_vertical_splitter.addWidget(group)

    def init_cyber_physical_panel(self):
        group = QGroupBox("软硬件协同的虚实交互验证")
        web_layout = QHBoxLayout()
        web_layout.setContentsMargins(5, 20, 5, 10)
        web_layout.setSpacing(5)

        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://satellitemap.space/vis/constellation/qianfan"))
        self.browser.setStyleSheet("background-color: transparent;")
        self.browser.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.flow_widget = CyberDataFlowWidget()

        self.usrp_pic_label = QLabel()
        self.usrp_pic_label.setAlignment(Qt.AlignCenter)
        self.usrp_pic_label.setStyleSheet("background-color: #050505; border: 2px solid #3A4A69; border-radius: 6px;")
        self.usrp_pic_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.usrp_pixmap = QPixmap("usrp_real.jpg")
        if not self.usrp_pixmap.isNull():
            scaled = self.usrp_pixmap.scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.usrp_pic_label.setPixmap(scaled)
            self.usrp_pic_label.setScaledContents(True)
        else:
            self.usrp_pic_label.setText("[ 硬件实物接入端 ]\n\n缺少 usrp_real.jpg")
            self.usrp_pic_label.setStyleSheet("color: #A0AEC0; font-family: Consolas;")

        web_layout.addWidget(self.browser, stretch=10)
        web_layout.addWidget(self.flow_widget, stretch=0)
        web_layout.addWidget(self.usrp_pic_label, stretch=9)

        group.setLayout(web_layout)
        self.left_vertical_splitter.addWidget(group)

    def init_dashboard_panel(self, layout):
        group = QGroupBox("实时性能监测面板 (Real-time Dashboard)")
        dash_layout = QVBoxLayout()
        dash_layout.setContentsMargins(10, 20, 10, 10)

        pg.setConfigOption('background', '#1A202C')
        pg.setConfigOption('foreground', '#A0AEC0')
        pg.setConfigOptions(antialias=True)
        axis_font = QFont("Consolas", 10)

        # ==================== 极具科技感的雷达/示波器风 IQ 监控双屏 ====================
        iq_layout = QHBoxLayout()
        iq_layout.setSpacing(15)

        # ---------------- 1. 左屏：发端基准 (Tx) ----------------
        self.tx_plot = pg.PlotWidget(
            title="<span style='color: #FBBF24; font-size: 11pt; font-family: Microsoft YaHei;'>▶ USRP发送信号星座图 (Tx)</span>")
        self.tx_plot.setBackground('#080B14')
        self.tx_plot.showGrid(x=True, y=True, alpha=0.15)
        self.tx_plot.setXRange(-1.8, 1.8)
        self.tx_plot.setYRange(-1.8, 1.8)
        self.tx_plot.getAxis('bottom').setTickFont(axis_font)
        self.tx_plot.getAxis('left').setTickFont(axis_font)

        self.tx_plot.addItem(pg.InfiniteLine(angle=90, movable=False,
                                             pen=pg.mkPen(color=(255, 255, 255, 40), width=1, style=Qt.DashLine)))
        self.tx_plot.addItem(pg.InfiniteLine(angle=0, movable=False,
                                             pen=pg.mkPen(color=(255, 255, 255, 40), width=1, style=Qt.DashLine)))

        # 【修复重点 1】使用正确的 pen/brush 参数，金黄色高亮边框，无填充
        self.tx_scatter = pg.ScatterPlotItem(
            symbol='+',
            size=12,
            pen=pg.mkPen(color=QColor(251, 191, 36, 255), width=2),  # 金黄色 (Neon Gold)
            brush=None,  # 无填充
            name="Tx Ideal"
        )
        self.tx_plot.addItem(self.tx_scatter)

        # ---------------- 2. 右屏：收端带噪 (Rx) ----------------
        self.rx_plot = pg.PlotWidget(
            title="<span style='color: #00E5FF; font-size: 11pt; font-family: Microsoft YaHei;'>▶ USRP接收信号星座图 (Rx)</span>")
        self.rx_plot.setBackground('#080B14')
        self.rx_plot.showGrid(x=True, y=True, alpha=0.15)
        self.rx_plot.setXRange(-1.8, 1.8)
        self.rx_plot.setYRange(-1.8, 1.8)
        self.rx_plot.getAxis('bottom').setTickFont(axis_font)
        self.rx_plot.getAxis('left').setTickFont(axis_font)

        self.rx_plot.addItem(
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(0, 229, 255, 40), width=1, style=Qt.DashLine)))
        self.rx_plot.addItem(
            pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color=(0, 229, 255, 40), width=1, style=Qt.DashLine)))

        # 【修复重点 2】无边框，采用高透明度的霓虹青色填充，密集处自然发亮
        self.rx_scatter = pg.ScatterPlotItem(
            symbol='o',
            size=6,
            pen=None,  # 无边框
            brush=pg.mkBrush(QColor(0, 229, 255, 120)),  # 霓虹青色 (Cyber Cyan) + 透明度
            name="Rx Real"
        )
        self.rx_plot.addItem(self.rx_scatter)

        iq_layout.addWidget(self.tx_plot)
        iq_layout.addWidget(self.rx_plot)
        dash_layout.addLayout(iq_layout)
        # ==============================================================================

        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setTitle("实时频谱瀑布分析 (Power Spectral Density)", color='#E2E8F0', size='12pt', bold=True)
        self.spec_plot.setLabel('bottom', 'Frequency Offset', units='Hz')
        self.spec_plot.setLabel('left', 'Power', units='dB')
        self.spec_plot.getAxis('bottom').setTickFont(axis_font)
        self.spec_plot.getAxis('left').setTickFont(axis_font)
        self.spec_plot.showGrid(x=True, y=True, alpha=0.15)
        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen(color='#4ADE80', width=2))

        self.ber_plot = pg.PlotWidget()
        self.ber_plot.setTitle("实时译码性能曲线 (BER Performance)", color='#E2E8F0', size='12pt', bold=True)
        self.ber_plot.setLabel('bottom', 'Frame')
        self.ber_plot.setLabel('left', 'Rate')
        self.ber_plot.setLogMode(y=True)
        self.ber_plot.addLegend()
        self.ber_curve = self.ber_plot.plot(pen=pg.mkPen(color='#63B3ED', width=2), name="AI-Decoder")
        self.fer_curve = self.ber_plot.plot(pen=pg.mkPen(color='#F87171', width=2), name="FER")
        self.ber_plot.getAxis('bottom').setTickFont(axis_font)
        self.ber_plot.getAxis('left').setTickFont(axis_font)
        self.ber_plot.showGrid(x=True, y=True, alpha=0.15)
        self.metric_frames = []
        self.metric_ber = []
        self.metric_fer = []

        dash_layout.addWidget(self.spec_plot)
        dash_layout.addWidget(self.ber_plot)
        group.setLayout(dash_layout)
        layout.addWidget(group)

    def update_iq_plot(self, tx_i, tx_q, rx_i, rx_q):
        self.tx_scatter.setData(tx_i, tx_q)
        self.rx_scatter.setData(rx_i, rx_q)

    def update_spectrum_plot(self, freq_hz, psd_db):
        self.spec_curve.setData(freq_hz, psd_db)

    def update_metrics_plot(self, frames, ber, fer, preber, snr_db, goodput_mbps):
        self.metric_frames.append(frames)
        self.metric_ber.append(max(ber, 1e-9))
        self.metric_fer.append(max(fer, 1e-9))
        if len(self.metric_frames) > 800:
            self.metric_frames = self.metric_frames[-800:]
            self.metric_ber = self.metric_ber[-800:]
            self.metric_fer = self.metric_fer[-800:]
        self.ber_curve.setData(self.metric_frames, self.metric_ber)
        self.fer_curve.setData(self.metric_frames, self.metric_fer)

    def append_log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = SatelliteSystemUI()
    window.show()
    sys.exit(app.exec())
