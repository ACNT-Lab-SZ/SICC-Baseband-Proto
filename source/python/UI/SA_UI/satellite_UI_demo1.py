import sys
import os
import socket
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QGroupBox, QSplitter,
                               QLabel, QFormLayout, QComboBox, QLineEdit, QFrame, QTextEdit,
                               QSizePolicy, QStackedWidget, QGridLayout)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl, Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QPixmap, QPainter, QColor, QPen, QPainterPath, QImage
try:
    import pyqtgraph as pg
except ModuleNotFoundError:
    local_site_packages = os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.9", "site-packages")
    if os.path.isdir(local_site_packages):
        sys.path.append(local_site_packages)
    import pyqtgraph as pg

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

    def __init__(self):
        super().__init__()
        self.running = True
        self.sock = None

    def stop(self):
        self.running = False
        if self.sock is not None:
            self.sock.close()

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock = sock
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(0.2)
            sock.bind(("127.0.0.1", 65432))
        except Exception:
            return
        while self.running:
            try:
                data, _ = sock.recvfrom(65536)
                if len(data) % 16 != 0 or len(data) == 0: continue
                arr = np.frombuffer(data, dtype=np.float32)
                num_points = len(arr) // 4
                tx_i, tx_q = arr[0:num_points], arr[num_points:2 * num_points]
                rx_i, rx_q = arr[2 * num_points:3 * num_points], arr[3 * num_points:4 * num_points]
                self.iq_data_received.emit(tx_i, tx_q, rx_i, rx_q)
            except socket.timeout:
                continue
            except Exception:
                break


# ==================== 线程 2：UDP 频谱数据接收 ====================
class SpectrumDataReceiverThread(QThread):
    spec_data_received = Signal(np.ndarray, np.ndarray)
    log_msg = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.sock = None

    def stop(self):
        self.running = False
        if self.sock is not None:
            self.sock.close()

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock = sock
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(0.2)
            sock.bind(("127.0.0.1", 65433))
        except Exception:
            return
        while self.running:
            try:
                data, _ = sock.recvfrom(65536)
                if len(data) % 8 != 0 or len(data) == 0: continue
                arr = np.frombuffer(data, dtype=np.float32)
                num_points = len(arr) // 2
                freqs, powers = arr[0:num_points], arr[num_points:2 * num_points]
                self.spec_data_received.emit(freqs, powers)
            except socket.timeout:
                continue
            except Exception:
                break


# ==================== 线程 3：UDP 【双路】视频流接收 ====================
class VideoDataReceiverThread(QThread):
    video_frame_received = Signal(np.ndarray, np.ndarray)
    log_msg = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.sock = None

    def stop(self):
        self.running = False
        if self.sock is not None:
            self.sock.close()

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock = sock
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(0.2)
            sock.bind(("127.0.0.1", 65434))
        except Exception:
            return
        while self.running:
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
            except socket.timeout:
                continue
            except Exception:
                break


# ==================== 线程 4：UDP BER/FER 性能指标接收 ====================
class MetricsDataReceiverThread(QThread):
    metric_data_received = Signal(np.ndarray, np.ndarray)
    log_msg = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.sock = None

    def stop(self):
        self.running = False
        if self.sock is not None:
            self.sock.close()

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock = sock
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(0.2)
            sock.bind(("127.0.0.1", 65435))
        except Exception:
            return
        while self.running:
            try:
                data, _ = sock.recvfrom(65536)
                if len(data) == 0:
                    continue
                if len(data) % 16 == 0:
                    arr = np.frombuffer(data, dtype=np.float64)
                elif len(data) % 8 == 0:
                    arr = np.frombuffer(data, dtype=np.float32).astype(np.float64)
                else:
                    continue
                if len(arr) == 6:
                    frames, ber, fer, _preber, _snr_db, _goodput_mbps = arr
                    y = ber if np.isfinite(ber) and ber > 0 else fer
                    self.metric_data_received.emit(
                        np.array([frames], dtype=np.float64),
                        np.array([y], dtype=np.float64))
                    continue
                if len(arr) < 2 or len(arr) % 2 != 0:
                    continue
                pairs = arr.reshape(-1, 2)
                self.metric_data_received.emit(pairs[:, 0], pairs[:, 1])
            except socket.timeout:
                continue
            except Exception:
                break


# ==================== 极简阵列流水灯特效 ====================
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
class SatelliteSystemUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("卫星通用译码器与链路自适应传输系统")
        self.resize(1600, 950)
        self.setStyleSheet(ACADEMIC_STYLE)

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
        self.ber_x = np.array([], dtype=np.float64)
        self.ber_y = np.array([], dtype=np.float64)
        self.metric_thread = MetricsDataReceiverThread();
        self.metric_thread.metric_data_received.connect(self.update_ber_plot);
        self.metric_thread.log_msg.connect(self.append_log);
        self.metric_thread.start()

    def create_header(self, parent_layout):
        header_frame = QFrame();
        header_frame.setObjectName("HeaderFrame");
        header_layout = QVBoxLayout(header_frame)
        header_layout.setSpacing(6)

        # === 修复 1：精简为两行，竞赛名与组名放一行 ===
        title_label = QLabel("面向卫星互联网的异构高速通用译码与预测驱动自适应传输系统")
        title_label.setObjectName("MainTitle")
        title_label.setAlignment(Qt.AlignCenter)

        # 将赛事名称与组名用竖线分隔，并仅对组名做高亮
        sub_title = QLabel(
            "第二十一届中国研究生电子设计竞赛 &nbsp;|&nbsp; 团队名：<span style='color: #38BDF8;'>【问天译码】</span>")
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
        self.coding_cb.addItems(["Polar码", "LDPC码"])
        self.platform_cb = QComboBox();
        self.platform_cb.addItems(["启用", "禁用"])
        self.standard_cb = QComboBox();
        self.standard_cb.addItems(["3GPP", "DVB-S2X", "CCSDS"])
        form_decoder.addRow(self.create_label("编码方式:"), self.coding_cb);
        form_decoder.addRow(self.create_label("GPU译码:"), self.platform_cb);
        form_decoder.addRow(self.create_label("通信体制:"), self.standard_cb)

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
        self.browser = QWebEngineView();
        self.browser.setUrl(QUrl("https://satellitemap.space/vis/constellation/qianfan"));
        self.browser.setStyleSheet("background-color: transparent;");
        self.browser.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
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
            title="<span style='color: #818CF8; font-size: 11pt; font-family: Microsoft YaHei;'>▶ 实时译码性能曲线 (BER)</span>")
        style_plot(self.ber_plot)
        self.ber_plot.setLogMode(y=True);
        self.ber_plot.addLegend(offset=(-10, 10));
        self.ber_plot.setLabel('bottom', 'Frame', **{'font-family': 'Consolas'})
        self.ber_plot.setLabel('left', 'BER', **{'font-family': 'Consolas'})
        self.ber_curve = self.ber_plot.plot(pen=pg.mkPen(color='#63B3ED', width=2), name="AI-Decoder")
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
        self.csi_curve = self.csi_plot.plot(pen=pg.mkPen(color='#4ADE80', width=2), fillLevel=0,
                                            fillBrush=(74, 222, 128, 50))

        self.throughput_plot = pg.PlotWidget(
            title="<span style='color: #A78BFA; font-size: 11pt; font-family: Microsoft YaHei;'>▶ 系统动态吞吐量 (System Throughput)</span>")
        style_plot(self.throughput_plot)
        self.throughput_curve = self.throughput_plot.plot(pen=pg.mkPen(color='#A78BFA', width=2), fillLevel=0,
                                                          fillBrush=(167, 139, 250, 50))

        self.amc_state_plot = pg.PlotWidget(
            title="<span style='color: #F472B6; font-size: 11pt; font-family: Microsoft YaHei;'>▶ MCS 阶数自适应切换轨迹 (AMC State Tracker)</span>")
        style_plot(self.amc_state_plot)
        self.amc_state_curve = self.amc_state_plot.plot(pen=pg.mkPen(color='#F472B6', width=2), symbol='s',
                                                        symbolSize=8, symbolBrush='#F472B6')

        amc_v_splitter.addWidget(self.csi_plot)
        amc_v_splitter.addWidget(self.throughput_plot)
        amc_v_splitter.addWidget(self.amc_state_plot)

        amc_layout.addWidget(amc_v_splitter)

        self.dashboard_stack.addWidget(decoder_view);
        self.dashboard_stack.addWidget(amc_view)
        main_dash_layout.addWidget(self.dashboard_stack);
        group.setLayout(main_dash_layout);
        layout.addWidget(group)

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

    def update_ber_plot(self, x, ber):
        if self.dashboard_stack.currentIndex() != 0:
            return
        x = np.asarray(x, dtype=np.float64)
        ber = np.maximum(np.asarray(ber, dtype=np.float64), 1e-8)
        self.ber_x = np.concatenate((self.ber_x, x))[-240:]
        self.ber_y = np.concatenate((self.ber_y, ber))[-240:]
        self.ber_curve.setData(self.ber_x, self.ber_y)

    def append_log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def closeEvent(self, event):
        for thread in (
            getattr(self, "iq_thread", None),
            getattr(self, "spec_thread", None),
            getattr(self, "video_thread", None),
            getattr(self, "metric_thread", None),
        ):
            if thread is not None:
                thread.stop()
        for thread in (
            getattr(self, "iq_thread", None),
            getattr(self, "spec_thread", None),
            getattr(self, "video_thread", None),
            getattr(self, "metric_thread", None),
        ):
            if thread is not None:
                thread.wait(1000)
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
