import sys
import os
import socket
import json
import time
import math
import re
import secrets
import subprocess
from pathlib import Path
import numpy as np
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault(
    "QT_LOGGING_RULES",
    "qt.multimedia.ffmpeg.*=false;qt.multimedia.*.warning=false",
)
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--ignore-gpu-blocklist --enable-webgl --use-angle=d3d11",
)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QGroupBox, QSplitter,
                               QLabel, QFormLayout, QComboBox, QLineEdit, QFrame, QTextEdit,
                               QSizePolicy, QStackedWidget, QGridLayout, QProgressBar)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtMultimedia import QMediaPlayer, QVideoSink
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


def find_project_root(start_dir: Path) -> Path:
    for path in [start_dir, *start_dir.parents]:
        if (path / "requirements.txt").exists() and (path / "source").exists():
            return path
        if path.name == "organized_workspace_20260601":
            return path
    return start_dir


PROJECT_ROOT = find_project_root(APP_DIR)


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path and path.exists():
            return path
    return paths[0]


CODE_MATRICES_ROOT = first_existing(
    PROJECT_ROOT / "data" / "code_matrices" / "Code_Matrices_Lib",
    PROJECT_ROOT / "Code_Matrices_Lib",
)
AI_V3_SOURCE_ROOT = first_existing(
    PROJECT_ROOT / "source" / "python" / "Ai_process_V3",
    PROJECT_ROOT / "Ai_process_V3",
)
AI_V3_MEDIA_ROOT = first_existing(
    PROJECT_ROOT / "data" / "media" / "Ai_process_V3",
    PROJECT_ROOT / "Ai_process_V3",
)
AI_V3_MODEL_ROOT = first_existing(
    PROJECT_ROOT / "data" / "models" / "Ai_process_V3",
    PROJECT_ROOT / "Ai_process_V3",
)
SATELLITE_UI_SOURCE_ROOT = first_existing(
    PROJECT_ROOT / "scripts" / "python_tools" / "Satellite_UI",
    PROJECT_ROOT / "Satellite_UI",
)
UI_NEW_MEDIA_ROOT = first_existing(
    PROJECT_ROOT / "data" / "media" / "UI_NEW",
    PROJECT_ROOT / "UI_NEW",
)
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
TRANSMISSION_ROOT = RUNTIME_ROOT / "transmission_file"
TRANSMISSION_SAMPLE_ROOT = first_existing(
    PROJECT_ROOT / "data" / "media" / "transmission_file",
    PROJECT_ROOT / "transmission_file",
    TRANSMISSION_ROOT,
)


def path_from_env(name: str, *parts: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value)
    for part in parts:
        path = path / part
    return path


def configure_external_runtime_env(env: dict) -> dict:
    uhd_root = path_from_env("UHD_ROOT") or path_from_env("UHD_PKG_PATH")
    if uhd_root and uhd_root.exists():
        env["UHD_PKG_PATH"] = str(uhd_root)
        env["UHD_IMAGES_DIR"] = str(uhd_root / "share" / "uhd" / "images")
        env["UHD_RFNOC_DIR"] = str(uhd_root / "share" / "uhd" / "rfnoc")

    path_entries = []
    torch_lib = path_from_env("TORCH_LIB_DIR")
    if torch_lib is None:
        try:
            import torch
            torch_lib = Path(torch.__file__).resolve().parent / "lib"
        except Exception:
            torch_lib = None
    cuda_bin = path_from_env("CUDA_PATH", "bin")
    cuda_osd_dir = path_from_env("CUDA_OSD_DLL_DIR")
    if cuda_osd_dir is None:
        cuda_osd_root = path_from_env("CUDA_OSD_ROOT")
        if cuda_osd_root is not None:
            cuda_osd_dir = cuda_osd_root / "build" / "vs2022" / "Release"
    for candidate in (
        torch_lib,
        uhd_root / "bin" if uhd_root else None,
        cuda_bin,
        cuda_osd_dir,
    ):
        if candidate and Path(candidate).exists():
            path_entries.append(str(candidate))
    path_entries.append(env.get("PATH", ""))
    env["PATH"] = ";".join(path_entries)
    return env

try:
    from model_update import ModelRegistry, ModelUpdatePacketizer, ModelUpdateReassembler, TASKS as MODEL_UPDATE_TASKS
except Exception:
    ModelRegistry = None
    ModelUpdatePacketizer = None
    ModelUpdateReassembler = None
    MODEL_UPDATE_TASKS = {}

try:
    from usrp_sync_task import UsrpSyncTaskThread
except Exception:
    UsrpSyncTaskThread = None


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
    min-height: 46px;
    max-height: 56px;
}
#MainTitle {
    color: #FFFFFF;
    font-size: 20px;
    font-weight: 900; 
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    letter-spacing: 1px; 
}
#SubTitle {
    color: #A0AEC0;
    font-size: 11px;
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
            self.log_msg.emit("[Video] Listening for TX/RX video frames on UDP 65434...")
        except Exception as exc:
            self.log_msg.emit(f"[Video] UDP 65434 bind failed: {exc}")
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


class OfflineGpuPipelineThread(QThread):
    log_msg = Signal(str)
    finished_metrics = Signal(dict)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = dict(config)
        self.proc = None

    def stop(self):
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass

    def run(self):
        repo = Path(self.config.get("repo", APP_DIR.parent))
        exe = repo / "build" / "uhd_cpp_gpu_pipeline" / "Release" / "uhd_ldpc_ofdm_link.exe"
        input_file = Path(self.config["input_file"])
        output_file = Path(self.config["output_file"])
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if not exe.exists():
            self.log_msg.emit(f"[RX] 离线 GPU-Pipeline 可执行文件不存在: {exe}")
            self.finished_metrics.emit({"ok": False, "reason": "exe_missing"})
            return
        if not input_file.exists():
            self.log_msg.emit(f"[RX] 输入业务文件不存在: {input_file}")
            self.finished_metrics.emit({"ok": False, "reason": "input_missing"})
            return

        cmd = [
            str(exe),
            "--mode", "gpu-sim",
            "--traffic", "file",
            "--input", str(input_file),
            "--output", str(output_file),
            "--alist", self.config["alist"],
            "--modulation", self.config["modulation"],
            "--decoder", self.config["decoder"],
            "--sim-snr-db", str(self.config["snr_db"]),
            "--rate", str(self.config.get("rate_hz", 15.36e6)),
            "--nfft", "1024",
            "--cp", str(self.config["cp"]),
            "--num-symbols", str(self.config["num_symbols"]),
            "--active-sc", str(self.config["active_sc"]),
            "--pilot-period", "4",
            "--ldpc-iter", str(self.config["ldpc_iter"]),
            "--ldpc-normalization", "0.95",
            "--ldpc-offset", "0",
            "--ldpc-damping", "0",
            "--ldpc-schedule", "2",
            "--test-seed", str(self.config["test_seed"]),
            "--gpu-pipeline",
            "--ui-constellation",
            "--ui-spectrum",
            "--ui-ntn",
            "--profile-pipeline",
            "--suppress-error-frames",
        ]
        if self.config.get("sim_snr_trace"):
            cmd += ["--sim-snr-trace", str(self.config["sim_snr_trace"])]
        if self.config.get("gpu_sim_wall_clock_throughput"):
            cmd.append("--gpu-sim-wall-clock-throughput")
        if self.config.get("adaptive_power_trace"):
            cmd += [
                "--adaptive-power-trace", str(self.config["adaptive_power_trace"]),
                "--adaptive-power-base-snr-db", str(self.config.get("adaptive_power_base_snr_db", self.config["snr_db"])),
                "--adaptive-enable-16qam",
                "--tx-repeat-min", str(self.config.get("tx_repeat_min", 1)),
                "--tx-repeat-max", str(self.config.get("tx_repeat_max", 3)),
                "--adaptive-power-hold-frames", str(self.config.get("adaptive_power_hold_frames", 300)),
            ]
            if self.config.get("adaptive_power_trace_live"):
                cmd += [
                    "--adaptive-power-trace-live",
                    "--adaptive-power-live-wait-ms", str(self.config.get("adaptive_power_live_wait_ms", 2000)),
                ]
        if self.config.get("systematic_front_info", False):
            cmd.append("--systematic-front-info")

        env = configure_external_runtime_env(os.environ.copy())

        log_path = log_dir / "offline_gpu_pipeline.log"
        metrics = {
            "ok": False,
            "business_mode": self.config.get("business_mode", ""),
            "input_file": str(input_file),
            "output_file": str(output_file),
            "log_file": str(log_path),
            "cmd": " ".join(cmd),
            "elapsed_sec": 0.0,
            "frames": 0,
            "ok_frames": 0,
            "err_frames": 0,
            "fer": None,
            "goodput_mbps": None,
            "avg_snr_db": None,
        }

        self.log_msg.emit(f"[RX] 启动离线全 GPU-Pipeline: {self.config.get('business_mode', '--')}")
        self.log_msg.emit(f"[RX] SNR={self.config['snr_db']} dB, FEC={Path(self.config['alist']).name}, modulation={self.config['modulation']}, decoder={self.config['decoder']}, seed={self.config['test_seed']}")
        if self.config.get("adaptive_power_trace"):
            self.log_msg.emit(
                "[RX] AMC预测功率驱动已启用: trace={trace}, baseSNR={base} dB, repeat={rmin}..{rmax}".format(
                    trace=self.config["adaptive_power_trace"],
                    base=self.config.get("adaptive_power_base_snr_db", self.config["snr_db"]),
                    rmin=self.config.get("tx_repeat_min", 1),
                    rmax=self.config.get("tx_repeat_max", 3),
                )
            )
            self.log_msg.emit(
                f"[RX] 信道增益映射: 每个增益样本作用于 "
                f"{int(self.config.get('adaptive_power_hold_frames', 300))} 个传输帧。"
            )
            self.log_msg.emit("[RX] AMC功率映射说明: 使用模拟逐帧信道增益；不使用MATLAB预测结果，不启用repeat。")
        if self.config.get("gpu_sim_wall_clock_throughput"):
            self.log_msg.emit("[RX] 自适应离线仿真吞吐量统计已去除硬件采样率限制，按GPU实际处理耗时计算。")
        t0 = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8", errors="ignore") as log_f:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    env=env,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    line = line.rstrip()
                    log_f.write(line + "\n")
                    lower = line.lower()
                    if (
                        line.startswith("[MEDIA]") or
                        line.startswith("[GPU-SIM]") or
                        (line.startswith("[BENCH]") and "decode.reference_ber" not in line) or
                        "RX summary" in line or
                        " FER=" in line or
                        "goodput=" in line or
                        "error:" in lower or
                        "failed" in lower
                    ):
                        self.log_msg.emit("[RX] " + line)
                    m = re.search(
                        r"frames=(\d+)\s+ok=(\d+)\s+err=(\d+)\s+FER=([0-9.eE+\-]+).*?goodput=([0-9.eE+\-]+)\s+Mbps.*?avgSNR=([0-9.eE+\-]+)",
                        line,
                    )
                    if m:
                        metrics.update({
                            "frames": int(m.group(1)),
                            "ok_frames": int(m.group(2)),
                            "err_frames": int(m.group(3)),
                            "fer": float(m.group(4)),
                            "goodput_mbps": float(m.group(5)),
                            "avg_snr_db": float(m.group(6)),
                        })
                code = self.proc.wait()
                metrics["exit_code"] = int(code)
                metrics["output_exists"] = output_file.exists() and output_file.stat().st_size > 0
                metrics["ok"] = (code == 0)
        except Exception as exc:
            metrics["reason"] = str(exc)
            self.log_msg.emit(f"[RX] 离线 GPU-Pipeline 启动/运行失败: {exc}")
        finally:
            metrics["elapsed_sec"] = time.perf_counter() - t0
            self.finished_metrics.emit(metrics)


class UsrpFileTransferThread(QThread):
    log_msg = Signal(str)
    finished_metrics = Signal(dict)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = dict(config)
        self.proc = None

    def stop(self):
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass

    @staticmethod
    def parse_summary_line(metrics, line):
        m = re.search(
            r"frames=(\d+)\s+ok=(\d+)\s+err=(\d+)\s+FER=([0-9.eE+\-]+).*?goodput=([0-9.eE+\-]+)\s+Mbps",
            line,
        )
        if m:
            metrics.update({
                "frames": int(m.group(1)),
                "ok_frames": int(m.group(2)),
                "err_frames": int(m.group(3)),
                "fer": float(m.group(4)),
                "goodput_mbps": float(m.group(5)),
            })

    def run(self):
        repo = Path(self.config.get("repo", APP_DIR.parent))
        script = repo / "uhd_cpp" / "scripts" / "start_gated_usrp_pair.ps1"
        input_file = Path(self.config["input_file"])
        output_file = Path(self.config["output_file"])
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        metrics = {
            "ok": False,
            "transport": "usrp",
            "business_mode": self.config.get("business_mode", ""),
            "input_file": str(input_file),
            "output_file": str(output_file),
            "log_dir": str(log_dir),
            "elapsed_sec": 0.0,
            "frames": 0,
            "ok_frames": 0,
            "err_frames": 0,
            "fer": None,
            "goodput_mbps": None,
        }
        if not script.exists():
            self.log_msg.emit(f"[RX] USRP 启动脚本不存在: {script}")
            metrics["reason"] = "script_missing"
            self.finished_metrics.emit(metrics)
            return
        if not input_file.exists():
            self.log_msg.emit(f"[RX] USRP 输入业务文件不存在: {input_file}")
            metrics["reason"] = "input_missing"
            self.finished_metrics.emit(metrics)
            return

        powershell = "powershell"
        cmd = [
            powershell,
            "-ExecutionPolicy", "Bypass",
            "-File", str(script),
            "-Repo", str(repo),
            "-LogRoot", str(log_dir),
            "-Traffic", "file",
            "-InputFile", str(input_file),
            "-OutputFile", str(output_file),
            "-DurationSec", str(self.config.get("duration_sec", 60)),
            "-RxTailSec", str(self.config.get("rx_tail_sec", 12)),
            "-RateHz", str(self.config.get("rate_hz", 12.5e6)),
            "-McrHz", str(self.config.get("mcr_hz", 200e6)),
            "-FreqHz", str(self.config.get("freq_hz", 5.0e9)),
            "-TxGain", str(self.config.get("tx_gain", 31)),
            "-RxGain", str(self.config.get("rx_gain", 22)),
            "-TxDeviceArgs", str(self.config.get("tx_device_args", "resource=RIO1")),
            "-RxDeviceArgs", str(self.config.get("rx_device_args", "resource=RIO0")),
            "-TxSubdev", str(self.config.get("tx_subdev", "A:0")),
            "-RxSubdev", str(self.config.get("rx_subdev", "B:0")),
            "-TxAntenna", str(self.config.get("tx_antenna", "TX/RX")),
            "-RxAntenna", str(self.config.get("rx_antenna", "RX2")),
            "-Modulation", str(self.config.get("modulation", "qpsk")),
            "-Alist", str(self.config["alist"]),
            "-Decoder", "cuda-bp",
            "-Nfft", "1024",
            "-Cp", str(self.config.get("cp", 128)),
            "-NumSymbols", str(self.config.get("num_symbols", 96)),
            "-ActiveSc", str(self.config.get("active_sc", 720)),
            "-PilotPeriod", "4",
            "-LdpcIter", str(self.config.get("ldpc_iter", 30)),
            "-ReportEvery", "100",
            "-WarmupFrames", "200",
            "-RxQueueBlocks", "4096",
            "-RxFrameQueue", "4096",
            "-RxBlockSamps", "32768",
            "-SyncPreamble", "zc-ofdm",
            "-SyncZcRoot", "17",
            "-GpuPipeline",
            "-ProfilePipeline",
            "-UiDashboard",
        ]
        metrics["cmd"] = " ".join(cmd)
        self.log_msg.emit(
            "[RX] 启动 USRP 文件链路: {mode}, rate={rate} Msps, txGain={tx}, rxGain={rx}, symbols={sym}".format(
                mode=self.config.get("business_mode", "--"),
                rate=float(self.config.get("rate_hz", 12.5e6)) / 1e6,
                tx=self.config.get("tx_gain", 31),
                rx=self.config.get("rx_gain", 22),
                sym=self.config.get("num_symbols", 96),
            )
        )
        self.log_msg.emit(f"[RX] USRP TX={input_file.name}, RX输出={output_file}")
        t0 = time.perf_counter()
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if line.startswith("[START]") or line.startswith("[GATE]") or line.startswith("[DONE]") or "failed" in line.lower():
                    self.log_msg.emit("[RX] " + line)
                self.parse_summary_line(metrics, line)
            code = self.proc.wait()
            metrics["exit_code"] = int(code)
        except Exception as exc:
            metrics["reason"] = str(exc)
            self.log_msg.emit(f"[RX] USRP 文件链路运行失败: {exc}")
        finally:
            metrics["elapsed_sec"] = time.perf_counter() - t0
            for p in log_dir.rglob("*.log"):
                try:
                    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                        self.parse_summary_line(metrics, line)
                except Exception:
                    pass
            metrics["output_exists"] = output_file.exists() and output_file.stat().st_size > 0
            metrics["ok"] = metrics.get("exit_code", 1) == 0 and bool(metrics["output_exists"])
            self.finished_metrics.emit(metrics)


class UsrpRandomOfdmTxControlThread(QThread):
    log_msg = Signal(str)

    def __init__(self, repo_root, port=65439, autostart=True, parent=None):
        super().__init__(parent)
        self.repo_root = Path(repo_root)
        self.port = int(port)
        self.autostart = bool(autostart)
        self.proc = None
        self.running = True

    def stop(self):
        self.running = False
        self.stop_tx()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(b'{"cmd":"quit"}', ("127.0.0.1", self.port))
        except Exception:
            pass

    def stop_tx(self):
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except Exception:
                    self.proc.kill()
        except Exception:
            pass
        self.proc = None

    def start_tx(self, reason="task"):
        if self.proc is not None and self.proc.poll() is None:
            return
        exe = self.repo_root / "build" / "uhd_cpp_gpu_pipeline" / "Release" / "uhd_ldpc_ofdm_link.exe"
        alist = self.repo_root / "Code_Matrices_Lib" / "LDPC" / "CCSDS_ldpc_n128_k64.alist"
        if not exe.exists() or not alist.exists():
            self.log_msg.emit(f"[USRP-TX] 随机OFDM发送端未启动: exe/alist不存在。")
            return
        env = configure_external_runtime_env(os.environ.copy())
        cmd = [
            str(exe),
            "--mode", "tx",
            "--traffic", "test",
            "--args", "resource=RIO1",
            "--tx-subdev", "A:0",
            "--tx-channel", "0",
            "--antenna", "TX/RX",
            "--tx-gain", "31",
            "--freq", "5000000000",
            "--rate", "12500000",
            "--mcr", "200000000",
            "--alist", str(alist),
            "--decoder", "cuda-bp",
            "--modulation", "qpsk",
            "--nfft", "1024",
            "--cp", "128",
            "--num-symbols", "96",
            "--active-sc", "512",
            "--pilot-period", "4",
            "--sync-preamble", "zc-ofdm",
            "--sync-zc-root", "17",
            "--gpu-phy-backend",
            "--tx-host-format", "sc16",
            "--tx-queue-frames", "128",
            "--duration", "0",
            "--report-every", "200",
            "--suppress-error-frames",
        ]
        log_dir = self.repo_root / "transmission_file" / "Rx" / "usrp_random_ofdm_tx_service"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "tx_service.log"
        try:
            log_f = log_path.open("a", encoding="utf-8", errors="ignore")
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                env=env,
            )
            self.log_msg.emit(f"[USRP-TX] 全GPU随机OFDM发送已启动: port={self.port}, reason={reason}")
        except Exception as exc:
            self.log_msg.emit(f"[USRP-TX] 全GPU随机OFDM发送启动失败: {exc}")

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.port))
            sock.settimeout(0.5)
        except Exception as exc:
            self.log_msg.emit(f"[USRP-TX] 控制端口 {self.port} 监听失败: {exc}")
            return
        self.log_msg.emit(f"[USRP-TX] 控制服务已启动，监听 UDP 127.0.0.1:{self.port}")
        if self.autostart:
            self.start_tx("ui-prestart")
        with sock:
            while self.running:
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except Exception:
                    break
                try:
                    payload = json.loads(data.decode("utf-8", errors="ignore"))
                except Exception:
                    payload = {"cmd": data.decode("utf-8", errors="ignore").strip()}
                cmd = str(payload.get("cmd", "")).lower()
                if cmd == "start":
                    self.start_tx(str(payload.get("reason", "task")))
                elif cmd == "stop":
                    self.stop_tx()
                    self.log_msg.emit("[USRP-TX] 全GPU随机OFDM发送已停止。")
                elif cmd == "quit":
                    break
        self.stop_tx()


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
        self.channel_prediction_control_path = APP_DIR / "runtime" / "channel_prediction_control.json"
        self.repo_root = PROJECT_ROOT
        self.offline_worker = None
        self.video_bridge_proc = None
        self.usrp_tx_control_port = 65439
        self.usrp_tx_service = None
        self.live_adaptive_trace_path = None
        self.live_adaptive_trace_active = False
        self.decoder_metric_state = {}
        self.fixed_decoder_snr_db = None
        self.current_model_update_package = None
        self.model_update_registry_path = RUNTIME_ROOT / "runtime_model_registry.json"
        self.model_update_work_root = TRANSMISSION_ROOT / "Tx" / "model_updates"
        self.model_update_staging_root = TRANSMISSION_ROOT / "Rx" / "model_update_staging"
        self.model_update_active_root = TRANSMISSION_ROOT / "Rx" / "active_models"

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
        self.apply_business_mode_preset(silent=True)
        self.update_semantic_metrics_panel()

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
        self.write_channel_prediction_control(silent=True)
        self.start_usrp_tx_control_service()

    def create_header(self, parent_layout):
        header_frame = QFrame();
        header_frame.setObjectName("HeaderFrame");
        header_frame.setMaximumHeight(56)
        header_frame.setMinimumHeight(46)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 4, 0, 4)
        header_layout.setSpacing(2)

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
        self.business_mode_cb = QComboBox();
        self.business_mode_cb.addItems([
            "星地紧急指令传输（短包低时延）",
            "卫星遥感视频文件传输（长包高吞吐量）",
            "语义任务传输",
            "星上模型更新传输",
        ])
        self.business_mode_cb.currentIndexChanged.connect(self.apply_business_mode_preset)
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
            "CUDA BP-OLD",
            "CPU BP/NMS",
        ])
        self.standard_cb = QComboBox();
        self.standard_cb.addItems(["BPSK", "QPSK", "16QAM", "64QAM"])
        self.semantic_task_cb = QComboBox();
        self.semantic_task_cb.addItems([
            "洪水检测 Sen1Floods11",
            "地震建筑损毁 QuickQuake",
            "海上船只 SSDD",
        ])
        self.semantic_task_cb.currentIndexChanged.connect(self.update_semantic_metrics_panel)
        self.model_update_task_cb = QComboBox()
        self.model_update_task_cb.addItem("flood：洪水检测 Sen1Floods11", "flood")
        self.model_update_task_cb.addItem("earthquake：地震建筑损毁 QuickQuake", "earthquake")
        self.model_update_task_cb.addItem("maritime_ship：海上船只检测 SSDD", "maritime_ship")
        self.model_update_task_cb.currentIndexChanged.connect(self.on_model_update_task_changed)
        form_decoder.addRow(self.create_label("业务模式:"), self.business_mode_cb);
        form_decoder.addRow(self.create_label("通信体制/FEC:"), self.coding_cb);
        form_decoder.addRow(self.create_label("译码后端:"), self.platform_cb);
        form_decoder.addRow(self.create_label("调制方式:"), self.standard_cb);
        self.semantic_task_label = self.create_label("语义任务:")
        form_decoder.addRow(self.semantic_task_label, self.semantic_task_cb)
        self.model_update_task_label = self.create_label("模型更新任务:")
        form_decoder.addRow(self.model_update_task_label, self.model_update_task_cb)

        panel_amc = QWidget();
        form_amc = QFormLayout(panel_amc);
        form_amc.setContentsMargins(5, 10, 5, 0);
        form_amc.setVerticalSpacing(15)
        self.ce_data_cb = QComboBox();
        self.ce_data_cb.addItems(["启用", "禁用"])
        self.cp_method_cb = QComboBox();
        # 默认禁用；用户点击“启用”后，MATLAB t-SNE 脚本才开始向 UI 推送预测可视化。
        self.cp_method_cb.addItems(["禁用", "启用"])
        self.cp_method_cb.currentIndexChanged.connect(self.write_channel_prediction_control)
        self.amc_mode_cb = QComboBox();
        self.amc_mode_cb.addItems(["启用", "禁用"])
        self.amc_mode_cb.setCurrentIndex(1)
        form_amc.addRow(self.create_label("信道状态信息采集:"), self.ce_data_cb);
        form_amc.addRow(self.create_label("信道状态信息预测:"), self.cp_method_cb);
        form_amc.addRow(self.create_label("基于AMC链路传输:"), self.amc_mode_cb)

        # Manual channel-anomaly injection is hidden for the current demo path.
        # Dynamic SNR/fade traces are generated by the link-adaptive test itself.
        self.channel_anomaly_cb = None

        self.config_stack.addWidget(panel_decoder);
        self.config_stack.addWidget(panel_amc);
        main_layout.addWidget(self.config_stack)

        self.btn_generate_model_update = QPushButton("生成更新包")
        self.btn_generate_model_update.clicked.connect(self.generate_model_update_package)
        main_layout.addWidget(self.btn_generate_model_update)

        self.model_update_status_widget = QFrame()
        self.model_update_status_widget.setStyleSheet(
            "QFrame { background-color: #0B1220; border: 1px solid #334155; border-radius: 6px; }")
        mu_layout = QVBoxLayout(self.model_update_status_widget)
        mu_layout.setContentsMargins(8, 8, 8, 8)
        mu_layout.setSpacing(6)
        self.model_update_progress = QProgressBar()
        self.model_update_progress.setRange(0, 100)
        self.model_update_progress.setValue(0)
        self.model_update_progress.setFormat("Chunk progress: %v / %m")
        self.model_update_status_text = QTextEdit()
        self.model_update_status_text.setReadOnly(True)
        self.model_update_status_text.setMaximumHeight(160)
        self.model_update_status_text.setStyleSheet(
            "background-color: #080B14; color: #D1FAE5; border: 1px solid #3A4A69; "
            "border-radius: 4px; font-family: Consolas, 'Microsoft YaHei'; font-size: 12px; padding: 5px;")
        mu_layout.addWidget(self.model_update_progress)
        mu_layout.addWidget(self.model_update_status_text)
        main_layout.addWidget(self.model_update_status_widget)
        main_layout.addStretch();
        self.btn_run = QPushButton("▶ 下发配置参数并启动测试");
        self.btn_run.clicked.connect(self.start_configured_transmission)
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
        pix = QPixmap(str(UI_NEW_MEDIA_ROOT / "usrp_real.jpg"))
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

        decoder_metric_bar = QFrame()
        decoder_metric_bar.setStyleSheet(
            "QFrame { background-color: #0B1220; border: 1px solid #243247; border-radius: 6px; }"
            "QLabel { border: none; }")
        decoder_metric_layout = QHBoxLayout(decoder_metric_bar)
        decoder_metric_layout.setContentsMargins(10, 6, 10, 6)
        decoder_metric_layout.setSpacing(8)
        self.decoder_metric_labels = {}

        def add_decoder_metric(key, title, color):
            card = QFrame()
            card.setStyleSheet("QFrame { background-color: #111827; border: 1px solid #334155; border-radius: 5px; }")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 4, 8, 4)
            card_layout.setSpacing(2)
            title_label = QLabel(title)
            title_label.setStyleSheet(
                "color: #94A3B8; font-family: 'Microsoft YaHei'; font-size: 10px; font-weight: bold;")
            value_label = QLabel("--")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet(
                f"color: {color}; font-family: Consolas, 'Microsoft YaHei'; font-size: 15px; font-weight: bold;")
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            decoder_metric_layout.addWidget(card)
            self.decoder_metric_labels[key] = value_label

        add_decoder_metric("throughput", "Throughput", "#A78BFA")
        add_decoder_metric("mcs", "MCS", "#F472B6")
        add_decoder_metric("power", "Rel. Power", "#FBBF24")
        add_decoder_metric("snr", "SNR", "#38BDF8")

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
        self.spec_plot.setLabel('bottom', 'Baseband frequency', units='MHz', **{'font-family': 'Consolas'});
        self.spec_plot.setLabel('left', 'Power', units='dB', **{'font-family': 'Consolas'})
        self.spec_plot.setYRange(-70, 5)
        self.spec_plot.setXRange(-8, 8)
        self.spec_occ_left = pg.InfiniteLine(angle=90, movable=False,
                                             pen=pg.mkPen(color=(56, 189, 248, 120), width=1, style=Qt.DashLine))
        self.spec_occ_right = pg.InfiniteLine(angle=90, movable=False,
                                              pen=pg.mkPen(color=(56, 189, 248, 120), width=1, style=Qt.DashLine))
        self.spec_plot.addItem(self.spec_occ_left)
        self.spec_plot.addItem(self.spec_occ_right)
        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen(color='#9F7AEA', width=2), fillLevel=-40,
                                              fillBrush=(159, 122, 234, 60), name="Live Spectrum")
        self.spec_ref_curve = self.spec_plot.plot(
            pen=pg.mkPen(color=QColor(74, 222, 128, 180), width=2, style=Qt.DashLine),
            name="OFDM occupied band")
        self.spec_avg_powers = None
        self.update_spectrum_reference(active_sc=512, nfft=1024, rate_hz=15.36e6)

        self.ber_plot = pg.PlotWidget(
            title="<span style='color: #818CF8; font-size: 11pt; font-family: Microsoft YaHei;'>▶ 实时译码性能曲线 (BER / FER)</span>")
        style_plot(self.ber_plot)
        self.ber_plot.setLabel('bottom', '帧', **{'font-family': 'Microsoft YaHei'})
        self.ber_plot.setLabel('left', 'log10(BER / FER)', **{'font-family': 'Consolas'})
        self.ber_plot.setXRange(1, 100)
        self.ber_plot.setYRange(-9, 0)
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
        self.tx_vid_title = QLabel(
            "<span style='color: #FBBF24; font-family: Microsoft YaHei; font-weight: bold;'>▶ 发送端原始数据</span>")
        self.tx_vid_title.setAlignment(Qt.AlignCenter)
        self.tx_video_label = QLabel()
        self.tx_video_label.setAlignment(Qt.AlignCenter)
        self.tx_video_label.setStyleSheet("background-color: #080B14; border: 1px solid #3A4A69; border-radius: 4px;")
        self.tx_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.tx_video_stack = QStackedWidget()
        self.tx_video_stack.addWidget(self.tx_video_label)
        tx_vid_layout.addWidget(self.tx_vid_title)
        tx_vid_layout.addWidget(self.tx_video_stack, stretch=1)

        rx_vid_widget = QWidget();
        rx_vid_layout = QVBoxLayout(rx_vid_widget);
        rx_vid_layout.setContentsMargins(0, 5, 0, 0);
        rx_vid_layout.setSpacing(5)
        self.rx_vid_title = QLabel(
            "<span style='color: #00E5FF; font-family: Microsoft YaHei; font-weight: bold;'>▶ 接收端恢复数据</span>")
        self.rx_vid_title.setAlignment(Qt.AlignCenter)
        self.rx_video_label = QLabel()
        self.rx_video_label.setAlignment(Qt.AlignCenter)
        self.rx_video_label.setStyleSheet("background-color: #080B14; border: 1px solid #3A4A69; border-radius: 4px;")
        self.rx_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.rx_video_stack = QStackedWidget()
        self.rx_video_stack.addWidget(self.rx_video_label)
        self.tx_media_player = QMediaPlayer(self)
        self.rx_media_player = QMediaPlayer(self)
        self.tx_video_sink = QVideoSink(self)
        self.rx_video_sink = QVideoSink(self)
        self.tx_video_sink.videoFrameChanged.connect(lambda frame: self.update_video_label_from_frame(self.tx_video_label, frame))
        self.rx_video_sink.videoFrameChanged.connect(lambda frame: self.update_video_label_from_frame(self.rx_video_label, frame))
        self.tx_media_player.setVideoOutput(self.tx_video_sink)
        self.rx_media_player.setVideoOutput(self.rx_video_sink)
        self.tx_media_player.mediaStatusChanged.connect(lambda status: self.loop_media_player(self.tx_media_player, status))
        self.rx_media_player.mediaStatusChanged.connect(lambda status: self.loop_media_player(self.rx_media_player, status))
        rx_vid_layout.addWidget(self.rx_vid_title)
        rx_vid_layout.addWidget(self.rx_video_stack, stretch=1)

        self.semantic_widget = QWidget();
        semantic_layout = QVBoxLayout(self.semantic_widget);
        semantic_layout.setContentsMargins(0, 5, 0, 0);
        semantic_layout.setSpacing(5)
        semantic_title = QLabel(
            "<span style='color: #4ADE80; font-family: Microsoft YaHei; font-weight: bold;'>▶ 语义任务指标 (RS-AI Task Metrics)</span>")
        semantic_title.setAlignment(Qt.AlignCenter)
        self.semantic_metrics_text = QTextEdit()
        self.semantic_metrics_text.setReadOnly(True)
        self.semantic_metrics_text.setStyleSheet(
            "background-color: #080B14; color: #D1FAE5; border: 1px solid #3A4A69; "
            "border-radius: 4px; font-family: Consolas, 'Microsoft YaHei'; font-size: 12px; padding: 6px;")
        semantic_layout.addWidget(semantic_title)
        semantic_layout.addWidget(self.semantic_metrics_text, stretch=1)

        row3_h_splitter.addWidget(tx_vid_widget)
        row3_h_splitter.addWidget(rx_vid_widget)
        row3_h_splitter.addWidget(self.semantic_widget)
        row3_h_splitter.setSizes([360, 360, 260])

        decoder_v_splitter.addWidget(row1_h_splitter)
        decoder_v_splitter.addWidget(row2_h_splitter)
        decoder_v_splitter.addWidget(row3_h_splitter)
        decoder_v_splitter.setSizes([250, 250, 400])

        decoder_layout.addWidget(decoder_metric_bar)
        decoder_layout.addWidget(decoder_v_splitter, stretch=1)

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
        self.channel_power_display_offset_db = 10.0
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
            offset=(-8, -8),
            horSpacing=4,
            verSpacing=-5,
            labelTextSize="7pt",
            labelTextColor="#E5E7EB",
        )
        try:
            self.tsne_legend.setBrush(pg.mkBrush(8, 11, 20, 220))
            self.tsne_legend.setPen(pg.mkPen(229, 231, 235, 110))
            self.tsne_legend.layout.setContentsMargins(4, 1, 4, 1)
            # 固定图例在 t-SNE 图右下角，避免遮挡点云主体。
            self.tsne_legend.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-8, -8))
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
        self.throughput_plot.setXRange(1, 1200, padding=0.02)
        self.throughput_curve = self.throughput_plot.plot(pen=pg.mkPen(color='#A78BFA', width=2), fillLevel=0,
                                                          fillBrush=(167, 139, 250, 50))
        self.throughput_history_x = []
        self.throughput_history_y = []

        self.amc_state_plot = pg.PlotWidget(
            title="<span style='color: #F472B6; font-size: 11pt; font-family: Microsoft YaHei;'>▶ MCS 阶数自适应切换轨迹 (AMC State Tracker)</span>")
        style_plot(self.amc_state_plot)
        self.amc_state_plot.setLabel('bottom', 'Frame', **{'font-family': 'Consolas'})
        self.amc_state_plot.setLabel('left', 'MCS index', **{'font-family': 'Consolas'})
        self.amc_state_plot.setXRange(1, 1200, padding=0.02)
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

    def semantic_task_catalog(self):
        root = AI_V3_MEDIA_ROOT
        return {
            "洪水检测 Sen1Floods11": {
                "task": "flood",
                "classes": "flood_water",
                "model": "emergency_sen1floods11_flood_best.pt",
                "metrics": "P=0.437, R=0.295, mAP50=0.271, mAP50-95=0.157",
                "scheme": "uniform / roi-layered",
                "gpu_direct": "uniform_roundtrip.json / roi_layered_roundtrip.json",
                "tx_video": root / "tasks" / "flood" / "videos" / "original_satellite_sequence.mp4",
                "rx_video": root / "tasks" / "flood" / "videos" / "receiver_detected_sequence.mp4",
            },
            "地震建筑损毁 QuickQuake": {
                "task": "earthquake",
                "classes": "intact_building, damaged_building",
                "model": "emergency_quickquake_damage_best.pt",
                "metrics": "held-out: P=0.889, R=0.704, mAP50=0.763, mAP50-95=0.763",
                "scheme": "roi-layered GPU-direct",
                "gpu_direct": "gpu_direct_roundtrip.json: frames=40, fps=8.79",
                "tx_video": root / "tasks" / "earthquake" / "videos" / "original_earthquake_building_mosaic.mp4",
                "rx_video": root / "tasks" / "earthquake" / "videos" / "receiver_detected_building_damage_mosaic.mp4",
            },
            "海上船只 SSDD": {
                "task": "maritime_ship",
                "classes": "ship",
                "model": "security_maritime_ship_best.pt",
                "metrics": "held-out: P=0.986, R=0.981, mAP50=0.991, mAP50-95=0.691",
                "scheme": "roi-layered GPU-direct",
                "gpu_direct": "gpu_direct_roundtrip.json: frames=40, fps=8.78",
                "tx_video": root / "tasks" / "maritime_ship" / "videos" / "original_maritime_ship_mosaic.mp4",
                "rx_video": root / "tasks" / "maritime_ship" / "videos" / "receiver_detected_ship_mosaic.mp4",
            },
        }

    def task_id_to_semantic_label(self, task_id):
        mapping = {
            "flood": "洪水检测 Sen1Floods11",
            "earthquake": "地震建筑损毁 QuickQuake",
            "maritime_ship": "海上船只 SSDD",
        }
        return mapping.get(str(task_id), "地震建筑损毁 QuickQuake")

    def semantic_label_to_task_id(self, label):
        info = self.semantic_task_catalog().get(label, {})
        return info.get("task", "earthquake")

    def active_semantic_task_id(self):
        try:
            registry = self.model_update_registry().load()
            last = registry.get("last_update", {}) or {}
            task_id = last.get("task_id")
            if task_id and task_id in {"flood", "earthquake", "maritime_ship"}:
                return task_id
            active_models = registry.get("active_models", {}) or {}
            if active_models:
                task_id = sorted(active_models.keys())[-1]
                if task_id in {"flood", "earthquake", "maritime_ship"}:
                    return task_id
        except Exception:
            pass
        if hasattr(self, "semantic_task_cb"):
            return self.semantic_label_to_task_id(self.semantic_task_cb.currentText())
        return "earthquake"

    def sync_semantic_task_to_active_model(self, task_id=None):
        task_id = task_id or self.active_semantic_task_id()
        if hasattr(self, "semantic_task_cb"):
            label = self.task_id_to_semantic_label(task_id)
            idx = self.semantic_task_cb.findText(label)
            if idx >= 0 and idx != self.semantic_task_cb.currentIndex():
                self.semantic_task_cb.blockSignals(True)
                self.semantic_task_cb.setCurrentIndex(idx)
                self.semantic_task_cb.blockSignals(False)
        return task_id

    def set_video_panel_titles(self, mode=None, task_id=None):
        mode = mode or (self.business_mode_cb.currentText() if hasattr(self, "business_mode_cb") else "")
        task_id = task_id or self.active_semantic_task_id()
        task_name = self.model_update_packetizer().task_info(task_id).get("task_name", self.task_id_to_semantic_label(task_id)) \
            if ("语义任务" in mode or "模型更新" in mode) and ModelUpdatePacketizer is not None else ""
        if "语义任务" in mode or "模型更新" in mode:
            tx_text = "▶ 原始数据"
            rx_text = f"▶ {task_name}处理后的数据"
        elif "紧急指令" in mode:
            tx_text = "▶ 发送端指令数据"
            rx_text = "▶ 接收端指令数据"
        else:
            tx_text = "▶ 发送端原始数据"
            rx_text = "▶ 接收端恢复数据"
        if hasattr(self, "tx_vid_title"):
            self.tx_vid_title.setText(
                f"<span style='color: #FBBF24; font-family: Microsoft YaHei; font-weight: bold;'>{tx_text}</span>")
        if hasattr(self, "rx_vid_title"):
            self.rx_vid_title.setText(
                f"<span style='color: #00E5FF; font-family: Microsoft YaHei; font-weight: bold;'>{rx_text}</span>")

    def model_update_task_id(self):
        if hasattr(self, "model_update_task_cb"):
            value = self.model_update_task_cb.currentData()
            if value:
                return str(value)
        return "earthquake"

    def start_usrp_tx_control_service(self):
        if UsrpSyncTaskThread is None:
            self.append_log("[USRP-SYNC] usrp_sync_task.py 未能导入，同步任务禁用。")
            return
        if self.usrp_tx_service is not None and self.usrp_tx_service.isRunning():
            return
        self.usrp_tx_service = UsrpSyncTaskThread(
            self.repo_root,
            port=self.usrp_tx_control_port,
            autostart=False,
            parent=self,
        )
        self.usrp_tx_service.log_msg.connect(self.append_log)
        self.usrp_tx_service.start()

    def signal_usrp_random_tx(self, cmd, reason="task"):
        if self.usrp_tx_service is None:
            return
        try:
            if cmd == "start":
                self.usrp_tx_service.request_start(reason)
            elif cmd == "stop":
                self.usrp_tx_service.request_stop(reason)
        except Exception as exc:
            self.append_log(f"[USRP-SYNC] 控制指令发送失败: {exc}")

    def model_update_packetizer(self):
        if ModelUpdatePacketizer is None:
            raise RuntimeError("model_update.py 未能导入，无法生成模型更新包。")
        return ModelUpdatePacketizer(
            AI_V3_MODEL_ROOT,
            self.model_update_work_root,
            chunk_size=4096,
        )

    def model_update_registry(self):
        if ModelRegistry is None:
            raise RuntimeError("model_update.py 未能导入，无法更新模型注册表。")
        return ModelRegistry(self.model_update_registry_path, self.model_update_active_root)

    def model_update_task_info(self, task_id=None):
        task_id = task_id or self.model_update_task_id()
        return self.model_update_packetizer().task_info(task_id)

    def append_model_update_status(self, text):
        line = f"[ModelUpdate] {text}"
        self.append_log(line)
        if hasattr(self, "model_update_status_text"):
            self.model_update_status_text.append(line)
            self.model_update_status_text.verticalScrollBar().setValue(
                self.model_update_status_text.verticalScrollBar().maximum())

    def update_model_update_status_panel(self, data=None):
        data = data or {}
        pkg = self.current_model_update_package
        task_id = data.get("task_id") or (pkg.task_id if pkg else self.model_update_task_id())
        try:
            info = self.model_update_packetizer().task_info(task_id)
        except Exception:
            info = {}
        chunk_count = int(data.get("chunk_count", getattr(pkg, "chunk_count", 0) or 0) or 0)
        received = int(data.get("received_chunks", 0) or 0)
        if chunk_count > 0:
            self.model_update_progress.setRange(0, chunk_count)
            self.model_update_progress.setValue(min(received, chunk_count))
        else:
            self.model_update_progress.setRange(0, 100)
            self.model_update_progress.setValue(0)
        manifest = data.get("manifest") or (pkg.manifest if pkg else {})
        active = data.get("active_model") or {}
        crc_status = "待校验"
        if data.get("crc_errors"):
            crc_status = "失败 " + str(data.get("crc_errors"))
        elif int(data.get("received_chunks", 0) or 0) > 0 and chunk_count > 0:
            crc_status = "通过"
        sha_status = "待校验" if "sha256_ok" not in data else ("通过" if data.get("sha256_ok") else "失败")
        smoke_status = "待执行"
        if "smoke_test_passed" in data:
            smoke_status = "通过" if data.get("smoke_test_passed") else data.get("smoke_message", "失败")
        lines = [
            f"当前任务: {task_id} / {info.get('task_name', manifest.get('task_name', '--'))}",
            f"模型文件: {manifest.get('model_name', info.get('model_name', '--'))}",
            f"模型版本: {manifest.get('version', info.get('version', '--'))}",
            f"Update ID: {manifest.get('update_id', getattr(pkg, 'update_id', '--') if pkg else '--')}",
            f"Chunk 接收进度: {received} / {chunk_count}",
            f"CRC 状态: {crc_status}",
            f"SHA256 状态: {sha_status}",
            f"Smoke Test: {smoke_status}",
            f"Active Model: {active.get('model_name', active.get('model_path', '--')) if active else '--'}",
            f"当前类别: {', '.join(manifest.get('classes', info.get('classes', []))) if (manifest or info) else '--'}",
        ]
        self.model_update_status_text.setPlainText("\n".join(lines))

    def on_model_update_task_changed(self, *args):
        self.current_model_update_package = None
        self.set_video_panel_titles(task_id=self.model_update_task_id())
        self.update_model_update_status_panel()
        self.update_model_update_demo_panel()
        if hasattr(self, "log_text"):
            self.append_model_update_status(f"selected task: {self.model_update_task_id()}")

    def generate_model_update_package(self):
        try:
            task_id = self.model_update_task_id()
            self.append_model_update_status(f"selected task: {task_id}")
            self.append_model_update_status("generating manifest / labels / infer_config")
            pkg = self.model_update_packetizer().build(task_id)
            self.current_model_update_package = pkg
            self.update_model_update_status_panel({
                "task_id": task_id,
                "manifest": pkg.manifest,
                "chunk_count": pkg.chunk_count,
                "received_chunks": 0,
            })
            self.append_model_update_status(
                f"package generated: {pkg.package_file.name}, chunks={pkg.chunk_count}, sha256={pkg.sha256[:12]}...")
            return pkg
        except Exception as exc:
            self.append_model_update_status(f"package generation failed: {exc}")
            return None

    def update_model_update_demo_panel(self, task_id=None, registry_entry=None):
        task_id = task_id or self.model_update_task_id()
        self.sync_semantic_task_to_active_model(task_id)
        self.set_video_panel_titles(task_id=task_id)
        if registry_entry is None:
            try:
                registry = self.model_update_registry().load()
                registry_entry = registry.get("active_models", {}).get(task_id)
            except Exception:
                registry_entry = None
        try:
            info = self.model_update_packetizer().task_info(task_id)
        except Exception:
            info = {}
        display = info.get("display", {})
        metrics = info.get("metrics", {})
        active_name = "--"
        if registry_entry:
            active_name = registry_entry.get("model_name") or Path(registry_entry.get("model_path", "")).name
        lines = [
            f"任务展示: {info.get('task_name', task_id)}",
            f"当前模型: {active_name}",
            f"模型版本: {registry_entry.get('version', info.get('version', '--')) if registry_entry else info.get('version', '--')}",
            f"类别: {', '.join(info.get('classes', []))}",
            f"SHA256: {'通过' if registry_entry else '待更新'}",
            f"模型加载: {'active' if registry_entry else '待加载'}",
            f"输入标题: {display.get('input_title', '--')}",
            f"输出标题: {display.get('output_title', '--')}",
            f"检测框样式: {display.get('style', '--')}",
            f"任务指标: mAP50={metrics.get('mAP50', '--')}, {metrics.get('summary', '')}",
            "检测框/类别/置信度由接收端检测视频叠加展示。",
        ]
        if hasattr(self, "semantic_metrics_text"):
            self.semantic_metrics_text.setPlainText("\n".join(lines))

        tx_video = info.get("tx_video_path")
        rx_video = info.get("rx_video_path")
        if tx_video and rx_video and Path(tx_video).exists() and Path(rx_video).exists():
            self.start_video_bridge(tx_video, rx_video)

    def load_model_update_registry_to_ui(self):
        try:
            registry = self.model_update_registry().load()
            last = registry.get("last_update", {}) or {}
            task_id = last.get("task_id")
            if task_id and hasattr(self, "model_update_task_cb"):
                idx = self.model_update_task_cb.findData(task_id)
                if idx >= 0 and idx != self.model_update_task_cb.currentIndex():
                    self.model_update_task_cb.setCurrentIndex(idx)
            active = registry.get("active_models", {}).get(task_id or self.model_update_task_id())
            self.update_model_update_status_panel({
                "task_id": task_id or self.model_update_task_id(),
                "active_model": active or {},
                "sha256_ok": bool(last.get("sha256_verified", False)),
                "smoke_test_passed": bool(last.get("smoke_test_passed", False)),
                "manifest": {
                    "task_id": task_id or self.model_update_task_id(),
                    "model_name": last.get("model_name", active.get("model_name", "--") if active else "--"),
                    "update_id": last.get("update_id", "--"),
                    "classes": active.get("classes", []) if active else [],
                    "version": active.get("version", "--") if active else "--",
                },
            })
            self.update_model_update_demo_panel(task_id or self.model_update_task_id(), active)
        except Exception as exc:
            self.append_model_update_status(f"registry load failed: {exc}")

    def finalize_model_update_after_transport(self, metrics):
        output_file = Path(metrics.get("output_file", ""))
        if not output_file.exists():
            self.append_model_update_status("receiver output missing; cannot reassemble")
            return
        try:
            self.append_model_update_status("manifest received")
            reassembler = ModelUpdateReassembler(self.model_update_staging_root)
            result = reassembler.reassemble_file(output_file)
            self.append_model_update_status(
                f"receiving chunks: {result['received_chunks']}/{result['chunk_count']}")
            if result.get("crc_errors"):
                self.append_model_update_status(f"CRC check failed: {result['crc_errors']}")
            else:
                self.append_model_update_status("CRC check passed")
            if result.get("missing_chunks"):
                self.append_model_update_status(f"missing chunks: {result['missing_chunks'][:20]}")
            if result.get("sha256_ok"):
                self.append_model_update_status("SHA256 verified")
            else:
                self.append_model_update_status("SHA256 mismatch")
            reg_result = self.model_update_registry().activate(result)
            result.update(reg_result)
            self.update_model_update_status_panel(result)
            if reg_result.get("activated"):
                task_id = result["task_id"]
                self.sync_semantic_task_to_active_model(task_id)
                self.append_model_update_status("smoke test passed")
                self.append_model_update_status(f"active model switched to {task_id}")
                self.append_model_update_status(f"opening {task_id} demo panel")
                self.update_model_update_demo_panel(task_id, reg_result.get("active_model"))
            else:
                self.append_model_update_status(
                    f"model load failed: {reg_result.get('smoke_message', 'unknown')}; rolled back to previous version")
        except Exception as exc:
            self.append_model_update_status(f"reassembly/register failed: {exc}")

    def update_semantic_metrics_panel(self, *args):
        if not hasattr(self, "semantic_metrics_text"):
            return
        if hasattr(self, "business_mode_cb") and "模型更新" in self.business_mode_cb.currentText():
            self.update_model_update_demo_panel()
            return
        if hasattr(self, "business_mode_cb") and "语义任务" in self.business_mode_cb.currentText():
            task_id = self.sync_semantic_task_to_active_model()
            self.set_video_panel_titles(task_id=task_id)
            name = self.task_id_to_semantic_label(task_id)
        else:
            name = self.semantic_task_cb.currentText() if hasattr(self, "semantic_task_cb") else "洪水检测 Sen1Floods11"
        info = self.semantic_task_catalog().get(name, {})
        lines = [
            f"任务: {name}",
            f"类别: {info.get('classes', '--')}",
            f"模型: {info.get('model', '--')}",
            f"指标: {info.get('metrics', '--')}",
            f"传输形态: {info.get('scheme', '--')}",
            f"GPU-direct验证: {info.get('gpu_direct', '--')}",
            "",
            "说明: 当前面板先展示同事交付包中的任务指标和检测视频；",
            "后续可把 rs-ai-gpu-direct-v1 的实时检测统计追加到这里。",
        ]
        self.semantic_metrics_text.setPlainText("\n".join(lines))

    def apply_business_mode_preset(self, *args, silent=False):
        if not hasattr(self, "business_mode_cb"):
            return
        mode = self.business_mode_cb.currentText()
        semantic_enabled = "语义任务" in mode
        model_update_enabled = "模型更新" in mode
        active_task_id = self.model_update_task_id() if model_update_enabled else (
            self.active_semantic_task_id() if semantic_enabled else None)
        if "紧急指令" in mode:
            self.coding_cb.setCurrentText("CCSDS LDPC n128/k64")
            self.platform_cb.setCurrentText("CUDA BP-OLD")
            self.standard_cb.setCurrentText("QPSK")
        elif "遥感视频" in mode:
            self.coding_cb.setCurrentText("DVB-S2 LDPC N16200 R5/6")
            self.platform_cb.setCurrentText("CUDA BP")
            self.standard_cb.setCurrentText("16QAM")
        elif model_update_enabled:
            self.coding_cb.setCurrentText("DVB-S2 LDPC N16200 R1/2")
            self.platform_cb.setCurrentText("CUDA BP")
            self.standard_cb.setCurrentText("QPSK")
        else:
            self.coding_cb.setCurrentText("DVB-S2 LDPC N16200 R1/2")
            self.platform_cb.setCurrentText("CUDA BP")
            self.standard_cb.setCurrentText("QPSK")
        self.semantic_task_cb.setVisible(semantic_enabled)
        self.semantic_task_cb.setEnabled(False)
        if semantic_enabled:
            self.sync_semantic_task_to_active_model(active_task_id)
        if hasattr(self, "semantic_task_label"):
            self.semantic_task_label.setVisible(semantic_enabled)
        if hasattr(self, "model_update_task_cb"):
            self.model_update_task_cb.setVisible(model_update_enabled)
            self.model_update_task_cb.setEnabled(model_update_enabled)
        if hasattr(self, "model_update_task_label"):
            self.model_update_task_label.setVisible(model_update_enabled)
        if hasattr(self, "btn_generate_model_update"):
            self.btn_generate_model_update.setVisible(model_update_enabled)
        if hasattr(self, "model_update_status_widget"):
            self.model_update_status_widget.setVisible(model_update_enabled)
        if hasattr(self, "semantic_widget"):
            self.semantic_widget.setVisible(semantic_enabled or model_update_enabled)
        if hasattr(self, "btn_run"):
            self.btn_run.setText("▶ 下发模型更新" if model_update_enabled else "▶ 下发配置参数并启动测试")
        self.set_video_panel_titles(mode, active_task_id)
        self.update_semantic_metrics_panel()
        if model_update_enabled:
            self.load_model_update_registry_to_ui()
        if not silent and hasattr(self, "log_text"):
            self.append_log(f"[System] 业务模式已切换: {mode}")

    def matrix_path_from_label(self, label):
        base = CODE_MATRICES_ROOT
        mapping = {
            "CCSDS LDPC n128/k64": base / "LDPC" / "CCSDS_ldpc_n128_k64.alist",
            "CCSDS LDPC n256/k128": base / "LDPC" / "CCSDS_ldpc_n256_k128.alist",
            "CCSDS LDPC n512/k256": base / "LDPC" / "CCSDS_ldpc_n512_k256.alist",
            "DVB-S2 LDPC N16200 R1/4": base / "LDPC" / "DVB_S2_short_N16200_rate_1_4.alist",
            "DVB-S2 LDPC N16200 R1/2": base / "LDPC" / "DVB_S2_short_N16200_rate_1_2.alist",
            "DVB-S2 LDPC N16200 R5/6": base / "LDPC" / "DVB_S2_short_N16200_rate_5_6.alist",
            "DVB-S2 LDPC N64800 R1/2": base / "LDPC" / "DVB_S2_N64800_R12.alist",
        }
        return mapping.get(label)

    def decoder_from_label(self, label, fec_label):
        short_bp_osd_ok = ("n128" in fec_label or "n256" in fec_label)
        if label == "CUDA BP":
            return "cuda-bp"
        if label in ("CUDA BP-OSD", "CUDA BP-OLD"):
            return "cuda-bp-osd" if short_bp_osd_ok else "cuda-bp"
        if label == "CPU BP/NMS":
            return "cpu"
        return "cuda-bp-osd" if short_bp_osd_ok else "cuda-bp"

    def emergency_command_payload(self):
        text = (
            "CMD:EMERGENCY_LINK_CHECK;"
            "TASK:GROUND_TO_SAT;"
            "ACTION:SAFE_MODE_ACK;"
            "PRIORITY:HIGH;"
            "SEQ:0001;"
            "AUTH:DEMO;"
            "END;"
        )
        raw = text.encode("ascii")
        if len(raw) > 128:
            raw = raw[:128]
        return raw + b" " * (128 - len(raw))

    def ensure_emergency_command_file(self):
        path = TRANSMISSION_ROOT / "Tx" / "emergency_command_128B.cmd"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.emergency_command_payload()
        if not path.exists() or path.read_bytes() != payload:
            path.write_bytes(payload)
        return path

    def format_command_payload(self, data):
        if isinstance(data, (str, Path)):
            data = Path(data).read_bytes()
        data = bytes(data)[:128]
        text = data.decode("ascii", errors="replace").rstrip(" \x00")
        grouped = "\n".join(part for part in text.split(";") if part)
        return f"{grouped}\n\nLength: {len(data)} bytes"

    def selected_media_paths(self):
        business = self.business_mode_cb.currentText()
        tx_dir = TRANSMISSION_ROOT / "Tx"
        sample_tx_dir = TRANSMISSION_SAMPLE_ROOT / "Tx"
        if "紧急指令" in business:
            return self.ensure_emergency_command_file(), None
        if "语义任务" in business:
            task_id = self.sync_semantic_task_to_active_model()
            info = self.semantic_task_catalog().get(self.task_id_to_semantic_label(task_id), {})
            return info.get("tx_video"), info.get("rx_video")
        if "模型更新" in business:
            if self.current_model_update_package is None:
                self.generate_model_update_package()
            if self.current_model_update_package is None:
                return None, None
            return self.current_model_update_package.package_file, None
        return first_existing(sample_tx_dir / "002.avi", tx_dir / "002.avi"), None

    def write_live_adaptive_power_trace(self):
        if not getattr(self, "live_adaptive_trace_active", False):
            return
        trace_path = getattr(self, "live_adaptive_trace_path", None)
        if not trace_path:
            return
        try:
            trace_path = Path(trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            power_trace = np.asarray(getattr(self, "power_history_y", []), dtype=float).reshape(-1)
            power_x = np.asarray(getattr(self, "power_history_x", []), dtype=float).reshape(-1)
            finite_mask = np.isfinite(power_trace)
            if power_trace.size and np.any(finite_mask):
                power_trace = np.clip(power_trace[finite_mask], -24.0, 8.0)
                if power_x.size == finite_mask.size:
                    power_x = power_x[finite_mask]
                else:
                    power_x = np.arange(1, power_trace.size + 1, dtype=float)
            else:
                power_trace = np.array([0.0], dtype=float)
                power_x = np.array([1.0], dtype=float)
            tmp_path = trace_path.with_suffix(trace_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                f.write("frame,relativePowerDb\n")
                for x, y in zip(power_x, power_trace):
                    f.write(f"{int(round(float(x)))},{float(y):.6f}\n")
            os.replace(tmp_path, trace_path)
        except Exception as exc:
            self.append_log(f"[RX] 实时预测功率 trace 刷新失败: {exc}")

    def generate_simulated_channel_trace(self, log_dir, frames, base_snr_db=24.0,
                                         fade_snr_db=4.0, fade_start=352,
                                         fade_end=641, jitter_db=2.0):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(secrets.randbelow(2_000_000_000) + 1)
        frames = int(frames)
        snr_trace = np.empty(frames, dtype=float)
        # Match the earlier dynamic-SNR demo profile, then inject one short
        # deep fade. The trace is one SNR/gain value per PHY frame.
        step_values = [8.0, 14.5, 18.0, 10.0, 15.0]
        step_len = max(1, int(math.ceil(frames / len(step_values))))
        for i in range(frames):
            snr_trace[i] = step_values[min(i // step_len, len(step_values) - 1)]
        snr_trace += rng.uniform(-float(jitter_db), float(jitter_db), size=snr_trace.size)
        left = max(1, int(fade_start)) - 1
        right = min(int(frames), int(fade_end))
        if right > left:
            snr_trace[left:right] = float(fade_snr_db)
        gain_trace = snr_trace - float(base_snr_db)

        gain_path = log_dir / "simulated_channel_gain_trace_db.csv"
        snr_path = log_dir / "simulated_snr_trace_db.csv"
        with gain_path.open("w", encoding="utf-8") as f:
            f.write("frame,relativePowerDb\n")
            for idx, value in enumerate(gain_trace, start=1):
                f.write(f"{idx},{float(value):.6f}\n")
        with snr_path.open("w", encoding="utf-8") as f:
            f.write("frame,snrDb\n")
            for idx, value in enumerate(snr_trace, start=1):
                f.write(f"{idx},{float(value):.6f}\n")
        self.append_log(
            f"[RX] 已生成自适应用模拟逐帧信道: SNR分段=8/14.5/18/10/15 dB, "
            f"jitter=±{jitter_db:.1f} dB, fade={fade_start}-{fade_end}帧->{fade_snr_db:.1f} dB")
        return gain_path, snr_path

    def build_offline_gpu_config(self):
        business = self.business_mode_cb.currentText()
        fec_label = self.coding_cb.currentText()
        if fec_label.startswith("Polar"):
            self.append_log("[RX] 当前离线 GPU-Pipeline 文件传输入口暂未启用 Polar，自动切换为 CCSDS LDPC n128/k64。")
            fec_label = "CCSDS LDPC n128/k64"
        alist = self.matrix_path_from_label(fec_label)
        if alist is None:
            raise RuntimeError(f"无法找到编码矩阵配置: {fec_label}")

        input_file, _ = self.selected_media_paths()
        if input_file is None:
            raise RuntimeError("未找到业务输入文件。")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = TRANSMISSION_ROOT / "Rx" / f"ui_offline_gpu_{stamp}"
        suffix = Path(input_file).suffix or ".bin"
        output_file = log_dir / ("rx_recovered" + suffix)
        modulation = self.standard_cb.currentText().lower()
        decoder = self.decoder_from_label(self.platform_cb.currentText(), fec_label)
        if self.platform_cb.currentText() in ("CUDA BP-OSD", "CUDA BP-OLD") and decoder != "cuda-bp-osd":
            self.append_log(
                f"[RX] 当前编码 {fec_label} 超出短码 BP-OLD 支持范围，已自动切换为 CUDA BP。")

        if "紧急指令" in business:
            snr_db, cp, num_symbols, active_sc, ldpc_iter = 8.0, 128, 20, 512, 30
        elif "遥感视频" in business:
            snr_db, cp, num_symbols, active_sc, ldpc_iter = 18.0, 128, 120, 720, 30
        elif "模型更新" in business:
            snr_db, cp, num_symbols, active_sc, ldpc_iter = 16.0, 128, 120, 720, 30
        else:
            snr_db, cp, num_symbols, active_sc, ldpc_iter = 12.0, 128, 96, 512, 30
        adaptive_base_snr_db = 24.0
        simulated_frames = 1200
        if "遥感视频" in business:
            simulated_frames = 1800
        elif "紧急指令" in business:
            simulated_frames = 900

        link_adaptive_mode = bool(
            hasattr(self, "mode_selector") and self.mode_selector.currentIndex() == 1)
        if link_adaptive_mode:
            simulated_frames = 6000
        amc_enabled = bool(
            link_adaptive_mode and
            hasattr(self, "amc_mode_cb") and
            self.amc_mode_cb.currentText() == "启用")
        adaptive_power_trace = ""
        sim_snr_trace = ""
        if link_adaptive_mode:
            channel_gain_trace, sim_snr_trace = self.generate_simulated_channel_trace(
                log_dir,
                simulated_frames,
                base_snr_db=adaptive_base_snr_db,
                fade_snr_db=4.0,
                fade_start=352,
                fade_end=641,
                jitter_db=2.0)
            self.live_adaptive_trace_active = False
            self.live_adaptive_trace_path = None
            if amc_enabled:
                adaptive_power_trace = channel_gain_trace
                self.append_log(
                    f"[RX] 链路自适应模式: 动态SNR已启用，AMC使用模拟逐帧信道增益 "
                    f"{Path(adaptive_power_trace).name}；352-641帧强衰落到4 dB，低SNR切BPSK，不启用重传。")
            else:
                self.append_log(
                    "[RX] 链路自适应模式: 动态SNR已启用，但AMC链路传输禁用；"
                    "本轮作为固定调制对照组。")
        else:
            self.live_adaptive_trace_active = False
            self.live_adaptive_trace_path = None
            self.append_log("[RX] 通用译码器配置模式: 不注入动态模拟信道，使用固定SNR配置。")

        config = {
            "repo": str(self.repo_root),
            "business_mode": business,
            "input_file": str(input_file),
            "output_file": str(output_file),
            "log_dir": str(log_dir),
            "alist": str(alist),
            "modulation": modulation,
            "decoder": decoder,
            "snr_db": snr_db,
            "cp": cp,
            "num_symbols": num_symbols,
            "active_sc": active_sc,
            "rate_hz": 15.36e6,
            "ldpc_iter": ldpc_iter,
            "test_seed": secrets.randbelow(2_000_000_000) + 1,
            "systematic_front_info": decoder == "cuda-bp-osd",
            "gpu_sim_wall_clock_throughput": link_adaptive_mode,
        }
        if sim_snr_trace:
            config["sim_snr_trace"] = str(sim_snr_trace)
        if adaptive_power_trace:
            config.update({
                "adaptive_power_trace": str(adaptive_power_trace),
                "adaptive_power_trace_live": False,
                "adaptive_power_live_wait_ms": 0,
                "adaptive_power_hold_frames": 1,
                "adaptive_power_base_snr_db": adaptive_base_snr_db,
                "tx_repeat_min": 1,
                "tx_repeat_max": 1,
            })
        return config

    def should_use_usrp_transport(self, config):
        # All business payloads now return to the offline full-GPU pipeline.
        # USRP is kept as a separate sync/placeholder task and never owns the
        # business file transport path from the UI.
        return False

    def estimate_usrp_duration_sec(self, input_file, config):
        try:
            size_bytes = max(Path(input_file).stat().st_size, 1)
        except Exception:
            size_bytes = 1
        business = config.get("business_mode", "")
        if "紧急指令" in business:
            return 10
        nfft = 1024
        cp = int(config.get("cp", 128))
        symbols = int(config.get("num_symbols", 96))
        active_sc = int(config.get("active_sc", 720))
        pilot_period = 4
        bits_per_sym = {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6}.get(
            str(config.get("modulation", "qpsk")).lower(), 2)
        data_symbols = symbols - len(range(0, symbols, pilot_period))
        code_rate = 0.5
        try:
            dims = Path(config["alist"]).read_text(encoding="utf-8", errors="ignore").splitlines()[0].split()
            n = int(dims[0])
            m = int(dims[1])
            code_rate = max(0.05, min(0.95, float(n - m) / float(n)))
        except Exception:
            pass
        coded_bits = active_sc * max(data_symbols, 1) * bits_per_sym
        payload_bytes = max(256, int(coded_bits * code_rate / 8.0) - 96)
        frame_samples = 512 + symbols * (nfft + cp)
        fps = float(config.get("rate_hz", 12.5e6)) / max(float(frame_samples), 1.0)
        nominal = (size_bytes / float(payload_bytes)) / max(fps, 1.0)
        margin = 4.0 if "遥感视频" in business else 2.5
        return int(max(20, min(1800, math.ceil(nominal * margin + 20))))

    def choose_usrp_active_sc(self, config):
        try:
            dims = Path(config["alist"]).read_text(encoding="utf-8", errors="ignore").splitlines()[0].split()
            code_n = int(dims[0])
        except Exception:
            code_n = 128
        symbols = int(config.get("num_symbols", 96))
        pilot_period = 4
        data_symbols = symbols - len(range(0, symbols, pilot_period))
        bits_per_sym = {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6}.get(
            str(config.get("modulation", "qpsk")).lower(), 2)
        preferred = [720, 768, 512, 450, 900, 600, 384, 256]
        if code_n >= 1000:
            # DVB-S2 short frames need 450 or 900 active carriers with 96 symbols
            # so that the OFDM coded-bit capacity is an integer number of 16200-bit codewords.
            preferred = [450, 900, 600, 720, 768, 512, 384, 256]
        for active_sc in preferred:
            if active_sc <= 0 or active_sc >= 1024 or active_sc % 2:
                continue
            coded_bits = active_sc * max(data_symbols, 1) * bits_per_sym
            if coded_bits % max(code_n, 1) == 0:
                return active_sc
        for active_sc in range(900, 127, -2):
            coded_bits = active_sc * max(data_symbols, 1) * bits_per_sym
            if coded_bits % max(code_n, 1) == 0:
                return active_sc
        return 512

    def build_usrp_file_config(self, config):
        cfg = dict(config)
        input_file = Path(cfg["input_file"])
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = TRANSMISSION_ROOT / "Rx" / f"ui_usrp_link_{stamp}"
        output_file = log_dir / ("rx_recovered" + (input_file.suffix or ".bin"))
        cfg.update({
            "transport": "usrp",
            "log_dir": str(log_dir),
            "output_file": str(output_file),
            "rate_hz": 12.5e6,
            "mcr_hz": 200e6,
            "freq_hz": 5.0e9,
            "tx_gain": 31,
            "rx_gain": 22,
            "num_symbols": 96,
            "baseline_num_symbols": 96,
            "tx_device_args": "resource=RIO1",
            "rx_device_args": "resource=RIO0",
            "tx_subdev": "A:0",
            "rx_subdev": "B:0",
            "tx_antenna": "TX/RX",
            "rx_antenna": "RX2",
            "rx_tail_sec": 12,
        })
        cfg["active_sc"] = self.choose_usrp_active_sc(cfg)
        cfg["duration_sec"] = self.estimate_usrp_duration_sec(input_file, cfg)
        if cfg["active_sc"] != config.get("active_sc"):
            self.append_log(
                f"[RX] USRP OFDM参数自适配: activeSC {config.get('active_sc')} -> {cfg['active_sc']}，"
                "保证每帧 coded bits 可整除当前 FEC 码长，避免视频链路启动后被截断。")
        return cfg

    def update_spectrum_reference(self, active_sc, nfft, rate_hz):
        try:
            nyquist_mhz = float(rate_hz) / 2.0 / 1e6
            occupied_mhz = float(active_sc) / max(float(nfft), 1.0) * float(rate_hz) / 1e6
            half_occ = occupied_mhz / 2.0
            self.spec_plot.setXRange(-nyquist_mhz, nyquist_mhz, padding=0.02)
            self.spec_occ_left.setValue(-half_occ)
            self.spec_occ_right.setValue(half_occ)
            x = np.array([-nyquist_mhz, -half_occ, -half_occ, half_occ, half_occ, nyquist_mhz], dtype=float)
            y = np.array([-48.0, -48.0, -3.0, -3.0, -48.0, -48.0], dtype=float)
            self.spec_ref_curve.setData(x, y)
        except Exception:
            pass

    def error_rate_to_log10(self, value):
        try:
            value = float(value)
            if not np.isfinite(value):
                value = 0.0
            return math.log10(max(value, 1e-9))
        except Exception:
            return -9.0

    def update_decoder_metric_strip(self, **values):
        """Mirror link-adaptive telemetry on the decoder dashboard page."""
        labels = getattr(self, "decoder_metric_labels", None)
        if not labels:
            return
        state = getattr(self, "decoder_metric_state", {})
        for key, value in values.items():
            if value is not None:
                state[key] = value
        self.decoder_metric_state = state

        def finite_float(key):
            try:
                value = float(state.get(key))
                return value if np.isfinite(value) else None
            except Exception:
                return None

        frame = state.get("frame")
        total_frames = state.get("total_frames")
        if "frame" in labels:
            try:
                display_frame = self.display_frame_index(frame, total_frames)
                display_total = self.display_frame_total(total_frames)
                labels["frame"].setText(f"{display_frame}/{display_total}")
            except Exception:
                labels["frame"].setText("--")

        throughput = finite_float("throughput_mbps")
        labels["throughput"].setText(f"{throughput:.2f} Mbps" if throughput is not None else "--")

        mcs = finite_float("mcs")
        labels["mcs"].setText(f"{mcs:.0f}" if mcs is not None else "--")

        power = finite_float("relative_power_display_db")
        labels["power"].setText(f"{power:+.1f} dB" if power is not None else "--")

        snr = finite_float("snr_db")
        labels["snr"].setText(f"{snr:.1f} dB" if snr is not None else "--")

    def display_frame_total(self, total_frames=None):
        """Initial UI frame window. Expand with the received frame, not totalFrames."""
        return 1200

    def display_frame_index(self, frame, total_frames=None):
        try:
            frame = int(round(float(frame)))
        except Exception:
            return 1
        return max(frame, 1)

    def reset_runtime_plots(self, config=None):
        self.ber_history_x.clear()
        self.ber_history_y.clear()
        self.fer_history_x.clear()
        self.fer_history_y.clear()
        self.throughput_history_x.clear()
        self.throughput_history_y.clear()
        self.mcs_history_x.clear()
        self.mcs_history_y.clear()
        self.ber_curve.setData([], [])
        self.fer_curve.setData([], [])
        self.throughput_curve.setData([], [])
        self.amc_state_curve.setData([], [])
        self.spec_curve.setData([], [])
        self.power_history_x = np.array([], dtype=float)
        self.power_history_y = np.array([], dtype=float)
        self.power_history_mask = np.array([], dtype=bool)
        self.channel_power_curve.setData([], [])
        self.channel_power_current.setData([], [])
        self.channel_power_region.setVisible(False)
        self.feature_buffer_status.setText("")
        self.feature_buffer_status.setVisible(False)
        self.spec_avg_powers = None
        self.ber_plot.setXRange(1, 100, padding=0.02)
        self.ber_plot.setYRange(-9, 0, padding=0.0)
        self.throughput_plot.setXRange(1, 1200, padding=0.02)
        self.amc_state_plot.setXRange(1, 1200, padding=0.02)
        self.decoder_metric_state = {}
        try:
            self.fixed_decoder_snr_db = float(config.get("snr_db")) if config is not None else None
        except Exception:
            self.fixed_decoder_snr_db = None
        if self.fixed_decoder_snr_db is not None:
            self.update_decoder_metric_strip(snr_db=self.fixed_decoder_snr_db)
        else:
            self.update_decoder_metric_strip()
        self.spec_plot.setYRange(-70, 5, padding=0.0)
        if config is not None:
            self.update_spectrum_reference(
                active_sc=config.get("active_sc", 512),
                nfft=1024,
                rate_hz=config.get("rate_hz", 15.36e6),
            )

    def stop_video_bridge(self):
        proc = getattr(self, "video_bridge_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        self.video_bridge_proc = None
        for player in (getattr(self, "tx_media_player", None), getattr(self, "rx_media_player", None)):
            try:
                if player is not None:
                    player.stop()
            except Exception:
                pass

    def loop_media_player(self, player, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            player.setPosition(0)
            player.play()

    def reset_media_label_style(self):
        media_style = "background-color: #080B14; border: 1px solid #3A4A69; border-radius: 4px;"
        for label in (self.tx_video_label, self.rx_video_label):
            label.setStyleSheet(media_style)
            label.setAlignment(Qt.AlignCenter)
            label.setText("")

    def update_video_label_from_frame(self, label, frame):
        if not frame.isValid():
            return
        image = frame.toImage()
        if image.isNull():
            return
        pix = QPixmap.fromImage(image)
        if pix.isNull() or label.width() <= 0 or label.height() <= 0:
            return
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def play_local_video_pair(self, tx_video, rx_video=None):
        tx_video = Path(tx_video)
        rx_video = Path(rx_video) if rx_video else tx_video
        if not tx_video.exists() or not rx_video.exists():
            self.append_log(f"[RX] 原分辨率视频播放文件不存在: TX={tx_video}, RX={rx_video}")
            return False
        try:
            self.tx_video_stack.setCurrentIndex(0)
            self.rx_video_stack.setCurrentIndex(0)
            self.reset_media_label_style()
            self.tx_media_player.setSource(QUrl.fromLocalFile(str(tx_video)))
            self.rx_media_player.setSource(QUrl.fromLocalFile(str(rx_video)))
            self.tx_media_player.play()
            self.rx_media_player.play()
            self.append_log(f"[RX] 原分辨率视频播放已启动: TX={tx_video.name}, RX={rx_video.name}")
            return True
        except Exception as exc:
            self.append_log(f"[RX] 原分辨率视频播放启动失败，回退UDP桥接: {exc}")
            return False

    def play_tx_video_wait_rx(self, tx_video):
        tx_video = Path(tx_video)
        if not tx_video.exists():
            self.append_log(f"[RX] 发送端预览视频不存在: {tx_video}")
            return False
        try:
            self.tx_video_stack.setCurrentIndex(0)
            self.rx_video_stack.setCurrentIndex(0)
            self.reset_media_label_style()
            self.tx_media_player.setSource(QUrl.fromLocalFile(str(tx_video)))
            self.tx_media_player.play()
            self.rx_media_player.stop()
            self.rx_video_label.clear()
            self.rx_video_label.setText("等待接收端文件恢复完成...")
            self.rx_video_label.setStyleSheet(
                "background-color: #080B14; color: #A0AEC0; border: 1px solid #3A4A69; "
                "border-radius: 4px; font-family: 'Microsoft YaHei'; font-size: 15px;")
            self.rx_video_label.setAlignment(Qt.AlignCenter)
            self.append_log(f"[RX] 发送端原始视频预览已启动，接收端等待恢复: TX={tx_video.name}")
            return True
        except Exception as exc:
            self.append_log(f"[RX] 发送端原始视频预览启动失败: {exc}")
            return False

    def play_rx_recovered_video(self, rx_video):
        rx_video = Path(rx_video)
        if not rx_video.exists():
            self.append_log(f"[RX] 接收端恢复视频不存在: {rx_video}")
            return False
        try:
            self.rx_video_stack.setCurrentIndex(0)
            self.reset_media_label_style()
            self.rx_media_player.setSource(QUrl.fromLocalFile(str(rx_video)))
            self.rx_media_player.play()
            self.append_log(f"[RX] 接收端恢复视频播放已启动: RX={rx_video.name}")
            return True
        except Exception as exc:
            self.append_log(f"[RX] 接收端恢复视频播放启动失败: {exc}")
            return False

    def show_static_image_pair(self, tx_path, rx_path=None):
        self.tx_video_stack.setCurrentIndex(0)
        self.rx_video_stack.setCurrentIndex(0)
        self.tx_media_player.stop()
        self.rx_media_player.stop()
        self.reset_media_label_style()
        pix = QPixmap(str(tx_path))
        if pix.isNull():
            return
        if self.tx_video_label.width() > 0:
            self.tx_video_label.setPixmap(pix.scaled(self.tx_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        rx_pix = QPixmap(str(rx_path)) if rx_path and Path(rx_path).exists() else pix
        if self.rx_video_label.width() > 0 and not rx_pix.isNull():
            self.rx_video_label.setPixmap(rx_pix.scaled(self.rx_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_rx_recovered_image(self, rx_path):
        self.rx_video_stack.setCurrentIndex(0)
        self.rx_media_player.stop()
        self.reset_media_label_style()
        rx_pix = QPixmap(str(rx_path))
        if self.rx_video_label.width() > 0 and not rx_pix.isNull():
            self.rx_video_label.setPixmap(rx_pix.scaled(self.rx_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.append_log(f"[RX] 接收端恢复图片已显示: RX={Path(rx_path).name}")

    def command_display_delay_ms(self):
        return 8 + (secrets.randbelow(8))

    def show_command_text_pair(self, tx_path, rx_path=None, *, delay_tx_ms=0, delay_rx=False):
        self.stop_video_bridge()
        self.tx_video_stack.setCurrentIndex(0)
        self.rx_video_stack.setCurrentIndex(0)
        tx_text = self.format_command_payload(tx_path)
        style = (
            "background-color: #080B14; color: #D1FAE5; border: 1px solid #3A4A69; "
            "border-radius: 4px; font-family: Consolas, 'Microsoft YaHei'; font-size: 15px; padding: 10px;")
        self.tx_video_label.setStyleSheet(style)
        self.rx_video_label.setStyleSheet(style)
        self.tx_video_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.rx_video_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        if delay_tx_ms and delay_tx_ms > 0:
            self.tx_video_label.setText("发送端指令等待装载...")
            QTimer.singleShot(int(delay_tx_ms), lambda text=tx_text: self.tx_video_label.setText(text))
        else:
            self.tx_video_label.setText(tx_text)
        if rx_path and Path(rx_path).exists():
            delay_ms = self.command_display_delay_ms()
            rx_text = self.format_command_payload(rx_path) + f"\n\nPacket delay: {delay_ms} ms"
            if delay_rx:
                self.rx_video_label.setText("接收端指令解调完成，等待显示...")
                QTimer.singleShot(15, lambda text=rx_text: self.rx_video_label.setText(text))
            else:
                self.rx_video_label.setText(rx_text)
        else:
            self.rx_video_label.setText("等待接收端指令恢复完成...")
        self.append_log(f"[RX] 星地紧急指令展示: TX={Path(tx_path).name}, bytes={Path(tx_path).stat().st_size}")

    def start_video_bridge(self, tx_video, rx_video=None, *, max_frames=0):
        self.stop_video_bridge()
        if not tx_video:
            return
        tx_video = Path(tx_video)
        rx_video = Path(rx_video) if rx_video else tx_video
        if tx_video.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            self.show_static_image_pair(tx_video, rx_video)
            return
        if tx_video.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
            if self.play_local_video_pair(tx_video, rx_video):
                return
        script = SATELLITE_UI_SOURCE_ROOT / "examples" / "send_video_udp_bridge.py"
        python_exe = Path(sys.executable)
        if not script.exists() or not python_exe.exists():
            self.append_log("[RX] 视频桥接脚本或 Python 环境不存在，跳过视频播放。")
            return
        cmd = [
            str(python_exe), str(script),
            "--tx-video", str(tx_video),
            "--rx-video", str(rx_video),
            "--mode", "pair",
            "--fps", "20",
            "--loop",
        ]
        if max_frames > 0:
            cmd += ["--max-frames", str(max_frames)]
        try:
            self.video_bridge_proc = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.append_log(f"[RX] UI视频桥接已启动: TX={tx_video.name}, RX={rx_video.name}")
        except Exception as exc:
            self.append_log(f"[RX] UI视频桥接启动失败: {exc}")

    def start_configured_transmission(self):
        if self.offline_worker is not None and self.offline_worker.isRunning():
            self.append_log("[RX] 已有传输任务正在运行。")
            return
        try:
            config = self.build_offline_gpu_config()
            if self.should_use_usrp_transport(config):
                config = self.build_usrp_file_config(config)
        except Exception as exc:
            self.append_log(f"[RX] 配置生成失败: {exc}")
            return

        tx_media, rx_preview = self.selected_media_paths()
        self.reset_runtime_plots(config)
        self.start_usrp_tx_control_service()
        self.append_log("[RX] 业务数据走离线全GPU-Pipeline；USRP仅发送同步OFDM信号。")
        self.signal_usrp_random_tx("start", config.get("business_mode", "file-task"))
        tx_path = Path(tx_media) if tx_media else None
        if "模型更新" in config.get("business_mode", ""):
            self.append_model_update_status("接收模型更新中")
            self.btn_generate_model_update.setEnabled(False)
            self.model_update_progress.setValue(0)
            self.rx_video_label.clear()
            self.rx_video_label.setText("等待模型更新包接收与重组完成...")
            self.update_model_update_status_panel()
        elif "紧急指令" in config.get("business_mode", ""):
            self.show_command_text_pair(tx_path, delay_tx_ms=200)
        elif tx_path and tx_path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
            self.play_tx_video_wait_rx(tx_path)
        elif tx_path and tx_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            self.show_static_image_pair(tx_path, None)
            self.rx_video_label.clear()
            self.rx_video_label.setText("等待接收端图片恢复完成...")
        else:
            self.start_video_bridge(tx_media, rx_preview)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("▶ 传输中...")
        self.offline_worker = OfflineGpuPipelineThread(config, self)
        self.offline_worker.log_msg.connect(self.append_log)
        self.offline_worker.finished_metrics.connect(self.on_offline_gpu_finished)
        self.offline_worker.start()

    def on_offline_gpu_finished(self, metrics):
        self.live_adaptive_trace_active = False
        self.live_adaptive_trace_path = None
        self.btn_run.setEnabled(True)
        if hasattr(self, "business_mode_cb") and "模型更新" in self.business_mode_cb.currentText():
            self.btn_run.setText("▶ 下发模型更新")
        else:
            self.btn_run.setText("▶ 下发配置参数并启动测试")
        if hasattr(self, "btn_generate_model_update"):
            self.btn_generate_model_update.setEnabled(True)
        fer = metrics.get("fer")
        goodput = metrics.get("goodput_mbps")
        elapsed = metrics.get("elapsed_sec", 0.0)
        frames = int(metrics.get("frames", 0) or 0)
        ok_frames = int(metrics.get("ok_frames", 0) or 0)
        err_frames = int(metrics.get("err_frames", 0) or 0)
        status = "成功" if metrics.get("ok") else ("有输出但未完全成功" if metrics.get("output_exists") else "未完成")
        self.append_log(
            "[RX] 传输完成: status={status}, frames={frames}, ok={ok}, err={err}, "
            "FER={fer}, avgThroughput={thr} Mbps, latency={lat:.3f} s, output={out}".format(
                status=status,
                frames=frames,
                ok=ok_frames,
                err=err_frames,
                fer=("{:.3e}".format(fer) if fer is not None else "n/a"),
                thr=("{:.2f}".format(goodput) if goodput is not None else "n/a"),
                lat=elapsed,
                out=metrics.get("output_file", "--"),
            )
        )
        if frames > 0 and not self.fer_history_x:
            self.fer_history_x.append(frames)
            self.fer_history_y.append(self.error_rate_to_log10(fer or 0.0))
            self.fer_curve.setData(self.fer_history_x, self.fer_history_y)
            self.ber_plot.setXRange(1, max(100, frames), padding=0.02)
            self.ber_plot.setYRange(-9, 0, padding=0.0)
        output_file = Path(metrics.get("output_file", ""))
        tx_media, rx_preview = self.selected_media_paths()
        self.signal_usrp_random_tx("stop", "rx-complete")
        if "模型更新" in metrics.get("business_mode", ""):
            self.finalize_model_update_after_transport(metrics)
        elif "紧急指令" in metrics.get("business_mode", ""):
            if output_file.exists():
                self.show_command_text_pair(tx_media, output_file, delay_rx=True)
                self.append_log(f"[RX] 接收端紧急指令恢复完成: RX={output_file.name}, bytes={output_file.stat().st_size}")
        elif output_file.exists() and output_file.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
            if not metrics.get("ok"):
                self.append_log(f"[RX] 视频未完整恢复，播放接收端失败/截断视频用于演示: {output_file}")
            self.play_rx_recovered_video(output_file)
        elif output_file.exists() and output_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            self.show_rx_recovered_image(output_file)

    # ---------------- 核心槽函数区 ----------------
    def update_iq_plot(self, tx_i, tx_q, rx_i, rx_q):
        if self.dashboard_stack.currentIndex() == 0:
            self.tx_scatter.setData(tx_i, tx_q)
            self.rx_scatter.setData(rx_i, rx_q)

    def update_spec_plot(self, freqs, powers):
        if self.dashboard_stack.currentIndex() == 0:
            freqs = np.asarray(freqs, dtype=float)
            powers = np.asarray(powers, dtype=float)
            if freqs.size == 0 or powers.size == 0:
                return
            n = min(freqs.size, powers.size)
            freqs = freqs[:n]
            powers = powers[:n]
            if np.nanmax(np.abs(freqs)) > 1e5:
                freqs = freqs / 1e6
            powers = np.nan_to_num(powers, nan=-90.0, posinf=5.0, neginf=-90.0)
            powers = powers - float(np.nanmax(powers))
            powers = np.clip(powers, -90.0, 5.0)
            if self.spec_avg_powers is None or len(self.spec_avg_powers) != len(powers):
                self.spec_avg_powers = powers
            else:
                self.spec_avg_powers = 0.75 * self.spec_avg_powers + 0.25 * powers
            self.spec_curve.setData(freqs, self.spec_avg_powers)

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
        if not self.is_channel_prediction_enabled():
            self.clear_tsne_view("Prediction disabled. Click 启用 to start t-SNE streaming.")
            return

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

        # channelPowerDb is generated by MATLAB channel-visualization scripts.
        # It is intentionally ignored here so MATLAB no longer drives the
        # "Relative Total Channel Power" display. Baseband telemetry can still
        # update that panel through the scalar relativePowerDb field below.
        channel_power_db = np.asarray([], dtype=float)
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
            self.write_live_adaptive_power_trace()
            x_all = self.power_history_x
            y_all = self.power_history_y
            y_display_all = y_all + float(getattr(self, "channel_power_display_offset_db", 0.0))
            mask_all = self.power_history_mask

            self.feature_buffer_image.setVisible(False)
            self.channel_power_curve.setVisible(True)
            self.channel_power_current.setVisible(True)
            self.channel_power_curve.setData(x_all, y_display_all)

            current_x = float(x_vals[-1])
            current_y = float(channel_power_db[-1])
            current_y_display = current_y + float(getattr(self, "channel_power_display_offset_db", 0.0))
            self.channel_power_current.setData([current_x], [current_y_display])
            self.update_decoder_metric_strip(
                frame=int(current_x),
                total_frames=int(state.get("totalFrames", max(1000, int(current_x))) or max(1000, int(current_x))),
                relative_power_display_db=current_y_display,
            )

            y_lim = state.get("channelPowerYLim", [-22.0, 8.0])
            try:
                y_min, y_max = float(y_lim[0]), float(y_lim[1])
            except Exception:
                y_min, y_max = -22.0, 8.0
            if y_max <= y_min:
                y_min, y_max = -22.0, 8.0
            y_offset = float(getattr(self, "channel_power_display_offset_db", 0.0))
            y_min += y_offset
            y_max += y_offset
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

        # Do not use MATLAB noisyFeatureBuffer/featureBuffer as a fallback in
        # the Relative Total Channel Power panel. That panel is reserved for
        # scalar baseband relativePowerDb telemetry.

        frame = int(state.get("frame", 1) or 1)
        total_frames = int(state.get("totalFrames", max(1000, frame)) or max(1000, frame))
        display_frame = self.display_frame_index(frame, total_frames)
        sample_count = int(state.get("sampleCount", 0) or 0)
        snr_value = state.get("snrDb", state.get("snr_db", state.get("snr", None)))
        try:
            snr_value = float(snr_value) if snr_value is not None else None
            if snr_value is not None and not np.isfinite(snr_value):
                snr_value = None
        except Exception:
            snr_value = None
        if hasattr(self, "mode_selector") and self.mode_selector.currentIndex() == 0:
            fixed_snr = getattr(self, "fixed_decoder_snr_db", None)
            if fixed_snr is not None:
                snr_value = fixed_snr
        self.update_decoder_metric_strip(
            frame=frame,
            total_frames=total_frames,
            snr_db=snr_value,
        )

        relative_power_value = state.get(
            "relativePowerDb",
            state.get("relative_power_db", state.get("relativePower", None)),
        )
        has_relative_power = False
        if relative_power_value is not None:
            try:
                relative_power_value = float(relative_power_value)
                if np.isfinite(relative_power_value):
                    has_relative_power = True
                    anomaly_mask = np.array([relative_power_value <= -15.0], dtype=bool)
                    self.merge_channel_power_history(
                        np.array([frame], dtype=float),
                        np.array([relative_power_value], dtype=float),
                        anomaly_mask)
                    x_all = self.power_history_x
                    y_all = self.power_history_y
                    y_offset = float(getattr(self, "channel_power_display_offset_db", 0.0))
                    y_display_all = y_all + y_offset
                    relative_power_display = relative_power_value + y_offset
                    mask_all = self.power_history_mask
                    self.feature_buffer_image.setVisible(False)
                    self.channel_power_curve.setVisible(True)
                    self.channel_power_current.setVisible(True)
                    self.channel_power_curve.setData(x_all, y_display_all)
                    self.channel_power_current.setData([frame], [relative_power_display])
                    self.feature_buffer_plot.setYRange(-24.0 + y_offset, 8.0 + y_offset, padding=0.0)
                    self.apply_channel_power_x_range(total_frames, frame)
                    if mask_all.size == x_all.size and np.any(mask_all):
                        idx = np.flatnonzero(mask_all)
                        self.channel_power_region.setRegion([x_all[idx[0]] - 0.5, x_all[idx[-1]] + 0.5])
                        self.channel_power_region.setVisible(True)
                    else:
                        self.channel_power_region.setVisible(False)
                    self.feature_buffer_status.setText(
                        f"Frame {frame}: {relative_power_display:+.1f} dB")
                    self.feature_buffer_status.setVisible(True)
                    self.update_decoder_metric_strip(relative_power_display_db=relative_power_display)
                    try:
                        x_range = self.feature_buffer_plot.viewRange()[0]
                        x_left, x_right = float(x_range[0]), float(x_range[1])
                        x_pad = 0.02 * max(1.0, x_right - x_left)
                        self.feature_buffer_status.setPos(x_left + x_pad, 6.5)
                    except Exception:
                        self.feature_buffer_status.setPos(float(frame), 6.5)
            except Exception:
                pass
        if (
            not has_relative_power
            and channel_power_db.size == 0
            and getattr(self, "power_history_x", np.array([], dtype=float)).size == 0
        ):
            try:
                self.feature_buffer_image.setVisible(False)
                self.channel_power_curve.setVisible(False)
                self.channel_power_current.setVisible(False)
                self.channel_power_region.setVisible(False)
                self.feature_buffer_status.setText("Waiting for baseband relativePowerDb...")
                self.feature_buffer_status.setVisible(True)
                self.feature_buffer_status.setPos(1.0, 6.5)
                self.feature_buffer_plot.setXRange(1, 1200, padding=0.02)
                y_offset = float(getattr(self, "channel_power_display_offset_db", 0.0))
                self.feature_buffer_plot.setYRange(-24.0 + y_offset, 8.0 + y_offset, padding=0.0)
            except Exception:
                pass

        throughput_value = state.get("throughputMbps", state.get("throughput_Mbps", state.get("throughput")))
        if throughput_value is not None:
            try:
                throughput_value = float(throughput_value)
                if np.isfinite(throughput_value):
                    if self.throughput_history_x and frame < self.throughput_history_x[-1]:
                        self.throughput_history_x.clear()
                        self.throughput_history_y.clear()
                    self.throughput_history_x.append(display_frame)
                    self.throughput_history_y.append(throughput_value)
                    self.throughput_curve.setData(self.throughput_history_x, self.throughput_history_y)
                    self.throughput_plot.setXRange(1, max(1200, display_frame), padding=0.02)
                    ymax = max(1.0, max(self.throughput_history_y) * 1.20)
                    self.throughput_plot.setYRange(0.0, ymax, padding=0.0)
                    self.update_decoder_metric_strip(throughput_mbps=throughput_value)
            except Exception:
                pass

        fer_value = state.get("fer", state.get("FER"))
        if fer_value is not None:
            try:
                fer_value = float(fer_value)
                if np.isfinite(fer_value):
                    self.fer_history_x.append(frame)
                    self.fer_history_y.append(self.error_rate_to_log10(fer_value))
                    self.fer_history_x = self.fer_history_x[-500:]
                    self.fer_history_y = self.fer_history_y[-500:]
                    self.fer_curve.setData(self.fer_history_x, self.fer_history_y)
                    right = max(100, max(self.fer_history_x))
                    self.ber_plot.setXRange(1, right, padding=0.02)
            except Exception:
                pass

        ber_value = state.get("ber", state.get("BER"))
        if state.get("berValid", True) and ber_value is not None:
            try:
                ber_value = float(ber_value)
                if np.isfinite(ber_value):
                    self.ber_history_x.append(frame)
                    self.ber_history_y.append(self.error_rate_to_log10(ber_value))
                    self.ber_history_x = self.ber_history_x[-500:]
                    self.ber_history_y = self.ber_history_y[-500:]
                    self.ber_curve.setData(self.ber_history_x, self.ber_history_y)
                    right = max(100, max(self.ber_history_x))
                    self.ber_plot.setXRange(1, right, padding=0.02)
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
                    self.mcs_history_x.append(display_frame)
                    self.mcs_history_y.append(mcs_value)
                    self.amc_state_curve.setData(self.mcs_history_x, self.mcs_history_y)
                    self.amc_state_plot.setXRange(1, max(1200, display_frame), padding=0.02)
                    ymax = max(8.0, max(self.mcs_history_y) + 1.0)
                    self.amc_state_plot.setYRange(0.0, ymax, padding=0.0)
                    self.update_decoder_metric_strip(mcs=mcs_value)
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

    def is_channel_prediction_enabled(self):
        try:
            return self.cp_method_cb.currentText().strip() == "启用"
        except Exception:
            return False

    def clear_tsne_view(self, message="Waiting for 20 dB t-SNE data..."):
        try:
            self.tsne_true_scatter.setData([], [])
            self.tsne_pred_scatter.setData([], [])
            self.tsne_status_text.setText(message)
            fixed_x = getattr(self, "tsne_fixed_x_range", (-32.0, 32.0))
            fixed_y = getattr(self, "tsne_fixed_y_range", (-32.0, 32.0))
            self.tsne_plot.setXRange(fixed_x[0], fixed_x[1], padding=0.0)
            self.tsne_plot.setYRange(fixed_y[0], fixed_y[1], padding=0.0)
            self.tsne_status_text.setPos(fixed_x[0] + 2.0, fixed_y[1] - 5.0)
        except Exception:
            pass

    def write_channel_prediction_control(self, *args, silent=False):
        enabled = self.is_channel_prediction_enabled()
        payload = {
            "type": "channel_prediction_control",
            "enabled": enabled,
            "updatedAt": time.time(),
        }
        try:
            self.channel_prediction_control_path.parent.mkdir(parents=True, exist_ok=True)
            self.channel_prediction_control_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            if enabled:
                self.clear_tsne_view("Waiting for 20 dB t-SNE data...")
            else:
                self.clear_tsne_view("Prediction disabled. Click 启用 to start t-SNE streaming.")
            if not silent and hasattr(self, "log_text"):
                state_text = "enabled" if enabled else "disabled"
                self.append_log(f"[t-SNE] Channel prediction visualization {state_text}.")
        except Exception as exc:
            if hasattr(self, "log_text"):
                self.append_log(f"[t-SNE] Failed to write prediction control: {exc}")

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
            self.stop_video_bridge()
        except Exception:
            pass
        try:
            if self.offline_worker is not None and self.offline_worker.isRunning():
                self.offline_worker.stop()
        except Exception:
            pass
        try:
            if self.usrp_tx_service is not None:
                self.usrp_tx_service.stop()
                self.usrp_tx_service.wait(2000)
        except Exception:
            pass
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
