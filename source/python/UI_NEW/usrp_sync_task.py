import json
import os
import socket
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal


def find_project_root(start_dir: Path) -> Path:
    for path in [start_dir, *start_dir.parents]:
        if (path / "requirements.txt").exists() and (path / "source").exists():
            return path
        if path.name == "organized_workspace_20260601":
            return path
    return start_dir


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(APP_DIR)
CODE_MATRICES_ROOT = (
    PROJECT_ROOT / "data" / "code_matrices" / "Code_Matrices_Lib"
    if (PROJECT_ROOT / "data" / "code_matrices" / "Code_Matrices_Lib").exists()
    else PROJECT_ROOT / "Code_Matrices_Lib"
)
TRANSMISSION_ROOT = PROJECT_ROOT / "runtime" / "transmission_file"


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


class UsrpSyncTaskThread(QThread):
    """Background USRP sync/placeholder OFDM task.

    This helper intentionally does not carry user payload. Business data stays on
    the offline full-GPU pipeline; USRP only emits a continuous GPU-generated
    random OFDM waveform so that the RF side can be warmed up or used as a
    synchronization/visual task during file-transfer demos.
    """

    log_msg = Signal(str)

    def __init__(self, repo_root, port=65439, autostart=False, parent=None):
        super().__init__(parent)
        self.repo_root = Path(repo_root) if repo_root else PROJECT_ROOT
        self.port = int(port)
        self.autostart = bool(autostart)
        self.proc = None
        self.running = True

    def stop(self):
        self.running = False
        self.stop_sync()
        self._send_local({"cmd": "quit"})

    def start_sync(self, reason="file-transfer"):
        if self.proc is not None and self.proc.poll() is None:
            return
        exe = self.repo_root / "build" / "uhd_cpp_gpu_pipeline" / "Release" / "uhd_ldpc_ofdm_link.exe"
        alist = CODE_MATRICES_ROOT / "LDPC" / "CCSDS_ldpc_n128_k64.alist"
        if not exe.exists() or not alist.exists():
            self.log_msg.emit("[USRP-SYNC] 未启动: uhd_ldpc_ofdm_link.exe 或 CCSDS 矩阵不存在。")
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

        log_dir = TRANSMISSION_ROOT / "Rx" / "usrp_sync_task"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            log_f = (log_dir / "tx_sync.log").open("a", encoding="utf-8", errors="ignore")
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
            self.log_msg.emit(f"[USRP-SYNC] 全GPU随机OFDM同步任务已启动: reason={reason}")
        except Exception as exc:
            self.log_msg.emit(f"[USRP-SYNC] 启动失败: {exc}")

    def stop_sync(self):
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

    def request_start(self, reason="file-transfer"):
        self._send_local({"cmd": "start", "reason": reason})

    def request_stop(self, reason="rx-complete"):
        self._send_local({"cmd": "stop", "reason": reason})

    def _send_local(self, payload):
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(data, ("127.0.0.1", self.port))
        except Exception:
            pass

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.port))
            sock.settimeout(0.5)
        except Exception as exc:
            self.log_msg.emit(f"[USRP-SYNC] 控制端口 {self.port} 监听失败: {exc}")
            return

        self.log_msg.emit(f"[USRP-SYNC] 控制服务已启动，监听 UDP 127.0.0.1:{self.port}")
        if self.autostart:
            self.start_sync("ui-prestart")

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
                reason = str(payload.get("reason", "file-transfer"))
                if cmd == "start":
                    self.start_sync(reason)
                elif cmd == "stop":
                    self.stop_sync()
                    self.log_msg.emit(f"[USRP-SYNC] 同步任务已停止: reason={reason}")
                elif cmd == "quit":
                    break

        self.stop_sync()
