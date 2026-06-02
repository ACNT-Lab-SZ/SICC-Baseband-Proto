# SICC Baseband Prototype

SICC Baseband Prototype 是一个面向星地/USRP 基带链路验证的原型工程，覆盖 PySide6 可视化 UI、UHD C++/CUDA 收发链路、LDPC/Polar 编译码矩阵、MATLAB NTN/链路算法入口，以及遥感 AI 业务与模型更新流程的代码骨架。

本仓库来自原始实验工程的整理上传版，保留了必要源码、脚本、构建入口、配置、编码矩阵和性能结果图表。大型模型、训练数据、视频素材、原始链路日志和运行生成物未纳入本次 Git 上传。

## Repository Layout

```text
.
├─ START_UI_NEW.bat              Windows UI 启动入口
├─ SETUP_UI_ENV.bat              Windows UI 环境安装入口
├─ requirements.txt              Python 依赖清单
├─ source/
│  ├─ python/                    PySide6 UI、AI 业务链路、工具脚本
│  ├─ cpp_cuda/                  UHD/C++/CUDA/LDPC-OFDM 源码与测试
│  ├─ matlab/                    MATLAB 编译码与 NTN 算法源码
│  └─ web_ui/                    UI 内嵌 Web/Three.js 可视化页面
├─ scripts/
│  ├─ powershell/                USRP、UHD、性能测试与批处理启动脚本
│  ├─ batch_cmd/                 Windows bat/cmd 启动器
│  ├─ matlab_entrypoints/        MATLAB 一键运行入口
│  ├─ python_tools/              Python 辅助工具
│  └─ shell_js/                  Shell/Node 辅助脚本
├─ build_system/                 CMake/Visual Studio 构建入口
├─ configs/                      UI、链路、模型更新和星历配置
├─ data/code_matrices/           LDPC/Polar/WiMAX/DVB-S2 编码矩阵
└─ performance_results/
   ├─ figures/                   性能结果图
   ├─ tables/                    汇总表和三线表
   └─ tools/                     性能结果抽取/绘图脚本
```

## Main Components

- `source/python/UI_NEW/main.py`：当前主要 PySide6 UI，集成星地可视化、业务选择、模型更新、离线 GPU pipeline 与 USRP 控制入口。
- `source/python/UI_NEW/usrp_sync_task.py`：UI 触发 USRP 同步任务的辅助线程。
- `source/python/UI_NEW/model_update.py`：遥感 AI 模型更新包生成、切片、manifest 与接收端部署逻辑。
- `source/cpp_cuda/uhd_cpp/src/uhd_ldpc_ofdm_link.cpp`：UHD LDPC-OFDM 收发链路核心实现。
- `source/cpp_cuda/uhd_cpp/src/gpu_ofdm_pipeline.cu`：GPU OFDM pipeline。
- `source/python/Ai_process_V2` 与 `source/python/Ai_process_V3`：遥感 AI 业务编码、ROI/语义层处理与 RSBF bitstream 工具。
- `source/matlab/`：MATLAB 编译码、X310 链路调试与 NTN 相关算法。

## Quick Start: UI

Windows 下可直接从仓库根目录运行：

```powershell
.\SETUP_UI_ENV.bat
.\START_UI_NEW.bat
```

也可以手动创建 Python 环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python .\source\python\UI_NEW\main.py
```

如果只启动 UI，可重点确认 `PySide6`、`pyqtgraph`、`matplotlib`、`PyOpenGL` 等依赖。遥感 AI/YOLO 流程还需要 `torch`、`ultralytics`、`opencv-python` 等包。

## External Dependencies

以下依赖与本机硬件/安装路径有关，不写入 `requirements.txt`：

- MATLAB R2024a 或更新版本，建议包含 Communications Toolbox、5G Toolbox、Satellite Communications Toolbox/Aerospace Toolbox、Instrument Control Toolbox。
- Visual Studio 2022 C++ Build Tools。
- CMake。
- NVIDIA CUDA Toolkit，与本机 GPU 驱动匹配。
- UHD 4.9.x、X310/NI-RIO 驱动与 FPGA images。
- FFmpeg，用于视频封装、转码或 UI 视频桥接。

## Environment Variables

上传版代码已移除本机绝对路径，外部工具路径通过环境变量或命令行参数指定：

```powershell
$env:PROJECT_ROOT = (Get-Location).Path
$env:UHD_ROOT = "<uhd-install-root>"
$env:UHD_PKG_PATH = "<uhd-install-root>"
$env:CUDA_PATH = "<cuda-toolkit-root>"
$env:CUDA_OSD_ROOT = "<cuda-osd-project-root>"
$env:CUDA_OSD_DLL_DIR = "<cuda-osd-runtime-dll-dir>"
$env:TORCH_LIB_DIR = "<torch-lib-dir>"
$env:MATLAB_EXE = "matlab"
```

未设置这些变量时，UI 和脚本会尽量使用仓库相对路径；涉及 USRP、CUDA OSD、MATLAB 或 Torch 动态库的功能仍需本机正确安装对应依赖。

## Build UHD C++/CUDA Link

示例 CMake 配置：

```powershell
cmake -S .\build_system\uhd_cpp -B .\build\uhd_cpp -G "Visual Studio 17 2022" -A x64 -DUHD_ROOT="$env:UHD_ROOT"
cmake --build .\build\uhd_cpp --config Release
```

常用脚本位于：

```text
scripts/powershell/uhd_cpp/scripts/start_uhd_cpp_pair.ps1
scripts/powershell/uhd_cpp/scripts/start_video_pair.ps1
scripts/powershell/uhd_cpp/scripts/run_codec_matrix_sweep.ps1
scripts/powershell/uhd_cpp/scripts/run_usrp_gain_frame_sweep.ps1
```

## Performance Results

`performance_results/` 中保留了论文/报告可直接引用的图表与汇总表：

- `performance_results/figures/`：MATLAB 绘制的性能图。
- `performance_results/tables/performance_three_line_tables.md`：GitHub 可读表格。
- `performance_results/tables/performance_three_line_tables.tex`：LaTeX booktabs 三线表。
- `performance_results/tools/`：结果抽取与绘图脚本。

重新生成表格和图：

```powershell
.\performance_results\tools\extract_performance_results.ps1
matlab -batch "cd('performance_results/tools'); plot_performance_results;"
```

## Data And Assets Policy

为保证仓库可上传、可 clone，本次 Git 版本不包含以下大型或运行时文件：

```text
data/media/
data/models/
data/datasets/
data/transmission/
data/runtime_samples/
performance_results/raw/
runtime/
generated/
build/
.venv/
```

如果需要复现实验演示，可按需补充模型、视频和原始数据。模型文件建议使用 Git LFS 管理：

```powershell
git lfs install
git lfs track "*.pt" "*.pth" "*.onnx" "*.engine"
git add .gitattributes
```

## Notes

- 仓库根目录已提供 `.gitignore`，运行 UI、构建工程或生成测试结果后，`runtime/`、`generated/`、`build/`、虚拟环境和缓存不会被默认提交。
- 上传版已对关键代码、脚本、CMake 和配置执行绝对路径扫描，避免依赖原始实验机的 Windows 盘符绝对路径。
- 部分历史配置和性能表反映既有实验结果，不代表 clone 后无需硬件即可复现实测 USRP 链路。
