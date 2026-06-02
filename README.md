# USRP NTN Project Organized Workspace


## Directory Layout

```text
organized_workspace_20260601/
  source/
    python/                 Python 主程序与模块源码
    matlab/                 MATLAB 函数与算法源码
    cpp_cuda/               C/C++/CUDA/UHD/LDPC-OFDM 源码
    web_ui/                 HTML/CSS/QML/JS UI 源码
    other/                  Fortran/Go/typing 等其他源码
  scripts/
    powershell/             PowerShell 运行、构建、USRP 启动脚本
    batch_cmd/              Windows bat/cmd 环境与启动脚本
    matlab_entrypoints/     MATLAB run/start/x310/plot 入口脚本
    python_tools/           Python 工具脚本、示例发送器
    shell_js/               Shell/Node 辅助脚本
  data/
    code_matrices/          LDPC/Polar/WiMAX/DVB-S2 矩阵与码表
    datasets/               MAT/CSV/NumPy/Pickle 数据集与结果表
    models/                 PyTorch 模型、模型更新包、任务模型
    media/                  图片、视频、UI 纹理、演示素材
    transmission/           bin/rsbf/channel 等链路传输数据
    runtime_samples/        zip 等便携样例包
    other/                  其他数据文件
  configs/                  JSON/INI/CFG/XML/TLE 配置与星历
  docs/                     README、技术文档、报告、PDF/DOCX
  build_system/             CMake/Visual Studio 工程入口文件
  manifests/                本次整理的机器可读清单
  performance_results/      测试结果、性能表格与 MATLAB 绘图输出
  tools/                    整理脚本
```

## Project Areas

原工程可以按下面几个方向理解：

- MATLAB NTN/Coded-OTFS 原型：根目录、`project/`、`+project/`、`matlab_qianfan_ntn/`。
- UHD C++/CUDA 链路：`uhd_cpp/`、`Ai_process_V3/src/`、`Ai_process_V3/tests/`。
- 遥感 AI 与模型更新链路：`Ai_process_V2/`、`Ai_process_V3/`、`transmission_file/`。
- PySide6 星地展示 UI：`Satellite_UI/`、`UI_NEW/`、`UI/SA_UI/`。
- 信道编码矩阵库：`Code_Matrices_Lib/`、`ChannelCodeFile/`。
- 便携交付包：`portable/usrp_video_link_portable_20260513/`。

## Manifest Files

`manifests/` 目录是本次扫描最重要的索引：

- `file_manifest.csv`：已归类文件清单，包含类别、原路径、扩展名、大小、复制方法、目标路径。
- `excluded_manifest.csv`：未纳入主整理目录的文件清单及原因。
- `category_summary.csv`：按分类统计的文件数量和体积。
- `extension_summary.csv`：按扩展名统计的文件数量和体积。
- `largest_included_files.csv`：纳入整理目录的最大 50 个文件。
- `organize_result.json`：本次整理总体结果。

本次统计：

```text
included files: 2405
excluded files: 42967
included size: 5952.83 MB
excluded size: 4969.07 MB
```

## Python Environment

推荐先为 UI/AI 部分创建新的虚拟环境：

```powershell
cd <clone-root>
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果只运行 UI，可以只关心 `PySide6`、`pyqtgraph`、`matplotlib`、`PyOpenGL`。如果运行遥感 AI/YOLO/模型更新流程，还需要 `torch`、`ultralytics`、`opencv-python` 等包。

## System Dependencies

这些依赖不应写进 pip requirements，需要按机器环境单独安装或确认：

- MATLAB R2024a 或更新版本，建议包含 Communications Toolbox、5G Toolbox、Satellite Communications Toolbox/Aerospace Toolbox、Instrument Control Toolbox。
- Visual Studio 2022 C++ Build Tools。
- CMake。
- NVIDIA CUDA Toolkit，与本机 GPU 驱动匹配。
- UHD 4.9.x 及 X310/NI-RIO 相关驱动、FPGA images。
- FFmpeg，用于视频封装、转码或 UI 视频桥接。

## Main Entrypoints

常用入口按方向列出：

```text
MATLAB NTN demo:
  run_ntn_otfs_demo.m
  project/startup_ntn_otfs_project.m
  matlab_qianfan_ntn/run_qianfan_ntn_demo.m

USRP C++/CUDA:
  uhd_cpp/CMakeLists.txt
  uhd_cpp/scripts/start_uhd_cpp_pair.ps1
  uhd_cpp/scripts/start_video_pair.ps1
  Ai_process_V3/src/uhd_ldpc_ofdm_link.cpp

UI:
  Satellite_UI/main.py
  UI_NEW/main.py
  SETUP_UI_ENV.bat
  START_UI_NEW.bat

AI/model update:
  Ai_process_V3/rs_ai_link/cli.py
  Ai_process_V3/tests/fec_offline_validation.cpp
  transmission_file/Tx/model_updates/
```

## Re-run Organization

如需重新扫描并刷新本目录：

```powershell
cd <source-project-root>
.\organized_workspace_20260601\tools\organize_workspace.ps1
```

脚本会重建目标文件副本，并刷新 `manifests/` 下的统计清单。
