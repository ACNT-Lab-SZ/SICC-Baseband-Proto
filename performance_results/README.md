# Performance Results

本目录整理了当前工程中可识别的测试、验证、训练和链路性能结果。

## Layout

```text
performance_results/
  raw/       原始结果文件副本，按结果类型归档
  tables/    汇总 CSV、Markdown 表格、LaTeX booktabs 三线表
  figures/   MATLAB 生成的性能结果图
  tools/     可重复生成脚本
```

## Figures

- `figures/yolo_training_performance.png`：YOLO 检测训练指标，包括 best mAP、最终 precision/recall。
- `figures/gpu_direct_roundtrip_performance.png`：GPU Direct roundtrip FPS 与平均 payload。
- `figures/link_decoder_performance.png`：USRP 链路 goodput/FPS/译码耗时与 AWGN decoder BER/FER。

## Tables

- `tables/performance_three_line_tables.md`：便于 GitHub 查看的一组汇总表。
- `tables/performance_three_line_tables.tex`：LaTeX booktabs 三线表，可直接放入论文或报告。
- `tables/yolo_training_summary.csv`：训练结果汇总。
- `tables/gpu_direct_roundtrip_summary.csv`：GPU Direct/ROI roundtrip 汇总。
- `tables/usrp_link_summary.csv`：USRP 链路实测汇总。
- `tables/decoder_awgn_summary.csv`：decoder AWGN 汇总。
- `tables/ui_metric_summary.csv`：UI metric JSON 汇总。
- `tables/power_trace_summary.csv`：预测功率 trace 汇总。
- `tables/result_file_inventory.csv`：纳入本次整理的原始结果文件索引。

## Current Highlights

- Maritime ship detection: best mAP50 = 0.9861, best mAP50-95 = 0.7024.
- Earthquake damage detection: best mAP50 = 0.8562, best mAP50-95 = 0.8562.
- VISO ROI-layered payload saving: payload 从 1700 KB/frame 降至 907.87 KB/frame，检测数保持 5277。
- USRP gated GPU pipeline: 14531 frames, FER = 0.009153, goodput = 0.69 Mbps, decode = 1.97 ms/frame.
- OSD-only AWGN point: SNR = 3 dB, BER = 2.1875e-4, FER = 1.0e-3.

## Reproduce

在 PowerShell 中重新抽取表格：

```powershell
cd <clone-root>
.\performance_results\tools\extract_performance_results.ps1
```

用 MATLAB 重新绘图：

```powershell
matlab -batch "cd('<clone-root>/performance_results/tools'); plot_performance_results;"
```
