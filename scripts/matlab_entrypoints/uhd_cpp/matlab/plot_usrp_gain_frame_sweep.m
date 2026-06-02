function outputPath = plot_usrp_gain_frame_sweep(metricsFile, saveDir, visible)
%PLOT_USRP_GAIN_FRAME_SWEEP Plot one-factor USRP gain and frame-length test.

if nargin < 1 || strlength(string(metricsFile)) == 0
    error("metricsFile is required.");
end
if nargin < 2 || strlength(string(saveDir)) == 0
    saveDir = fullfile(fileparts(metricsFile), "figures");
end
if nargin < 3 || strlength(string(visible)) == 0
    visible = "on";
end
if ~isfile(metricsFile)
    error("Metric file not found: %s", metricsFile);
end
if ~isfolder(saveDir)
    mkdir(saveDir);
end

T = readtable(metricsFile, "TextType", "string", "VariableNamingRule", "preserve");
fields = ["tx_gain_db", "rx_gain_db", "num_symbols", "frame_samples", "frames", ...
    "FER", "goodput_mbps", "fps", "decode_total_ms_per_frame", ...
    "fec_ms_per_frame", "sync_miss", "tracking_miss", "sync_incomplete", ...
    "uhd_overflow_markers"];
for field = fields
    T.(field) = str2double(string(T.(field)));
end

baselineTx = 25;
baselineRx = 30;
baselineSymbols = 120;
tx = sortrows(T(T.rx_gain_db == baselineRx & T.num_symbols == baselineSymbols, :), "tx_gain_db");
rx = sortrows(T(T.tx_gain_db == baselineTx & T.num_symbols == baselineSymbols, :), "rx_gain_db");
fl = sortrows(T(T.tx_gain_db == baselineTx & T.rx_gain_db == baselineRx, :), "num_symbols");

f = figure("Color", "w", "Visible", char(string(visible)), "Name", "USRP gain frame sweep");
f.Position(3:4) = [1420, 900];
layout = tiledlayout(f, 2, 2, "Padding", "loose", "TileSpacing", "compact");

ax = nexttile(layout);
yyaxis(ax, "left");
plot(ax, tx.tx_gain_db, tx.goodput_mbps, "-o", "LineWidth", 1.6, ...
    "MarkerFaceColor", [0.00 0.45 0.74]);
ylabel(ax, "有效吞吐率 (Mbps)");
ylim(ax, [0 2.6]);
yyaxis(ax, "right");
plot(ax, tx.tx_gain_db, tx.uhd_overflow_markers, "--s", "LineWidth", 1.3);
ylabel(ax, "UHD overflow 标记数");
xlabel(ax, "TX gain (dB)");
title(ax, "发射增益扫描（RX=30 dB, symbols=120）");
grid(ax, "on");

ax = nexttile(layout);
yyaxis(ax, "left");
plot(ax, rx.rx_gain_db, rx.goodput_mbps, "-o", "LineWidth", 1.6, ...
    "MarkerFaceColor", [0.00 0.45 0.74]);
ylabel(ax, "有效吞吐率 (Mbps)");
ylim(ax, [0 2.6]);
yyaxis(ax, "right");
plot(ax, rx.rx_gain_db, rx.uhd_overflow_markers, "--s", "LineWidth", 1.3);
ylabel(ax, "UHD overflow 标记数");
xlabel(ax, "RX gain (dB)");
title(ax, "接收增益扫描（TX=25 dB, symbols=120）");
grid(ax, "on");

ax = nexttile(layout);
yyaxis(ax, "left");
plot(ax, fl.num_symbols, fl.goodput_mbps, "-o", "LineWidth", 1.6, ...
    "MarkerFaceColor", [0.47 0.67 0.19]);
ylabel(ax, "有效吞吐率 (Mbps)");
ylim(ax, [0 2.6]);
yyaxis(ax, "right");
shownFer = fl.FER;
zero = shownFer == 0;
shownFer(zero) = 3 ./ fl.frames(zero);
semilogy(ax, fl.num_symbols, shownFer, "--s", "LineWidth", 1.4);
yline(ax, 1e-3, ":", "FER=10^{-3}", "HandleVisibility", "off");
ylabel(ax, "FER（零误帧点按 3/N 上界）");
ylim(ax, [1e-4 2e-3]);
xlabel(ax, "每帧 OFDM 符号数");
title(ax, "帧长扫描（TX=25 dB, RX=30 dB）");
grid(ax, "on");

ax = nexttile(layout);
caseNames = compose("%g/%g/%d", T.tx_gain_db, T.rx_gain_db, T.num_symbols);
anomalies = [T.uhd_overflow_markers, T.sync_miss + T.tracking_miss];
bar(ax, categorical(caseNames, caseNames, "Ordinal", true), anomalies);
ylabel(ax, "事件数");
title(ax, "实时链路异常统计（TX/RX/symbols）");
legend(ax, ["UHD overflow 标记", "同步 miss + tracking miss"], ...
    "Location", "northwest");
ax.XTickLabelRotation = 30;
ax.FontSize = 9.5;
grid(ax, "on");

title(layout, "USRP 全 GPU 链路：增益与帧长度单因素扫描（QPSK, DVB-S2 short R1/4, 20 s/组）");
outputPath = fullfile(saveDir, "fig6_usrp_gain_frame_onefactor.png");
exportgraphics(f, outputPath, "Resolution", 230);
if strcmpi(string(visible), "off")
    close(f);
end
fprintf("Saved: %s\n", outputPath);
end
