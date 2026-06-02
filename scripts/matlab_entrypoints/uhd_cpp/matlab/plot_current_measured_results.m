function outputs = plot_current_measured_results(logRoot, saveDir, visible)
%PLOT_CURRENT_MEASURED_RESULTS Consolidate measured offline and USRP results.
% This function reads completed result CSV files only and writes new figures
% and summary tables into saveDir. It does not alter any experiment output.

if nargin < 1 || strlength(string(logRoot)) == 0
    here = fileparts(mfilename("fullpath"));
    repoRoot = fileparts(fileparts(here));
    logRoot = fullfile(repoRoot, "build", "uhd_cpp_gpu_pipeline", "logs");
end
if nargin < 2 || strlength(string(saveDir)) == 0
    saveDir = fullfile(logRoot, "integrated_results_20260527");
end
if nargin < 3 || strlength(string(visible)) == 0
    visible = "on";
end
visible = char(string(visible));

paths.fer = fullfile(logRoot, ...
    "fer_qpsk_dvbs2_500k_ebn0_0_05_8_20260524_232200", "offline_metrics.csv");
paths.throughput = fullfile(logRoot, ...
    "throughput_all_ldpc_mcs_ebn0_0_2_16_20260524_161606", "offline_metrics.csv");
paths.usrpCode = fullfile(logRoot, ...
    "usrp_qpsk_dvbs2_12p5m_200s_20260524_170156", "usrp_metrics.csv");
paths.usrpMcs = fullfile(logRoot, "Test_before_2026_05_24_16_15", ...
    "usrp_fullgpu_modulation_600s", "usrp_metrics.csv");
paths.gainFiles = [
    string(fullfile(logRoot, "usrp_min_gain_check_tx9_rx14_25msps_20s_20260526", "usrp_gain_frame_metrics.csv"));
    string(fullfile(logRoot, "usrp_min_gain_confirm_tx11_rx12_25msps_20s_20260526", "usrp_gain_frame_metrics.csv"));
    string(fullfile(logRoot, "usrp_min_gain_cross_25msps_20260526", "tx13_rx12", "usrp_gain_frame_metrics.csv"));
    string(fullfile(logRoot, "usrp_min_gain_final_tx11_rx14_25msps_60s_20260526", "usrp_gain_frame_metrics.csv"));
    string(fullfile(logRoot, "usrp_min_gain_confirm_tx13_rx14_25msps_20s_20260526", "usrp_gain_frame_metrics.csv"));
    string(fullfile(logRoot, "usrp_min_gain_coarse_25msps_sc16_20260526", "tx15_rx16", "usrp_gain_frame_metrics.csv"));
    string(fullfile(logRoot, "usrp_min_gain_coarse_25msps_sc16_20260526", "tx17_rx18", "usrp_gain_frame_metrics.csv"));
    string(fullfile(logRoot, "usrp_min_gain_coarse_25msps_sc16_20260526", "tx19_rx20", "usrp_gain_frame_metrics.csv"));
    string(fullfile(logRoot, "usrp_min_gain_coarse_25msps_sc16_20260526", "tx21_rx22", "usrp_gain_frame_metrics.csv"));
    string(fullfile(logRoot, "usrp_tx_queue128_sc16_highprio_25msps_cp128_r56_tx23_rx24_60s_20260526", "usrp_gain_frame_metrics.csv"))
];
mustExist({paths.fer, paths.throughput, paths.usrpCode, paths.usrpMcs});
mustExist(cellstr(paths.gainFiles));
if ~isfolder(saveDir)
    mkdir(saveDir);
end

fer = readtable(paths.fer, "TextType", "string", "VariableNamingRule", "preserve");
throughput = readtable(paths.throughput, "TextType", "string", "VariableNamingRule", "preserve");
usrpCode = readtable(paths.usrpCode, "TextType", "string", "VariableNamingRule", "preserve");
usrpMcs = readtable(paths.usrpMcs, "TextType", "string", "VariableNamingRule", "preserve");
gain = readGainResults(paths.gainFiles);

outputs = strings(5, 1);
outputs(1) = fullfile(saveDir, "fig6_offline_qpsk_fer_all_codes.png");
outputs(2) = fullfile(saveDir, "fig6_offline_throughput_16db_heatmap.png");
outputs(3) = fullfile(saveDir, "fig6_usrp_code_mcs_measured_comparison.png");
outputs(4) = fullfile(saveDir, "fig6_usrp_25msps_gain_boundary.png");
outputs(5) = fullfile(saveDir, "fig6_usrp_25msps_reliability_goodput.png");

thresholds = buildFerThresholds(fer);
rank16 = sortrows(throughput(throughput.ebn0_db == 16, ...
    ["code", "N", "K", "rate", "modulation", "decoder", "ebn0_db", ...
     "goodput_mbps", "fps", "fec_ms_per_frame", "fec_cap_info_mbps"]), ...
    "goodput_mbps", "descend");
writetable(thresholds, fullfile(saveDir, "qpsk_fer_threshold_summary.csv"));
writetable(rank16, fullfile(saveDir, "offline_throughput_16db_rank.csv"));
writetable(usrpCode, fullfile(saveDir, "usrp_12p5msps_code_comparison.csv"));
writetable(usrpMcs, fullfile(saveDir, "usrp_600s_mcs_comparison.csv"));
writetable(gain, fullfile(saveDir, "usrp_25msps_gain_boundary_summary.csv"));

plotOfflineFer(fer, outputs(1), visible);
plotThroughputHeatmap(throughput, outputs(2), visible);
plotUsrpCodeMcs(usrpCode, usrpMcs, outputs(3), visible);
plotGainBoundary(gain, outputs(4), outputs(5), visible);

fprintf("\nQPSK FER < 1e-3 threshold summary:\n");
disp(thresholds);
fprintf("\nHighest measured offline throughput combinations at Eb/N0 = 16 dB:\n");
disp(rank16(1:min(10, height(rank16)), :));
fprintf("\n25 MSps USRP gain boundary summary:\n");
disp(gain(:, ["tx_gain_db", "rx_gain_db", "frames", "err", "FER", "goodput_mbps"]));
fprintf("Saved measured-data consolidation to: %s\n", saveDir);
end

function T = readGainResults(files)
T = table();
wanted = ["tx_gain_db", "rx_gain_db", "frames", "ok", "err", "FER", "BER", ...
    "fps", "goodput_mbps", "decode_total_ms_per_frame", "fec_ms_per_frame", ...
    "gpu_iq_buffer_dropped", "device_frame_queue_dropped"];
for i = 1:numel(files)
    row = readtable(files(i), "TextType", "string", "VariableNamingRule", "preserve");
    row = row(:, wanted);
    T = [T; row]; %#ok<AGROW>
end
T = sortrows(T, ["tx_gain_db", "rx_gain_db"]);
end

function T = buildFerThresholds(M)
codes = unique(M.code, "stable");
out = cell(numel(codes), 8);
for i = 1:numel(codes)
    S = sortrows(M(M.code == codes(i), :), "ebn0_db");
    hit = S(S.FER < 1e-3, :);
    zero = S(S.err == 0, :);
    if isempty(hit)
        out(i, :) = {codes(i), S.N(1), S.K(1), S.rate(1), NaN, NaN, NaN, NaN};
    else
        firstZero = NaN;
        if ~isempty(zero)
            firstZero = zero.ebn0_db(1);
        end
        out(i, :) = {codes(i), S.N(1), S.K(1), S.rate(1), ...
            hit.ebn0_db(1), hit.FER(1), hit.goodput_mbps(1), firstZero};
    end
end
T = cell2table(out, "VariableNames", ["code", "N", "K", "rate", ...
    "first_fer_lt_1e_3_ebn0_db", "fer_at_threshold", ...
    "goodput_at_threshold_mbps", "first_zero_error_ebn0_db"]);
end

function plotOfflineFer(T, out, visible)
f = figure("Color", "w", "Visible", visible, "Name", "Offline QPSK FER");
f.Position(3:4) = [1300, 700];
ax = axes(f); hold(ax, "on"); grid(ax, "on");
codes = unique(T.code, "stable");
colors = lines(numel(codes));
for i = 1:numel(codes)
    S = sortrows(T(T.code == codes(i), :), "ebn0_db");
    y = S.FER;
    y(y == 0) = 3 ./ S.frames(y == 0);
    semilogy(ax, S.ebn0_db, y, "-o", "LineWidth", 1.3, ...
        "Color", colors(i, :), "DisplayName", codeLabel(codes(i)));
end
yline(ax, 1e-3, "--k", "FER = 10^{-3}", "HandleVisibility", "off");
xlabel(ax, "E_b/N_0 (dB)"); ylabel(ax, "FER（零误帧点显示 95% 上界 3/N）");
title(ax, "QPSK 全 GPU 离线链路：不同 LDPC 码型可靠性");
ylim(ax, [1e-6, 1.2]); xlim(ax, [0, 8]); set(ax, "YScale", "log");
legend(ax, "Location", "southoutside", "Orientation", "horizontal", "NumColumns", 4);
exportgraphics(f, out, "Resolution", 240);
if strcmpi(visible, "off"), close(f); end
end

function plotThroughputHeatmap(T, out, visible)
codes = unique(T.code, "stable");
mods = ["bpsk", "qpsk", "16qam", "64qam"];
S = T(T.ebn0_db == 16, :);
values = nan(numel(codes), numel(mods));
for i = 1:numel(codes)
    for j = 1:numel(mods)
        hit = S(S.code == codes(i) & S.modulation == mods(j), :);
        if ~isempty(hit), values(i, j) = hit.goodput_mbps(1); end
    end
end
f = figure("Color", "w", "Visible", visible, "Name", "Offline throughput heatmap");
f.Position(3:4) = [1100, 700];
h = heatmap(upper(mods), labelCodes(codes), values);
h.Title = "E_b/N_0 = 16 dB 时全 GPU 离线有效吞吐率 (Mbps)";
h.XLabel = "调制方式"; h.YLabel = "码型"; h.CellLabelFormat = "%.2f";
h.Colormap = parula;
exportgraphics(f, out, "Resolution", 240);
if strcmpi(visible, "off"), close(f); end
end

function plotUsrpCodeMcs(C, M, out, visible)
f = figure("Color", "w", "Visible", visible, "Name", "USRP measured comparisons");
f.Position(3:4) = [1450, 720];
layout = tiledlayout(f, 2, 2, "Padding", "compact", "TileSpacing", "compact");
codeLabels = labelCodes(C.code);
codeNames = categorical(codeLabels, codeLabels, "Ordinal", true);
nexttile(layout);
bar(codeNames, C.goodput_mbps); grid on; ylabel("有效吞吐率 (Mbps)");
title("12.5 MSps, QPSK, 200 s：不同 DVB-S2 编码");
nexttile(layout);
semilogy(codeNames, C.FER, "-o", "LineWidth", 1.3); grid on;
yline(1e-3, "--k", "目标 FER=10^{-3}"); ylabel("FER");
title("12.5 MSps 编码可靠性");
modNames = categorical(upper(M.modulation), upper(M.modulation), "Ordinal", true);
nexttile(layout);
bar(modNames, M.goodput_mbps); grid on; ylabel("有效吞吐率 (Mbps)");
title("600 s 长测：调制阶数与有效吞吐率");
nexttile(layout);
semilogy(modNames, M.FER, "-o", "LineWidth", 1.3); grid on;
yline(1e-3, "--k", "目标 FER=10^{-3}"); ylabel("FER");
title("600 s 长测：高阶调制的可靠性代价");
exportgraphics(f, out, "Resolution", 240);
if strcmpi(visible, "off"), close(f); end
end

function plotGainBoundary(G, outFer, outTradeoff, visible)
labels = compose("%g/%g", G.tx_gain_db, G.rx_gain_db);
gainSum = G.tx_gain_db + G.rx_gain_db;
shownFer = G.FER;
shownFer(shownFer == 0) = 3 ./ G.frames(shownFer == 0);
passed = G.FER < 1e-3;

f = figure("Color", "w", "Visible", visible, "Name", "25MSps gain boundary");
f.Position(3:4) = [1350, 620];
layout = tiledlayout(f, 1, 2, "Padding", "compact", "TileSpacing", "compact");
ax = nexttile(layout); hold(ax, "on"); grid(ax, "on");
for i = 1:height(G)
    mark = "o";
    if passed(i), mark = "s"; end
    semilogy(ax, gainSum(i), shownFer(i), mark, "MarkerSize", 8, ...
        "LineWidth", 1.5, "MarkerFaceColor", ternaryColor(passed(i)));
    text(ax, gainSum(i) + 0.3, shownFer(i), labels(i), "FontSize", 9);
end
yline(ax, 1e-3, "--k", "FER = 10^{-3}");
xlabel(ax, "TX gain + RX gain (dB)");
ylabel(ax, "FER（零误帧点显示 95% 上界 3/N）");
title(ax, "可用增益边界（标注为 TX/RX）");
set(ax, "YScale", "log"); ylim(ax, [1e-5, 0.2]);
ax = nexttile(layout); hold(ax, "on"); grid(ax, "on");
scatter(ax, G.rx_gain_db, G.tx_gain_db, 100, log10(max(shownFer, 1e-6)), "filled");
for i = 1:height(G)
    text(ax, G.rx_gain_db(i) + 0.15, G.tx_gain_db(i), sprintf("%.1g", G.FER(i)), ...
        "FontSize", 8.5);
end
colorbar(ax); xlabel(ax, "RX gain (dB)"); ylabel(ax, "TX gain (dB)");
title(ax, "测试点 FER 分布（颜色为 log_{10}(FER)）");
exportgraphics(f, outFer, "Resolution", 240);
if strcmpi(visible, "off"), close(f); end

f = figure("Color", "w", "Visible", visible, "Name", "25MSps tradeoff");
f.Position(3:4) = [1300, 560];
layout = tiledlayout(f, 1, 2, "Padding", "compact", "TileSpacing", "compact");
ax = nexttile(layout); hold(ax, "on"); grid(ax, "on");
bar(ax, categorical(labels, labels, "Ordinal", true), G.goodput_mbps);
ylabel(ax, "有效吞吐率 (Mbps)"); xlabel(ax, "TX/RX gain (dB)");
title(ax, "25 MSps 全 GPU 链路有效吞吐率");
ax = nexttile(layout); hold(ax, "on"); grid(ax, "on");
yyaxis(ax, "left"); plot(ax, categorical(labels, labels, "Ordinal", true), ...
    G.decode_total_ms_per_frame, "-o", "LineWidth", 1.4);
ylabel(ax, "总消费时延 (ms/frame)");
yyaxis(ax, "right"); plot(ax, categorical(labels, labels, "Ordinal", true), ...
    G.fec_ms_per_frame, "-s", "LineWidth", 1.4);
ylabel(ax, "FEC 时延 (ms/frame)");
title(ax, "GPU 接收处理时延");
exportgraphics(f, outTradeoff, "Resolution", 240);
if strcmpi(visible, "off"), close(f); end
end

function color = ternaryColor(pass)
if pass
    color = [0.20, 0.60, 0.32];
else
    color = [0.84, 0.28, 0.24];
end
end

function label = codeLabel(code)
code = string(code);
switch code
    case "CCSDS_ldpc_n128_k64", label = "CCSDS (128,64)";
    case "CCSDS_ldpc_n256_k128", label = "CCSDS (256,128)";
    case "CCSDS_ldpc_n512_k256", label = "CCSDS (512,256)";
    case "DVB_S2_N64800_R12", label = "DVB-S2 normal R1/2";
    case "DVB_S2_short_N16200_rate_1_2", label = "DVB-S2 short R1/2";
    case "DVB_S2_short_N16200_rate_1_4", label = "DVB-S2 short R1/4";
    case "DVB_S2_short_N16200_rate_5_6", label = "DVB-S2 short R5/6";
    otherwise, label = code;
end
end

function labels = labelCodes(codes)
labels = strings(numel(codes), 1);
for i = 1:numel(codes)
    labels(i) = codeLabel(codes(i));
end
end

function mustExist(files)
for i = 1:numel(files)
    if ~isfile(files{i})
        error("Missing measured result file: %s", files{i});
    end
end
end
