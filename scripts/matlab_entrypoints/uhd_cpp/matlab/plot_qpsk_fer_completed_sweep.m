function outputPath = plot_qpsk_fer_completed_sweep(metricsFile, saveDir, visible)
%PLOT_QPSK_FER_COMPLETED_SWEEP Generate a paper-ready FER comparison plot.

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
    error("Offline metric table not found: %s", metricsFile);
end
if ~isfolder(saveDir)
    mkdir(saveDir);
end

T = readtable(metricsFile, "TextType", "string", "VariableNamingRule", "preserve");
numericFields = ["N", "K", "ebn0_db", "frames", "err", "FER"];
for field = numericFields
    T.(field) = str2double(string(T.(field)));
end
T = T(lower(T.modulation) == "qpsk" & ~contains(lower(T.code), "smoke"), :);

codes = unique(T.code, "stable");
colors = lines(numel(codes));
styles = ["-", "-", "-", "--", "--", "--", "-."];
markers = ["o", "s", "d", "^", "v", ">", "<"];

f = figure("Name", "QPSK LDPC FER", "Color", "w", "Visible", char(string(visible)));
f.Position(3:4) = [1260, 720];
ax = axes(f);
hold(ax, "on");
grid(ax, "on");
ax.GridAlpha = 0.20;
ax.MinorGridAlpha = 0.10;
ax.YMinorGrid = "on";

curves = gobjects(numel(codes), 1);
labels = strings(numel(codes), 1);
for i = 1:numel(codes)
    S = sortrows(T(T.code == codes(i), :), "ebn0_db");
    zeroMask = S.FER == 0;
    measuredFer = S.FER;
    measuredFer(zeroMask) = NaN;
    upperBound = 3 ./ S.frames(zeroMask);
    labels(i) = codeLabel(codes(i), S.N(1), S.K(1));
    curves(i) = semilogy(ax, S.ebn0_db, measuredFer, ...
        "Color", colors(i, :), "LineStyle", styles(i), "LineWidth", 1.55, ...
        "Marker", markers(i), "MarkerSize", 5.5, "MarkerFaceColor", "w", ...
        "DisplayName", labels(i));
    if any(zeroMask)
        semilogy(ax, S.ebn0_db(zeroMask), upperBound, "v", ...
            "Color", colors(i, :), "MarkerFaceColor", colors(i, :), ...
            "MarkerSize", 5.5, "HandleVisibility", "off");
    end
end

yline(ax, 1e-3, ":", "FER = 10^{-3}", ...
    "Color", [0.15 0.15 0.15], "LineWidth", 1.1, ...
    "LabelHorizontalAlignment", "left", "HandleVisibility", "off");
xlabel(ax, "E_b/N_0 (dB)");
ylabel(ax, "帧误码率 FER");
title(ax, "QPSK 下不同 LDPC 码型的全 GPU 离线 FER 性能");
xlim(ax, [0 8]);
xticks(ax, 0:0.5:8);
ylim(ax, [1e-6 1.2]);
set(ax, "YScale", "log", "FontSize", 11);

lgd = legend(ax, curves, labels, "Location", "southoutside", ...
    "Orientation", "horizontal", "NumColumns", 4, "Interpreter", "none");
lgd.FontSize = 9.5;
annotation(f, "textbox", [0.10 0.005 0.80 0.035], ...
    "String", "实心倒三角：观测到 0 个误帧，按 95% 置信上界 3/N_{frames} 显示", ...
    "EdgeColor", "none", "HorizontalAlignment", "center", "FontSize", 9);

outputPath = fullfile(saveDir, "fig6_qpsk_all_ldpc_fer_vs_ebn0_final.png");
exportgraphics(f, outputPath, "Resolution", 240);
if strcmpi(string(visible), "off")
    close(f);
end
fprintf("Saved: %s\n", outputPath);
end

function label = codeLabel(code, n, k)
switch code
    case "CCSDS_ldpc_n128_k64"
        label = "CCSDS (128,64)";
    case "CCSDS_ldpc_n256_k128"
        label = "CCSDS (256,128)";
    case "CCSDS_ldpc_n512_k256"
        label = "CCSDS (512,256)";
    case "DVB_S2_N64800_R12"
        label = "DVB-S2 normal R1/2 (64800,32400)";
    case "DVB_S2_short_N16200_rate_1_2"
        label = "DVB-S2 short R1/2 (16200,8100)";
    case "DVB_S2_short_N16200_rate_1_4"
        label = "DVB-S2 short R1/4 (16200,4050)";
    case "DVB_S2_short_N16200_rate_5_6"
        label = "DVB-S2 short R5/6 (16200,13500)";
    otherwise
        label = code + " (" + string(n) + "," + string(k) + ")";
end
end
