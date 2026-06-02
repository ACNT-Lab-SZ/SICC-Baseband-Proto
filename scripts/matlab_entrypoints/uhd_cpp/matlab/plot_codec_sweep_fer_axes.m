function report = plot_codec_sweep_fer_axes(metricsFile, varargin)
%PLOT_CODEC_SWEEP_FER_AXES Plot one offline codec sweep in two SNR coordinates.
%
% report = plot_codec_sweep_fer_axes(metricsFile)
% report = plot_codec_sweep_fer_axes(metricsFile, "SaveDir", outDir)
%
% The Eb/N0 figure evaluates coding performance after energy normalization.
% The injected-SNR figure is intended for practical MCS/adaptation lookup.

p = inputParser;
addParameter(p, "SaveDir", "", @(x) isstring(x) || ischar(x));
addParameter(p, "Visible", "on", @(x) any(strcmpi(string(x), ["on", "off"])));
addParameter(p, "Modulation", "qpsk", @(x) isstring(x) || ischar(x));
addParameter(p, "ExcludeSmoke", true, @(x) islogical(x) && isscalar(x));
parse(p, varargin{:});

saveDir = string(p.Results.SaveDir);
visible = char(string(p.Results.Visible));
modulation = lower(string(p.Results.Modulation));

if ~isfile(metricsFile)
    error("Offline metric table not found: %s", metricsFile);
end

T = readtable(metricsFile, "TextType", "string", "VariableNamingRule", "preserve");
T = normalizeMetricTypes(T);
T = T(lower(T.modulation) == modulation, :);
if p.Results.ExcludeSmoke
    T = T(~contains(lower(T.code), "smoke"), :);
end
if isempty(T)
    error("No %s results remain in %s after filtering.", upper(modulation), metricsFile);
end

report.filtered = T;
report.modulation = modulation;
report.ebn0Figure = plotCoordinateView(T, "ebn0_db", ...
    "E_b/N_0 (dB)", upper(modulation) + " 编码性能评价");
report.injectedSnrFigure = plotCoordinateView(T, "injected_snr_db", ...
    "实际注入数据子载波 SNR (dB)", upper(modulation) + " 链路自适应选档");
set(report.ebn0Figure, "Visible", visible);
set(report.injectedSnrFigure, "Visible", visible);

if strlength(saveDir) > 0
    if ~isfolder(saveDir)
        mkdir(saveDir);
    end
    exportgraphics(report.ebn0Figure, fullfile(saveDir, "fer_goodput_vs_ebn0.png"), "Resolution", 200);
    exportgraphics(report.injectedSnrFigure, fullfile(saveDir, "fer_goodput_vs_injected_snr.png"), "Resolution", 200);
    writetable(T, fullfile(saveDir, "plotted_offline_metrics.csv"));
    fprintf("FER/goodput figures saved to: %s\n", saveDir);
end
end

function T = normalizeMetricTypes(T)
numericNames = ["N", "K", "rate", "ebn0_db", "injected_snr_db", ...
    "injected_noise_var", "llr_demod_noise_var", "frames", "err", ...
    "FER", "BER", "fps", "goodput_mbps", "fec_cap_info_mbps"];
for name = numericNames
    if ismember(name, string(T.Properties.VariableNames))
        T.(name) = str2double(string(T.(name)));
    end
end
end

function f = plotCoordinateView(T, xName, xLabel, figureName)
f = figure("Name", figureName, "Color", "w", "Visible", "off");
f.Position(3:4) = [1300, 520];
layout = tiledlayout(f, 1, 2, "Padding", "compact", "TileSpacing", "compact");

codes = unique(T.code, "stable");
colors = lines(numel(codes));
labels = strings(numel(codes), 1);
h = gobjects(numel(codes), 1);

axFer = nexttile(layout, 1);
hold(axFer, "on");
grid(axFer, "on");
for i = 1:numel(codes)
    S = sortrows(T(T.code == codes(i), :), xName);
    y = S.FER;
    y(y == 0) = 1 ./ S.frames(y == 0);
    labels(i) = codeLabel(codes(i), S.N(1), S.K(1));
    h(i) = semilogy(axFer, S.(xName), y, "-o", ...
        "Color", colors(i, :), "LineWidth", 1.15, "DisplayName", labels(i));
end
yline(axFer, 1e-3, "--k", "FER = 10^{-3}", ...
    "LabelHorizontalAlignment", "left", "HandleVisibility", "off");
ylim(axFer, [1e-5, 1]);
set(axFer, "YScale", "log");
xlabel(axFer, xLabel);
ylabel(axFer, "FER (零错误点显示为 1/测试帧数)");
title(axFer, figureName + " - 可靠性");

axGoodput = nexttile(layout, 2);
hold(axGoodput, "on");
grid(axGoodput, "on");
for i = 1:numel(codes)
    S = sortrows(T(T.code == codes(i), :), xName);
    plot(axGoodput, S.(xName), S.goodput_mbps, "-o", ...
        "Color", colors(i, :), "LineWidth", 1.15, "HandleVisibility", "off");
end
xlabel(axGoodput, xLabel);
ylabel(axGoodput, "有效吞吐率 (Mbps)");
title(axGoodput, figureName + " - 吞吐量");

lgd = legend(axFer, h, labels, "Orientation", "horizontal", ...
    "NumColumns", min(numel(codes), 4), "Interpreter", "none");
lgd.Layout.Tile = "south";
end

function name = codeLabel(code, n, k)
if startsWith(code, "CCSDS_ldpc_")
    standard = "CCSDS LDPC";
elseif startsWith(code, "DVB_S2")
    standard = "DVB-S2 LDPC";
else
    standard = "LDPC";
end
name = standard + " (" + string(n) + ", " + string(k) + ")";
end
