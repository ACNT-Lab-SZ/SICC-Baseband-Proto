function outputs = plot_throughput_all_ldpc_mcs_results(metricsPath, saveDir, visible)
%PLOT_THROUGHPUT_ALL_LDPC_MCS_RESULTS Plot completed throughput-only sweep.

if nargin < 1 || strlength(string(metricsPath)) == 0
    here = fileparts(mfilename("fullpath"));
    repoRoot = fileparts(fileparts(here));
    metricsPath = fullfile(repoRoot, "build", "uhd_cpp_gpu_pipeline", "logs", ...
        "throughput_all_ldpc_mcs_ebn0_0_2_16_20260524_161606", ...
        "offline_metrics.csv");
end
if nargin < 2 || strlength(string(saveDir)) == 0
    saveDir = fullfile(fileparts(metricsPath), "figures");
end
if nargin < 3 || strlength(string(visible)) == 0
    visible = "on";
end
visible = char(string(visible));
if ~isfile(metricsPath)
    error("Missing throughput metrics: %s", metricsPath);
end
if ~isfolder(saveDir)
    mkdir(saveDir);
end

T = readtable(metricsPath, "TextType", "string", "VariableNamingRule", "preserve");
codes = ["CCSDS_ldpc_n128_k64", "CCSDS_ldpc_n256_k128", "CCSDS_ldpc_n512_k256", ...
    "DVB_S2_N64800_R12", "DVB_S2_short_N16200_rate_1_2", ...
    "DVB_S2_short_N16200_rate_1_4", "DVB_S2_short_N16200_rate_5_6"];
mods = ["bpsk", "qpsk", "16qam", "64qam"];
palette = lines(numel(codes));
outputs = strings(2, 1);
outputs(1) = fullfile(saveDir, "fig6_offline_throughput_all_codes_by_mcs.png");
outputs(2) = fullfile(saveDir, "fig6_offline_throughput_heatmap_16db.png");

% Four-panel throughput view. Each panel keeps modulation fixed and compares codes.
f = figure("Name", "全码型全调制离线吞吐率", "Color", "w", "Visible", visible);
f.Position(3:4) = [1500, 930];
layout = tiledlayout(2, 2, "Padding", "compact", "TileSpacing", "compact");
handles = gobjects(numel(codes), 1);
for mi = 1:numel(mods)
    ax = nexttile(layout); hold(ax, "on"); grid(ax, "on");
    for ci = 1:numel(codes)
        S = sortrows(T(T.code == codes(ci) & T.modulation == mods(mi), :), "ebn0_db");
        if isempty(S)
            continue;
        end
        p = plot(ax, S.ebn0_db, S.goodput_mbps, "-o", "Color", palette(ci, :), ...
            "LineWidth", 1.25, "MarkerSize", 4, "DisplayName", codeLabel(codes(ci)));
        if mi == 1
            handles(ci) = p;
        end
    end
    xlabel(ax, "E_b/N_0 (dB)");
    ylabel(ax, "有效吞吐率 (Mbps)");
    title(ax, upper(mods(mi)));
    xlim(ax, [0, 16]);
    ylim(ax, [0, 200]);
end
title(layout, "全 GPU 离线链路：不同 LDPC 与调制方式的有效吞吐率");
lgd = legend(handles, arrayfun(@codeLabel, codes), "Orientation", "horizontal", ...
    "NumColumns", 4, "Interpreter", "none");
lgd.Layout.Tile = "south";
exportgraphics(f, outputs(1), "Resolution", 220);
close(f);

% Heatmap for the high-Eb/N0 endpoint used for combination ranking.
S = T(T.ebn0_db == 16, :);
values = nan(numel(codes), numel(mods));
for ci = 1:numel(codes)
    for mi = 1:numel(mods)
        hit = S(S.code == codes(ci) & S.modulation == mods(mi), :);
        if ~isempty(hit)
            values(ci, mi) = hit.goodput_mbps(1);
        end
    end
end
f = figure("Name", "16dB 组合吞吐率热力图", "Color", "w", "Visible", visible);
f.Position(3:4) = [1050, 700];
h = heatmap(upper(mods), arrayfun(@codeLabel, codes), values);
h.Title = "E_b/N_0 = 16 dB 时的全 GPU 有效吞吐率";
h.XLabel = "调制方式";
h.YLabel = "LDPC 码型";
h.ColorbarVisible = "on";
h.CellLabelFormat = "%.2f";
h.Colormap = parula;
exportgraphics(f, outputs(2), "Resolution", 220);
close(f);

disp(outputs);
end

function name = codeLabel(code)
code = string(code);
if code == "CCSDS_ldpc_n128_k64"
    name = "CCSDS (128,64)";
elseif code == "CCSDS_ldpc_n256_k128"
    name = "CCSDS (256,128)";
elseif code == "CCSDS_ldpc_n512_k256"
    name = "CCSDS (512,256)";
elseif code == "DVB_S2_N64800_R12"
    name = "DVB-S2 normal R1/2";
elseif code == "DVB_S2_short_N16200_rate_1_2"
    name = "DVB-S2 short R1/2";
elseif code == "DVB_S2_short_N16200_rate_1_4"
    name = "DVB-S2 short R1/4";
elseif code == "DVB_S2_short_N16200_rate_5_6"
    name = "DVB-S2 short R5/6";
else
    name = code;
end
end
