function outputs = plot_chapter6_measured_results(logRoot, saveDir, visible)
%PLOT_CHAPTER6_MEASURED_RESULTS Generate Chapter 6 figures from completed runs.
% Only consumes existing measured CSV/log files; it does not synthesize
% constellation or spectrum data that was not captured during the USRP run.

if nargin < 1 || strlength(string(logRoot)) == 0
    here = fileparts(mfilename("fullpath"));
    repoRoot = fileparts(fileparts(here));
    logRoot = fullfile(repoRoot, "build", "uhd_cpp_gpu_pipeline", "logs");
end
if nargin < 2 || strlength(string(saveDir)) == 0
    saveDir = fullfile(logRoot, "chapter6_measured_figures");
end
if nargin < 3 || strlength(string(visible)) == 0
    visible = "on";
end
visible = char(string(visible));

paths.formalDvb = fullfile(logRoot, "Test_before_2026_05_24_16_15", ...
    "qpsk_dvbs2_tuned_formal_v3", "offline_metrics.csv");
paths.throughput = fullfile(logRoot, ...
    "throughput_all_ldpc_mcs_ebn0_0_2_16_20260524_161606", "offline_metrics.csv");
paths.usrp = fullfile(logRoot, ...
    "usrp_qpsk_dvbs2_12p5m_200s_20260524_170156", "usrp_metrics.csv");
mustExist(struct2cell(paths));
if ~isfolder(saveDir)
    mkdir(saveDir);
end

D = readtable(paths.formalDvb, "TextType", "string", "VariableNamingRule", "preserve");
T = readtable(paths.throughput, "TextType", "string", "VariableNamingRule", "preserve");
U = readtable(paths.usrp, "TextType", "string", "VariableNamingRule", "preserve");

outputs = strings(4, 1);
outputs(1) = fullfile(saveDir, "fig6_1_dvbs2_qpsk_fer_ebn0.png");
outputs(2) = fullfile(saveDir, "fig6_x_offline_throughput_mcs.png");
outputs(3) = fullfile(saveDir, "fig6_x_usrp_dvbs2_fer_goodput.png");
outputs(4) = fullfile(saveDir, "fig6_x_usrp_pipeline_runtime.png");

% Figure 6-1: formal DVB-S2 FER versus Eb/N0.
f = figure("Color", "w", "Visible", visible, "Name", "DVB-S2 QPSK FER");
hold on; grid on;
codes = unique(D.code, "stable");
for i = 1:numel(codes)
    S = sortrows(D(D.code == codes(i), :), "ebn0_db");
    fer = S.FER;
    fer(fer == 0) = 1 ./ S.frames(fer == 0);
    semilogy(S.ebn0_db, fer, "-o", "LineWidth", 1.3, ...
        "DisplayName", codeLabel(codes(i)));
end
yline(1e-3, "--k", "FER=10^{-3}", "HandleVisibility", "off");
xlabel("E_b/N_0 (dB)");
ylabel("误帧率 FER（零错误点按 1/N 显示）");
title("QPSK 下 DVB-S2 全 GPU 基带可靠性");
set(gca, "YScale", "log");
ylim([1e-5, 1]);
legend("Location", "southoutside", "Orientation", "horizontal", "NumColumns", 2);
exportgraphics(f, outputs(1), "Resolution", 220);
close(f);

% Offline throughput: the best-performing DVB-S2 short 5/6 curve by MCS.
target = T(T.code == "DVB_S2_short_N16200_rate_5_6", :);
mods = ["bpsk", "qpsk", "16qam", "64qam"];
f = figure("Color", "w", "Visible", visible, "Name", "Offline throughput by MCS");
hold on; grid on;
for i = 1:numel(mods)
    S = sortrows(target(target.modulation == mods(i), :), "ebn0_db");
    plot(S.ebn0_db, S.goodput_mbps, "-o", "LineWidth", 1.3, ...
        "DisplayName", upper(mods(i)));
end
xlabel("E_b/N_0 (dB)");
ylabel("有效吞吐率 (Mbps)");
title("DVB-S2 short rate 5/6 全 GPU 离线吞吐率");
legend("Location", "southoutside", "Orientation", "horizontal");
exportgraphics(f, outputs(2), "Resolution", 220);
close(f);

% USRP: fixed RF condition, code-rate reliability/goodput tradeoff.
labels = arrayfun(@codeLabel, U.code, "UniformOutput", false);
labels = categorical(string(labels), string(labels), "Ordinal", true);
f = figure("Color", "w", "Visible", visible, "Name", "USRP code comparison");
f.Position(3:4) = [1250, 500];
tiledlayout(1, 2, "Padding", "compact", "TileSpacing", "compact");
nexttile;
bar(labels, U.goodput_mbps);
grid on; ylabel("有效吞吐率 (Mbps)");
title("USRP QPSK 有效吞吐率");
nexttile;
semilogy(labels, U.FER, "-o", "LineWidth", 1.3);
yline(1e-3, "--k", "目标 FER=10^{-3}");
grid on; ylabel("FER"); ylim([1e-3, 1e-2]);
title("USRP QPSK 误帧率");
exportgraphics(f, outputs(3), "Resolution", 220);
close(f);

% Runtime profile: derive real-time ratio and parse decode capacity.
n = height(U);
realtimePct = zeros(n, 1);
decodeCapFps = zeros(n, 1);
for i = 1:n
    raw = fileread(U.log(i));
    nominal = regexp(raw, 'nominal: frameSamples=\d+ txFrameRate=([0-9.]+)', ...
        'tokens', 'once');
    capacity = regexp(raw, 'decode_consume:.*?cap_fps=([0-9.]+)', ...
        'tokens', 'once');
    realtimePct(i) = 100 * U.fps(i) / str2double(nominal{1});
    decodeCapFps(i) = str2double(capacity{1});
end
f = figure("Color", "w", "Visible", visible, "Name", "USRP runtime profile");
f.Position(3:4) = [1250, 500];
tiledlayout(1, 2, "Padding", "compact", "TileSpacing", "compact");
nexttile;
bar(labels, realtimePct);
grid on; ylabel("实际 fps / 名义 fps (%)"); ylim([0, 105]);
title("实时处理比例");
nexttile;
yyaxis left;
bar(labels, U.uhd_overflow_markers);
ylabel("UHD U 标记次数");
yyaxis right;
plot(labels, decodeCapFps, "-o", "LineWidth", 1.3);
ylabel("GPU decode 容量 (fps)");
grid on; title("采集标记与 GPU 消费余量");
exportgraphics(f, outputs(4), "Resolution", 220);
close(f);

disp("Generated measured-data figures:");
disp(outputs);
end

function name = codeLabel(code)
code = string(code);
if code == "DVB_S2_N64800_R12"
    name = "normal R1/2";
elseif code == "DVB_S2_short_N16200_rate_1_2"
    name = "short R1/2";
elseif code == "DVB_S2_short_N16200_rate_1_4"
    name = "short R1/4";
elseif code == "DVB_S2_short_N16200_rate_5_6"
    name = "short R5/6";
else
    name = code;
end
end

function mustExist(files)
for i = 1:numel(files)
    if ~isfile(files{i})
        error("Missing measured result file: %s", files{i});
    end
end
end
