function report = plot_integrated_gpu_usrp_results(logRoot, varargin)
%PLOT_INTEGRATED_GPU_USRP_RESULTS 绘制已验证的 GPU 物理层与 USRP 结果。
%
% report = plot_integrated_gpu_usrp_results()
% report = plot_integrated_gpu_usrp_results(logRoot, "SaveDir", outputDir)
%
% 数据使用策略：
%   - qpsk_noisevar_fixed_formal_v2 是 GPU demod/LLR 噪声方差修复后的
%     QPSK 可靠性扫描，保留其中 CCSDS LDPC 数据用于对照。
%   - qpsk_dvbs2_tuned_formal_v3 是经标准矩阵核对及 BP-NMS 参数验证后的
%     DVB-S2 正式数据（iter=50, alpha=0.95, offset=0, damping=0, schedule=2），
%     用于替换此前默认译码参数下不成立的 DVB-S2 可靠性数据。
%   - codec_matrix_sweep_20260522_162732 在运行时开启了 ThroughputOnly=True。
%     它跳过了确定性的参考误码率计算，仅统计校验成功的校验和（parity-ok codewords），
%     因此仅用于处理吞吐量的对比。
%   - usrp_fullgpu_modulation_600s 是 600 秒的硬件结果集。
%
% 本函数仅读取现有的日志文件。只有在提供了可选的 SaveDir 参数时，
% 才会将图表和 CSV 表格写入磁盘。
if nargin < 1 || strlength(string(logRoot)) == 0
    here = fileparts(mfilename("fullpath"));
    repoRoot = fileparts(fileparts(here));
    logRoot = fullfile(repoRoot, "build", "uhd_cpp_gpu_pipeline", "logs");
end
p = inputParser;
addParameter(p, "SaveDir", "", @(x) isstring(x) || ischar(x));
addParameter(p, "Visible", "on", @(x) any(strcmpi(string(x), ["on", "off"])));
parse(p, varargin{:});
saveDir = string(p.Results.SaveDir);
visible = char(string(p.Results.Visible));
paths.reliableBaseline = fullfile(logRoot, "qpsk_noisevar_fixed_formal_v2", "offline_metrics.csv");
paths.reliableDvbFormal = fullfile(logRoot, "qpsk_dvbs2_tuned_formal_v3", "offline_metrics.csv");
paths.throughputSweep = fullfile(logRoot, "codec_matrix_sweep_20260522_162732", "manifest.csv");
paths.usrp = fullfile(logRoot, "usrp_fullgpu_modulation_600s", "usrp_metrics.csv");
paths.pipeline = fullfile(logRoot, "pipeline_compare_20260523_5s", "usrp_metrics.csv");
paths.decoder = fullfile(logRoot, "full_gpu_offline_0_10db_20260521", "decoder_awgn_0_10_summary.csv");
paths.batch = fullfile(logRoot, "full_gpu_offline_0_10db_20260521", "decoder_batch_efficiency_7db_summary.csv");
paths.linkAwgn = fullfile(logRoot, "gpu_sim_cuda_inject_0_10db_20260521", "gpu_link_awgn_cuda_0_10_summary.csv");
paths.linkChannel = fullfile(logRoot, "gpu_sim_cuda_inject_0_10db_20260521", "gpu_link_channel_cuda_0_10_summary.csv");
mustExist(struct2cell(paths));
baseline = readOfflineMetrics(paths.reliableBaseline, true);
dvbFormal = readOfflineMetrics(paths.reliableDvbFormal, true);
baseline = baseline(~startsWith(baseline.Code, "DVB_S2") & ...
    baseline.Code ~= "systematic_ldpc_n256_k128_smoke", :);
report.qpskReliability = [baseline; dvbFormal];
report.throughputOnly = readSweepLogs(paths.throughputSweep, false);
report.reliabilityThreshold = buildThresholdTable(report.qpskReliability);
report.throughputAt16 = buildThroughputTable(report.throughputOnly, 16);
report.usrp = readUsrpMetrics(paths.usrp);
report.pipeline = readtable(paths.pipeline, "TextType", "string");
report.decoder = readtable(paths.decoder, "TextType", "string");
report.batch = readtable(paths.batch, "TextType", "string");
report.linkAwgn = readtable(paths.linkAwgn, "TextType", "string");
report.linkChannel = readtable(paths.linkChannel, "TextType", "string");
report.paths = paths;
fprintf("\n验证的 QPSK 可靠性门限（已启用参考 BER）：\n");
disp(report.reliabilityThreshold);
fprintf("\n在 Eb/N0 = 16 dB 下的离线处理吞吐量排名（仅吞吐量模式，不含 FER）：\n");
disp(report.throughputAt16(1:min(12, height(report.throughputAt16)), :));
fprintf("\nUSRP 600 秒全 GPU 调制测试：\n");
disp(report.usrp);
figs = gobjects(0);
figs(end+1) = plotReliabilitySweep(report.qpskReliability, visible);
figs(end+1) = plotReliabilityByActualSnr(report.qpskReliability, visible);
figs(end+1) = plotThroughputOnlySweep(report.throughputOnly, visible);
figs(end+1) = plotUsrpLongTest(report.usrp, visible);
figs(end+1) = plotCudaLinkComparison(report.linkAwgn, report.linkChannel, visible);
figs(end+1) = plotDecoderAndBatch(report.decoder, report.batch, visible);
report.figures = figs;
if strlength(saveDir) > 0
    if ~isfolder(saveDir)
        mkdir(saveDir);
    end
    names = ["qpsk_reliability_ebn0_sweep", "qpsk_reliability_actual_snr_sweep", ...
        "throughput_only_all_modulations", ...
        "usrp_600s_modulation", "cuda_link_awgn_channel", "decoder_batch"];
    for i = 1:numel(figs)
        exportgraphics(figs(i), fullfile(saveDir, names(i) + ".png"), "Resolution", 200);
    end
    writetable(report.reliabilityThreshold, fullfile(saveDir, "qpsk_reliability_threshold.csv"));
    writetable(report.throughputAt16, fullfile(saveDir, "throughput_only_ebn0_16_rank.csv"));
    writetable(report.usrp, fullfile(saveDir, "usrp_600s_metrics_derived.csv"));
    fprintf("已将图表和表格保存至: %s\n", saveDir);
end
end

function T = readOfflineMetrics(metricsPath, reliabilityValid)
M = readtable(metricsPath, "TextType", "string", "VariableNamingRule", "preserve");
M = M(M.modulation == "qpsk", :);
n = height(M);
T = table(M.code, M.N, M.K, M.rate, M.modulation, M.ebn0_db, ...
    M.injected_snr_db, M.decoder, repmat("OK", n, 1), M.frames, M.err, ...
    M.FER, M.BER, M.fps, M.goodput_mbps, M.fec_cap_info_mbps, ...
    M.sync_miss_total, repmat(logical(reliabilityValid), n, 1), ...
    'VariableNames', {'Code', 'N', 'K', 'Rate', 'Modulation', 'EbN0dB', ...
    'SimSNRdB', 'Decoder', 'Status', 'Frames', 'Err', 'FER', 'BER', ...
    'FPS', 'GoodputMbps', 'FECcapMbps', 'SyncMissTotal', 'ReliabilityValid'});
end

function T = readSweepLogs(manifestPath, reliabilityValid)
M = readtable(manifestPath, "TextType", "string", "VariableNamingRule", "preserve");
keep = M.family == "LDPC" & strlength(M.log) > 0;
M = M(keep, :);
n = height(M);
frames = zeros(n, 1);
err = zeros(n, 1);
fer = nan(n, 1);
ber = nan(n, 1);
fps = nan(n, 1);
goodput = nan(n, 1);
fecCap = nan(n, 1);
syncMiss = nan(n, 1);
for i = 1:n
    text = fileread(M.log(i));
    tokens = regexp(text, ...
        'frames=(\d+) ok=(\d+) err=(\d+) FER=([0-9.eE+\-]+) BER=([^ ]+).*?fps=([0-9.]+) goodput=([0-9.]+) Mbps', ...
        'tokens');
    if isempty(tokens)
        error("在 %s 中未找到接收（RX）摘要", M.log(i));
    end
    s = tokens{end};
    frames(i) = str2double(s{1});
    err(i) = str2double(s{3});
    fer(i) = str2double(s{4});
    ber(i) = str2double(s{5});
    fps(i) = str2double(s{6});
    goodput(i) = str2double(s{7});
    f = regexp(text, ...
        'decode\.fec_decode: ms/frame=([0-9.]+) cap_fps=([0-9.]+) cap_info=([0-9.]+) Mbps', ...
        'tokens');
    if ~isempty(f)
        fecCap(i) = str2double(f{end}{3});
    end
    sy = regexp(text, ...
        '\[PROFILE\] gpu_sync: found=(\d+) miss=(\d+) trackingMiss=(\d+) incomplete=(\d+)', ...
        'tokens');
    if ~isempty(sy)
        syncMiss(i) = str2double(sy{end}{2}) + str2double(sy{end}{3}) + str2double(sy{end}{4});
    end
end
T = table(M.code, M.N, M.K, M.rate, M.modulation, M.ebn0_db, M.sim_snr_db, ...
    M.decoder, M.status, frames, err, fer, ber, fps, goodput, fecCap, syncMiss, ...
    repmat(logical(reliabilityValid), n, 1), ...
    'VariableNames', {'Code', 'N', 'K', 'Rate', 'Modulation', 'EbN0dB', ...
    'SimSNRdB', 'Decoder', 'Status', 'Frames', 'Err', 'FER', 'BER', ...
    'FPS', 'GoodputMbps', 'FECcapMbps', 'SyncMissTotal', 'ReliabilityValid'});
T.N = str2double(string(T.N));
T.K = str2double(string(T.K));
T.Rate = str2double(string(T.Rate));
T.EbN0dB = str2double(string(T.EbN0dB));
T.SimSNRdB = str2double(string(T.SimSNRdB));
end

function T = buildThresholdTable(sweep)
codes = unique(sweep.Code, "stable");
out = cell(numel(codes), 9);
for i = 1:numel(codes)
    S = sortrows(sweep(sweep.Code == codes(i), :), "EbN0dB");
    hit = S(S.FER < 1e-3, :);
    zero = S(S.Err == 0, :);
    top = S(end, :);
    if isempty(hit)
        hitEbN0 = NaN; hitSnr = NaN; hitFer = NaN; hitGoodput = NaN;
    else
        hitEbN0 = hit.EbN0dB(1); hitSnr = hit.SimSNRdB(1);
        hitFer = hit.FER(1); hitGoodput = hit.GoodputMbps(1);
    end
    if isempty(zero)
        zeroEbN0 = NaN;
    else
        zeroEbN0 = zero.EbN0dB(1);
    end
    out(i, :) = {codes(i), top.N, top.K, top.Decoder, hitEbN0, hitSnr, ...
        hitFer, hitGoodput, zeroEbN0};
end
T = cell2table(out, 'VariableNames', {'Code', 'N', 'K', 'Decoder', ...
    'FirstFERlt1e3_EbN0dB', 'FirstFERlt1e3_SimSNRdB', ...
    'ThresholdFER', 'ThresholdGoodputMbps', 'FirstZeroError_EbN0dB'});
end

function T = buildThroughputTable(sweep, ebn0)
T = sweep(sweep.EbN0dB == ebn0, {'Code', 'N', 'K', 'Modulation', ...
    'Decoder', 'EbN0dB', 'SimSNRdB', 'GoodputMbps', 'FECcapMbps'});
T = sortrows(T, "GoodputMbps", "descend");
end

function T = readUsrpMetrics(path)
M = readtable(path, "TextType", "string", "VariableNamingRule", "preserve");
n = height(M);
txFrames = nan(n, 1);
for i = 1:n
    text = fileread(M.log(i));
    token = regexp(text, 'tx: frames=(\d+)', 'tokens');
    if ~isempty(token)
        txFrames(i) = str2double(token{end}{1});
    end
end
frames = str2double(string(M.frames));
ok = str2double(string(M.ok));
goodput = str2double(string(M.goodput_mbps));
T = table(M.modulation, txFrames, frames, ok, str2double(string(M.FER)), ...
    goodput, goodput / 25, 100 * (1 - frames ./ txFrames), ...
    100 * (1 - ok ./ txFrames), str2double(string(M.uhd_overflow_markers)), ...
    'VariableNames', {'Modulation', 'TxFrames', 'DecodedFrames', 'OkFrames', ...
    'DecodeFER', 'GoodputMbps', 'SpectralEfficiency_bpsHz', ...
    'CaptureLossPct', 'EndToEndLossPct', 'UhdOverflowMarkers'});
end

function f = plotReliabilitySweep(T, visible)
f = figure("Name", "QPSK 验证的可靠性扫描", "Color", "w", "Visible", visible);
f.Position(3:4) = [1320, 520];
layout = tiledlayout(1, 2, "Padding", "compact", "TileSpacing", "compact");
T = T(T.Code ~= "systematic_ldpc_n256_k128_smoke", :);
codes = unique(T.Code, "stable");
colors = lines(numel(codes));
labels = strings(numel(codes), 1);
h = gobjects(numel(codes), 1);
axFer = nexttile(layout, 1); hold(axFer, "on"); grid(axFer, "on");
for i = 1:numel(codes)
    S = sortrows(T(T.Code == codes(i), :), "EbN0dB");
    y = S.FER;
    y(S.Err == 0) = 1 ./ S.Frames(S.Err == 0);
    labels(i) = codeLabel(codes(i), S.N(1), S.K(1));
    h(i) = semilogy(axFer, S.EbN0dB, y, "-o", "Color", colors(i, :), ...
        "LineWidth", 1.1, "DisplayName", labels(i));
end
yline(axFer, 1e-3, "--k", "FER = 10^{-3}", "LabelHorizontalAlignment", "left", ...
    "HandleVisibility", "off");
set(axFer, "YScale", "log");
ylim(axFer, [1e-5, 1]);
xlabel(axFer, "E_b/N_0 (dB)"); ylabel(axFer, "物理层帧误帧率 (FER, 零错误显示为 1/N)");
title(axFer, "QPSK 可靠性（已启用参考 BER）");
axGoodput = nexttile(layout, 2); hold(axGoodput, "on"); grid(axGoodput, "on");
for i = 1:numel(codes)
    S = sortrows(T(T.Code == codes(i), :), "EbN0dB");
    plot(axGoodput, S.EbN0dB, S.GoodputMbps, "-o", "Color", colors(i, :), ...
        "LineWidth", 1.1, "HandleVisibility", "off");
end
xlabel(axGoodput, "E_b/N_0 (dB)"); ylabel(axGoodput, "吞吐量 (Mbps)");
title(axGoodput, "QPSK 吞吐量");
lgd = legend(axFer, h, labels, "Orientation", "horizontal", "NumColumns", 4, ...
    "Interpreter", "none");
lgd.Layout.Tile = "south";
end

function f = plotReliabilityByActualSnr(T, visible)
f = figure("Name", "QPSK 实际注入信噪比扫描", "Color", "w", "Visible", visible);
f.Position(3:4) = [1320, 520];
layout = tiledlayout(1, 2, "Padding", "compact", "TileSpacing", "compact");
codes = unique(T.Code, "stable");
colors = lines(numel(codes));
labels = strings(numel(codes), 1);
h = gobjects(numel(codes), 1);
axFer = nexttile(layout, 1); hold(axFer, "on"); grid(axFer, "on");
for i = 1:numel(codes)
    S = sortrows(T(T.Code == codes(i), :), "SimSNRdB");
    y = S.FER;
    y(S.Err == 0) = 1 ./ S.Frames(S.Err == 0);
    labels(i) = codeLabel(codes(i), S.N(1), S.K(1));
    h(i) = semilogy(axFer, S.SimSNRdB, y, "-o", "Color", colors(i, :), ...
        "LineWidth", 1.1, "DisplayName", labels(i));
end
yline(axFer, 1e-3, "--k", "FER = 10^{-3}", "LabelHorizontalAlignment", "left", ...
    "HandleVisibility", "off");
set(axFer, "YScale", "log");
ylim(axFer, [1e-5, 1]);
xlabel(axFer, "实际注入 E_s/N_0 (dB)");
ylabel(axFer, "物理层帧误帧率 (FER, 零错误显示为 1/N)");
title(axFer, "QPSK 实际信道门限");
axGoodput = nexttile(layout, 2); hold(axGoodput, "on"); grid(axGoodput, "on");
for i = 1:numel(codes)
    S = sortrows(T(T.Code == codes(i), :), "SimSNRdB");
    plot(axGoodput, S.SimSNRdB, S.GoodputMbps, "-o", "Color", colors(i, :), ...
        "LineWidth", 1.1, "HandleVisibility", "off");
end
xlabel(axGoodput, "实际注入 E_s/N_0 (dB)"); ylabel(axGoodput, "吞吐量 (Mbps)");
title(axGoodput, "实际信道吞吐量");
lgd = legend(axFer, h, labels, "Orientation", "horizontal", "NumColumns", 4, ...
    "Interpreter", "none");
lgd.Layout.Tile = "south";
end

function f = plotThroughputOnlySweep(T, visible)
f = figure("Name", "纯吞吐量调制扫描", "Color", "w", "Visible", visible);
target = T(T.Code == "DVB_S2_short_N16200_rate_5_6", :);
mods = ["bpsk", "qpsk", "16qam", "64qam"];
hold on; grid on;
for i = 1:numel(mods)
    S = sortrows(target(target.Modulation == mods(i), :), "EbN0dB");
    plot(S.EbN0dB, S.GoodputMbps, "-o", "LineWidth", 1.2, "DisplayName", upper(mods(i)));
end
xlabel("E_b/N_0 (dB)"); ylabel("处理吞吐量 (Mbps)");
title({"DVB-S2 短码 16200/13320 全调制吞吐量", ...
    "ThroughputOnly=True: 此处输出的 FER 不作为可靠性指标"});
legend("Location", "northwest");
end

function f = plotUsrpLongTest(T, visible)
f = figure("Name", "USRP 600 秒调制测试结果", "Color", "w", "Visible", visible);
tiledlayout(1, 2, "Padding", "compact", "TileSpacing", "compact");
nexttile;
modOrder = ["bpsk", "qpsk", "16qam", "64qam"];
x = categorical(string(T.Modulation), modOrder, 'Ordinal', true);
Y = [ ...
    T.GoodputMbps, ...
    T.GoodputMbps .* (1 - T.EndToEndLossPct / 100) ...
];
bar(x, Y);
grid on; ylabel("Mbps"); title("USRP 600 秒 吞吐量");
legend("记录的吞吐量", "根据发送/成功接收调整后的吞吐量", "Location", "northwest");
nexttile;
yyaxis left
semilogy(x, T.DecodeFER, "-o", "LineWidth", 1.2);
ylabel("译码误帧率 (FER)"); grid on; ylim([1e-3 1]);
yyaxis right
plot(x, T.EndToEndLossPct, "-s", "LineWidth", 1.2);
ylabel("端到端丢失率 (%)");
title("可靠性随调制阶数的变化");
end

function f = plotCudaLinkComparison(A, C, visible)
f = figure("Name", "CUDA 链路信道对比", "Color", "w", "Visible", visible);
tiledlayout(1, 2, "Padding", "compact", "TileSpacing", "compact");
snrA = str2double(string(A.snr));
snrC = str2double(string(C.snr));
ferA = str2double(string(A.FER));
ferC = str2double(string(C.FER));
nexttile; semilogy(snrA, max(ferA, 1e-6), "-o", snrC, max(ferC, 1e-6), "-s");
grid on; xlabel("配置的信噪比 SNR (dB)"); ylabel("误帧率 (FER)"); yline(1e-3, "--k");
title("完整 GPU 链路可靠性"); legend("AWGN 高斯白噪声", "三径失配信道", "目标门限");
nexttile; plot(snrA, str2double(string(A.goodput)), "-o", ...
    snrC, str2double(string(C.goodput)), "-s");
grid on; xlabel("配置的信噪比 SNR (dB)"); ylabel("吞吐量 (Mbps)");
title("完整 GPU 链路吞吐量"); legend("AWGN 高斯白噪声", "三径失配信道");
end

function f = plotDecoderAndBatch(D, B, visible)
f = figure("Name", "译码器与批处理效率", "Color", "w", "Visible", visible);
tiledlayout(1, 2, "Padding", "compact", "TileSpacing", "compact");
nexttile;
snr = str2double(string(D.snr_db));
semilogy(snr, max(str2double(string(D.FER)), 1e-7), "-o", ...
    snr, max(str2double(string(D.BER)), 1e-8), "-s");
grid on; xlabel("信噪比 SNR (dB)"); ylabel("误码/误帧率");
title("纯 CUDA BP-OSD 性能"); legend("误帧率 (FER)", "误码率 (BER)");
nexttile;
batch = str2double(string(B.batch));
yyaxis left
semilogx(batch, str2double(string(B.info_Mbps)), "-o", "LineWidth", 1.2);
ylabel("信息吞吐量 (Mbps)"); grid on;
yyaxis right
semilogx(batch, str2double(string(B.latency_us_per_codeword)), "-s", "LineWidth", 1.2);
ylabel("时延 (微秒/码字)");
xlabel("批处理码字数量"); title("在 7 dB 下的批处理效率");
end

function mustExist(files)
for i = 1:numel(files)
    if ~isfile(files{i})
        error("未找到所需的测试结果文件: %s", files{i});
    end
end
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
