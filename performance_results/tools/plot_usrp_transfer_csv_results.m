function plot_usrp_transfer_csv_results()
% Plot line charts for decoder_awgn and rx_offline_or_usrp_link CSV results.

rootDir = fileparts(fileparts(mfilename('fullpath')));
tablesDir = fullfile(rootDir, 'tables');
figuresDir = fullfile(rootDir, 'figures');
if ~exist(figuresDir, 'dir')
    mkdir(figuresDir);
end

plotOfflineTraceLines(tablesDir, figuresDir);
plotOfflineRunSummary(tablesDir, figuresDir);
plotUsrplinkAndDecoder(tablesDir, figuresDir);
end

function plotOfflineTraceLines(tablesDir, figuresDir)
longFile = fullfile(tablesDir, 'usrp_offline_trace_long.csv');
if ~isfile(longFile)
    return;
end
T = readtable(longFile, 'TextType', 'string', 'VariableNamingRule', 'preserve');
if isempty(T)
    return;
end

runs = unique(T.run_index);
colors = turbo(numel(runs));

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1200 620]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
hold on;
for i = 1:numel(runs)
    idx = T.run_index == runs(i);
    plot(T.frame(idx), T.snr_db(idx), 'Color', [colors(i,:) 0.28], 'LineWidth', 0.8);
end
meanTrace = groupsummary(T, 'frame', 'mean', 'snr_db');
plot(meanTrace.frame, meanTrace.mean_snr_db, 'k-', 'LineWidth', 2.2);
grid on;
xlabel('Frame');
ylabel('SNR (dB)');
title('Offline GPU simulated SNR traces');

nexttile;
hold on;
for i = 1:numel(runs)
    idx = T.run_index == runs(i);
    plot(T.frame(idx), T.channel_gain_db(idx), 'Color', [colors(i,:) 0.28], 'LineWidth', 0.8);
end
meanGain = groupsummary(T, 'frame', 'mean', 'channel_gain_db');
plot(meanGain.frame, meanGain.mean_channel_gain_db, 'k-', 'LineWidth', 2.2);
grid on;
xlabel('Frame');
ylabel('Relative channel gain (dB)');
title('Offline GPU simulated channel gain traces');

exportgraphics(fig, fullfile(figuresDir, 'usrp_offline_snr_channel_trace_lines.png'), 'Resolution', 220);
close(fig);
end

function plotOfflineRunSummary(tablesDir, figuresDir)
summaryFile = fullfile(tablesDir, 'usrp_offline_trace_summary.csv');
if ~isfile(summaryFile)
    return;
end
T = readtable(summaryFile, 'TextType', 'string', 'VariableNamingRule', 'preserve');
if isempty(T)
    return;
end

sim = T(T.mode == "offline_gpu_simulated_channel", :);
pred = T(T.mode == "offline_gpu_predicted_power", :);

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1200 760]);
tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
hold on;
plot(sim.run_index, sim.snr_mean_db, 'o-', 'LineWidth', 1.4, 'DisplayName', 'Mean SNR');
plot(sim.run_index, sim.snr_min_db, 'v--', 'LineWidth', 1.0, 'DisplayName', 'Min SNR');
plot(sim.run_index, sim.snr_max_db, '^--', 'LineWidth', 1.0, 'DisplayName', 'Max SNR');
plot(sim.run_index, sim.nominal_snr_db, 'k-', 'LineWidth', 1.8, 'DisplayName', 'Nominal SNR');
grid on;
xlabel('Run index');
ylabel('SNR (dB)');
title('Offline run-level SNR configuration');
legend('Location', 'best');

nexttile;
hold on;
plot(sim.run_index, sim.channel_gain_mean_db, 'o-', 'LineWidth', 1.4, 'DisplayName', 'Mean channel gain');
plot(sim.run_index, sim.channel_gain_min_db, 'v--', 'LineWidth', 1.0, 'DisplayName', 'Min channel gain');
plot(sim.run_index, sim.channel_gain_max_db, '^--', 'LineWidth', 1.0, 'DisplayName', 'Max channel gain');
if ~isempty(pred)
    plot(pred.run_index, pred.channel_gain_mean_db, 's', 'MarkerSize', 7, 'LineWidth', 1.3, 'DisplayName', 'Predicted power only');
end
grid on;
xlabel('Run index');
ylabel('Relative gain / power (dB)');
title('Offline run-level channel gain configuration');
legend('Location', 'best');

exportgraphics(fig, fullfile(figuresDir, 'usrp_offline_run_summary_lines.png'), 'Resolution', 220);
close(fig);
end

function plotUsrplinkAndDecoder(tablesDir, figuresDir)
usrpFile = fullfile(tablesDir, 'usrp_radio_configuration_performance.csv');
decoderFile = fullfile(tablesDir, 'decoder_awgn_configuration_performance.csv');

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1200 600]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
if isfile(usrpFile)
    U = readtable(usrpFile, 'TextType', 'string', 'VariableNamingRule', 'preserve');
    x = 1:5;
    y = [U.goodput_mbps, U.fps, U.decode_ms_per_frame, U.fer * 1000, U.err / max(U.frames, 1) * 1000];
    plot(x, y, 'o-', 'LineWidth', 1.8, 'MarkerSize', 8);
    set(gca, 'XTick', x, 'XTickLabel', {'Goodput', 'FPS', 'Decode ms', 'FER x1000', 'ERR/frame x1000'});
    grid on;
    ylabel('Metric value');
    title('Real USRP gated GPU link performance');
else
    axis off;
    text(0.1, 0.5, 'No USRP CSV found');
end

nexttile;
if isfile(decoderFile)
    D = readtable(decoderFile, 'TextType', 'string', 'VariableNamingRule', 'preserve');
    yyaxis left;
    semilogy(D.snr_db, D.ber, 'o-', 'LineWidth', 1.8, 'MarkerSize', 8);
    ylabel('BER');
    yyaxis right;
    semilogy(D.snr_db, D.fer, 's-', 'LineWidth', 1.8, 'MarkerSize', 8);
    ylabel('FER');
    grid on;
    xlabel('SNR (dB)');
    title('Decoder-only AWGN performance');
else
    axis off;
    text(0.1, 0.5, 'No decoder AWGN CSV found');
end

exportgraphics(fig, fullfile(figuresDir, 'usrp_link_decoder_line_performance.png'), 'Resolution', 220);
close(fig);
end
