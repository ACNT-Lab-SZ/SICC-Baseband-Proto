clear functions
cd(fileparts(mfilename('fullpath')))
stateLog = run_local_qianfan_demo();

% Do not start the channel-energy UI bridge here.
% run_gc_channel_ui_bridge.m publishes ntn_channel_subplots/channelPowerDb
% to UDP 65436, which drives the UI "Relative Total Channel Power" panel.
% Keep this disabled so start_matlab.m only launches the Qianfan demo and
% local performance plots.
% cd('<project-root>/source/python/Satellite_UI')
% run('run_gc_channel_ui_bridge.m')

% 1. 数据准备 (根据你的表格手动填入)
snr_db = [0, 0.5, 1, 1.5, 2, 2.5, 3];
ber = [1.09e-1, 6.33e-2, 2.98e-2, 1.08e-2, 2.90e-3, 6.17e-4, 9.84e-5];
fer = [4.66e-1, 2.78e-1, 1.34e-1, 4.97e-2, 1.38e-2, 3.02e-3, 5.06e-04];
throughput = [26.136, 27.643, 27.809, 28.207, 27.909, 27.824, 27.87];

% 2. 创建图形窗口
figure('Color', [1 1 1]);

%% 绘图 A: BER 与 FER (左侧 Y 轴)
subplot(1, 2, 1);
semilogy(snr_db, ber, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'BER');
hold on;
semilogy(snr_db, fer, 'rs--', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'FER (BLER)');
grid on;
xlabel('SNR (dB)');
ylabel('Error Rate');
title('Error Rate Performance');
legend('Location', 'southwest');
ylim([1e-5, 1]); % 设置 Y 轴范围以突出显示变化

%% 绘图 B: 吞吐量 (右侧 Y 轴)
subplot(1, 2, 2);
plot(snr_db, throughput, 'k^-', 'LineWidth', 1.5, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)');
ylabel('Throughput (Mbps)');
title('System Throughput');
% 由于你的吞吐量在 1.5dB 后趋于平稳，可以稍微调整下 Y 轴范围
ylim([min(throughput)-1, max(throughput)+1]); 

% 整体美化
set(gcf, 'Position', [100, 100, 1000, 400]); % 调整窗口大小
