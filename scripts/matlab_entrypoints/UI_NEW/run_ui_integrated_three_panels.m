%% run_ui_integrated_three_panels.m
% 一键集成启动脚本：一个 MATLAB 文件启动 UI 三个图的数据后端。
%
% 功能：
%   1) 可选启动 Python UI main.py；
%   2) 启动信道后端 gc_vis_gen_data.m：
%        - 左图：3D Delay-Doppler Signal Intensity
%        - 中图：Relative Total Channel Power / Noisy CSI Buffer
%      该后端通过 UDP 65436 发送 payload.type = 'ntn_channel_subplots'。
%   3) 启动 t-SNE 后端 plot_t_sne_ui_publish_modified.m：
%        - 右图：Normal 20 dB t-SNE
%      该后端通过 UDP 65436 发送 payload.type = 'ntn_tsne_frame'。
%
% 使用方式：
%   在 clone 后的整理目录中运行本文件，例如：
%       scripts/matlab_entrypoints/UI_NEW/run_ui_integrated_three_panels.m
%   然后在 MATLAB 命令行运行：
%       run('run_ui_integrated_three_panels.m')
%
% 前提：
%   - main.py 已经是支持 ntn_channel_subplots + ntn_tsne_frame 路由的版本；
%   - gc_vis_gen_data.m 是你当前用于左图/中图的信道展示脚本；
%   - plot_t_sne_ui_publish_modified.m 是你当前用于右图 t-SNE 的脚本；
%   - predict_results 文件夹位于本目录下，或自行修改 cfg.PredictDir。
%
% 说明：
%   这个文件是“集成启动器”，不是把两个超长脚本硬拼成一个 3000 行文件。
%   这样做更稳：两个重后端分别在独立 MATLAB 进程里运行，不会互相阻塞。
%   UI 端靠 payload.type 区分两类数据：
%       'ntn_channel_subplots' -> 左图 + 中图
%       'ntn_tsne_frame'       -> 右图

clc;

%% ===================== 0. 用户配置 =====================
cfg = struct();

% 当前工程目录。默认取本脚本所在目录。
cfg.ProjectDir = fileparts(mfilename('fullpath'));
if isempty(cfg.ProjectDir)
    cfg.ProjectDir = pwd;
end

% 是否自动启动 Python UI。
% 如果你已经手动打开 UI，可以设为 false。
cfg.StartPythonUI = false;

% Python UI 入口。
cfg.PythonExe = fullfile(cfg.ProjectDir, '.venv', 'Scripts', 'python.exe');
cfg.MainPy = fullfile(cfg.ProjectDir, 'main.py');

% 两个后端脚本。
% 说明：t-SNE 脚本文件名如果你本地叫 plot_t_sne_ui_prediction_gate_legend.m，
% 这里会优先使用它；如果不存在，再兼容旧文件名 plot_t_sne_ui_publish_modified.m。
cfg.ChannelScript = fullfile(cfg.ProjectDir, 'gc_vis_gen_data.m');
cfg.TsneScriptCandidates = { ...
    fullfile(cfg.ProjectDir, 'plot_t_sne_ui_prediction_gate_legend.m'), ...
    fullfile(cfg.ProjectDir, 'plot_t_sne_ui_prediction_gate_legend_fixed.m'), ...
    fullfile(cfg.ProjectDir, 'plot_t_sne_ui_publish_modified.m') ...
};
cfg.TsneScript = pick_existing_file(cfg.TsneScriptCandidates, 'TsneScript');

% 预测结果目录。
cfg.PredictDir = fullfile(cfg.ProjectDir, 'predict_results');

% runtime 控制文件目录。
cfg.RuntimeDir = fullfile(cfg.ProjectDir, 'runtime');
cfg.PredictionControlFile = fullfile(cfg.RuntimeDir, 'channel_prediction_control.json');
cfg.AnomalyControlFile = fullfile(cfg.RuntimeDir, 'channel_anomaly_control.json');

% t-SNE 是否等待 UI 中“信道状态信息预测=启用”。
% 你的 plot_t_sne_ui_publish_modified.m 如果已经包含 WaitForPredictionEnable=true，
% 这里就保持 true；否则它仍可能直接开始推送。
cfg.InitPredictionEnabled = false;

% 异常注入初始状态。
cfg.InitAnomalyEnabled = false;
cfg.InitAnomalyAttenuationDb = 12;

% 启动间隔，避免 UI / UDP 端口初始化拥挤。
cfg.DelayAfterUIStartSec = 3.0;
cfg.DelayBetweenBackendsSec = 1.0;

% 是否新开独立 MATLAB 窗口运行后端。
% Windows 下建议 true；这样主 MATLAB 不会被阻塞。
cfg.DetachedBackend = true;

%% ===================== 1. 基础检查 =====================
fprintf('\n============================================================\n');
fprintf('[INTEGRATED-DEMO] ProjectDir:\n  %s\n', cfg.ProjectDir);
fprintf('============================================================\n');

assert_file_exists(cfg.ChannelScript, 'ChannelScript');
assert_file_exists(cfg.TsneScript, 'TsneScript');

if ~exist(cfg.RuntimeDir, 'dir')
    mkdir(cfg.RuntimeDir);
end

if ~exist(cfg.PredictDir, 'dir')
    warning('[INTEGRATED-DEMO] predict_results not found:\n  %s\n右侧 t-SNE 可能无法启动。', cfg.PredictDir);
end

%% ===================== 2. 初始化 UI 控制文件 =====================
write_json_text(cfg.PredictionControlFile, struct( ...
    'enabled', logical(cfg.InitPredictionEnabled), ...
    'updatedAt', now, ...
    'source', 'run_ui_integrated_three_panels'));

write_json_text(cfg.AnomalyControlFile, struct( ...
    'enabled', logical(cfg.InitAnomalyEnabled), ...
    'attenuationDb', cfg.InitAnomalyAttenuationDb, ...
    'updatedAt', now, ...
    'source', 'run_ui_integrated_three_panels'));

fprintf('[INTEGRATED-DEMO] Control files initialized:\n');
fprintf('  Prediction: %s | enabled=%d\n', cfg.PredictionControlFile, cfg.InitPredictionEnabled);
fprintf('  Anomaly   : %s | enabled=%d | attenuation=%.1f dB\n', ...
    cfg.AnomalyControlFile, cfg.InitAnomalyEnabled, cfg.InitAnomalyAttenuationDb);

%% ===================== 3. 可选启动 Python UI =====================
if cfg.StartPythonUI
    assert_file_exists(cfg.PythonExe, 'PythonExe');
    assert_file_exists(cfg.MainPy, 'MainPy');

    fprintf('\n[INTEGRATED-DEMO] Starting Python UI...\n');
    start_python_ui(cfg);

    fprintf('[INTEGRATED-DEMO] Waiting %.1f s for UI initialization...\n', cfg.DelayAfterUIStartSec);
    pause(cfg.DelayAfterUIStartSec);
else
    fprintf('\n[INTEGRATED-DEMO] Python UI auto-start is disabled.\n');
    fprintf('请确认 UI 已经启动，并且 main.py 正在监听 UDP 65436。\n');
end

%% ===================== 4. 启动两个 MATLAB 后端 =====================
fprintf('\n[INTEGRATED-DEMO] Starting channel backend...\n');
channelBatch = sprintf( ...
    "cd('%s'); run('%s');", ...
    escape_matlab_string(cfg.ProjectDir), ...
    escape_matlab_string(cfg.ChannelScript));
start_matlab_backend(channelBatch, 'GC_Channel_Backend', cfg.DetachedBackend);

pause(cfg.DelayBetweenBackendsSec);

fprintf('\n[INTEGRATED-DEMO] Starting t-SNE backend...\n');
tsneBatch = sprintf( ...
    "cd('%s'); setenv('PREDICT_DIR','%s'); run('%s');", ...
    escape_matlab_string(cfg.ProjectDir), ...
    escape_matlab_string(cfg.PredictDir), ...
    escape_matlab_string(cfg.TsneScript));
start_matlab_backend(tsneBatch, 'GC_TSNE_Backend', cfg.DetachedBackend);

fprintf('\n============================================================\n');
fprintf('[INTEGRATED-DEMO] 后端已启动。\n');
fprintf('左图/中图：由 gc_vis_gen_data.m 推送 ntn_channel_subplots。\n');
fprintf('右图 t-SNE：由 plot_t_sne_ui_publish_modified.m 推送 ntn_tsne_frame。\n');
fprintf('\n如果右图显示 Waiting for 20 dB t-SNE data：\n');
fprintf('  1) 在 UI 左侧把“信道状态信息预测”切换为“启用”；\n');
fprintf('  2) 确认 predict_results 下有 predictions_*.mat；\n');
fprintf('  3) 查看 GC_TSNE_Backend 窗口是否仍在等待控制文件。\n');
fprintf('\n停止方式：关闭两个后端 MATLAB 窗口，或在任务管理器结束 MATLAB 进程。\n');
fprintf('============================================================\n\n');

%% ===================== Local functions =====================

function pathName = pick_existing_file(candidates, label)
    for ii = 1:numel(candidates)
        if exist(candidates{ii}, 'file') == 2
            pathName = candidates{ii};
            fprintf('[INTEGRATED-DEMO] %s selected:\n  %s\n', label, pathName);
            return;
        end
    end

    msg = sprintf('[INTEGRATED-DEMO] %s not found. Tried:', label);
    for ii = 1:numel(candidates)
        msg = sprintf('%s\n  %s', msg, candidates{ii});
    end
    error('%s', msg);
end

function assert_file_exists(pathName, label)
    if exist(pathName, 'file') ~= 2
        error('[INTEGRATED-DEMO] %s not found:\n  %s', label, pathName);
    end
end

function write_json_text(filePath, s)
    txt = jsonencode(s);
    fid = fopen(filePath, 'w');
    if fid < 0
        error('Cannot write control file:\n  %s', filePath);
    end
    cleaner = onCleanup(@() fclose(fid));
    fwrite(fid, txt, 'char');
end

function s = escape_matlab_string(s)
    % Escape single quotes for MATLAB -batch command strings.
    s = char(s);
    s = strrep(s, '''', '''''');
end

function matlabExe = get_matlab_executable()
    if ispc
        matlabExe = fullfile(matlabroot, 'bin', 'matlab.exe');
    else
        matlabExe = fullfile(matlabroot, 'bin', 'matlab');
    end

    if exist(matlabExe, 'file') ~= 2
        matlabExe = 'matlab';
    end
end

function start_python_ui(cfg)
    if ispc
        cmd = sprintf('start "Satellite_UI" /D "%s" "%s" "%s"', ...
            cfg.ProjectDir, cfg.PythonExe, cfg.MainPy);
    else
        cmd = sprintf('cd "%s" && "%s" "%s" &', ...
            cfg.ProjectDir, cfg.PythonExe, cfg.MainPy);
    end

    fprintf('  %s\n', cmd);
    [status, msg] = system(cmd);
    if status ~= 0
        warning('[INTEGRATED-DEMO] Failed to start Python UI:\n%s', msg);
    end
end

function start_matlab_backend(batchCmd, windowTitle, detached)
    matlabExe = get_matlab_executable();

    if detached
        if ispc
            % start "Title" "matlab.exe" -nosplash -nodesktop -minimize -batch "..."
            cmd = sprintf('start "%s" "%s" -nosplash -nodesktop -minimize -batch "%s"', ...
                windowTitle, matlabExe, batchCmd);
        else
            cmd = sprintf('"%s" -nosplash -nodesktop -batch "%s" &', ...
                matlabExe, batchCmd);
        end

        fprintf('  %s\n', cmd);
        [status, msg] = system(cmd);
        if status ~= 0
            warning('[INTEGRATED-DEMO] Failed to start backend %s:\n%s', windowTitle, msg);
        end
    else
        % 非 detached 模式会阻塞当前 MATLAB，只适合调试。
        fprintf('[INTEGRATED-DEMO] Running backend in current MATLAB: %s\n', windowTitle);
        eval(batchCmd);
    end
end
