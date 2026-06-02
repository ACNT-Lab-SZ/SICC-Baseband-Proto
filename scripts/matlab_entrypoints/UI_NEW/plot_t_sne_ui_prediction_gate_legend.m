%% plot_tsne_csi_features_fixed50_four_snr_v13_interactive_condition_sample_batches.m
% Standalone script: 2 x 2 t-SNE visualization for 5/10/15/20 dB SNR.
%
% 修改重点：
%   1) 只画 SNR = 5, 10, 15, 20 dB 四种情况；
%   2) 四个 SNR 子图放在同一张图中，布局为 2 x 2；
%   3) 只采用一种合适的 t-SNE 输入特征：
%        cfg.representationMode = 'window_flatten'
%        cfg.featureScope        = 'h_only'
%      即将整个预测窗口内的 [Re(h), Im(h)] 展平成一个向量。
%      若 Tout = 24, taps = 3，则输入特征维度为 24 x 2 x 3 = 144。
%
% 说明：
%   - t-SNE 横纵坐标没有直接物理意义，只表示二维嵌入坐标。
%   - 每个子图中，Ground-truth 与 Predicted 点云越重合，说明预测 CSI
%     feature 的分布越接近真实 CSI feature。
%   - t-SNE 只是辅助可视化，不能替代 NMSE / H_eff NMSE / EVM / SER。
%
% 使用：
%   将本脚本放在 NMSE 主脚本同级目录，或保证 predict_results 在当前脚本目录下。
%   如果预测结果在其他目录，可先运行：
%       setenv('PREDICT_DIR', 'your/path/to/predict_results')
%   然后运行：
%       plot_tsne_csi_features_fixed50_four_snr_v4

clc; clear; close all;

%% 1. Paths
SCRIPT_DIR = fileparts(mfilename('fullpath'));
if isempty(SCRIPT_DIR), SCRIPT_DIR = pwd; end
PROJECT_DIR = fileparts(SCRIPT_DIR);

predictDirFromEnv = getenv('PREDICT_DIR');
if ~isempty(predictDirFromEnv)
    PREDICT_DIR = predictDirFromEnv;
else
    PREDICT_DIR = fullfile(SCRIPT_DIR, 'predict_results');
end

% 输出目录：直接保存到当前 MATLAB 脚本所在文件夹。
OUTPUT_DIR = SCRIPT_DIR;

ORIGINAL_DIR = pwd;
cleanupObj = onCleanup(@() cd(ORIGINAL_DIR));
cd(SCRIPT_DIR);

addpath(SCRIPT_DIR, '-begin');
addpath(PROJECT_DIR, '-end');

fprintf('SCRIPT_DIR : %s\n', SCRIPT_DIR);
fprintf('PREDICT_DIR: %s\n', PREDICT_DIR);
fprintf('OUTPUT_DIR : %s\n', OUTPUT_DIR);

if exist('tsne', 'file') ~= 2
    error(['Cannot find MATLAB function tsne(). ', ...
           'This script requires Statistics and Machine Learning Toolbox. ', ...
           'If unavailable, use Python sklearn.manifold.TSNE instead.']);
end

%% 2. Config
cfg = struct();

% 与 NMSE 代码一致的基础设置。
cfg.N = 16;
cfg.M = 16;
cfg.default_snr_db = 15;
cfg.default_valid_feature_dim = 12;
cfg.taps_per_state = 3;

% Doppler 恢复，与 NMSE 评估脚本保持一致。
cfg.dopplerFeatureScale = 10;
cfg.dopplerScalePolicy = 'auto';       % 'auto' / 'force_unscale' / 'none'

% SNR 分组，与 NMSE 评估脚本保持一致。
cfg.requireSampleSNRLabel = true;
cfg.allowLegacyBlockSplit = false;
cfg.snrLabelTolerance = 1e-6;

% Prediction files.
cfg.predictionPattern = 'predictions_*_fixed50_ntn_tdl*.mat';

% 命令行交互选择信道类型。
% true : 运行脚本后在命令行输入编号/名称选择要绘制的信道类型；
% false: 使用 cfg.conditionsToPlot 的固定设置。
cfg.enableInteractiveConditionSelection = false;

% 非交互模式下使用；交互模式下会被命令行选择结果覆盖。
% 可选：'normal', 'step_loss', 'burst_shadow', 'firsttap_block', 'doppler_jump', 'mixed'
% 例如：cfg.conditionsToPlot = {'doppler_jump'};
% 空 cell {} 表示绘制全部。
cfg.conditionsToPlot = {'normal'};

% 只画 5/10/15/20 dB 四个 SNR。
cfg.snrToPlot = 20;
cfg.dynamicSnrToPlot = 20;      % UI mode: only stream/display the 20 dB t-SNE panel.

% Output control.
cfg.enableStaticTSNE = false;             % Original 2 x 2 static t-SNE figure.
cfg.enableDynamicTSNE = true;             % Frame-by-frame t-SNE video/GIF.

% 只采用一种表示，避免一张图里同时混入 time_step 和 window_flatten。
% 推荐：window_flatten + h_only
%   - window_flatten：保留整个预测窗口的时间演化信息；
%   - h_only：只使用复信道增益 Re/Im，避免 delay/Doppler 尺度或静态项主导 t-SNE。
cfg.representationMode = 'window_flatten';  % 'time_step' / 'window_flatten'
cfg.dynamicFeatureMode = 'time_step';        % Dynamic t-SNE always uses one prediction frame at a time.
% Dynamic t-SNE visual fix.
% single_frame  : 原始方案，只看当前时刻，若该时刻样本方差很小会自然聚成一点；
% sliding_window: 推荐方案，每一帧使用当前时刻附近的局部时间窗特征，避免单时刻 6 维特征退化；
% prefix        : 展示截至当前帧的累计预测窗口特征，视觉最稳定，但物理含义是“累计到第 t 帧”。
cfg.dynamicTemporalMode = 'sliding_window';  % 'single_frame' / 'sliding_window' / 'prefix'
cfg.dynamicHalfWindow = 2;                   % sliding_window 下使用 t-2:t+2，共最多 5 个时刻
cfg.dynamicTimeRange = [];                   % 电赛展示：[] 表示 1:Tout，24 个预测帧全部展示
cfg.dynamicAxisMode = 'current';             % 'current' / 'cumulative'
cfg.uiFixedAxisLim = [-32 32; -32 32];       % Fixed UI axes keep the live display from jumping between frames.
cfg.uiAxisPercentile = [2 98];               % UI uses robust limits so one t-SNE outlier does not flatten the cloud.
cfg.uiAxisPadding = 0.12;                    % Slightly wider than MATLAB's default padding for the smaller UI panel.
cfg.dynamicPlotContent = 'true_pred';         % 两种颜色同时展示：'true_pred' / 'prediction_only'
cfg.degenerateVarThreshold = 1e-10;          % 用于提示某帧输入特征是否近似常数
cfg.addTinyJitterForDegenerate = false;      % 仅调试可设 true；论文图不建议人为加扰动
cfg.tinyJitterScale = 1e-4;
cfg.featureScope = 'all';                   % 使用全部 tap 特征；若 delay/Doppler 尺度主导，可改回 h_only
cfg.timeIndex = [];                         % mode='time_step' 时有效，[] 表示取预测窗口中间时刻

% 每个 SNR 最多抽取的样本数。
% 每个子图实际 t-SNE 点数为 2 * maxSamplesPerSNR，因为 true/pred 联合降维。
cfg.maxSamplesPerSNR = 250;                 % Match the MATLAB presentation figure: 250 true + 250 predicted points.
cfg.sampleSelectionMode = 'random';         % 'random' / 'first'
cfg.randomSeed = 2026;

% t-SNE 参数。
cfg.tsnePerplexity = 30;
cfg.tsneExaggeration = 4;
cfg.tsneNumPCAComponents = 30;              % 高维输入先 PCA，再 t-SNE；自动不超过实际维度
cfg.tsneMaxIter = 1000;                     % Kept for record; not passed to tsne() for MATLAB release compatibility.
cfg.standardizeBeforeTSNE = true;

% Figure / save.
cfg.showFigure = false;
cfg.closeAfterSave = true;
cfg.savePNG = false;
cfg.savePDF = false;
cfg.saveFIG = false;
cfg.dpi = 300;

% Dynamic video / GIF save options.
cfg.videoFrameRate = 2;
cfg.saveMP4 = false;
cfg.saveGIF = false;                       % 优化显示速度：GIF 写入很慢，默认关闭；如需 GIF 可改回 true。

% Python UI streaming. The UI main.py listens on UDP 65436 and distinguishes
% t-SNE frames by payload.type = 'ntn_tsne_frame'.
cfg.enableUiPublish = true;
cfg.UiHost = '127.0.0.1';
cfg.UiPort = 65436;
cfg.UiFramePause = 0.12;
cfg.WaitForPredictionEnable = true;       % true: 等待 Python UI 中“信道状态信息预测=启用”后再开始输出 t-SNE
cfg.UiPredictionControlFile = fullfile(SCRIPT_DIR, 'runtime', 'channel_prediction_control.json');
cfg.alignTSNEByProcrustes = false;
cfg.useFastDrawNow = true;                 % 优化显示刷新：使用 drawnow limitrate。
cfg.figureRenderer = 'opengl';             % 优化散点图刷新速度。
cfg.videoFrameWidth = 1280;                % 固定视频帧宽度，避免 writeVideo 帧尺寸不一致报错。
cfg.videoFrameHeight = 920;                % 固定视频帧高度，避免 writeVideo 帧尺寸不一致报错。
cfg.repeatCycles = 10;                     % Keep MATLAB's sample-batch coverage while streaming frames to the UI.
cfg.varyTSNEEachCycle = false;             % false: 同一输入固定 t-SNE 随机种子；由于每轮样本批次不同，结果仍会不同。
cfg.legendLocation = 'southeast';          % 固定图例在右下角，避免遮挡中心点云。

%% 3. Locate prediction files
prediction_files = dir(fullfile(PREDICT_DIR, cfg.predictionPattern));
if isempty(prediction_files)
    altPredictDir = SCRIPT_DIR;
    altPredictionFiles = dir(fullfile(altPredictDir, cfg.predictionPattern));
    if ~isempty(altPredictionFiles)
        fprintf('No files in default PREDICT_DIR. Use SCRIPT_DIR as PREDICT_DIR instead.\n');
        PREDICT_DIR = altPredictDir;
        prediction_files = altPredictionFiles;
    end
end
if isempty(prediction_files)
    error('No prediction files found in %s with pattern %s. Put prediction .mat files in PREDICT_DIR or run setenv(''PREDICT_DIR'', ''your_predict_results_path'') before this script.', PREDICT_DIR, cfg.predictionPattern);
end
[~, sort_idx] = sort({prediction_files.name});
prediction_files = prediction_files(sort_idx);

fprintf('\nFound %d prediction file(s).\n', numel(prediction_files));
fprintf('SNR to plot          : %s\n', mat2str(cfg.snrToPlot));
fprintf('Feature scope        : %s\n', cfg.featureScope);
fprintf('Representation mode  : %s\n', cfg.representationMode);
fprintf('Max samples per SNR  : %d\n', cfg.maxSamplesPerSNR);

% 根据当前 predict_results 中实际存在的文件，在命令行选择信道类型。
[available_conditions, condition_counts] = collect_available_conditions_from_files(prediction_files);

if isfield(cfg, 'enableInteractiveConditionSelection') && cfg.enableInteractiveConditionSelection
    cfg.conditionsToPlot = select_conditions_interactive(available_conditions, condition_counts, cfg.conditionsToPlot);
else
    cfg.conditionsToPlot = normalize_conditions_to_cell(cfg.conditionsToPlot);
end

fprintf('Selected condition(s): %s\n', format_selected_conditions(cfg.conditionsToPlot));

%% 4. Process files
for file_idx = 1:numel(prediction_files)

    mat_path = fullfile(prediction_files(file_idx).folder, prediction_files(file_idx).name);
    file_name = prediction_files(file_idx).name;
    condition_name = infer_condition_from_prediction_filename(file_name);

    if isempty(condition_name)
        condition_name = 'unknown_condition';
        warning('Cannot infer channel condition from file name: %s. Use unknown_condition.', file_name);
    end

    % 只处理命令行选中的信道类型。
    if ~should_plot_condition(condition_name, cfg.conditionsToPlot)
        fprintf('Skip condition: %s | File: %s\n', condition_name, file_name);
        continue;
    end

    condOutDir = fullfile(OUTPUT_DIR, condition_name);
    if ~exist(condOutDir, 'dir'), mkdir(condOutDir); end

    fprintf('\n============================================================\n');
    fprintf('Processing file %d / %d\n', file_idx, numel(prediction_files));
    fprintf('Condition : %s\n', condition_name);
    fprintf('File      : %s\n', file_name);
    fprintf('============================================================\n');

    result = load_prediction_result_local(mat_path);

    [pred_h, true_h, snr_db, snr_label_test, N_eval, M_eval, taps_eval] = ...
        prepare_prediction_tensors_local(result, cfg);

    if cfg.enableStaticTSNE
        plot_tsne_four_snr_for_prediction_file(pred_h, true_h, snr_db, snr_label_test, ...
            N_eval, M_eval, taps_eval, result, condition_name, cfg, condOutDir);
    end

    if cfg.enableDynamicTSNE
        plot_dynamic_tsne_video_for_prediction_file(pred_h, true_h, snr_db, snr_label_test, ...
            N_eval, M_eval, taps_eval, result, condition_name, cfg, condOutDir);
    end

end

fprintf('\nDone. t-SNE outputs saved under:\n%s\n', OUTPUT_DIR);

%% ========================================================================
% Local functions
% ========================================================================

function [available_conditions, condition_counts] = collect_available_conditions_from_files(prediction_files)

    all_conditions = cell(numel(prediction_files), 1);

    for ii = 1:numel(prediction_files)
        c = infer_condition_from_prediction_filename(prediction_files(ii).name);
        if isempty(c)
            c = 'unknown_condition';
        end
        all_conditions{ii} = lower(strtrim(c));
    end

    canonical_order = {'normal', 'step_loss', 'burst_shadow', 'firsttap_block', 'doppler_jump', 'mixed', 'unknown_condition'};
    unique_conditions = unique(all_conditions, 'stable');

    available_conditions = {};
    condition_counts = [];

    for ii = 1:numel(canonical_order)
        c = canonical_order{ii};
        if any(strcmp(unique_conditions, c))
            available_conditions{end + 1} = c; %#ok<AGROW>
            condition_counts(end + 1) = sum(strcmp(all_conditions, c)); %#ok<AGROW>
        end
    end

    for ii = 1:numel(unique_conditions)
        c = unique_conditions{ii};
        if ~any(strcmp(canonical_order, c))
            available_conditions{end + 1} = c; %#ok<AGROW>
            condition_counts(end + 1) = sum(strcmp(all_conditions, c)); %#ok<AGROW>
        end
    end

end

function selected_conditions = select_conditions_interactive(available_conditions, ~, default_conditions)

    default_conditions = normalize_conditions_to_cell(default_conditions);

    fprintf('\n============================================================\n');
    fprintf('可选择的信道类型如下：\n');
    fprintf('  0) all / 全部信道类型\n');

    for ii = 1:numel(available_conditions)
        fprintf('  %d) %s\n', ii, available_conditions{ii});
    end

    while true
        if isempty(default_conditions)
            prompt = '请输入要绘制的信道类型序号 [默认 all]: ';
        else
            prompt = sprintf('请输入要绘制的信道类型序号 [默认 %s]: ', format_selected_conditions(default_conditions));
        end

        user_input = strtrim(input(prompt, 's'));

        if isempty(user_input)
            selected_conditions = default_conditions;
            break;
        end

        user_input_lower = lower(strtrim(user_input));

        if any(strcmp(user_input_lower, {'0', 'all', '全部', 'quanbu'}))
            selected_conditions = {};
            break;
        end

        tokens = regexp(user_input_lower, '[,，;；\s]+', 'split');
        tokens = tokens(~cellfun(@isempty, tokens));

        selected_conditions = {};
        invalid_tokens = {};

        for kk = 1:numel(tokens)
            token = normalize_condition_alias(tokens{kk});
            token_num = str2double(token);

            if ~isnan(token_num) && abs(token_num - round(token_num)) < eps
                idx = round(token_num);
                if idx >= 1 && idx <= numel(available_conditions)
                    selected_conditions{end + 1} = available_conditions{idx}; %#ok<AGROW>
                else
                    invalid_tokens{end + 1} = tokens{kk}; %#ok<AGROW>
                end
            else
                if any(strcmp(available_conditions, token))
                    selected_conditions{end + 1} = token; %#ok<AGROW>
                else
                    invalid_tokens{end + 1} = tokens{kk}; %#ok<AGROW>
                end
            end
        end

        selected_conditions = unique(selected_conditions, 'stable');

        if isempty(invalid_tokens) && ~isempty(selected_conditions)
            break;
        end

        if ~isempty(invalid_tokens)
            fprintf('输入中存在无效选项：%s\n', strjoin(invalid_tokens, ', '));
            fprintf('请重新输入编号或信道类型名称。\n');
        else
            fprintf('未选择有效信道类型，请重新输入。\n');
        end
    end

    fprintf('最终选择：%s\n', format_selected_conditions(selected_conditions));
    fprintf('============================================================\n\n');

end

function tf = should_plot_condition(condition_name, selected_conditions)

    selected_conditions = normalize_conditions_to_cell(selected_conditions);

    if isempty(selected_conditions)
        tf = true;
        return;
    end

    this_condition = strtrim(condition_name);
    tf = any(strcmpi(this_condition, selected_conditions));

end

function selected_conditions = normalize_conditions_to_cell(selected_conditions)

    if isempty(selected_conditions)
        selected_conditions = {};
        return;
    end

    if ischar(selected_conditions) || isstring(selected_conditions)
        selected_conditions = cellstr(selected_conditions);
    end

    selected_conditions = selected_conditions(:).';

    for ii = 1:numel(selected_conditions)
        selected_conditions{ii} = normalize_condition_alias(selected_conditions{ii});
    end

    selected_conditions = unique(selected_conditions, 'stable');

end

function condition_name = normalize_condition_alias(condition_name)

    condition_name = lower(strtrim(char(condition_name)));
    condition_name = strrep(condition_name, '-', '_');
    condition_name = strrep(condition_name, ' ', '_');

    switch condition_name
        case {'none', 'normal_channel', 'no_abnormal', 'no_anomaly', '正常', '无异常'}
            condition_name = 'normal';
        case {'step', 'steploss', 'step_loss_channel', '整体链路衰减'}
            condition_name = 'step_loss';
        case {'burst', 'shadow', 'burstshadow', 'burst_shadowing', '短时阴影衰落'}
            condition_name = 'burst_shadow';
        case {'firsttap', 'first_tap', 'firsttapblock', 'first_tap_block', '首径阻断'}
            condition_name = 'firsttap_block';
        case {'doppler', 'dopplerjump', 'doppler_jump_channel', '多普勒突变'}
            condition_name = 'doppler_jump';
        case {'mix', 'mixed_channel', '混合异常'}
            condition_name = 'mixed';
        otherwise
            % Keep the normalized name.
    end

end

function text_out = format_selected_conditions(selected_conditions)

    selected_conditions = normalize_conditions_to_cell(selected_conditions);

    if isempty(selected_conditions)
        text_out = 'all';
    else
        text_out = strjoin(selected_conditions, ', ');
    end

end

function result = load_prediction_result_local(mat_path)

    data = load(mat_path);

    if isfield(data, 'pred_h')
        result.pred_h = data.pred_h;
    elseif isfield(data, 'Y_pred')
        result.pred_h = data.Y_pred;
    elseif isfield(data, 'predicted_csi')
        result.pred_h = data.predicted_csi;
    else
        error('Cannot find pred_h/Y_pred/predicted_csi in %s.', mat_path);
    end

    result.has_true_h = true;
    if isfield(data, 'true_h')
        result.true_h = data.true_h;
    elseif isfield(data, 'Y_test')
        result.true_h = data.Y_test;
    elseif isfield(data, 'ground_truth_csi')
        result.true_h = data.ground_truth_csi;
    else
        result.true_h = result.pred_h;
        result.has_true_h = false;
        warning('Cannot find true_h/Y_test/ground_truth_csi in %s. Use pred_h as placeholder for prediction-only visualization.', mat_path);
    end

    [~, file_base, ~] = fileparts(mat_path);
    result.file_name = [file_base, '.mat'];
    result.file_base = file_base;

    result.method_name = regexprep(file_base, '^predictions_', '');
    result.method_name = regexprep(result.method_name, '_fixed50_ntn_tdl_diff_snr$', '');
    result.method_name = regexprep(result.method_name, '_fixed50_ntn_tdl$', '');
    result.method_name = upper(result.method_name);

    result.snr_db = [];
    if isfield(data, 'snr_db')
        result.snr_db = double(data.snr_db(:)).';
    end

    result.snr_label_test = [];
    if isfield(data, 'snr_label_test')
        result.snr_label_test = double(data.snr_label_test(:));
    elseif isfield(data, 'snr_label')
        result.snr_label_test = double(data.snr_label(:));
    elseif isfield(data, 'snr')
        result.snr_label_test = double(data.snr(:));
    end

    result.samples_per_snr_test = [];
    if isfield(data, 'samples_per_snr_test')
        result.samples_per_snr_test = double(data.samples_per_snr_test(1));
    end

    result.valid_feature_dim = [];
    if isfield(data, 'valid_feature_dim')
        result.valid_feature_dim = double(data.valid_feature_dim(1));
    end

    result.N = [];
    if isfield(data, 'N')
        result.N = double(data.N(1));
    end

    result.M = [];
    if isfield(data, 'M')
        result.M = double(data.M(1));
    end

    result.doppler_is_physical = [];
    if isfield(data, 'doppler_is_physical')
        result.doppler_is_physical = double(data.doppler_is_physical(1));
    end

    result.doppler_feature_scale = [];
    if isfield(data, 'doppler_feature_scale')
        result.doppler_feature_scale = double(data.doppler_feature_scale(1));
    elseif isfield(data, 'doppler_feature_scale_original')
        result.doppler_feature_scale = double(data.doppler_feature_scale_original(1));
    end

end

function [pred_h, true_h, snr_db, snr_label_test, N_eval, M_eval, taps_eval] = ...
    prepare_prediction_tensors_local(result, cfg)

    pred_h = double(result.pred_h);
    true_h = double(result.true_h);

    if size(pred_h, 1) ~= size(true_h, 1) || ...
            size(pred_h, 2) ~= size(true_h, 2) || ...
            size(pred_h, 3) ~= size(true_h, 3)
        error('pred_h and true_h size mismatch in %s: pred=%s, true=%s.', ...
            result.file_name, mat2str(size(pred_h)), mat2str(size(true_h)));
    end

    snr_label_test = result.snr_label_test;
    snr_db = result.snr_db;

    if isempty(snr_db)
        if ~isempty(snr_label_test)
            snr_db = unique(double(snr_label_test(:))).';
        else
            snr_db = cfg.default_snr_db;
        end
    end
    snr_db = double(snr_db(:)).';

    valid_feature_dim = result.valid_feature_dim;
    if isempty(valid_feature_dim)
        valid_feature_dim = cfg.default_valid_feature_dim;
    end
    valid_feature_dim = double(valid_feature_dim(1));

    if size(pred_h, 3) < valid_feature_dim || size(true_h, 3) < valid_feature_dim
        error('%s has feature_dim smaller than valid_feature_dim=%d.', ...
            result.file_name, valid_feature_dim);
    end

    pred_h = pred_h(:, :, 1:valid_feature_dim);
    true_h = true_h(:, :, 1:valid_feature_dim);

    inferred_taps = valid_feature_dim / 4;
    if abs(inferred_taps - round(inferred_taps)) > eps
        error('valid_feature_dim=%d is not divisible by 4 in %s.', ...
            valid_feature_dim, result.file_name);
    end
    taps_eval = round(inferred_taps);

    if taps_eval ~= cfg.taps_per_state
        warning('Using inferred taps=%d from %s instead of configured taps=%d.', ...
            taps_eval, result.file_name, cfg.taps_per_state);
    end

    N_eval = result.N;
    if isempty(N_eval), N_eval = cfg.N; end

    M_eval = result.M;
    if isempty(M_eval), M_eval = cfg.M; end

    [pred_h, true_h, applied_scale] = recover_physical_doppler_local( ...
        pred_h, true_h, taps_eval, cfg.dopplerFeatureScale, cfg.dopplerScalePolicy, result);

    fprintf('  Method              : %s\n', result.method_name);
    fprintf('  Samples / Tout / Dim: %d / %d / %d\n', ...
        size(pred_h, 1), size(pred_h, 2), size(pred_h, 3));
    fprintf('  N / M / taps        : %d / %d / %d\n', N_eval, M_eval, taps_eval);
    fprintf('  SNR list            : %s\n', mat2str(snr_db));
    fprintf('  Doppler scale used  : %.6g\n', applied_scale);

    if cfg.requireSampleSNRLabel
        if isempty(snr_label_test)
            error('%s does not contain snr_label_test/snr_label/snr. Current script requires sample-level SNR labels.', ...
                result.file_name);
        end

        if numel(snr_label_test) ~= size(pred_h, 1)
            error('snr_label_test length mismatch in %s: labels=%d, samples=%d.', ...
                result.file_name, numel(snr_label_test), size(pred_h, 1));
        end
    end

end

function plot_tsne_four_snr_for_prediction_file(pred_h, true_h, snr_db, snr_label_test, ...
    N_eval, M_eval, taps_eval, result, condition_name, cfg, output_dir) %#ok<INUSD>

    snr_to_plot = cfg.snrToPlot(:).';

    % 检查当前文件实际存在的 SNR。
    valid_snr = [];
    missing_snr = [];
    for k = 1:numel(snr_to_plot)
        if any(abs(snr_db - snr_to_plot(k)) <= cfg.snrLabelTolerance)
            valid_snr(end + 1) = snr_to_plot(k); %#ok<AGROW>
        else
            missing_snr(end + 1) = snr_to_plot(k); %#ok<AGROW>
            warning('SNR=%g dB does not exist in %s. This panel will be marked as missing.', ...
                snr_to_plot(k), result.file_name);
        end
    end

    if isempty(valid_snr)
        warning('No valid SNR to plot for %s. Skip.', result.file_name);
        return;
    end

    if cfg.showFigure
        fig = figure('Color', 'w', 'Position', [80, 60, 1180, 900]);
    else
        fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80, 60, 1180, 900]);
    end

    try
        tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
        useTile = true;
    catch
        useTile = false;
    end

    summary_rows = repmat(empty_tsne_summary_row(), 0, 1);

    for p = 1:4

        this_snr = snr_to_plot(p);

        if useTile
            nexttile;
        else
            subplot(2, 2, p);
        end

        if ~any(abs(snr_db - this_snr) <= cfg.snrLabelTolerance)
            axis off;
            text(0.5, 0.5, sprintf('SNR = %g dB missing', this_snr), ...
                'HorizontalAlignment', 'center', ...
                'FontName', 'Times New Roman', ...
                'FontSize', 13);
            continue;
        end

        if ~isempty(snr_label_test)
            idx = abs(double(snr_label_test(:)) - this_snr) <= cfg.snrLabelTolerance;
            true_part = true_h(idx, :, :);
            pred_part = pred_h(idx, :, :);
        else
            if ~cfg.allowLegacyBlockSplit
                error('Legacy block split is disabled and no snr_label_test is available.');
            end
            error('Legacy block split is not implemented in this t-SNE script. Please export snr_label_test.');
        end

        if isempty(true_part)
            warning('No samples for SNR=%g dB in %s. Skip this panel.', this_snr, result.file_name);
            axis off;
            text(0.5, 0.5, sprintf('No samples for SNR = %g dB', this_snr), ...
                'HorizontalAlignment', 'center', ...
                'FontName', 'Times New Roman', ...
                'FontSize', 13);
            continue;
        end

        [X, label_is_pred, selected_count, feature_dim, time_idx] = ...
            build_tsne_input(true_part, pred_part, taps_eval, cfg.representationMode, cfg);

        if size(X, 1) < 10
            warning('Too few samples for t-SNE: SNR=%g. Skip this panel.', this_snr);
            axis off;
            text(0.5, 0.5, sprintf('Too few samples for SNR = %g dB', this_snr), ...
                'HorizontalAlignment', 'center', ...
                'FontName', 'Times New Roman', ...
                'FontSize', 13);
            continue;
        end

        [Y, tsne_perplexity] = run_tsne_2d(X, cfg);

        show_legend = (p == 1);
        draw_tsne_panel(Y, label_is_pred, this_snr, cfg.representationMode, ...
            cfg, time_idx, feature_dim, selected_count, show_legend);

        row = empty_tsne_summary_row();
        row.Condition = condition_name;
        row.Method = result.method_name;
        row.FileName = result.file_name;
        row.SNR_dB = this_snr;
        row.Mode = cfg.representationMode;
        row.FeatureScope = cfg.featureScope;
        row.TimeIndex = time_idx;
        row.NumSelectedSamples = selected_count;
        row.NumTSNEPoints = size(X, 1);
        row.FeatureDim = feature_dim;
        row.Perplexity = tsne_perplexity;
        summary_rows(end + 1) = row; %#ok<AGROW>

    end

    try
        sgtitle(sprintf('t-SNE of CSI Prediction Features | %s | %s', ...
            condition_name, result.method_name), ...
            'Interpreter', 'none', ...
            'FontName', 'Times New Roman', ...
            'FontSize', 15, ...
            'FontWeight', 'bold');
    catch
    end

    drawnow;

    safe_method = make_safe_filename_local(result.method_name);
    safe_file = make_safe_filename_local(result.file_base);
    safe_scope = make_safe_filename_local(cfg.featureScope);
    safe_mode = make_safe_filename_local(cfg.representationMode);
    snr_tag = make_safe_filename_local(strrep(mat2str(snr_to_plot), ' ', '_'));

    output_base = fullfile(output_dir, sprintf('tsne_2x2_%s_%s_%s_snr_%s_%s', ...
        safe_method, safe_scope, safe_mode, snr_tag, safe_file));

    if cfg.saveFIG
        savefig(fig, [output_base, '.fig']);
    end

    if cfg.savePNG
        try
            exportgraphics(fig, [output_base, '.png'], 'Resolution', cfg.dpi);
        catch
            saveas(fig, [output_base, '.png']);
        end
    end

    if cfg.savePDF
        try
            exportgraphics(fig, [output_base, '.pdf'], 'ContentType', 'vector');
        catch
            saveas(fig, [output_base, '.pdf']);
        end
    end

    if ~isempty(summary_rows)
        summary_table = struct2table(summary_rows);
        try
            writetable(summary_table, [output_base, '_summary.csv']);
        catch ME
            % warning('Failed to save t-SNE summary CSV: %s', ME.message);
        end
    end

    if cfg.closeAfterSave && ishandle(fig)
        close(fig);
    end

    fprintf('  Saved four-SNR t-SNE figure: %s.png\n', output_base);

    if ~isempty(missing_snr)
        fprintf('  Missing SNR panel(s): %s\n', mat2str(missing_snr));
    end

end

function plot_dynamic_tsne_video_for_prediction_file(pred_h, true_h, snr_db, snr_label_test, ...
    N_eval, M_eval, taps_eval, result, condition_name, cfg, output_dir) %#ok<INUSD>

    if isfield(cfg, 'dynamicSnrToPlot') && ~isempty(cfg.dynamicSnrToPlot)
        snr_to_plot = cfg.dynamicSnrToPlot(:).';
    else
        snr_to_plot = cfg.snrToPlot(:).';
    end

    if isempty(snr_to_plot)
        snr_to_plot = snr_db(:).';
    end

    if isempty(snr_label_test)
        warning('No snr_label_test found in %s. Skip dynamic t-SNE.', result.file_name);
        return;
    end

    if ~strcmpi(cfg.dynamicFeatureMode, 'time_step')
        error('Dynamic t-SNE currently supports cfg.dynamicFeatureMode=''time_step'' only.');
    end

    % 动态视频/GIF/MAT 也直接保存在当前代码所在文件夹。
    dynamicOutDir = output_dir;

    repeat_cycles = get_repeat_cycles_from_cfg(cfg);

    % selected_idx_cell{s, c}: 第 s 个 SNR 在第 c 轮使用的样本索引。
    % 例如每个 SNR 有 2500 个样本，cfg.maxSamplesPerSNR=250 且 cfg.repeatCycles=10 时，
    % 第 1 轮使用样本 1:250，第 2 轮使用 251:500，...，第 10 轮使用 2251:2500。
    selected_idx_cell = cell(numel(snr_to_plot), repeat_cycles);
    valid_panel = false(numel(snr_to_plot), 1);

    for s = 1:numel(snr_to_plot)
        this_snr = snr_to_plot(s);
        snr_mask = abs(double(snr_label_test(:)) - this_snr) <= cfg.snrLabelTolerance;
        all_idx = find(snr_mask);

        if isempty(all_idx)
            warning('SNR = %g dB not found in %s. This dynamic panel will be marked as missing.', ...
                this_snr, result.file_name);
            continue;
        end

        selected_batches = select_dynamic_tsne_sample_batches(all_idx, cfg, repeat_cycles);
        selected_idx_cell(s, :) = selected_batches;
        valid_panel(s) = true;

        selected_sizes = cellfun(@numel, selected_batches);
        fprintf('Dynamic t-SNE panel | condition=%s | SNR=%g dB | total samples=%d | samples per cycle=%s\n', ...
            condition_name, this_snr, numel(all_idx), mat2str(selected_sizes));
    end

    if ~any(valid_panel)
        warning('No requested SNR is available in %s. Skip four-panel dynamic t-SNE.', result.file_name);
        return;
    end

    make_dynamic_tsne_video_multi_snr(pred_h, true_h, selected_idx_cell, valid_panel, snr_to_plot, ...
        taps_eval, result, condition_name, cfg, dynamicOutDir);

end

function selected_batches = select_dynamic_tsne_sample_batches(all_idx, cfg, repeat_cycles)

    all_idx = all_idx(:);
    num_all = numel(all_idx);

    if isempty(all_idx)
        selected_batches = cell(1, repeat_cycles);
        return;
    end

    if isinf(cfg.maxSamplesPerSNR)
        batch_size = num_all;
    else
        batch_size = min(num_all, max(1, round(cfg.maxSamplesPerSNR)));
    end

    switch lower(cfg.sampleSelectionMode)
        case 'random'
            % 先对全部样本做一次固定随机打乱，然后按顺序分批。
            % 这样每轮不是重复同一批，而是依次覆盖不同样本。
            rng(cfg.randomSeed);
            ordered_idx = all_idx(randperm(num_all));
        case 'first'
            ordered_idx = all_idx;
        otherwise
            error('Unknown cfg.sampleSelectionMode=%s. Use random or first.', ...
                cfg.sampleSelectionMode);
    end

    selected_batches = cell(1, repeat_cycles);

    for cycle_idx = 1:repeat_cycles
        start_pos = (cycle_idx - 1) * batch_size + 1;
        stop_pos  = cycle_idx * batch_size;

        if stop_pos <= num_all
            selected_batches{cycle_idx} = ordered_idx(start_pos:stop_pos);
        else
            % 如果 repeatCycles * batch_size 超过样本总数，则循环回到开头补齐。
            wrapped_pos = mod((start_pos:stop_pos) - 1, num_all) + 1;
            selected_batches{cycle_idx} = ordered_idx(wrapped_pos);
        end

        selected_batches{cycle_idx} = selected_batches{cycle_idx}(:);
    end

end

function make_dynamic_tsne_video_multi_snr(pred_h, true_h, selected_idx_cell, valid_panel, snr_to_plot, ...
    taps, result, condition_name, cfg, output_dir)

    Tout_all = size(pred_h, 2);

    if isfield(cfg, 'dynamicTimeRange') && ~isempty(cfg.dynamicTimeRange)
        time_list = cfg.dynamicTimeRange(:).';
        time_list = time_list(time_list >= 1 & time_list <= Tout_all);
        time_list = unique(time_list, 'stable');
        if isempty(time_list)
            warning('cfg.dynamicTimeRange is empty after range check. Fall back to all frames.');
            time_list = 1:Tout_all;
        end
    else
        time_list = 1:Tout_all;
    end

    num_frames = numel(time_list);
    num_panels = numel(snr_to_plot);
    feat_idx = get_feature_indices(taps, cfg.featureScope);

    repeat_cycles = get_repeat_cycles_from_cfg(cfg);

    % 第三维对应样本批次/循环轮数。
    Y_cell = cell(num_frames, num_panels, repeat_cycles);
    perplexity_list = nan(num_frames, num_panels, repeat_cycles);
    feature_dim_list = nan(num_frames, num_panels, repeat_cycles);
    selected_count_list = zeros(num_panels, repeat_cycles);
    x_lim_cell = cell(num_panels, repeat_cycles);
    y_lim_cell = cell(num_panels, repeat_cycles);

    for p = 1:num_panels
        if valid_panel(p)
            for cycle_idx = 1:repeat_cycles
                selected_count_list(p, cycle_idx) = numel(selected_idx_cell{p, cycle_idx});
            end
        end
    end

    fprintf('\nDynamic four-SNR true-vs-pred t-SNE video | condition=%s | method=%s | frames=%d | sample batches=%d | total frames=%d\n', ...
        condition_name, result.method_name, num_frames, repeat_cycles, num_frames * repeat_cycles);

    safe_condition = make_safe_filename_local(condition_name);
    safe_method = make_safe_filename_local(result.method_name);
    safe_scope = make_safe_filename_local(cfg.featureScope);
    snr_tag = make_safe_filename_local(strrep(mat2str(snr_to_plot), ' ', '_'));
    file_tag = sprintf('%s_%s_SNR_%s_%s_dynamic_tsne_2x2', ...
        safe_condition, safe_method, snr_tag, safe_scope);

    mp4_path = fullfile(output_dir, [file_tag, '.mp4']);
    avi_path = fullfile(output_dir, [file_tag, '.avi']);
    gif_path = fullfile(output_dir, [file_tag, '.gif']);
    mat_path = fullfile(output_dir, [file_tag, '_embedding.mat']);

    if cfg.showFigure
        fig = figure('Color', 'w', 'Units', 'pixels', ...
            'Position', [80, 50, cfg.videoFrameWidth, cfg.videoFrameHeight]);
    else
        fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'pixels', ...
            'Position', [80, 50, cfg.videoFrameWidth, cfg.videoFrameHeight]);
    end

    % 固定窗口尺寸，避免 getframe 每一帧尺寸轻微变化导致 writeVideo 报错。
    try
        set(fig, 'Resize', 'off');
    catch
    end
    set(fig, 'Units', 'pixels', ...
        'Position', [80, 50, cfg.videoFrameWidth, cfg.videoFrameHeight]);

    if isfield(cfg, 'figureRenderer') && ~isempty(cfg.figureRenderer)
        set(fig, 'Renderer', cfg.figureRenderer);
    end

    videoObj = [];
    video_path = '';
    write_video = isfield(cfg, 'saveMP4') && cfg.saveMP4;
    video_is_open = false;

    write_gif = isfield(cfg, 'saveGIF') && cfg.saveGIF;
    align_available = exist('procrustes', 'file') == 2;
    warned_no_procrustes = false;

    target_frame_size = [cfg.videoFrameHeight, cfg.videoFrameWidth];

    uiUdp = [];
    if isfield(cfg, 'enableUiPublish') && cfg.enableUiPublish
        try
            uiUdp = udpport('datagram', 'IPV4');
            try
                uiUdp.OutputDatagramSize = 65507;
            catch
            end
            fprintf('[GC-UI] Publishing dynamic t-SNE frames to UDP %d.\n', cfg.UiPort);
        catch ME
            % warning('Failed to create UDP publisher for Python UI: %s. Continue without UI streaming.', ME.message);
            uiUdp = [];
        end
    end

    for cycle_idx = 1:repeat_cycles

        fprintf('\nStart repeat cycle %d / %d.\n', cycle_idx, repeat_cycles);

        for frame_idx = 1:num_frames

            total_frame_idx = (cycle_idx - 1) * num_frames + frame_idx;
            t = time_list(frame_idx);

            % 交互控制：只有 UI 左侧“信道状态信息预测”切换为“启用”后，才开始计算/输出 t-SNE 帧。
            wait_for_ui_prediction_enable(cfg);

        clf(fig);
        try
            tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
            useTile = true;
        catch
            useTile = false;
        end

        for p = 1:min(4, num_panels)

            if useTile
                nexttile;
            else
                subplot(2, 2, p);
            end

            if ~valid_panel(p)
                axis off;
                text(0.5, 0.5, sprintf('SNR = %g dB missing', snr_to_plot(p)), ...
                    'HorizontalAlignment', 'center', ...
                    'FontName', 'Times New Roman', ...
                    'FontSize', 13);
                continue;
            end

            selected_idx = selected_idx_cell{p, cycle_idx};
            num_samples = numel(selected_idx);

            if num_samples < 5
                axis off;
                text(0.5, 0.5, sprintf('Too few samples for SNR = %g dB', snr_to_plot(p)), ...
                    'HorizontalAlignment', 'center', ...
                    'FontName', 'Times New Roman', ...
                    'FontSize', 13);
                valid_panel(p) = false;
                continue;
            end

            [X_true, X_pred, used_t_list] = build_dynamic_tsne_frame_feature( ...
                true_h, pred_h, selected_idx, t, feat_idx, cfg);

            plot_prediction_only = isfield(cfg, 'dynamicPlotContent') && ...
                strcmpi(cfg.dynamicPlotContent, 'prediction_only');

            if plot_prediction_only
                X = X_pred;
            else
                X = [X_true; X_pred];
            end

            [X, x_stat] = preprocess_tsne_input_dynamic(X, cfg);

            if x_stat.total_var < cfg.degenerateVarThreshold
                warning('Near-degenerate dynamic t-SNE input | source t=%d | SNR=%g dB | total variance=%.3e. The embedding may collapse.', ...
                    t, snr_to_plot(p), x_stat.total_var);
                if isfield(cfg, 'addTinyJitterForDegenerate') && cfg.addTinyJitterForDegenerate
                    rng(cfg.randomSeed + 900000 + 1000 * p + t);
                    X = X + cfg.tinyJitterScale * randn(size(X));
                end
            end

            feature_dim_list(frame_idx, p, cycle_idx) = size(X, 2);

            tsne_seed = cfg.randomSeed + 1000 * p + t;
            if isfield(cfg, 'varyTSNEEachCycle') && cfg.varyTSNEEachCycle
                tsne_seed = tsne_seed + 100000 * cycle_idx;
            end
            rng(tsne_seed);
            [Y, perplexity] = run_tsne_2d(X, cfg);
            perplexity_list(frame_idx, p, cycle_idx) = perplexity;

            if isfield(cfg, 'alignTSNEByProcrustes') && cfg.alignTSNEByProcrustes && frame_idx > 1
                if align_available && ~isempty(Y_cell{frame_idx - 1, p, cycle_idx})
                    try
                        [~, Y_aligned] = procrustes(Y_cell{frame_idx - 1, p, cycle_idx}, Y, ...
                            'Scaling', true, ...
                            'Reflection', false);
                        Y = Y_aligned;
                    catch ME
                        warning('Procrustes alignment failed at t=%d, SNR=%g: %s', ...
                            t, snr_to_plot(p), ME.message);
                    end
                elseif ~warned_no_procrustes
                    warning('Cannot find procrustes(). Dynamic t-SNE alignment is skipped.');
                    warned_no_procrustes = true;
                end
            end

            Y_cell{frame_idx, p, cycle_idx} = Y;

            if isfield(cfg, 'dynamicAxisMode') && strcmpi(cfg.dynamicAxisMode, 'current')
                [x_lim_cell{p, cycle_idx}, y_lim_cell{p, cycle_idx}] = current_dynamic_tsne_axis_limits(Y, cfg);
            else
                [x_lim_cell{p, cycle_idx}, y_lim_cell{p, cycle_idx}] = update_dynamic_tsne_axis_limits( ...
                    x_lim_cell{p, cycle_idx}, y_lim_cell{p, cycle_idx}, Y, cfg);
            end

            fprintf('  cycle = %02d / %02d | frame = %02d / %02d | source t = %02d | SNR = %g dB | points = %d | feature dim = %d | perplexity = %.1f | used t = %s\n', ...
                cycle_idx, repeat_cycles, frame_idx, num_frames, t, snr_to_plot(p), size(Y, 1), feature_dim_list(frame_idx, p, cycle_idx), perplexity, mat2str(used_t_list));

            Y = Y_cell{frame_idx, p, cycle_idx};
            num_samples = selected_count_list(p, cycle_idx);

            if ~isempty(uiUdp)
                try
                    publish_ui_tsne_frame(uiUdp, cfg, Y, plot_prediction_only, num_samples, ...
                        condition_name, result, snr_to_plot(p), cycle_idx, repeat_cycles, ...
                        frame_idx, num_frames, t, used_t_list, feature_dim_list(frame_idx, p, cycle_idx), ...
                        perplexity, x_lim_cell{p, cycle_idx}, y_lim_cell{p, cycle_idx});
                    if isfield(cfg, 'UiFramePause') && cfg.UiFramePause > 0
                        pause(cfg.UiFramePause);
                    end
                catch ME
                    % warning('Failed to publish t-SNE frame to Python UI: %s', ME.message);
                end
            end

            if plot_prediction_only
                scatter(Y(:, 1), Y(:, 2), ...
                    18, [0.0000, 0.4470, 0.7410], 'filled', ...
                    'MarkerFaceAlpha', 0.62, 'MarkerEdgeAlpha', 0.62, ...
                    'DisplayName', 'Predicted CSI feature');
                hold on;
            else
                idx_true = 1:num_samples;
                idx_pred = num_samples + (1:num_samples);

                scatter(Y(idx_pred, 1), Y(idx_pred, 2), ...
                    16, [0.0000, 0.4470, 0.7410], 'filled', ...
                    'MarkerFaceAlpha', 0.55, 'MarkerEdgeAlpha', 0.55, ...
                    'DisplayName', 'Predicted CSI feature');
                hold on;
                scatter(Y(idx_true, 1), Y(idx_true, 2), ...
                    16, [0.8500, 0.3250, 0.0980], 'filled', ...
                    'MarkerFaceAlpha', 0.55, 'MarkerEdgeAlpha', 0.55, ...
                    'DisplayName', 'Ground-truth CSI feature');
            end

            grid on;
            box on;

            if ~isempty(x_lim_cell{p, cycle_idx})
                xlim(x_lim_cell{p, cycle_idx});
            end
            if ~isempty(y_lim_cell{p, cycle_idx})
                ylim(y_lim_cell{p, cycle_idx});
            end

            if repeat_cycles > 1
                title(sprintf('SNR = %g dB | cycle %d/%d | frame %d/%d, source t=%d, used t=%s\nfeature dim = %d, samples = %d', ...
                    snr_to_plot(p), cycle_idx, repeat_cycles, frame_idx, num_frames, t, mat2str(used_t_list), feature_dim_list(frame_idx, p, cycle_idx), selected_count_list(p, cycle_idx)), ...
                    'Interpreter', 'none', ...
                    'FontName', 'Times New Roman', ...
                    'FontSize', 11);
            else
                title(sprintf('SNR = %g dB | frame %d/%d, source t=%d, used t=%s\nfeature dim = %d, samples = %d', ...
                    snr_to_plot(p), frame_idx, num_frames, t, mat2str(used_t_list), feature_dim_list(frame_idx, p, cycle_idx), selected_count_list(p, cycle_idx)), ...
                    'Interpreter', 'none', ...
                    'FontName', 'Times New Roman', ...
                    'FontSize', 11);
            end
            
            lgd = legend('Location', cfg.legendLocation, ...
                'Interpreter', 'none', ...
                'Box', 'on', ...
                'FontName', 'Times New Roman', ...
                'FontSize', 8.5);
            
            set(lgd, ...
                'Color', 'w', ...        % 白色背景
                'EdgeColor', [0.7 0.7 0.7]);  % 浅灰边框

            set(gca, ...
                'FontName', 'Times New Roman', ...
                'FontSize', 10.5, ...
                'LineWidth', 1.0);

        end

        try
            sgtitle(sprintf('Dynamic t-SNE of True and Predicted CSI Features | %s | %s', ...
                condition_name, result.method_name), ...
                'Interpreter', 'none', ...
                'FontName', 'Times New Roman', ...
                'FontSize', 15, ...
                'FontWeight', 'bold');
        catch
        end

        if isfield(cfg, 'useFastDrawNow') && cfg.useFastDrawNow
            drawnow limitrate;
        else
            drawnow;
        end
        frame = getframe(fig);
        frame_rgb = fix_video_frame_size(frame.cdata, target_frame_size);

        if write_video
            if ~video_is_open
                [videoObj, video_path, write_video] = open_dynamic_tsne_video_writer( ...
                    mp4_path, avi_path, cfg);
                video_is_open = write_video;
            end
            writeVideo(videoObj, frame_rgb);
        end

        if write_gif
            try
                [im, map] = rgb2ind(frame_rgb, 256);
                if total_frame_idx == 1
                    imwrite(im, map, gif_path, 'gif', ...
                        'LoopCount', inf, ...
                        'DelayTime', 1 / cfg.videoFrameRate);
                else
                    imwrite(im, map, gif_path, 'gif', ...
                        'WriteMode', 'append', ...
                        'DelayTime', 1 / cfg.videoFrameRate);
                end
            catch ME
                warning('GIF writing failed at cycle=%d, frame=%d, source t=%d: %s. GIF output is disabled.', cycle_idx, frame_idx, t, ME.message);
                write_gif = false;
            end
        end

        end
    end

    if video_is_open && ~isempty(videoObj)
        close(videoObj);
        fprintf('Saved dynamic t-SNE video: %s\n', video_path);
    end

    if isfield(cfg, 'saveGIF') && cfg.saveGIF && exist(gif_path, 'file') == 2
        fprintf('Saved dynamic t-SNE GIF  : %s\n', gif_path);
    end

    save(mat_path, ...
        'Y_cell', ...
        'selected_idx_cell', ...
        'snr_to_plot', ...
        'condition_name', ...
        'perplexity_list', ...
        'feature_dim_list', ...
        'selected_count_list', ...
        'time_list', ...
        'repeat_cycles', ...
        'x_lim_cell', ...
        'y_lim_cell', ...
        'cfg');

    fprintf('Saved dynamic t-SNE embedding MAT: %s\n', mat_path);

    if isfield(cfg, 'closeAfterSave') && cfg.closeAfterSave && ishandle(fig)
        close(fig);
    end

end


function publish_ui_tsne_frame(uiUdp, cfg, Y, plot_prediction_only, num_samples, ...
    condition_name, result, snr_dB, cycle_idx, repeat_cycles, ...
    frame_idx, num_frames, source_t, used_t_list, feature_dim, perplexity, x_lim, y_lim)
%PUBLISH_UI_TSNE_FRAME Send one dynamic t-SNE frame to the Python UI.
% Payload is consumed by main.py as type = 'ntn_tsne_frame'.

    payload = struct();
    payload.type = 'ntn_tsne_frame';
    payload.condition = condition_name;
    payload.method = result.method_name;
    payload.fileName = result.file_name;
    payload.snrDb = snr_dB;
    payload.cycle = cycle_idx;
    payload.totalCycles = repeat_cycles;
    payload.frame = frame_idx;
    payload.totalFrames = num_frames;
    payload.sourceTime = source_t;
    payload.usedTimeList = used_t_list;
    payload.featureDim = feature_dim;
    payload.samples = num_samples;
    payload.perplexity = perplexity;

    if plot_prediction_only
        payload.trueXY = zeros(0, 2);
        payload.predXY = double(Y(:, 1:2));
    else
        idx_true = 1:num_samples;
        idx_pred = num_samples + (1:num_samples);
        payload.trueXY = double(Y(idx_true, 1:2));
        payload.predXY = double(Y(idx_pred, 1:2));
    end

    if ~isempty(x_lim)
        payload.xLim = double(x_lim(:).');
    else
        payload.xLim = [];
    end

    if ~isempty(y_lim)
        payload.yLim = double(y_lim(:).');
    else
        payload.yLim = [];
    end

    write(uiUdp, uint8(jsonencode(payload)), cfg.UiHost, cfg.UiPort);

end



function wait_for_ui_prediction_enable(cfg)
%WAIT_FOR_UI_PREDICTION_ENABLE Pause t-SNE streaming until Python UI enables prediction.
% The Python UI writes runtime/channel_prediction_control.json:
%   {"enabled": true/false, ...}
% When disabled, this function waits without publishing any t-SNE frames.

    if ~isfield(cfg, 'WaitForPredictionEnable') || ~cfg.WaitForPredictionEnable
        return;
    end

    lastPrintTic = tic;
    while true
        ctrl = read_ui_prediction_control(cfg);
        if ctrl.enabled
            return;
        end

        if toc(lastPrintTic) > 2.0
            fprintf('[GC-UI] Waiting for UI control: set "信道状态信息预测" to 启用 to start t-SNE streaming...\n');
            lastPrintTic = tic;
        end
        pause(0.20);
    end
end


function ctrl = read_ui_prediction_control(cfg)
%READ_UI_PREDICTION_CONTROL Read Python UI prediction/t-SNE control file.

    ctrl = struct();
    ctrl.enabled = false;

    if ~isfield(cfg, 'UiPredictionControlFile') || isempty(cfg.UiPredictionControlFile)
        return;
    end

    controlFile = cfg.UiPredictionControlFile;
    if exist(controlFile, 'file') ~= 2
        return;
    end

    try
        rawText = fileread(controlFile);
        if isempty(strtrim(rawText))
            return;
        end
        data = jsondecode(rawText);
        if isfield(data, 'enabled')
            ctrl.enabled = logical(data.enabled);
        end
    catch
        ctrl.enabled = false;
    end
end

function repeat_cycles = get_repeat_cycles_from_cfg(cfg)

    if isfield(cfg, 'repeatCycles') && ~isempty(cfg.repeatCycles)
        repeat_cycles = cfg.repeatCycles;
    else
        repeat_cycles = 1;
    end

    if ~isnumeric(repeat_cycles) || numel(repeat_cycles) ~= 1 || ~isfinite(repeat_cycles)
        error('cfg.repeatCycles must be a finite positive integer when saving a video.');
    end

    repeat_cycles = max(1, round(repeat_cycles));

end


function [X_true, X_pred, used_t_list] = build_dynamic_tsne_frame_feature(true_h, pred_h, selected_idx, t, feat_idx, cfg)

    num_samples = numel(selected_idx);
    Tout = size(true_h, 2);

    if ~isfield(cfg, 'dynamicTemporalMode') || isempty(cfg.dynamicTemporalMode)
        dynamicTemporalMode = 'single_frame';
    else
        dynamicTemporalMode = lower(cfg.dynamicTemporalMode);
    end

    switch dynamicTemporalMode

        case 'single_frame'
            used_t_list = t;

        case 'sliding_window'
            if isfield(cfg, 'dynamicHalfWindow') && ~isempty(cfg.dynamicHalfWindow)
                half_win = max(0, round(cfg.dynamicHalfWindow));
            else
                half_win = 2;
            end
            used_t_list = max(1, t - half_win) : min(Tout, t + half_win);

        case 'prefix'
            used_t_list = 1:t;

        otherwise
            error('Unknown cfg.dynamicTemporalMode=%s. Use single_frame / sliding_window / prefix.', ...
                cfg.dynamicTemporalMode);

    end

    X_true = reshape(true_h(selected_idx, used_t_list, feat_idx), num_samples, []);
    X_pred = reshape(pred_h(selected_idx, used_t_list, feat_idx), num_samples, []);

end

function [x_lim, y_lim] = current_dynamic_tsne_axis_limits(Y, cfg)

    if isfield(cfg, 'uiFixedAxisLim') && isequal(size(cfg.uiFixedAxisLim), [2 2])
        fixed_lim = double(cfg.uiFixedAxisLim);
        if all(isfinite(fixed_lim(:))) && fixed_lim(1, 2) > fixed_lim(1, 1) && fixed_lim(2, 2) > fixed_lim(2, 1)
            x_lim = fixed_lim(1, :);
            y_lim = fixed_lim(2, :);
            return;
        end
    end

    finite_rows = all(isfinite(Y(:, 1:2)), 2);
    Y = Y(finite_rows, 1:2);

    if isempty(Y)
        x_lim = [-1 1];
        y_lim = [-1 1];
        return;
    end

    if isfield(cfg, 'uiAxisPercentile') && numel(cfg.uiAxisPercentile) >= 2
        pct = sort(double(cfg.uiAxisPercentile(1:2)));
    else
        pct = [2 98];
    end
    pct(1) = max(0, min(49, pct(1)));
    pct(2) = min(100, max(51, pct(2)));

    if size(Y, 1) >= 20
        x_min = prctile(Y(:, 1), pct(1));
        x_max = prctile(Y(:, 1), pct(2));
        y_min = prctile(Y(:, 2), pct(1));
        y_max = prctile(Y(:, 2), pct(2));
    else
        x_min = min(Y(:, 1));
        x_max = max(Y(:, 1));
        y_min = min(Y(:, 2));
        y_max = max(Y(:, 2));
    end

    if isfield(cfg, 'uiAxisPadding') && ~isempty(cfg.uiAxisPadding)
        pad_frac = max(0.02, double(cfg.uiAxisPadding(1)));
    else
        pad_frac = 0.12;
    end

    x_span = x_max - x_min;
    y_span = y_max - y_min;

    if ~isfinite(x_span) || x_span < eps
        x_center = mean(Y(:, 1));
        x_lim = [x_center - 1, x_center + 1];
    else
        x_pad = pad_frac * x_span;
        x_lim = [x_min - x_pad, x_max + x_pad];
    end

    if ~isfinite(y_span) || y_span < eps
        y_center = mean(Y(:, 2));
        y_lim = [y_center - 1, y_center + 1];
    else
        y_pad = pad_frac * y_span;
        y_lim = [y_min - y_pad, y_max + y_pad];
    end

end

function frame_rgb_fixed = fix_video_frame_size(frame_rgb, target_size)

    % VideoWriter 要求所有帧尺寸完全一致。
    % getframe 在不同刷新时刻可能产生 1~数个像素的尺寸波动，因此这里做白底裁剪/补边。
    target_h = target_size(1);
    target_w = target_size(2);

    frame_rgb = uint8(frame_rgb);
    [h, w, c] = size(frame_rgb);

    if c ~= 3
        error('Captured frame must be an RGB image with 3 channels.');
    end

    if h == target_h && w == target_w
        frame_rgb_fixed = frame_rgb;
        return;
    end

    frame_rgb_fixed = uint8(255 * ones(target_h, target_w, 3));

    copy_h = min(h, target_h);
    copy_w = min(w, target_w);

    src_r0 = floor((h - copy_h) / 2) + 1;
    src_c0 = floor((w - copy_w) / 2) + 1;

    dst_r0 = floor((target_h - copy_h) / 2) + 1;
    dst_c0 = floor((target_w - copy_w) / 2) + 1;

    frame_rgb_fixed(dst_r0:dst_r0 + copy_h - 1, dst_c0:dst_c0 + copy_w - 1, :) = ...
        frame_rgb(src_r0:src_r0 + copy_h - 1, src_c0:src_c0 + copy_w - 1, :);

end


function [videoObj, video_path, write_video] = open_dynamic_tsne_video_writer(mp4_path, avi_path, cfg)

    write_video = true;

    try
        videoObj = VideoWriter(mp4_path, 'MPEG-4');
        video_path = mp4_path;
        videoObj.FrameRate = cfg.videoFrameRate;
        open(videoObj);
    catch ME1
        warning('dynamicTSNE:MP4VideoWriterFailed', ...
            'MPEG-4 VideoWriter failed: %s. Retry with Motion JPEG AVI.', ME1.message);
        try
            videoObj = VideoWriter(avi_path, 'Motion JPEG AVI');
            video_path = avi_path;
            videoObj.FrameRate = cfg.videoFrameRate;
            open(videoObj);
        catch ME2
            warning('dynamicTSNE:AVIVideoWriterFailed', ...
                'AVI VideoWriter also failed: %s. Video output is disabled.', ME2.message);
            videoObj = [];
            video_path = '';
            write_video = false;
        end
    end

end

function [x_lim, y_lim] = update_dynamic_tsne_axis_limits(x_lim, y_lim, Y, cfg)

    [new_x_lim, new_y_lim] = current_dynamic_tsne_axis_limits(Y, cfg);

    if isempty(x_lim)
        x_lim = new_x_lim;
    else
        x_lim = [min(x_lim(1), new_x_lim(1)), max(x_lim(2), new_x_lim(2))];
    end

    if isempty(y_lim)
        y_lim = new_y_lim;
    else
        y_lim = [min(y_lim(1), new_y_lim(1)), max(y_lim(2), new_y_lim(2))];
    end

end

function [X, x_stat] = preprocess_tsne_input_dynamic(X, cfg)

    X = double(X);
    X(~isfinite(X)) = 0;

    x_stat = struct();
    x_stat.raw_total_var = sum(var(X, 0, 1));
    x_stat.raw_mean_pairwise_std = mean(std(X, 0, 1));

    if isfield(cfg, 'standardizeBeforeTSNE') && cfg.standardizeBeforeTSNE
        mu = mean(X, 1);
        sigma = std(X, 0, 1);
        sigma(~isfinite(sigma) | sigma < eps) = 1;
        X = (X - mu) ./ sigma;
        X(~isfinite(X)) = 0;
    end

    x_stat.total_var = sum(var(X, 0, 1));
    x_stat.mean_pairwise_std = mean(std(X, 0, 1));

end

function [X, label_is_pred, selected_count, feature_dim, time_idx] = ...
    build_tsne_input(true_part, pred_part, taps, mode, cfg)

    num_samples = size(true_part, 1);
    pred_len = size(true_part, 2);
    feat_dim_all = size(true_part, 3);

    if isempty(cfg.timeIndex)
        time_idx = ceil(pred_len / 2);
    else
        time_idx = max(1, min(pred_len, cfg.timeIndex));
    end

    max_keep = min(num_samples, cfg.maxSamplesPerSNR);

    rng(cfg.randomSeed + pred_len + feat_dim_all + max_keep);

    switch lower(cfg.sampleSelectionMode)
        case 'first'
            selected_idx = 1:max_keep;
        case 'random'
            selected_idx = randperm(num_samples, max_keep);
        otherwise
            error('Unknown cfg.sampleSelectionMode=%s. Use random or first.', cfg.sampleSelectionMode);
    end

    feat_idx = get_feature_indices(taps, cfg.featureScope);

    true_sel = true_part(selected_idx, :, feat_idx);
    pred_sel = pred_part(selected_idx, :, feat_idx);

    switch lower(mode)

        case 'time_step'

            X_true = squeeze(true_sel(:, time_idx, :));
            X_pred = squeeze(pred_sel(:, time_idx, :));

            if isvector(X_true), X_true = X_true(:); end
            if isvector(X_pred), X_pred = X_pred(:); end

        case 'window_flatten'

            X_true = reshape(true_sel, max_keep, []);
            X_pred = reshape(pred_sel, max_keep, []);

        otherwise

            error('Unknown representation mode: %s.', mode);

    end

    X_true = double(X_true);
    X_pred = double(X_pred);

    % 联合 t-SNE：必须将 true/pred 拼在一起降维，否则二维坐标不可比较。
    X = [X_true; X_pred];
    label_is_pred = [false(size(X_true, 1), 1); true(size(X_pred, 1), 1)];

    selected_count = max_keep;
    feature_dim = size(X, 2);

    X(~isfinite(X)) = 0;

    if cfg.standardizeBeforeTSNE
        mu = mean(X, 1);
        sigma = std(X, 0, 1);
        sigma(~isfinite(sigma) | sigma < eps) = 1;
        X = (X - mu) ./ sigma;
        X(~isfinite(X)) = 0;
    end

end

function feat_idx = get_feature_indices(taps, featureScope)

    switch lower(featureScope)

        case 'all'
            feat_idx = 1:(4 * taps);

        case 'h_only'
            feat_idx = 1:(2 * taps);

        case 'gain_delay_doppler'
            feat_idx = 1:(4 * taps);

        otherwise
            error('Unknown cfg.featureScope=%s. Use all / h_only / gain_delay_doppler.', featureScope);

    end

end

function [Y, perplexity] = run_tsne_2d(X, cfg)

    n = size(X, 1);

    % t-SNE 要求 perplexity 明显小于样本数。
    perplexity = min(cfg.tsnePerplexity, max(5, floor((n - 1) / 3)));
    perplexity = max(1, min(perplexity, n - 1));

    if ~isempty(cfg.tsneNumPCAComponents)
        numPCA = min([cfg.tsneNumPCAComponents, size(X, 2), n - 1]);
    else
        numPCA = min(size(X, 2), n - 1);
    end

    Y = tsne(X, ...
        'NumDimensions', 2, ...
        'Perplexity', perplexity, ...
        'Exaggeration', cfg.tsneExaggeration, ...
        'NumPCAComponents', numPCA, ...
        'Standardize', false);

end

function draw_tsne_panel(Y, label_is_pred, snr_dB, mode, ~, time_idx, feature_dim, selected_count, show_legend)

    hold on;
    grid on;
    box on;

    Y_true = Y(~label_is_pred, :);
    Y_pred = Y(label_is_pred, :);

    try
        scatter(Y_pred(:, 1), Y_pred(:, 2), 16, [0.0000 0.4470 0.7410], 'filled', ...
            'MarkerFaceAlpha', 0.55, 'MarkerEdgeAlpha', 0.55, ...
            'DisplayName', 'Predicted CSI feature');
        scatter(Y_true(:, 1), Y_true(:, 2), 16, [0.8500 0.3250 0.0980], 'filled', ...
            'MarkerFaceAlpha', 0.55, 'MarkerEdgeAlpha', 0.55, ...
            'DisplayName', 'Ground-truth CSI feature');
    catch
        scatter(Y_pred(:, 1), Y_pred(:, 2), 16, [0.0000 0.4470 0.7410], 'filled', ...
            'DisplayName', 'Predicted CSI feature');
        scatter(Y_true(:, 1), Y_true(:, 2), 16, [0.8500 0.3250 0.0980], 'filled', ...
            'DisplayName', 'Ground-truth CSI feature');
    end

    % xlabel('t-SNE dimension 1');
    % ylabel('t-SNE dimension 2');

    switch lower(mode)
        case 'time_step'
            mode_name = sprintf('t = %d CSI feature', time_idx);
        case 'window_flatten'
            mode_name = 'prediction-window CSI feature';
        otherwise
            mode_name = mode;
    end

    title(sprintf('SNR = %g dB | %s\nfeature dim = %d, samples = %d', ...
        snr_dB, mode_name, feature_dim, selected_count), ...
        'Interpreter', 'none', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 11);

    if show_legend
        legend('Location', 'southeast', 'Interpreter', 'none', 'Box', 'on');
    end

    set(gca, ...
        'FontName', 'Times New Roman', ...
        'FontSize', 10.5, ...
        'LineWidth', 1.0);

end

function row = empty_tsne_summary_row()

    row = struct( ...
        'Condition', '', ...
        'Method', '', ...
        'FileName', '', ...
        'SNR_dB', NaN, ...
        'Mode', '', ...
        'FeatureScope', '', ...
        'TimeIndex', NaN, ...
        'NumSelectedSamples', NaN, ...
        'NumTSNEPoints', NaN, ...
        'FeatureDim', NaN, ...
        'Perplexity', NaN);

end

function condition_name = infer_condition_from_prediction_filename(file_name)

    nameUpper = upper(file_name);

    if contains(nameUpper, 'PHY_GATE_NORMAL')
        condition_name = 'normal';
    elseif contains(nameUpper, 'PHY_GATE_STEP_LOSS')
        condition_name = 'step_loss';
    elseif contains(nameUpper, 'PHY_GATE_BURST_SHADOW')
        condition_name = 'burst_shadow';
    elseif contains(nameUpper, 'PHY_GATE_FIRSTTAP_BLOCK')
        condition_name = 'firsttap_block';
    elseif contains(nameUpper, 'PHY_GATE_DOPPLER_JUMP')
        condition_name = 'doppler_jump';
    elseif contains(nameUpper, 'PHY_GATE_MIXED')
        condition_name = 'mixed';
    else
        condition_name = '';
    end

end

function [pred_h, true_h, applied_scale] = recover_physical_doppler_local( ...
    pred_h, true_h, K, configuredScale, policy, result)

    dop_idx = 3 * K + 1 : 4 * K;
    applied_scale = 1.0;

    switch lower(policy)

        case 'none'
            return;

        case 'force_unscale'
            applied_scale = choose_scale_local(configuredScale, result);
            pred_h(:, :, dop_idx) = pred_h(:, :, dop_idx) / applied_scale;
            true_h(:, :, dop_idx) = true_h(:, :, dop_idx) / applied_scale;
            return;

        case 'auto'
            if ~isempty(result.doppler_is_physical)
                if result.doppler_is_physical == 1
                    applied_scale = 1.0;
                    return;
                elseif result.doppler_is_physical == 0
                    applied_scale = choose_scale_local(configuredScale, result);
                    pred_h(:, :, dop_idx) = pred_h(:, :, dop_idx) / applied_scale;
                    true_h(:, :, dop_idx) = true_h(:, :, dop_idx) / applied_scale;
                    return;
                else
                    warning('Unknown doppler_is_physical=%g in %s. Fall back to numeric auto detection.', ...
                        result.doppler_is_physical, result.file_name);
                end
            end

            applied_scale = choose_scale_local(configuredScale, result);
            dop_true = true_h(:, :, dop_idx);
            abs_max = max(abs(dop_true(:)));

            if abs_max > 0.05
                pred_h(:, :, dop_idx) = pred_h(:, :, dop_idx) / applied_scale;
                true_h(:, :, dop_idx) = true_h(:, :, dop_idx) / applied_scale;
            else
                applied_scale = 1.0;
            end
            return;

        otherwise
            error('Unknown dopplerScalePolicy: %s', policy);

    end

end

function scale = choose_scale_local(configuredScale, result)

    if ~isempty(result.doppler_feature_scale) && result.doppler_feature_scale > 0
        scale = result.doppler_feature_scale;
    else
        scale = configuredScale;
    end

end

function safe_name = make_safe_filename_local(name_in)

    safe_name = char(name_in);
    safe_name = regexprep(safe_name, '[^\w\d\-]+', '_');
    safe_name = regexprep(safe_name, '_+', '_');

    if numel(safe_name) > 100
        safe_name = safe_name(1:100);
    end

end
