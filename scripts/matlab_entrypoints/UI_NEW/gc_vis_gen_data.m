clc;
clear;
close all;
rng(20260513, 'twister');

% 依赖检查
if exist('nrTDLChannel', 'class') ~= 8
    error(['未检测到 nrTDLChannel。请确认 MATLAB 已安装并授权 5G Toolbox，', ...
        '建议使用 MATLAB R2021a 或更新版本。']);
end

%% ===================== 1. 参数配置 =====================
cfg = struct();

% 信道列表
cfg.AllProfiles = {'NTN-TDL-A', 'NTN-TDL-B', 'NTN-TDL-C', 'NTN-TDL-D'};

% 终端输入选择信道类型
cfg.ProfileList = choose_profile_list_terminal(cfg.AllProfiles);

% 输出目录
cfg.SaveFolder = 'clean_dataset_output';
cfg.VideoFolder = 'clean_visualization_output';

% OTFS 网格参数（只用于 delay/Doppler 可视化）
cfg.N = 32;                         % Doppler 维度
cfg.M = 32;                         % Delay 维度
cfg.DeltaF = 15e3;                  % 子载波间隔
cfg.SampleRate = 1.25 * cfg.M * cfg.DeltaF;
cfg.DelayRes = 1 / (cfg.M * cfg.DeltaF);
cfg.FrameDur = cfg.N / cfg.DeltaF;
cfg.DopplerResHz = 1 / cfg.FrameDur;

% NTN-TDL 信道参数
cfg.DelaySpread = 300e-9;
cfg.SmallScaleDopplerHz = 5;
cfg.PathDelayOffset = 0;

% 数据集窗口
cfg.InputWindow = 96;
cfg.TargetWindow = 24;
cfg.Stride = 12;

% 数据集规模
cfg.NumScenesPerProfile = 50;
cfg.TotalStepsPerScene = 1000;

% noisy CSI 观测噪声
cfg.ObsSNRdB = 0;
cfg.DelayObsNoiseStdBin = 0.05;
cfg.DopplerObsNoiseStdBin = 0.05;

% train / val / test 划分
cfg.TrainRatio = 0.8;
cfg.ValRatio = 0.1;
cfg.TestRatio = 0.1;

% 可视化设置
cfg.EnableVisualization = true;
cfg.RecordVideo = true;
cfg.MaxVideoFrames = 5000;
cfg.VideoFrameRate = 3;
cfg.TxSymbolsPerFrame = 12;
cfg.QamOrder = 4;                   % QPSK
cfg.EnableUiPublish = true;
cfg.UiHost = '127.0.0.1';
cfg.UiPort = 65436;
cfg.UiFramePause = 0.05;
cfg.EnableUiAnomalyControl = true;
cfg.UiAnomalyControlFile = fullfile(fileparts(mfilename('fullpath')), 'runtime', 'channel_anomaly_control.json');
cfg.UiAnomalyAttenuationDbDefault = 12;

% DD 图异常增强显示：固定参考功率归一化 + 可选结构型异常
% 说明：'frame' 会逐帧归一化，适合看路径结构；'absolute' 使用固定参考功率，适合展示异常衰减。
cfg.DDGridNormalizeMode = 'frame';         % UI图1使用 frame-normalized 形状，再用dB压缩缩放高度
cfg.DDGridVisualGain = 1.0;                % frame模式下保持1即可，避免异常时直接趴平
cfg.EnableDDStructuralAnomaly = true;      % 比赛展示增强：异常时叠加首径阻断 + Doppler 偏移
cfg.UiAnomalyFirstTapExtraDb = 2;          % 首径额外阻断 dB，建议 2~4，过大会导致曲面消失
cfg.UiAnomalyDopplerJumpBins = 2;          % Doppler 方向偏移 bin 数
cfg.DDDisplayDbMin = -24;                 % dB压缩显示下限：低于该值仍保留最小可见高度
cfg.DDDisplayDbMax = 3;                   % dB压缩显示上限
cfg.DDFloorScale = 0.18;                  % DD曲面最低显示高度，避免异常后完全看不见
cfg.DDAnomalyExtraScale = 0.92;           % 异常时额外轻微压低，但不让曲面消失

% 创建输出目录
if ~exist(cfg.SaveFolder, 'dir')
    mkdir(cfg.SaveFolder);
end
if cfg.RecordVideo && ~exist(cfg.VideoFolder, 'dir')
    mkdir(cfg.VideoFolder);
end

fprintf('\n=== 数据集生成 + 采集过程可视化 ===\n');
fprintf('选择的信道类型:\n');
for iProfile = 1:numel(cfg.ProfileList)
    fprintf('  - %s\n', cfg.ProfileList{iProfile});
end
fprintf('数据集输出目录 : %s\n', cfg.SaveFolder);
fprintf('视频输出目录   : %s\n\n', cfg.VideoFolder);

%% ===================== 2. 逐个信道生成数据集 =====================
for ip = 1:numel(cfg.ProfileList)
    profileName = cfg.ProfileList{ip};

    XTrainCell = {};
    YTrainCell = {};
    MetaTrainCell = {};
    XValCell = {};
    YValCell = {};
    MetaValCell = {};
    XTestCell = {};
    YTestCell = {};
    MetaTestCell = {};

    firstSceneForUi = [];

    fprintf('\n--- 开始生成 %s 数据集 ---\n', profileName);

    for sceneIdx = 1:cfg.NumScenesPerProfile
        seed = 20260513 + 10000 * ip + sceneIdx;
        rng(seed, 'twister');

        scene = generate_one_scene(profileName, sceneIdx, seed, cfg);
        [XScene, YScene, MetaScene] = cut_scene_to_xy(scene, cfg);

        if sceneIdx == 1
            firstSceneForUi = scene;
        end

        splitRatio = sceneIdx / cfg.NumScenesPerProfile;
        if splitRatio <= cfg.TrainRatio
            XTrainCell{end+1, 1} = XScene;
            YTrainCell{end+1, 1} = YScene;
            MetaTrainCell{end+1, 1} = MetaScene;
        elseif splitRatio <= cfg.TrainRatio + cfg.ValRatio
            XValCell{end+1, 1} = XScene;
            YValCell{end+1, 1} = YScene;
            MetaValCell{end+1, 1} = MetaScene;
        else
            XTestCell{end+1, 1} = XScene;
            YTestCell{end+1, 1} = YScene;
            MetaTestCell{end+1, 1} = MetaScene;
        end

        fprintf('scene %03d/%03d | paths=%d | samples=%d\n', ...
            sceneIdx, cfg.NumScenesPerProfile, scene.numPaths, size(XScene, 1));
    end

    X_train = cat_or_empty(XTrainCell);
    Y_train = cat_or_empty(YTrainCell);
    Meta_train = cat_meta_or_empty(MetaTrainCell);

    X_val = cat_or_empty(XValCell);
    Y_val = cat_or_empty(YValCell);
    Meta_val = cat_meta_or_empty(MetaValCell);

    X_test = cat_or_empty(XTestCell);
    Y_test = cat_or_empty(YTestCell);
    Meta_test = cat_meta_or_empty(MetaTestCell);

    featureDim = size(X_train, 3);
    numPaths = featureDim / 4;

    saveName = fullfile(cfg.SaveFolder, ...
        sprintf('dataset_%s_clean.mat', strrep(profileName, '-', '_')));

    save(saveName, ...
        'X_train', 'Y_train', 'Meta_train', ...
        'X_val', 'Y_val', 'Meta_val', ...
        'X_test', 'Y_test', 'Meta_test', ...
        'profileName', 'featureDim', 'numPaths', 'cfg', ...
        '-v7.3');

    % ================== summary ==================
    fprintf('\n=== 数据集信息 ===\n');
    fprintf('信道类型          : %s\n', profileName);
    fprintf('路径数            : %d\n', numPaths);
    fprintf('特征维度          : %d\n', featureDim);
    fprintf('训练样本数        : %d\n', size(X_train,1));
    fprintf('验证样本数        : %d\n', size(X_val,1));
    fprintf('测试样本数        : %d\n', size(X_test,1));
    fprintf('OTFS Doppler 维度  : %d\n', cfg.N);
    fprintf('OTFS Delay 维度    : %d\n', cfg.M);
    fprintf('子载波间隔 Δf      : %.1f kHz\n', cfg.DeltaF/1e3);
    fprintf('采样率 SampleRate  : %.1f kHz\n', cfg.SampleRate/1e3);
    fprintf('Delay 分辨率       : %.3e s\n', cfg.DelayRes);
    fprintf('Doppler 分辨率     : %.3f Hz\n', cfg.DopplerResHz);
    fprintf('数据集保存路径     : %s\n', saveName);
    fprintf('===============================\n\n');

    % 可视化
    if cfg.EnableVisualization && ~isempty(firstSceneForUi)
        videoFile = fullfile(cfg.VideoFolder, ...
            sprintf('acquisition_%s_clean.mp4', strrep(profileName, '-', '_')));
        visualize_acquisition_process(firstSceneForUi, cfg, videoFile);
    end
end

fprintf('\n完成。干净版数据集和可视化结果已经生成。\n');

%% ===================== 子函数 =====================
function profileList = choose_profile_list_terminal(allProfiles)
fprintf('\n请选择信道类型：\n');
fprintf('  1 -> NTN-TDL-A\n');
fprintf('  2 -> NTN-TDL-B\n');
fprintf('  3 -> NTN-TDL-C\n');
fprintf('  4 -> NTN-TDL-D\n');
fprintf('  5 -> 全部\n');

choice = strtrim(getenv('GC_PROFILE_CHOICE'));
if isempty(choice)
    choice = strtrim(getenv('GC_PROFILE'));
end

if isempty(choice) && is_interactive_matlab_session()
    choice = input('请输入序号，默认3：', 's');
elseif isempty(choice)
    choice = '3';
    fprintf('后台/批处理模式未提供输入，默认使用 3 -> NTN-TDL-C。\n');
else
    fprintf('使用环境变量指定信道类型: %s\n', choice);
end

choice = strtrim(choice);
if isempty(choice)
    choice = '3';
end

choiceLower = lower(choice);
switch choiceLower
    case {'1', 'a', 'ntn-tdl-a', 'ntn_tdl_a'}
        profileList = allProfiles(1);
    case {'2', 'b', 'ntn-tdl-b', 'ntn_tdl_b'}
        profileList = allProfiles(2);
    case {'3', 'c', 'ntn-tdl-c', 'ntn_tdl_c'}
        profileList = allProfiles(3);
    case {'4', 'd', 'ntn-tdl-d', 'ntn_tdl_d'}
        profileList = allProfiles(4);
    case {'5', 'all', '全部'}
        profileList = allProfiles;
    otherwise
        warning('输入无效，默认使用 NTN-TDL-C。');
        profileList = allProfiles(3);
end
end

function tf = is_interactive_matlab_session()
tf = usejava('desktop') && feature('ShowFigureWindows');
end

function scene = generate_one_scene(profileName, sceneIdx, seed, cfg)
%GENERATE_ONE_SCENE 生成一段连续的 clean/noisy CSI 采集场景。

    ch = create_ntn_tdl_channel(profileName, cfg, seed);

    inputSig = complex(ones(cfg.TotalStepsPerScene, 1), 0);
    reset(ch);
    [~, pathGains] = ch(inputSig);

    pathGains = squeeze(pathGains);
    if isvector(pathGains)
        pathGains = pathGains(:);
    end

    T = size(pathGains, 1);
    K = size(pathGains, 2);

    delayBinsOne = ch.PathDelays(:).' ./ cfg.DelayRes;
    delayBins = repmat(delayBinsOne, T, 1);

    % 生成平滑变化的 residual Doppler，用于展示路径随时间变化。
    dopplerHz = zeros(T, K);
    tapOffsetHz = 0.2 * cfg.SmallScaleDopplerHz * randn(1, K);
    for k = 1:K
        for t = 2:T
            dopplerHz(t, k) = 0.98 * dopplerHz(t - 1, k) + ...
                0.05 * cfg.SmallScaleDopplerHz * randn();
        end
    end
    dopplerHz = dopplerHz + repmat(tapOffsetHz, T, 1);
    dopplerBins = dopplerHz ./ cfg.DopplerResHz;

    obsGains = add_complex_noise(pathGains, cfg.ObsSNRdB);
    obsDelayBins = delayBins + cfg.DelayObsNoiseStdBin * randn(size(delayBins));
    obsDopplerBins = dopplerBins + cfg.DopplerObsNoiseStdBin * randn(size(dopplerBins));

    txIdx = randi([0, cfg.QamOrder - 1], cfg.TotalStepsPerScene * cfg.TxSymbolsPerFrame, 1);
    txQamSymbols = exp(1i * (pi / 4 + pi / 2 * txIdx));

    scene = struct();
    scene.profileName = profileName;
    scene.sceneIdx = sceneIdx;
    scene.seed = seed;
    scene.numPaths = K;
    scene.txQamSymbols = txQamSymbols;
    scene.cleanGains = pathGains;
    scene.obsGains = obsGains;
    scene.delayBins = delayBins;
    scene.obsDelayBins = obsDelayBins;
    scene.dopplerBins = dopplerBins;
    scene.obsDopplerBins = obsDopplerBins;
    scene.cleanFeat = pack_features(pathGains, delayBins, dopplerBins);
    scene.obsFeat = pack_features(obsGains, obsDelayBins, obsDopplerBins);
end

function ch = create_ntn_tdl_channel(profileName, cfg, seed)
%CREATE_NTN_TDL_CHANNEL 创建 3GPP NTN-TDL-A/B/C/D 信道对象。

    ch = nrTDLChannel;
    ch.DelayProfile = 'Custom';
    ch.MaximumDopplerShift = cfg.SmallScaleDopplerHz;

    switch upper(profileName)
        case 'NTN-TDL-A'
            ch.FadingDistribution = 'Rayleigh';
            ch.PathDelays = [0, 1.0811, 2.8416] * cfg.DelaySpread + cfg.PathDelayOffset;
            ch.AveragePathGains = [0, -4.675, -6.482];
        case 'NTN-TDL-B'
            ch.FadingDistribution = 'Rayleigh';
            ch.PathDelays = [0, 0.7249, 0.7410, 5.7392] * cfg.DelaySpread + cfg.PathDelayOffset;
            ch.AveragePathGains = [0, -1.973, -4.332, -11.914];
        case 'NTN-TDL-C'
            ch.FadingDistribution = 'Rician';
            ch.PathDelays = [0, 0, 14.8124] * cfg.DelaySpread + cfg.PathDelayOffset;
            ch.AveragePathGains = [-0.394, -10.618, -23.373];
            ch.KFactorFirstTap = 10.224;
        case 'NTN-TDL-D'
            ch.FadingDistribution = 'Rician';
            ch.PathDelays = [0, 0, 0.5596, 7.3340] * cfg.DelaySpread + cfg.PathDelayOffset;
            ch.AveragePathGains = [-0.284, -11.991, -9.887, -16.771];
            ch.KFactorFirstTap = 11.707;
        otherwise
            error('Unsupported profile: %s', profileName);
    end

    ch.SampleRate = cfg.SampleRate;
    ch.MIMOCorrelation = 'Low';
    ch.Polarization = 'Co-Polar';
    ch.NumTransmitAntennas = 1;
    ch.NumReceiveAntennas = 1;
    ch.RandomStream = 'mt19937ar with seed';
    ch.Seed = seed;
end

function feat = pack_features(gains, delayBins, dopplerBins)
%PACK_FEATURES 将路径参数打包成 [T, 4K] 特征矩阵。
% 每条路径格式：[real(h), imag(h), delayBin, dopplerBin]。

    [T, K] = size(gains);
    feat = zeros(T, 4 * K);
    for k = 1:K
        base = 4 * (k - 1);
        feat(:, base + 1) = real(gains(:, k));
        feat(:, base + 2) = imag(gains(:, k));
        feat(:, base + 3) = delayBins(:, k);
        feat(:, base + 4) = dopplerBins(:, k);
    end
end

function y = add_complex_noise(x, snr_dB)
%ADD_COMPLEX_NOISE 对复路径增益加入复高斯观测噪声。

    snrLin = 10^(snr_dB / 10);
    signalPower = mean(abs(x(:)).^2);
    noisePower = signalPower / snrLin;
    noise = sqrt(noisePower / 2) * (randn(size(x)) + 1i * randn(size(x)));
    y = x + noise;
end

function [X, Y, Meta] = cut_scene_to_xy(scene, cfg)
%CUT_SCENE_TO_XY 将连续采集过程切成监督学习数据集。
% X = 过去 InputWindow 帧 noisy CSI，Y = 未来 TargetWindow 帧 clean CSI。

    obsFeat = scene.obsFeat;
    cleanFeat = scene.cleanFeat;
    T = size(obsFeat, 1);
    D = size(obsFeat, 2);
    L = cfg.InputWindow;
    P = cfg.TargetWindow;

    numSamples = floor((T - L - P) / cfg.Stride) + 1;
    if numSamples <= 0
        error('TotalStepsPerScene must be at least InputWindow + TargetWindow.');
    end

    X = zeros(numSamples, L, D);
    Y = zeros(numSamples, P, D);
    Meta = repmat(struct( ...
        'profileName', scene.profileName, ...
        'sceneIdx', scene.sceneIdx, ...
        'seed', scene.seed, ...
        'inputStart', 0, ...
        'inputEnd', 0, ...
        'targetStart', 0, ...
        'targetEnd', 0), numSamples, 1);

    for i = 1:numSamples
        t0 = 1 + (i - 1) * cfg.Stride;
        idxX = t0:t0 + L - 1;
        idxY = t0 + L:t0 + L + P - 1;
        X(i, :, :) = obsFeat(idxX, :);
        Y(i, :, :) = cleanFeat(idxY, :);
        Meta(i).inputStart = idxX(1);
        Meta(i).inputEnd = idxX(end);
        Meta(i).targetStart = idxY(1);
        Meta(i).targetEnd = idxY(end);
    end
end

function visualize_acquisition_process(scene, cfg, videoFile)
%VISUALIZE_ACQUISITION_PROCESS 用 2 x 3 六张图展示采集过程。

    T = size(scene.obsFeat, 1);
    frameIdx = unique(round(linspace(1, T, min(T, cfg.MaxVideoFrames))));
    uiAnomalyScale = ones(T, 1);
    lastUiAnomalyEnabled = false;
    lastUiAnomalyAttenuationDb = cfg.UiAnomalyAttenuationDbDefault;

    writer = [];
    if cfg.RecordVideo
        writer = VideoWriter(videoFile, 'MPEG-4');
        writer.FrameRate = cfg.VideoFrameRate;
        open(writer);
    end

    uiUdp = [];
    if isfield(cfg, 'EnableUiPublish') && cfg.EnableUiPublish
        uiUdp = udpport('datagram', 'IPV4');
        try
            uiUdp.OutputDatagramSize = 65507;
        catch
        end
        fprintf('[GC-UI] Publishing channel subplots to UDP %d\n', cfg.UiPort);
    end

    fig = figure('Color', 'w', 'Position', [80, 60, 1500, 880]);
    for ii = 1:numel(frameIdx)
        t = frameIdx(ii);
        uiAnomaly = read_ui_channel_anomaly_control(cfg);
        if uiAnomaly.enabled
            uiAnomalyScale(t) = 10^(-uiAnomaly.attenuationDb / 20);
        else
            uiAnomalyScale(t) = 1;
        end
        if uiAnomaly.enabled ~= lastUiAnomalyEnabled || abs(uiAnomaly.attenuationDb - lastUiAnomalyAttenuationDb) > eps
            fprintf('[GC-UI] Manual channel anomaly %s | attenuation = %.1f dB | frame = %d\n', ...
                ternary_text(uiAnomaly.enabled, 'enabled', 'disabled'), uiAnomaly.attenuationDb, t);
            lastUiAnomalyEnabled = uiAnomaly.enabled;
            lastUiAnomalyAttenuationDb = uiAnomaly.attenuationDb;
        end
        sceneFrame = apply_ui_power_anomaly_to_scene(scene, uiAnomalyScale, t, uiAnomaly, cfg);
        clf(fig);

        subplot(2, 3, 1);
        plot_tx_symbol_block(sceneFrame, cfg, t);

        subplot(2, 3, 2);
        plot_3d_delay_doppler_channel(sceneFrame, cfg, t);

        subplot(2, 3, 3);
        plot_current_path_gain(sceneFrame, t);

        subplot(2, 3, 4);
        plot_csi_observation_buffer(sceneFrame, t);

        subplot(2, 3, 5);
        plot_noisy_feature_buffer(sceneFrame, t);

        subplot(2, 3, 6);
        draw_xy_window(t, T, cfg);

        sgtitle(sprintf('TX QAM and Online CSI Dataset Acquisition | %s | SNR = %.1f dB', ...
            scene.profileName, cfg.ObsSNRdB), ...
            'FontWeight', 'bold', 'FontSize', 14);
        drawnow;

        if cfg.RecordVideo
            writeVideo(writer, getframe(fig));
        end
        if ~isempty(uiUdp)
            publish_ui_channel_subplots(sceneFrame, cfg, t, uiUdp);
            pause(cfg.UiFramePause);
        end
    end

    if cfg.RecordVideo
        close(writer);
        fprintf('已保存可视化视频:\n  %s\n', videoFile);
    end
end

function uiAnomaly = read_ui_channel_anomaly_control(cfg)
%READ_UI_CHANNEL_ANOMALY_CONTROL Poll the Python UI control file.
    uiAnomaly = struct();
    uiAnomaly.enabled = false;
    uiAnomaly.attenuationDb = cfg.UiAnomalyAttenuationDbDefault;

    if ~isfield(cfg, 'EnableUiAnomalyControl') || ~cfg.EnableUiAnomalyControl
        return;
    end
    if ~isfield(cfg, 'UiAnomalyControlFile') || isempty(cfg.UiAnomalyControlFile)
        return;
    end
    if exist(cfg.UiAnomalyControlFile, 'file') ~= 2
        return;
    end

    try
        rawText = fileread(cfg.UiAnomalyControlFile);
        if isempty(strtrim(rawText))
            return;
        end
        data = jsondecode(rawText);
        if isfield(data, 'enabled')
            uiAnomaly.enabled = logical(data.enabled);
        end
        if isfield(data, 'attenuationDb') && isfinite(double(data.attenuationDb))
            uiAnomaly.attenuationDb = max(0, double(data.attenuationDb));
        end
    catch
        uiAnomaly.enabled = false;
        uiAnomaly.attenuationDb = cfg.UiAnomalyAttenuationDbDefault;
    end
end

function sceneOut = apply_ui_power_anomaly_to_scene(scene, anomalyScale, t, uiAnomaly, cfg)
%APPLY_UI_POWER_ANOMALY_TO_SCENE Apply manual anomaly for UI display.
% 基础异常：整体功率衰减；
% 展示增强：仅对异常帧叠加首径阻断 + Doppler 偏移，使 DD 图一眼可见变化。

    sceneOut = scene;

    T = size(scene.obsGains, 1);
    idx = 1:min(max(1, t), T);

    scale = anomalyScale(idx);
    K = size(scene.obsGains, 2);
    scaleMat = repmat(scale(:), 1, K);

    % ===== 1) 基础 power attenuation：整体链路功率衰减 =====
    sceneOut.cleanGains(idx, :) = scene.cleanGains(idx, :) .* scaleMat;
    sceneOut.obsGains(idx, :)   = scene.obsGains(idx, :) .* scaleMat;

    % 保持原始 delay/doppler，后面仅对异常帧按需增强
    sceneOut.dopplerBins = scene.dopplerBins;
    sceneOut.obsDopplerBins = scene.obsDopplerBins;

    % ===== 2) 展示增强：异常帧出现“主径塌陷 + Doppler 跳变” =====
    anomIdx = idx(scale(:) < 0.999);
    if ~isempty(anomIdx) && isfield(cfg, 'EnableDDStructuralAnomaly') && cfg.EnableDDStructuralAnomaly

        % 2.1 首径额外阻断：让 DD 主峰明显塌陷
        if K >= 1
            if isfield(cfg, 'UiAnomalyFirstTapExtraDb')
                firstTapExtraDb = cfg.UiAnomalyFirstTapExtraDb;
            else
                firstTapExtraDb = 10;
            end
            firstTapScale = 10^(-firstTapExtraDb / 20);
            sceneOut.cleanGains(anomIdx, 1) = sceneOut.cleanGains(anomIdx, 1) .* firstTapScale;
            sceneOut.obsGains(anomIdx, 1)   = sceneOut.obsGains(anomIdx, 1) .* firstTapScale;
        end

        % 2.2 Doppler 偏移：让 DD 峰在 Doppler 方向发生明显跳变
        if isfield(cfg, 'UiAnomalyDopplerJumpBins')
            dopplerJumpBins = cfg.UiAnomalyDopplerJumpBins;
        else
            dopplerJumpBins = 4;
        end
        sceneOut.dopplerBins(anomIdx, :)    = scene.dopplerBins(anomIdx, :) + dopplerJumpBins;
        sceneOut.obsDopplerBins(anomIdx, :) = scene.obsDopplerBins(anomIdx, :) + dopplerJumpBins;
    end

    % ===== 3) 重新打包特征 =====
    sceneOut.cleanFeat = pack_features(sceneOut.cleanGains, scene.delayBins, sceneOut.dopplerBins);
    sceneOut.obsFeat   = pack_features(sceneOut.obsGains, scene.obsDelayBins, sceneOut.obsDopplerBins);

    % ===== 4) UI 状态字段 =====
    sceneOut.uiAnomalyEnabled = uiAnomaly.enabled;
    sceneOut.uiAnomalyAttenuationDb = uiAnomaly.attenuationDb;
    sceneOut.uiAnomalyScale = anomalyScale(:);
    sceneOut.uiFeatureReference = max(abs(scene.obsFeat), [], 1);

    % 固定参考功率：用于 DD 图 absolute normalization，避免每帧归一化抵消异常衰减
    refPower = max(sum(abs(scene.obsGains).^2, 2));
    if ~isfinite(refPower) || refPower <= eps
        refPower = 1;
    end
    sceneOut.uiDDReferencePower = refPower;

    % 中图相对功率参考：用原始非异常场景的中位功率作为 0 dB 基准
    basePowerDb = 10 * log10(sum(abs(scene.obsGains).^2, 2) + eps);
    basePowerDb = basePowerDb(isfinite(basePowerDb));
    if isempty(basePowerDb)
        sceneOut.uiPowerReferenceDb = 0;
    else
        sceneOut.uiPowerReferenceDb = median(basePowerDb);
    end
end

function scale = get_ui_anomaly_display_scale(scene, t)
%GET_UI_ANOMALY_DISPLAY_SCALE Lower normalized DD surface height during manual attenuation.
    scale = 1;
    if isfield(scene, 'uiAnomalyScale') && numel(scene.uiAnomalyScale) >= t
        scale = max(0, min(1, double(scene.uiAnomalyScale(t))));
    end
end

function out = ternary_text(cond, trueText, falseText)
    if cond
        out = trueText;
    else
        out = falseText;
    end
end

function plot_tx_symbol_block(scene, cfg, t)
%PLOT_TX_SYMBOL_BLOCK 显示当前采集帧对应的 QPSK 发射符号块。
    txSym = scene.txQamSymbols(:);
    txSym = scene.txQamSymbols(:);
    numSym = numel(txSym);
    blockLen = min(cfg.TxSymbolsPerFrame, numSym);
    startIdx = mod((t - 1) * blockLen, numSym) + 1;
    idx = mod((startIdx:startIdx + blockLen - 1) - 1, numSym) + 1;
    blockSym = txSym(idx);

    idealSym = exp(1i * (pi / 4 + pi / 2 * (0:cfg.QamOrder - 1))).';
    symbolClass = zeros(blockLen, 1);
    for i = 1:blockLen
        [~, symbolClass(i)] = min(abs(blockSym(i) - idealSym));
    end

    stem(1:blockLen, symbolClass - 1, 'filled', 'LineWidth', 1.2);
    hold on;
    ylim([-0.4, cfg.QamOrder - 0.6]);
    yticks(0:cfg.QamOrder - 1);
    xlabel('Symbol position inside current block');
    ylabel('QAM symbol index');
    title('TX Symbol Block Used by Current Frame');
    text(0.03, 0.93, sprintf('Frame %d uses TX symbols %d-%d', ...
        t, idx(1), idx(end)), ...
        'Units', 'normalized', 'FontSize', 9, 'FontWeight', 'bold');
end

function plot_3d_delay_doppler_channel(scene, cfg, t)
%PLOT_3D_DELAY_DOPPLER_CHANNEL 显示三维 Delay-Doppler 信道强度曲面。
% 说明：图1不直接画负dB，而是用 dB 压缩后的 0~1 高度显示。
% 这样异常 -12 dB 时，曲面会明显降低，但不会完全趴平消失。

    [Z, powerDropDb, displayScale] = build_visible_dd_display_grid(scene, cfg, t);

    [MDelay, NDopp] = size(Z);
    [dopplerGrid, delayGrid] = meshgrid(1:NDopp, 1:MDelay);

    fineN = max(80, 5 * NDopp);
    fineM = max(80, 5 * MDelay);
    fineDoppler = linspace(1, NDopp, fineN);
    fineDelay = linspace(1, MDelay, fineM);
    [dopplerFineGrid, delayFineGrid] = meshgrid(fineDoppler, fineDelay);
    Zfine = interp2(dopplerGrid, delayGrid, Z, dopplerFineGrid, delayFineGrid, 'cubic');
    Zfine = max(0, min(1, Zfine));

    surf(dopplerFineGrid, delayFineGrid, Zfine, ...
        'EdgeColor', 'none', ...
        'FaceColor', 'interp');
    view(42, 28);
    axis tight;
    grid on; box on;
    xlabel('Doppler index');
    ylabel('Delay index');
    zlabel('Display intensity');
    title('3D Delay-Doppler Signal Intensity');
    ddCmap = interp1( ...
        [0, 0.32, 0.68, 1], ...
        [0.08 0.10 0.32; ...
         0.05 0.38 0.70; ...
         0.15 0.78 0.72; ...
         0.98 0.86 0.22], ...
        linspace(0, 1, 256));
    colormap(gca, ddCmap);
    shading interp;
    clim([0, 1]);
    zlim([0, 1.05]);
    xlim([0.5, NDopp + 0.5]);
    ylim([0.5, MDelay + 0.5]);
    grid on;
    box on;

    if isfield(scene, 'uiAnomalyEnabled') && scene.uiAnomalyEnabled
        statusText = sprintf('ANOMALY ON | drop %.1f dB | scale %.2f', powerDropDb, displayScale);
        statusColor = [0.80 0.05 0.05];
    else
        statusText = sprintf('NORMAL | drop %.1f dB | scale %.2f', powerDropDb, displayScale);
        statusColor = [0.02 0.08 0.18];
    end

    text(0.03, 0.95, sprintf('Equivalent TDL DD taps | Frame = %d', t), ...
        'Units', 'normalized', ...
        'Color', [0.02 0.08 0.18], ...
        'FontWeight', 'bold', ...
        'FontSize', 10);
    text(0.03, 0.88, statusText, ...
        'Units', 'normalized', ...
        'Color', statusColor, ...
        'FontWeight', 'bold', ...
        'FontSize', 9);
    text(0.03, 0.81, 'dB-compressed DD surface: keeps anomaly visible', ...
        'Units', 'normalized', ...
        'Color', [0.02 0.08 0.18], ...
        'FontWeight', 'bold', ...
        'FontSize', 9);
end

function [ddGridDisplay, powerDropDb, displayScale] = build_visible_dd_display_grid(scene, cfg, t)
%BUILD_VISIBLE_DD_DISPLAY_GRID 构造给MATLAB图1和Python UI使用的DD显示网格。
%
% 设计目的：
%   1) DD形状用 frame-normalized grid 保持峰/路径结构可见；
%   2) 当前帧相对功率下降 powerDropDb 用 dB 压缩映射到显示高度；
%   3) 异常 -12 dB 时曲面降低到约0.35~0.45，而不是直接变成0.01贴地。

    ddShapeGrid = build_dd_signal_intensity_grid( ...
        scene.obsGains(t, :), ...
        scene.obsDelayBins(t, :), ...
        scene.obsDopplerBins(t, :), ...
        cfg.M, cfg.N, ...
        'frame', [], 1.0);
    ddShapeGrid = max(0, min(1, ddShapeGrid));

    currentPowerDb = 10 * log10(sum(abs(scene.obsGains(t, :)).^2) + eps);

    if isfield(scene, 'uiPowerReferenceDb') && isfinite(scene.uiPowerReferenceDb)
        powerRefDb = scene.uiPowerReferenceDb;
    else
        basePowerDb = 10 * log10(sum(abs(scene.obsGains).^2, 2) + eps);
        basePowerDb = basePowerDb(isfinite(basePowerDb));
        if isempty(basePowerDb)
            powerRefDb = currentPowerDb;
        else
            powerRefDb = median(basePowerDb);
        end
    end

    powerDropDb = currentPowerDb - powerRefDb;
    displayScale = map_power_drop_db_to_display_scale(powerDropDb, cfg);

    if isfield(scene, 'uiAnomalyEnabled') && scene.uiAnomalyEnabled
        if isfield(cfg, 'DDAnomalyExtraScale') && isfinite(cfg.DDAnomalyExtraScale)
            extraScale = cfg.DDAnomalyExtraScale;
        else
            extraScale = 0.92;
        end
        if isfield(cfg, 'DDFloorScale') && isfinite(cfg.DDFloorScale)
            floorScale = cfg.DDFloorScale;
        else
            floorScale = 0.18;
        end
        displayScale = max(floorScale, extraScale * displayScale);
    end

    ddGridDisplay = ddShapeGrid .* displayScale;
    ddGridDisplay = max(0, min(1, ddGridDisplay));
end

function displayScale = map_power_drop_db_to_display_scale(powerDropDb, cfg)
%MAP_POWER_DROP_DB_TO_DISPLAY_SCALE 将相对dB功率下降映射为0~1的显示高度。
% 0 dB -> 接近 1；-12 dB -> 约 0.45；-24 dB及以下 -> 保留 floorScale。

    if isfield(cfg, 'DDDisplayDbMin') && isfinite(cfg.DDDisplayDbMin)
        dbMin = cfg.DDDisplayDbMin;
    else
        dbMin = -24;
    end
    if isfield(cfg, 'DDDisplayDbMax') && isfinite(cfg.DDDisplayDbMax)
        dbMax = cfg.DDDisplayDbMax;
    else
        dbMax = 3;
    end
    if isfield(cfg, 'DDFloorScale') && isfinite(cfg.DDFloorScale)
        floorScale = cfg.DDFloorScale;
    else
        floorScale = 0.18;
    end

    if dbMax <= dbMin
        dbMin = -24;
        dbMax = 3;
    end

    s = (powerDropDb - dbMin) / (dbMax - dbMin);
    s = max(0, min(1, s));
    displayScale = floorScale + (1 - floorScale) * s;
end

function gridOut = build_dd_signal_intensity_grid(gains, delayBins, dopplerBins, M, N, normalizeMode, refPower, visualGain)
%BUILD_DD_SIGNAL_INTENSITY_GRID 构造 DD 域强度图。
%
% normalizeMode:
%   'frame'    : 每帧归一化，适合看路径结构，但会掩盖整体功率衰减；
%   'absolute' : 固定参考功率归一化，适合展示异常衰减。

    if nargin < 6 || isempty(normalizeMode)
        normalizeMode = 'frame';
    end
    if nargin < 7 || isempty(refPower)
        refPower = [];
    end
    if nargin < 8 || isempty(visualGain)
        visualGain = 1.0;
    end

    [delayAxis, dopplerAxis] = ndgrid(1:M, 1:N);
    gridOut = zeros(M, N);

    gains = gains(:).';
    tapPowerRaw = abs(gains).^2;

    if isempty(tapPowerRaw) || max(tapPowerRaw) <= eps
        return;
    end

    switch lower(normalizeMode)
        case 'frame'
            % 原始方式：每帧按最大 tap 归一化
            tapPower = tapPowerRaw ./ (max(tapPowerRaw) + eps);

        case 'absolute'
            % 固定参考功率归一化，异常衰减会真实体现在 DD 曲面高度上
            if isempty(refPower) || ~isfinite(refPower) || refPower <= eps
                refPower = max(tapPowerRaw);
            end
            tapPower = visualGain .* tapPowerRaw ./ (refPower + eps);

        otherwise
            error('Unknown normalizeMode: %s. Use frame or absolute.', normalizeMode);
    end

    sigmaDelayMain = 0.72;
    sigmaDopplerMain = 0.95;
    sigmaDelaySkirt = 1.75;
    sigmaDopplerSkirt = 2.35;

    for k = 1:numel(gains)
        delayCenter = mod(delayBins(k), M) + 1;
        dopplerCenter = mod(dopplerBins(k) + floor(N / 2), N) + 1;

        delayDist = min(abs(delayAxis - delayCenter), M - abs(delayAxis - delayCenter));
        dopplerDist = min(abs(dopplerAxis - dopplerCenter), N - abs(dopplerAxis - dopplerCenter));

        mainLobe = exp(-0.5 * ( ...
            (delayDist ./ sigmaDelayMain).^2 + ...
            (dopplerDist ./ sigmaDopplerMain).^2));

        leakageSkirt = 0.16 * exp(-0.5 * ( ...
            (delayDist ./ sigmaDelaySkirt).^2 + ...
            (dopplerDist ./ sigmaDopplerSkirt).^2));

        gridOut = gridOut + tapPower(k) .* (mainLobe + leakageSkirt);
    end

    switch lower(normalizeMode)
        case 'frame'
            gridOut = gridOut ./ (max(gridOut(:)) + eps);
        case 'absolute'
            % 不再按当前帧最大值重新归一化，否则异常衰减又会被抵消
            gridOut = gridOut;
    end

    gridOut = max(0, min(1, gridOut));
end

function plot_current_path_gain(scene, t)
%PLOT_CURRENT_PATH_GAIN 显示当前帧 clean/noisy 路径幅度对比。

    bar([abs(scene.cleanGains(t, :)).', abs(scene.obsGains(t, :)).']);
    grid on; box on;
    xlabel('Path index');
    ylabel('|h|');
    legend({'Clean', 'Noisy'}, 'Location', 'best');
    title('Current Path Magnitudes');
end

function plot_csi_observation_buffer(scene, t)
%PLOT_CSI_OBSERVATION_BUFFER 显示 noisy CSI 采集缓冲区。

    imagesc(1:t, 1:scene.numPaths, abs(scene.obsGains(1:t, :)).');
    axis xy;
    colorbar;
    xlabel('Collected frame');
    ylabel('Path index');
    title('CSI Observation Buffer: |h_k(t)|');
    hold on;
    xline(t, '--w', 'LineWidth', 1.2);
    text(0.03, 0.93, sprintf('Collected %d / %d frames', t, size(scene.obsGains, 1)), ...
        'Units', 'normalized', 'Color', 'w', 'FontSize', 9, 'FontWeight', 'bold');
end

function plot_noisy_feature_buffer(scene, t)
%PLOT_NOISY_FEATURE_BUFFER 显示 noisy CSI 特征缓冲区。

    totalPowerDb = 10 * log10(sum(abs(scene.obsGains(1:t, :)).^2, 2) + eps);
    if isfield(scene, 'uiPowerReferenceDb') && isfinite(scene.uiPowerReferenceDb)
        powerRefDb = scene.uiPowerReferenceDb;
    else
        finitePowerDb = totalPowerDb(isfinite(totalPowerDb));
        if isempty(finitePowerDb)
            powerRefDb = 0;
        else
            powerRefDb = median(finitePowerDb);
        end
    end
    relativePowerDb = totalPowerDb - powerRefDb;
    if numel(relativePowerDb) >= 3
        relativePowerDb = movmean(relativePowerDb, min(3, numel(relativePowerDb)));
    end
    plot(1:t, relativePowerDb, 'Color', [0.10 0.45 0.95], 'LineWidth', 1.8);
    grid on; box on;
    xlabel('Collected frame');
    ylabel('Relative power (dB)');
    title('Relative Total Channel Power');
    ylim([-22, 8]);
    hold on;
    yline(0, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
    if isfield(scene, 'uiAnomalyScale') && numel(scene.uiAnomalyScale) >= t
        mask = scene.uiAnomalyScale(1:t) < 0.999;
        add_anomaly_shading_to_current_axes(1:t, mask);
    end
    xline(t, '--', 'Color', [0.85 0.15 0.15], 'LineWidth', 1.2);
    if isfield(scene, 'uiAnomalyEnabled') && scene.uiAnomalyEnabled
        text(0.03, 0.90, sprintf('Anomaly: -%.0f dB', scene.uiAnomalyAttenuationDb), ...
            'Units', 'normalized', 'Color', [0.80 0.05 0.05], 'FontSize', 9, 'FontWeight', 'bold');
    end
end

function add_anomaly_shading_to_current_axes(x, anomalyMask)
%ADD_ANOMALY_SHADING_TO_CURRENT_AXES Shade the active manual attenuation interval.
    anomalyMask = logical(anomalyMask(:));
    x = x(:);
    if isempty(anomalyMask) || ~any(anomalyMask)
        return;
    end
    d = diff([false; anomalyMask; false]);
    startIdx = find(d == 1);
    endIdx = find(d == -1) - 1;
    yl = ylim;
    for k = 1:numel(startIdx)
        xs = x(startIdx(k)) - 0.5;
        xe = x(endIdx(k)) + 0.5;
        p = patch([xs xe xe xs], [yl(1) yl(1) yl(2) yl(2)], [1.0 0.45 0.45], ...
            'FaceAlpha', 0.16, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        try
            uistack(p, 'bottom');
        catch
        end
    end
    ylim(yl);
end

function draw_xy_window(t, T, cfg)
%DRAW_XY_WINDOW 显示 X/Y 样本窗口形成过程。

    hold on; box on;
    xlim([1, T]);
    ylim([0, 1]);
    set(gca, 'YTick', []);
    xlabel('Time frame');

    rectangle('Position', [1, 0.38, max(0, t - 1), 0.20], ...
        'FaceColor', [0.82 0.82 0.82], 'EdgeColor', 'none');

    sampleCount = floor((t - cfg.InputWindow - cfg.TargetWindow) / cfg.Stride) + 1;
    if sampleCount > 0
        xStart = 1 + (sampleCount - 1) * cfg.Stride;
        xEnd = xStart + cfg.InputWindow - 1;
        yStart = xEnd + 1;
        yEnd = yStart + cfg.TargetWindow - 1;

        rectangle('Position', [xStart, 0.18, cfg.InputWindow, 0.25], ...
            'FaceColor', [0.18 0.45 0.85], 'EdgeColor', 'none');
        rectangle('Position', [yStart, 0.55, cfg.TargetWindow, 0.25], ...
            'FaceColor', [0.95 0.55 0.16], 'EdgeColor', 'none');
        text(xStart, 0.10, sprintf('X: %d-%d', xStart, xEnd), 'FontSize', 9);
        text(yStart, 0.84, sprintf('Y: %d-%d', yStart, yEnd), 'FontSize', 9);
        statusText = sprintf('Generated samples: %d', sampleCount);
        phaseText = 'One sample = past noisy CSI + future clean CSI';
    else
        statusText = sprintf('Collecting CSI: %d / %d frames for first sample', ...
            t, cfg.InputWindow + cfg.TargetWindow);
        phaseText = 'Waiting for enough frames';
    end

    xline(t, 'r-', 'LineWidth', 1.4);
    text(min(T, t + 0.01 * T), 0.63, 'current', 'Color', 'r', 'FontWeight', 'bold');
    text(0.02 * T, 0.94, statusText, 'FontWeight', 'bold');
    text(0.02 * T, 0.86, phaseText, 'FontSize', 9, 'Color', [0.2 0.2 0.2]);
    title('Training Sample Construction');
end

function publish_ui_channel_subplots(scene, cfg, t, uiUdp)
%PUBLISH_UI_CHANNEL_SUBPLOTS Send the three CSI-process subplots to Python UI.

    payload = struct();
    payload.type = 'ntn_channel_subplots';
    payload.profile = scene.profileName;
    payload.frame = t;
    payload.totalFrames = size(scene.obsGains, 1);
    payload.numPaths = scene.numPaths;
    % ===== DD grid for UI: dB-compressed visible surface =====
    % 注意：UI图1不直接显示负dB，而是把相对功率下降映射为0~1高度。
    % 这样异常时曲面明显降低，但不会直接趴平成一条线。
    [payload.ddGrid, ddPowerDropDb, ddDisplayScale] = build_visible_dd_display_grid(scene, cfg, t);
    payload.ddGrid = max(0, min(1, payload.ddGrid));
    payload.ddPeak = max(payload.ddGrid(:));
    payload.ddPeakDb = ddPowerDropDb;
    payload.ddDisplayScale = ddDisplayScale;
    payload.ddGridMode = 'db_compressed_visible_surface';
    if isfield(scene, 'uiAnomalyEnabled') && scene.uiAnomalyEnabled
        payload.anomalyType = 'mixed_power_firsttap_doppler';
    else
        payload.anomalyType = 'normal';
    end
    payload.cleanPathMagnitude = abs(scene.cleanGains(t, :));
    payload.noisyPathMagnitude = abs(scene.obsGains(t, :));
    payload.anomalyEnabled = isfield(scene, 'uiAnomalyEnabled') && scene.uiAnomalyEnabled;
    if isfield(scene, 'uiAnomalyAttenuationDb')
        payload.anomalyAttenuationDb = scene.uiAnomalyAttenuationDb;
    else
        payload.anomalyAttenuationDb = 0;
    end

    % Do not publish channel-energy traces to the UI from this MATLAB path.
    % The UI's Relative Total Channel Power panel is reserved for baseband
    % telemetry (relativePowerDb) rather than Qianfan/GC visualization data.

    % Keep the feature-buffer payload as a fallback for older UI builds.
    featureDimShown = min(size(scene.obsFeat, 2), 16);
    featureTimeShown = min(t, cfg.InputWindow);
    tStart = max(1, t - featureTimeShown + 1);
    featView = scene.obsFeat(tStart:t, 1:featureDimShown);

    % Use a fixed per-feature reference so manual power attenuation stays visible.
    if isfield(scene, 'uiFeatureReference') && numel(scene.uiFeatureReference) >= featureDimShown
        featRef = scene.uiFeatureReference(1:featureDimShown);
    else
        featRef = max(abs(scene.obsFeat(:, 1:featureDimShown)), [], 1);
    end
    featRef(~isfinite(featRef) | featRef < eps) = 1;
    featView = abs(featView) ./ featRef;
    featView = max(0, min(1, featView));

    payload.noisyFeatureBuffer = featView.';  % [featureDimShown, collectedFrame]
    payload.featureDimShown = featureDimShown;
    payload.featureTimeStart = tStart;
    payload.featureTimeEnd = t;

    payload.inputWindow = cfg.InputWindow;
    payload.targetWindow = cfg.TargetWindow;
    payload.stride = cfg.Stride;
    payload.sampleCount = max(0, floor((t - cfg.InputWindow - cfg.TargetWindow) / cfg.Stride) + 1);

    write(uiUdp, uint8(jsonencode(payload)), cfg.UiHost, cfg.UiPort);
end

function out = cat_or_empty(c)
%CAT_OR_EMPTY 合并 cell 中的数据；空 cell 返回空矩阵。

    if isempty(c)
        out = [];
    else
        out = cat(1, c{:});
    end
end

function out = cat_meta_or_empty(c)
%CAT_META_OR_EMPTY 合并 Meta 结构体；空 cell 返回空结构体。

    if isempty(c)
        out = struct([]);
    else
        out = cat(1, c{:});
    end
end
