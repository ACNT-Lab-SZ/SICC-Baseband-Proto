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
choice = input('请输入序号，默认 1：', 's');
choice = strtrim(choice);
if isempty(choice)
    choice = '1';
end

switch choice
    case '1'
        profileList = allProfiles(1);
    case '2'
        profileList = allProfiles(2);
    case '3'
        profileList = allProfiles(3);
    case '4'
        profileList = allProfiles(4);
    case '5'
        profileList = allProfiles;
    otherwise
        warning('输入无效，默认使用 NTN-TDL-A。');
        profileList = allProfiles(1);
end
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

    writer = [];
    if cfg.RecordVideo
        writer = VideoWriter(videoFile, 'MPEG-4');
        writer.FrameRate = cfg.VideoFrameRate;
        open(writer);
    end

    fig = figure('Color', 'w', 'Position', [80, 60, 1500, 880]);
    for ii = 1:numel(frameIdx)
        t = frameIdx(ii);
        clf(fig);

        subplot(2, 3, 1);
        plot_tx_symbol_block(scene, cfg, t);

        subplot(2, 3, 2);
        plot_3d_delay_doppler_channel(scene, cfg, t);

        subplot(2, 3, 3);
        plot_current_path_gain(scene, t);

        subplot(2, 3, 4);
        plot_csi_observation_buffer(scene, t);

        subplot(2, 3, 5);
        plot_noisy_feature_buffer(scene, t);

        subplot(2, 3, 6);
        draw_xy_window(t, T, cfg);

        sgtitle(sprintf('TX QAM and Online CSI Dataset Acquisition | %s | SNR = %.1f dB', ...
            scene.profileName, cfg.ObsSNRdB), ...
            'FontWeight', 'bold', 'FontSize', 14);
        drawnow;

        if cfg.RecordVideo
            writeVideo(writer, getframe(fig));
        end
    end

    if cfg.RecordVideo
        close(writer);
        fprintf('已保存可视化视频:\n  %s\n', videoFile);
    end
end

function plot_tx_symbol_block(scene, cfg, t)
%PLOT_TX_SYMBOL_BLOCK 显示当前采集帧对应的 QPSK 发射符号块。
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
    Z = build_dd_signal_intensity_grid( ...
        scene.obsGains(t, :), ...
        scene.obsDelayBins(t, :), ...
        scene.obsDopplerBins(t, :), ...
        cfg.M, cfg.N);

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
    zlabel('Normalized intensity');
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
    text(0.03, 0.95, sprintf('Equivalent DD surface | Frame = %d', t), ...)
        'Units', 'normalized', ...
        'Color', [0.02 0.08 0.18], ...
        'FontWeight', 'bold', ...
        'FontSize', 10);
    text(0.03, 0.88, sprintf('Surface: Delay x Doppler = %d x %d', MDelay, NDopp), ...
        'Units', 'normalized', ...
        'Color', [0.02 0.08 0.18], ...
        'FontWeight', 'bold', ...
        'FontSize', 9);
    text(0.03, 0.81, 'Smoothed color surface, normalized to [0, 1]', ...
        'Units', 'normalized', ...
        'Color', [0.02 0.08 0.18], ...
        'FontWeight', 'bold', ...
        'FontSize', 9);
end

function gridOut = build_dd_signal_intensity_grid(gains, delayBins, dopplerBins, M, N)
%BUILD_DD_SIGNAL_INTENSITY_GRID 构造类似 pulse pilot 接收矩阵的 DD 域强度图。

    [delayAxis, dopplerAxis] = ndgrid(1:M, 1:N);
    complexGrid = zeros(M, N);

    for k = 1:numel(gains)
        delayCenter = mod(delayBins(k), M) + 1;
        dopplerCenter = mod(dopplerBins(k) + floor(N / 2), N) + 1;

        delayDist = min(abs(delayAxis - delayCenter), M - abs(delayAxis - delayCenter));
        dopplerDist = min(abs(dopplerAxis - dopplerCenter), N - abs(dopplerAxis - dopplerCenter));
        mainLobe = exp(-(delayDist.^2 / 2.8 + dopplerDist.^2 / 2.8));
        sideLobe = 0.22 * cos(1.45 * delayDist + angle(gains(k))) .* ...
            cos(1.30 * dopplerDist - angle(gains(k))) .* ...
            exp(-(delayDist.^2 + dopplerDist.^2) / 42);
        ripple = 0.08 * sin(0.95 * delayAxis + 0.38 * dopplerAxis + 2 * angle(gains(k)));

        complexGrid = complexGrid + gains(k) .* (mainLobe + sideLobe + ripple);
    end

    texture = 0.08 * sin(0.9 * delayAxis + 0.45 * dopplerAxis) + ...
        0.06 * cos(0.35 * delayAxis - 1.10 * dopplerAxis);

    gridOut = abs(complexGrid) + texture;
    gridOut = gridOut - min(gridOut(:));
    gridOut = gridOut ./ (max(gridOut(:)) + eps);
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

    dim = min(size(scene.obsFeat, 2), 40);
    featView = scene.obsFeat(1:t, 1:dim);
    featView = featView - min(featView, [], 1);
    featView = featView ./ (max(featView, [], 1) + eps);

    imagesc(1:t, 1:dim, featView.');
    axis xy;
    colorbar;
    xlabel('Collected frame');
    ylabel('Feature index');
    title('Noisy CSI Feature Buffer');
    hold on;
    xline(t, '--w', 'LineWidth', 1.2);
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
