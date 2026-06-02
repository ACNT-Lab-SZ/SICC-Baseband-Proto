function result = x310_uncoded_bpsk_debug(mode, overrides)
% x310_uncoded_bpsk_debug('tx')
% x310_uncoded_bpsk_debug('rx')
%
% 功能：
%   - 去掉编译码，只测试 uncoded BPSK + OFDM
%   - 固定测试比特帧，便于直接统计 BER
%   - 高频导频（pilot/data 交替）
%   - 接收端实时显示：
%       1) 等化后星座图
%       2) 验证后帧 BER 曲线
%
% 典型用途：
%   1) 先在同轴 + 衰减器下验证物理链路本身是否正常
%   2) 排除 LDPC / 软解调 / 译码状态机带来的额外影响
%   3) 专注观察：
%       - 帧同步是否稳定
%       - CFO 估计是否合理
%       - 导频信道估计是否平稳
%       - 等化后星座图是否收敛
%       - uncoded BER 是否足够低
%
% 输入：
%   mode      : 'tx' 或 'rx'
%   overrides : 可选结构体，用于覆盖默认参数，例如：
%               struct('showPlot',false,'rxGain',10)
%
% 输出：
%   result    : 运行结果结构体
%               TX 模式：发送帧数、运行时间
%               RX 模式：BER/FER/SNR/CFO/goodput 等统计量
%
% 依赖：
%   - Wireless Testbench Support Package for NI USRP Radios
%   - MATLAB 基本绘图
%
% 建议：
%   - 先同轴 + 衰减器联调，再上空口
%   - 若出现 overrun，先关闭绘图并降低打印频率
%   - 若星座图两团明显但较散，优先检查增益、衰减、同步精度

if nargin < 2
    overrides = struct();
end

% 构造默认配置，并允许外部覆盖
cfg = localCfg(overrides);

mode = lower(string(mode));
switch mode
    case "tx"
        result = runTx(cfg);
    case "rx"
        result = runRx(cfg);
    otherwise
        error('mode 必须是 ''tx'' 或 ''rx''');
end

end

%% =========================================================
function cfg = localCfg(overrides)

% ---------------- USRP 参数 ----------------
% 两台 X310 的 IP 地址
cfg.txIP = '192.168.40.2';
cfg.rxIP = '192.168.50.2';

% 射频中心频率
cfg.fc   = 2.45e9;

% X310 主时钟
cfg.mcr  = 184.32e6;

% 采样率：
% 这里选 7.68 Msps，恰好对应 15 kHz * 512 点 OFDM
cfg.fs   = 7.68e6;

% X310 发送/接收插值与抽取因子
cfg.interp = round(cfg.mcr / cfg.fs);   % 24
cfg.decim  = cfg.interp;

% 初始增益建议值
% 同轴联调时建议从较低 TX 增益开始，避免前端压缩
cfg.txGain = 5;
cfg.rxGain = 15;

% ---------------- OFDM 参数 ----------------
% FFT 点数
cfg.Nfft     = 512;

% 循环前缀长度
cfg.CP       = 36;

% 有效子载波数（不含 DC）
% 左右各 120，总计 240
cfg.activeSC = 240;

% 每帧承载的数据 OFDM 符号数
cfg.numDataSym = 8;

% 总 OFDM 符号数 = 导频符号 + 数据符号
% 这里采用 pilot/data 交替方式，所以总符号数为 16
cfg.numSym     = 16;

% 导频与数据符号索引
% 1,3,5,... 为导频符号
% 2,4,6,... 为数据符号
cfg.pilotSym = 1:2:cfg.numSym;
cfg.dataSym  = 2:2:cfg.numSym;

% 频域有效子载波映射位置
% 采用 fftshift 频域索引约定：
%   - 左半边负频率子载波
%   - 跳过 DC
%   - 右半边正频率子载波
negIdx = (cfg.Nfft/2 - cfg.activeSC/2 + 1) : (cfg.Nfft/2);
posIdx = (cfg.Nfft/2 + 2) : (cfg.Nfft/2 + 1 + cfg.activeSC/2);
cfg.usedIdx = [negIdx, posIdx];

% ---------------- 前导参数 ----------------
% 前导采用“重复半段”结构：
%   preamble = [half; half]
% 用途：
%   1) 帧检测（相关峰）
%   2) 粗频偏估计（两半段相位差）
cfg.preHalfLen = 256;
cfg.preamble   = buildPreamble(cfg.preHalfLen);

% ---------------- 测试帧比特 ----------------
% 一帧承载的数据比特数
% BPSK 下 1 个数据子载波 = 1 bit
cfg.bitsPerFrame = cfg.activeSC * cfg.numDataSym;

% 固定参考比特流：
% TX 和 RX 共享同一组参考比特，RX 可直接逐位统计 BER
cfg.refBits      = buildReferenceBits(cfg.bitsPerFrame);

% 导频矩阵：
% 每个导频 OFDM 符号在所有有效子载波上发送已知 BPSK 导频
cfg.pilotMat     = buildPilotMatrix(cfg.activeSC, numel(cfg.pilotSym));

% ---------------- RX 参数 ----------------
% 每帧总时域长度 = 前导 + 所有 OFDM 符号
cfg.frameLen = numel(cfg.preamble) + cfg.numSym * (cfg.Nfft + cfg.CP);

% 每次从 USRP 读取的采样数
% 这里设为 2 帧长度，便于缓冲区中跨帧搜索
cfg.rxSamplesPerFrame = 2 * cfg.frameLen;

% overrun 后是否直接清空缓存并重同步
cfg.resetOnOverrun  = true;

% 缓存最大保留帧数
% 防止缓冲区无限增长
cfg.maxBufferedFrames = 4;

% 帧同步相关峰阈值
cfg.syncPeakThresh  = 0.45;

% 导频相关阈值
% 用于剔除虽然检测到前导，但导频状态不一致的“伪有效帧”
cfg.pilotCorrThresh = 0.88;

% 前若干帧跳过统计
% 用于避开启动瞬态、缓存未稳定阶段
cfg.skipFrames  = 10;

% 每隔多少帧打印一次统计信息
cfg.reportEvery = 20;

% 每隔多少个检测帧打印一次逐帧日志
cfg.logEvery    = 10;

% 每隔多少个检测帧更新一次图
cfg.plotEvery   = 2;

% 最长运行时间
cfg.maxRunTimeSec = inf;

% 最多统计多少 validated frames / detected frames
cfg.maxValidatedFrames = inf;
cfg.maxDetectedFrames = inf;

% 程序退出前是否打印最终统计
cfg.printFinalStats = true;

% 是否实时绘图
% 若仍有 overrun，建议先关掉
cfg.showPlot = true;

% 使用外部参数覆盖默认配置
cfg = applyOverrides(cfg, overrides);

fprintf('Uncoded BPSK debug mode\n');
fprintf('bits/frame = %d, OFDM symbols = %d (pilot=%d, data=%d)\n', ...
    cfg.bitsPerFrame, cfg.numSym, numel(cfg.pilotSym), numel(cfg.dataSym));

end

%% =========================================================
function result = runTx(cfg)

% 创建 X310 发射对象
tx = comm.SDRuTransmitter( ...
    'Platform',            'X310', ...
    'IPAddress',           cfg.txIP, ...
    'CenterFrequency',     cfg.fc, ...
    'MasterClockRate',     cfg.mcr, ...
    'InterpolationFactor', cfg.interp, ...
    'Gain',                cfg.txGain);

cleanupObj = onCleanup(@() release(tx)); %#ok<NASGU>

fprintf('TX start | IP=%s | fc=%.3f GHz | fs=%.2f Msps | txGain=%g dB\n', ...
    cfg.txIP, cfg.fc/1e9, cfg.fs/1e6, cfg.txGain);

frameCnt = 0;
t0 = tic;

% TX 使用固定测试帧反复发送
txFrame = buildTxFrame(cfg.refBits, cfg);

while true
    underrun = tx(txFrame);

    % underrun 表示主机送数速度赶不上发射速度
    if underrun ~= 0
        fprintf('[TX] underrun @ frame=%d\n', frameCnt);
    end

    if mod(frameCnt, 100) == 0
        fprintf('[TX] frame=%d | elapsed=%.2f s\n', frameCnt, toc(t0));
    end

    frameCnt = frameCnt + 1;

    if toc(t0) >= cfg.maxRunTimeSec
        break;
    end
end

result = struct('mode', "tx", 'framesSent', frameCnt, 'elapsedSec', toc(t0));

end

%% =========================================================
function result = runRx(cfg)

% 创建 X310 接收对象
rx = comm.SDRuReceiver( ...
    'Platform',         'X310', ...
    'IPAddress',        cfg.rxIP, ...
    'CenterFrequency',  cfg.fc, ...
    'MasterClockRate',  cfg.mcr, ...
    'DecimationFactor', cfg.decim, ...
    'Gain',             cfg.rxGain, ...
    'SamplesPerFrame',  cfg.rxSamplesPerFrame, ...
    'OutputDataType',   'double');

cleanupObj = onCleanup(@() release(rx)); %#ok<NASGU>

fprintf('RX start | IP=%s | fc=%.3f GHz | fs=%.2f Msps | rxGain=%g dB\n', ...
    cfg.rxIP, cfg.fc/1e9, cfg.fs/1e6, cfg.rxGain);

stats = initStats(cfg);

% 原始接收缓存
buf = complex(zeros(0,1));

% 初始化绘图
if cfg.showPlot
    plotState = initPlots();
else
    plotState = [];
end

while true
    [y, ~, overrun] = rx();

    % overrun 表示主机来不及接收/处理数据，样本流可能断裂
    if overrun ~= 0
        stats.overrunCount = stats.overrunCount + 1;
        fprintf('[RX] overrun detected -> flush buffer and resync\n');
        if cfg.resetOnOverrun
            buf = complex(zeros(0,1));
            continue;
        end
    end

    % 追加到本地缓存
    buf = [buf; y]; %#ok<AGROW>

    % 限制缓存长度，防止内存无限增长
    maxBufLen = cfg.maxBufferedFrames * cfg.frameLen;
    if numel(buf) > maxBufLen
        buf = buf(end-maxBufLen+1:end);
    end

    % 只要缓存中还有足够长度，就持续搜帧和解调
    while numel(buf) >= cfg.frameLen
        % 1) 前导检测 + 粗 CFO 估计
        [startIdx, peak, cfoHz] = findFrame(buf, cfg);

        % 未找到有效前导：保留最后一帧长度附近的数据，等待下次继续
        if isempty(startIdx)
            buf = buf(max(1, numel(buf)-cfg.frameLen+1):end);
            break;
        end

        % 找到了帧头，但数据还不完整：等下一次更多样本进来
        if startIdx + cfg.frameLen - 1 > numel(buf)
            buf = buf(startIdx:end);
            break;
        end

        % 截取一整帧数据进行解调
        oneFrame = buf(startIdx : startIdx + cfg.frameLen - 1);

        % 2) CFO 校正 + OFDM 解调 + 导频估计 + MMSE 均衡 + BPSK 判决
        [bitsHat, eqSym, diagInfo] = decodeOneFrame(oneFrame, cfg, cfoHz);

        % 3) 逐位统计本帧 BER
        bitErr = sum(uint8(bitsHat(:)) ~= uint8(cfg.refBits(:)));
        ber = bitErr / cfg.bitsPerFrame;

        % 4) 当前帧是否计入 validated 统计
        validated = (peak >= cfg.syncPeakThresh) && (diagInfo.pilotCorr >= cfg.pilotCorrThresh);

        % 5) 更新统计器
        stats = updateStats(stats, cfg, bitErr, ber, validated, diagInfo);

        % validated 帧必打印；否则按 logEvery 打印
        doPrint = validated || mod(stats.detectedFrames, cfg.logEvery) == 0;
        if doPrint
            fprintf(['[RX] det=%5d | peak=%.3f | CFO=%8.1f Hz | SNR=%6.2f dB | ' ...
                     'pilotCorr=%.3f | BER=%8.3e | valid=%d\n'], ...
                stats.detectedFrames, peak, diagInfo.cfoHz, diagInfo.snrEst, ...
                diagInfo.pilotCorr, ber, validated);
        end

        % 更新星座图和 BER 曲线
        if cfg.showPlot && mod(stats.detectedFrames, cfg.plotEvery) == 0
            plotState = updatePlots(plotState, eqSym, stats);
        end

        % 周期性打印总体统计
        if stats.detectedFrames > cfg.skipFrames && mod(stats.detectedFrames, cfg.reportEvery) == 0
            printStats(stats, cfg);
        end

        % 当前帧消费掉，从后续缓存继续找下一帧
        buf = buf(startIdx + cfg.frameLen : end);

        % 满足退出条件时返回结果
        if stats.validatedFrames >= cfg.maxValidatedFrames || stats.detectedFrames >= cfg.maxDetectedFrames
            if cfg.printFinalStats
                printStats(stats, cfg);
            end
            result = finalizeResult(stats, cfg);
            return;
        end
    end

    % 超时退出
    if toc(stats.tStart) >= cfg.maxRunTimeSec
        if cfg.printFinalStats
            printStats(stats, cfg);
        end
        result = finalizeResult(stats, cfg);
        return;
    end
end

end

%% =========================================================
function txFrame = buildTxFrame(bits, cfg)

% 数据比特 -> BPSK 符号
dataSym = bitsToBPSK(bits);

% 频域数据网格：activeSC x numDataSym
dataGrid = reshape(dataSym, cfg.activeSC, cfg.numDataSym);

% 总资源网格：Nfft x numSym
grid = complex(zeros(cfg.Nfft, cfg.numSym));

% 导频符号位置填入已知导频
grid(cfg.usedIdx, cfg.pilotSym) = cfg.pilotMat;

% 数据符号位置填入数据
grid(cfg.usedIdx, cfg.dataSym)  = dataGrid;

% OFDM 调制
td = ifft(ifftshift(grid, 1), cfg.Nfft, 1);

% 加循环前缀
tdcp = [td(end-cfg.CP+1:end,:); td];
payloadWave = tdcp(:);

% 前导 + 数据载荷
txFrame = [cfg.preamble; payloadWave];

% 归一化，避免前端过驱
txFrame = 0.5 * txFrame / max(abs(txFrame) + 1e-12);

end

%% =========================================================
function [bitsHat, eqSym, diagInfo] = decodeOneFrame(rxFrame, cfg, cfoHz)

% 1) 粗频偏补偿
n = (0:numel(rxFrame)-1).';
rxFrame = rxFrame .* exp(-1j * 2*pi * cfoHz / cfg.fs * n);

% 2) 去掉前导，只保留 OFDM 载荷部分
r = rxFrame(numel(cfg.preamble)+1:end);

% 3) 重新整理成 [每列一个 OFDM 符号]
rMat = reshape(r, cfg.Nfft + cfg.CP, cfg.numSym);

% 4) 去 CP
rMat = rMat(cfg.CP+1:end, :);

% 5) FFT 到频域
R = fftshift(fft(rMat, cfg.Nfft, 1), 1);

% 只保留有效子载波
Y = R(cfg.usedIdx, :);

% ---------- 导频估计 ----------
% 在所有导频 OFDM 符号上，利用已知导频估计频域信道响应
Hp = Y(:, cfg.pilotSym) ./ cfg.pilotMat;

% 相邻导频信道估计的一致性
% 若值偏低，说明当前帧可能同步不准或估计不稳
if size(Hp,2) >= 2
    corrVals = zeros(1, size(Hp,2)-1);
    for k = 1:numel(corrVals)
        a = Hp(:,k);
        b = Hp(:,k+1);
        corrVals(k) = abs(a' * b) / (norm(a) * norm(b) + eps);
    end
    pilotCorr = mean(corrVals);
else
    pilotCorr = 1;
end

% 把导频位置上的信道估计插值到所有 OFDM 符号
Hest = complex(zeros(cfg.activeSC, cfg.numSym));
symAxis = 1:cfg.numSym;
for k = 1:cfg.activeSC
    Hest(k,:) = interp1(cfg.pilotSym, Hp(k,:), symAxis, 'linear', 'extrap');
end

% 只取数据符号上的接收值与信道
Yd = Y(:, cfg.dataSym);
Hd = Hest(:, cfg.dataSym);

% ---------- 噪声估计 ----------
% 先做一次粗均衡
eq0   = Yd ./ max(Hd, 1e-9);

% 粗硬判决
hard0 = bpskHardSlice(eq0(:));

% 用“判决后残差”估计噪声功率
resid = Yd(:) - Hd(:).*hard0;
noiseVar = max(mean(abs(resid).^2), 1e-8);

% ---------- 一拍 MMSE 均衡 ----------
eq = conj(Hd).*Yd ./ (abs(Hd).^2 + noiseVar);
eqSym = eq(:);

% ---------- BPSK 硬判决 ----------
% 实部 < 0 判为 bit=1，否则判为 bit=0
bitsHat = uint8(real(eqSym) < 0);

% 输出诊断量
diagInfo.cfoHz = cfoHz;
diagInfo.snrEst = 10*log10(mean(abs(Hd(:)).^2) / noiseVar);
diagInfo.pilotCorr = pilotCorr;

end

%% =========================================================
function [startIdx, peak, cfoHz] = findFrame(buf, cfg)

% 前导相关检测
pre = cfg.preamble(:);
Lp  = numel(pre);

% 匹配滤波相关
corrVal = conv(buf, flipud(conj(pre)), 'valid');

% 局部能量，用于归一化
engVal  = conv(abs(buf).^2, ones(Lp,1), 'valid');

% 归一化相关峰
metric = abs(corrVal).^2 ./ (engVal * sum(abs(pre).^2) + eps);
[peak, startIdx] = max(metric);

% 峰值过低则认为没找到帧
if isempty(peak) || peak < cfg.syncPeakThresh
    startIdx = [];
    cfoHz = 0;
    return;
end

% 利用重复半前导估计粗频偏
preRx = buf(startIdx : startIdx + 2*cfg.preHalfLen - 1);
r1 = preRx(1:cfg.preHalfLen);
r2 = preRx(cfg.preHalfLen+1:end);

% 两半段相位差 -> CFO
cfoHz = angle(sum(conj(r1).*r2)) * cfg.fs / (2*pi*cfg.preHalfLen);

end

%% =========================================================
function stats = initStats(cfg)

stats.tStart = tic;

% 检测到的总帧数（包括未通过 validated 判定的帧）
stats.detectedFrames  = 0;

% 通过 validated 判定并计入 BER/FER 的帧数
stats.validatedFrames = 0;

% validated 帧中，零误码帧数量
stats.goodFrames      = 0;

% validated 帧中，存在任意误码的帧数量
stats.errFrames       = 0;

% validated 帧累计总比特数与错误比特数
stats.totalBits = 0;
stats.errBits   = 0;

% 累计 SNR / CFO，用于求平均
stats.sumSnr = 0;
stats.sumCfo = 0;

% overrun 次数
stats.overrunCount = 0;

% 前 skipFrames 帧不参与正式统计
stats.skipFrames = cfg.skipFrames;

% BER 曲线绘图历史
stats.berHist = [];
stats.detHist = [];

end

function stats = updateStats(stats, cfg, bitErr, ber, validated, diagInfo)

stats.detectedFrames = stats.detectedFrames + 1;

% 启动初期跳过统计
if stats.detectedFrames <= stats.skipFrames
    return;
end

if validated
    stats.validatedFrames = stats.validatedFrames + 1;
    stats.totalBits = stats.totalBits + cfg.bitsPerFrame;
    stats.errBits   = stats.errBits + bitErr;

    stats.sumSnr = stats.sumSnr + diagInfo.snrEst;
    stats.sumCfo = stats.sumCfo + abs(diagInfo.cfoHz);

    stats.berHist(end+1) = ber; %#ok<AGROW>
    stats.detHist(end+1) = stats.detectedFrames; %#ok<AGROW>

    if bitErr == 0
        stats.goodFrames = stats.goodFrames + 1;
    else
        stats.errFrames = stats.errFrames + 1;
    end
end

end

function printStats(stats, cfg)

elapsed = toc(stats.tStart);
validated = stats.validatedFrames;

fprintf('\n========== Uncoded BPSK Statistics ==========\n');
fprintf('Elapsed time          : %.2f s\n', elapsed);
fprintf('Skipped frames        : %d\n', stats.skipFrames);
fprintf('Detected frames       : %d\n', max(stats.detectedFrames - stats.skipFrames, 0));
fprintf('Validated frames      : %d\n', validated);
fprintf('Good frames           : %d\n', stats.goodFrames);
fprintf('Error frames          : %d\n', stats.errFrames);
fprintf('Overruns              : %d\n', stats.overrunCount);

if validated > 0
    ber = stats.errBits / max(stats.totalBits, 1);
    fer = stats.errFrames / validated;
    avgSnr = stats.sumSnr / validated;
    avgCfo = stats.sumCfo / validated;

    % 这里的 goodput 定义为“零误码帧有效载荷速率”
    goodput = stats.goodFrames * cfg.bitsPerFrame / elapsed / 1e6;

    fprintf('BER (validated)       : %.3e\n', ber);
    fprintf('FER (validated)       : %.3e\n', fer);
    fprintf('Average SNR est       : %.2f dB\n', avgSnr);
    fprintf('Average |CFO|         : %.2f Hz\n', avgCfo);
    fprintf('Zero-error goodput    : %.3f Mb/s\n', goodput);
end

fprintf('=============================================\n\n');

end

%% =========================================================
function plotState = initPlots()

plotState.fig = figure('Name', 'Uncoded BPSK RX Debug', 'NumberTitle', 'off');

% 上图：等化后星座图
subplot(2,1,1);
plotState.hConst = plot(nan, nan, '.', 'MarkerSize', 8);
grid on;
xlim([-2 2]);
ylim([-2 2]);
xlabel('In-Phase');
ylabel('Quadrature');
title('Equalized BPSK Constellation');

% 下图：validated 帧 BER 曲线
subplot(2,1,2);
plotState.hBer = plot(nan, nan, '-o', 'LineWidth', 1);
grid on;
set(gca, 'YScale', 'log');
ylim([1e-5 1]);
xlabel('Detected frame index');
ylabel('BER');
title('Validated Frame BER');

drawnow;

end

function plotState = updatePlots(plotState, eqSym, stats)

% 若窗口被关闭，则重新创建
if ~ishandle(plotState.fig)
    plotState = initPlots();
end

% 星座图
subplot(2,1,1);
nShow = min(numel(eqSym), 1500);
set(plotState.hConst, 'XData', real(eqSym(1:nShow)), 'YData', imag(eqSym(1:nShow)));

% BER 曲线
subplot(2,1,2);
if ~isempty(stats.detHist)
    set(plotState.hBer, 'XData', stats.detHist, 'YData', max(stats.berHist, 1e-5));
end

drawnow limitrate;

end

%% =========================================================
function pre = buildPreamble(L)
% 构造重复半前导：
%   pre = [half; half]
% 作用：
%   - 帧检测
%   - CFO 估计
old = rng;
rng(11, 'twister');
bits = randi([0 1], L, 1, 'uint8');
rng(old);

half = bitsToBPSK(bits);
pre = [half; half];
pre = pre / sqrt(mean(abs(pre).^2));
end

function bits = buildReferenceBits(N)
% 固定参考测试比特
old = rng;
rng(20260418, 'twister');
bits = randi([0 1], N, 1, 'uint8');
rng(old);
end

function P = buildPilotMatrix(nSC, nPilot)
% 构造导频矩阵：
% 大小 = [有效子载波数 x 导频符号数]
old = rng;
rng(73, 'twister');
bits = randi([0 1], nSC * nPilot, 1, 'uint8');
rng(old);

P = reshape(bitsToBPSK(bits), nSC, nPilot);
end

function sym = bitsToBPSK(bits)
% BPSK 映射：
%   0 -> +1
%   1 -> -1
bits = uint8(bits(:));
sym = 1 - 2*double(bits);
sym = complex(sym, 0);
end

function sym = bpskHardSlice(x)
% BPSK 硬判决符号切片
sym = complex(ones(size(x)), 0);
sym(real(x) < 0) = -1;
end

function result = finalizeResult(stats, cfg)

elapsed = toc(stats.tStart);
validated = stats.validatedFrames;

result = struct();
result.elapsedSec = elapsed;
result.skippedFrames = stats.skipFrames;
result.detectedFrames = max(stats.detectedFrames - stats.skipFrames, 0);
result.validatedFrames = validated;
result.goodFrames = stats.goodFrames;
result.errorFrames = stats.errFrames;
result.overruns = stats.overrunCount;
result.bitsPerFrame = cfg.bitsPerFrame;
result.txGain = cfg.txGain;
result.rxGain = cfg.rxGain;

if validated > 0
    result.ber = stats.errBits / max(stats.totalBits, 1);
    result.fer = stats.errFrames / validated;
    result.avgSnrDb = stats.sumSnr / validated;
    result.avgAbsCfoHz = stats.sumCfo / validated;
    result.goodputMbps = stats.goodFrames * cfg.bitsPerFrame / max(elapsed, eps) / 1e6;
else
    result.ber = NaN;
    result.fer = NaN;
    result.avgSnrDb = NaN;
    result.avgAbsCfoHz = NaN;
    result.goodputMbps = 0;
end

end

function cfg = applyOverrides(cfg, overrides)

if isempty(overrides)
    return;
end

% 用 overrides 中的字段覆盖默认配置
overrideFields = fieldnames(overrides);
for idx = 1:numel(overrideFields)
    fieldName = overrideFields{idx};
    cfg.(fieldName) = overrides.(fieldName);
end

end