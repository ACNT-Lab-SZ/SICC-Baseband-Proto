function x310_uncoded_bpsk_multisync(mode)
% x310_uncoded_bpsk_multisync('tx')
% x310_uncoded_bpsk_multisync('rx')
%
% 多步骤同步版：
%   1) DC removal + AGC normalize
%   2) short preamble packet detect
%   3) coarse CFO from repeated short preamble
%   4) long preamble fine timing
%   5) long preamble channel estimation
%   6) per-symbol comb-pilot CPE correction
%   7) uncoded BPSK hard decision + BER
%
% 适用：
%   - X310 / Wireless Testbench
%   - uncoded BPSK OFDM联调
%   - 先验证同步链，再回到编码系统

cfg = localCfg();

mode = lower(string(mode));
switch mode
    case "tx"
        runTx(cfg);
    case "rx"
        runRx(cfg);
    otherwise
        error('mode 必须是 ''tx'' 或 ''rx''');
end

end

%% =========================================================
function cfg = localCfg()

% ---------------- USRP ----------------
cfg.txIP = '192.168.40.2';
cfg.rxIP = '192.168.50.2';

cfg.fc   = 2.45e9;
cfg.mcr  = 184.32e6;
cfg.fs   = 7.68e6;
cfg.interp = round(cfg.mcr / cfg.fs);
cfg.decim  = cfg.interp;

cfg.txGain = -20;
cfg.rxGain = 10;

% ---------------- OFDM ----------------
cfg.Nfft = 512;
cfg.CP   = 64;
cfg.symLen = cfg.Nfft + cfg.CP;

cfg.activeSC = 120;   % 左右各60，不含DC
negIdx = (cfg.Nfft/2 - cfg.activeSC/2 + 1) : (cfg.Nfft/2);
posIdx = (cfg.Nfft/2 + 2) : (cfg.Nfft/2 + 1 + cfg.activeSC/2);
cfg.usedIdx = [negIdx, posIdx];

% comb pilots
cfg.pilotSpacing = 12;
cfg.pilotActIdx  = 1:cfg.pilotSpacing:cfg.activeSC;
cfg.dataActIdx   = setdiff(1:cfg.activeSC, cfg.pilotActIdx);

cfg.numPilotSC = numel(cfg.pilotActIdx);
cfg.numDataSC  = numel(cfg.dataActIdx);

cfg.numDataSym = 12;

% ---------------- 短前导：重复半段 ----------------
cfg.shortHalfLen = 256;
cfg.shortPreamble = buildShortPreamble(cfg.shortHalfLen);

% ---------------- 长前导：1个已知OFDM训练符号 ----------------
cfg.longTrainAct = buildKnownBPSK(cfg.activeSC, 101);
cfg.longPreamble = buildOneOFDMSymbol(cfg.longTrainAct, cfg);

% ---------------- 数据导频 ----------------
cfg.pilotVals = buildKnownBPSK(cfg.numPilotSC, 202);

% ---------------- 测试数据 ----------------
cfg.bitsPerFrame = cfg.numDataSC * cfg.numDataSym;
cfg.refBits = buildReferenceBits(cfg.bitsPerFrame, 303);

% ---------------- 帧结构 ----------------
cfg.frameLen = numel(cfg.shortPreamble) + numel(cfg.longPreamble) + cfg.numDataSym * cfg.symLen;

% 额外多取一个CP长度，供长前导精同步向后搜索使用
cfg.frameLenExt = cfg.frameLen + cfg.CP;

cfg.rxSamplesPerFrame = 2 * cfg.frameLenExt;

% ---------------- 接收门限/搜索 ----------------
cfg.syncPeakThresh = 0.45;
cfg.longCorrThresh = 0.75;
cfg.pilotCorrThresh = 0.97;

cfg.resetOnOverrun = true;
cfg.maxBufferedFrames = 4;

cfg.skipFrames  = 10;
cfg.reportEvery = 20;
cfg.logEvery    = 10;
cfg.plotEvery   = 2;
cfg.showPlot    = true;

fprintf('Multi-step sync uncoded BPSK mode\n');
fprintf('Nfft=%d, CP=%d, activeSC=%d, dataSC=%d, pilotSC=%d, numDataSym=%d\n', ...
    cfg.Nfft, cfg.CP, cfg.activeSC, cfg.numDataSC, cfg.numPilotSC, cfg.numDataSym);

end

%% =========================================================
function runTx(cfg)

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

txFrame = buildTxFrame(cfg.refBits, cfg);

frameCnt = 0;
t0 = tic;
while true
    underrun = tx(txFrame);
    if underrun ~= 0
        fprintf('[TX] underrun @ frame=%d\n', frameCnt);
    end
    if mod(frameCnt, 100) == 0
        fprintf('[TX] frame=%d | elapsed=%.2f s\n', frameCnt, toc(t0));
    end
    frameCnt = frameCnt + 1;
end

end

%% =========================================================
function runRx(cfg)

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
buf = complex(zeros(0,1));

if cfg.showPlot
    plotState = initPlots();
else
    plotState = [];
end

while true
    [y, ~, overrun] = rx();

    % 首先做简单预处理
    y = y - mean(y);                     % DC removal
    y = y / sqrt(mean(abs(y).^2) + eps); % AGC-like normalize

    if overrun ~= 0
        stats.overrunCount = stats.overrunCount + 1;
        fprintf('[RX] overrun detected -> flush buffer and resync\n');
        if cfg.resetOnOverrun
            buf = complex(zeros(0,1));
            continue;
        end
    end

    buf = [buf; y]; %#ok<AGROW>

    maxBufLen = cfg.maxBufferedFrames * cfg.frameLenExt;
    if numel(buf) > maxBufLen
        buf = buf(end-maxBufLen+1:end);
    end

    while numel(buf) >= cfg.frameLenExt
        [coarseStart, peak, coarseCFO] = packetDetect(buf, cfg);

        if isempty(coarseStart)
            buf = buf(max(1, numel(buf)-cfg.frameLenExt+1):end);
            break;
        end

        if coarseStart + cfg.frameLenExt - 1 > numel(buf)
            buf = buf(coarseStart:end);
            break;
        end

        % 这里取扩展帧长，给长前导精同步留出后移空间
        oneFrameExt = buf(coarseStart : coarseStart + cfg.frameLenExt - 1);
        [bitsHat, eqSym, diagInfo] = decodeFrame(oneFrameExt, coarseCFO, cfg);

        bitErr = sum(uint8(bitsHat(:)) ~= uint8(cfg.refBits(:)));
        ber = bitErr / cfg.bitsPerFrame;

        validated = (peak >= cfg.syncPeakThresh) && ...
                    (diagInfo.longCorr >= cfg.longCorrThresh) && ...
                    (diagInfo.pilotCorr >= cfg.pilotCorrThresh);

        stats = updateStats(stats, cfg, bitErr, ber, validated, diagInfo);

        doPrint = validated || mod(stats.detectedFrames, cfg.logEvery) == 0;
        if doPrint
            fprintf(['[RX] det=%5d | peak=%.3f | CFO=%8.1f Hz | longCorr=%.3f | ' ...
                     'pilotCorr=%.3f | SNR=%6.2f dB | off=%2d | BER=%8.3e | valid=%d\n'], ...
                stats.detectedFrames, peak, diagInfo.cfoHz, diagInfo.longCorr, ...
                diagInfo.pilotCorr, diagInfo.snrEst, diagInfo.bestOffset, ...
                ber, validated);
        end

        if cfg.showPlot && mod(stats.detectedFrames, cfg.plotEvery) == 0
            plotState = updatePlots(plotState, eqSym, stats);
        end

        if stats.detectedFrames > cfg.skipFrames && mod(stats.detectedFrames, cfg.reportEvery) == 0
            printStats(stats, cfg);
        end

        % 仍然只消费主帧长度，额外CP样本留给下一次搜索
        buf = buf(coarseStart + cfg.frameLen : end);
    end
end

end

%% =========================================================
function txFrame = buildTxFrame(bits, cfg)

dataSym = bitsToBPSK(bits);

grid = complex(zeros(cfg.Nfft, cfg.numDataSym));

ptr = 1;
for m = 1:cfg.numDataSym
    act = complex(zeros(cfg.activeSC,1));
    act(cfg.pilotActIdx) = cfg.pilotVals;

    nDataThis = numel(cfg.dataActIdx);
    act(cfg.dataActIdx) = dataSym(ptr:ptr+nDataThis-1);
    ptr = ptr + nDataThis;

    grid(cfg.usedIdx,m) = act;
end

td = ifft(ifftshift(grid,1), cfg.Nfft, 1);
tdcp = [td(end-cfg.CP+1:end,:); td];
dataWave = tdcp(:);

txFrame = [cfg.shortPreamble; cfg.longPreamble; dataWave];
txFrame = 0.5 * txFrame / max(abs(txFrame) + 1e-12);

end

%% =========================================================
function [bitsHat, eqDataAll, diagInfo] = decodeFrame(rxFrame, coarseCFO, cfg)

% ---------- Step 1: 粗频偏补偿 ----------
n = (0:numel(rxFrame)-1).';
rxFrame = rxFrame .* exp(-1j * 2*pi * coarseCFO / cfg.fs * n);

% ---------- Step 2: 长前导精同步 ----------
expLongStart = numel(cfg.shortPreamble) + 1;

% 最多只能向后移动到“不会导致整帧越界”的位置
maxSearch = min(cfg.CP - 1, numel(rxFrame) - cfg.frameLen);
searchRange = 0:maxSearch;

bestMetric = -inf;
bestLongStart = expLongStart;
bestOffset = 0;

for off = searchRange
    s = expLongStart + off;
    e = s + numel(cfg.longPreamble) - 1;

    if e > numel(rxFrame)
        break;
    end

    r = rxFrame(s:e);
    c = abs(sum(r .* conj(cfg.longPreamble))) / ...
        (norm(r) * norm(cfg.longPreamble) + eps);

    if c > bestMetric
        bestMetric = c;
        bestLongStart = s;
        bestOffset = off;
    end
end

% ---------- Step 3: 长前导信道估计 ----------
longSeg = rxFrame(bestLongStart : bestLongStart + numel(cfg.longPreamble) - 1);
longNoCP = longSeg(cfg.CP+1:end);

Ylong = fftshift(fft(longNoCP, cfg.Nfft));
YlongAct = Ylong(cfg.usedIdx);

Hact = YlongAct ./ cfg.longTrainAct;

% ---------- Step 4: 数据符号处理 ----------
dataStart = bestLongStart + numel(cfg.longPreamble);

eqDataAll = complex(zeros(cfg.numDataSC * cfg.numDataSym, 1));
bitsHat = zeros(cfg.bitsPerFrame,1,'uint8');

pilotCorrList = zeros(cfg.numDataSym,1);
cpeList = zeros(cfg.numDataSym,1);
snrPilotList = zeros(cfg.numDataSym,1);

ptr = 1;
for m = 1:cfg.numDataSym
    s = dataStart + (m-1)*cfg.symLen;
    e = s + cfg.symLen - 1;

    if e > numel(rxFrame)
        error('decodeFrame: 数据符号越界，bestLongStart=%d, dataStart=%d, m=%d, e=%d, len=%d', ...
            bestLongStart, dataStart, m, e, numel(rxFrame));
    end

    symSeg = rxFrame(s:e);
    symNoCP = symSeg(cfg.CP+1:end);

    Y = fftshift(fft(symNoCP, cfg.Nfft));
    Yact = Y(cfg.usedIdx);

    % 初步均衡
    eqAct = Yact ./ Hact;

    % ---------- Step 5: comb pilot公共相位误差校正 ----------
    eqPilots = eqAct(cfg.pilotActIdx);
    theta = angle(sum(eqPilots .* conj(cfg.pilotVals)));
    eqAct = eqAct * exp(-1j * theta);

    eqPilots2 = eqAct(cfg.pilotActIdx);
    pilotCorr = abs(sum(eqPilots2 .* conj(cfg.pilotVals))) / ...
                (norm(eqPilots2) * norm(cfg.pilotVals) + eps);

    pilotErr = eqPilots2 - cfg.pilotVals;
    snrPilot = 10*log10(mean(abs(cfg.pilotVals).^2) / (mean(abs(pilotErr).^2) + eps));

    eqData = eqAct(cfg.dataActIdx);
    bhat = uint8(real(eqData) < 0);

    nDataThis = numel(eqData);
    eqDataAll(ptr:ptr+nDataThis-1) = eqData;
    bitsHat(ptr:ptr+nDataThis-1) = bhat;
    ptr = ptr + nDataThis;

    pilotCorrList(m) = pilotCorr;
    cpeList(m) = theta;
    snrPilotList(m) = snrPilot;
end

diagInfo.cfoHz = coarseCFO;
diagInfo.longCorr = bestMetric;
diagInfo.pilotCorr = mean(pilotCorrList);
diagInfo.snrEst = mean(snrPilotList);
diagInfo.cpeStdDeg = std(unwrap(cpeList)) * 180/pi;
diagInfo.bestLongStart = bestLongStart;
diagInfo.bestOffset = bestOffset;

end

%% =========================================================
function [pktStart, peak, coarseCFO] = packetDetect(buf, cfg)

L = cfg.shortHalfLen;
if numel(buf) < 2*L
    pktStart = [];
    peak = [];
    coarseCFO = 0;
    return;
end

r1 = buf(1:end-2*L+1);
P = zeros(numel(buf)-2*L+1,1);
R = zeros(numel(buf)-2*L+1,1);

for n = 1:numel(P)
    a = buf(n:n+L-1);
    b = buf(n+L:n+2*L-1);
    P(n) = sum(a .* conj(b));
    R(n) = sum(abs(b).^2);
end

M = abs(P).^2 ./ (R.^2 + eps);
[peak, idx] = max(M);

if isempty(peak) || peak < cfg.syncPeakThresh
    pktStart = [];
    coarseCFO = 0;
    return;
end

pktStart = idx;
coarseCFO = -angle(P(idx)) * cfg.fs / (2*pi*L);

end

%% =========================================================
function stats = initStats(cfg)

stats.tStart = tic;

stats.detectedFrames  = 0;
stats.validatedFrames = 0;
stats.goodFrames      = 0;
stats.errFrames       = 0;

stats.totalBits = 0;
stats.errBits   = 0;

stats.sumSnr      = 0;
stats.sumCfo      = 0;
stats.sumLongCorr = 0;
stats.sumPilotCorr = 0;

stats.overrunCount = 0;
stats.skipFrames = cfg.skipFrames;

stats.berHist = [];
stats.detHist = [];

end

function stats = updateStats(stats, cfg, bitErr, ber, validated, diagInfo)

stats.detectedFrames = stats.detectedFrames + 1;

if stats.detectedFrames <= stats.skipFrames
    return;
end

if validated
    stats.validatedFrames = stats.validatedFrames + 1;
    stats.totalBits = stats.totalBits + cfg.bitsPerFrame;
    stats.errBits   = stats.errBits + bitErr;

    stats.sumSnr       = stats.sumSnr + diagInfo.snrEst;
    stats.sumCfo       = stats.sumCfo + abs(diagInfo.cfoHz);
    stats.sumLongCorr  = stats.sumLongCorr + diagInfo.longCorr;
    stats.sumPilotCorr = stats.sumPilotCorr + diagInfo.pilotCorr;

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

fprintf('\n========== Multi-Step Sync Statistics ==========\n');
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
    avgLongCorr = stats.sumLongCorr / validated;
    avgPilotCorr = stats.sumPilotCorr / validated;
    goodput = stats.goodFrames * cfg.bitsPerFrame / elapsed / 1e6;

    fprintf('BER (validated)       : %.3e\n', ber);
    fprintf('FER (validated)       : %.3e\n', fer);
    fprintf('Average SNR est       : %.2f dB\n', avgSnr);
    fprintf('Average |CFO|         : %.2f Hz\n', avgCfo);
    fprintf('Average longCorr      : %.3f\n', avgLongCorr);
    fprintf('Average pilotCorr     : %.3f\n', avgPilotCorr);
    fprintf('Zero-error goodput    : %.3f Mb/s\n', goodput);
end

fprintf('===============================================\n\n');

end

%% =========================================================
function plotState = initPlots()

plotState.fig = figure('Name', 'X310 Multi-Step Sync RX Debug', 'NumberTitle', 'off');

subplot(2,1,1);
plotState.hConst = plot(nan, nan, '.', 'MarkerSize', 8);
grid on;
xlim([-2 2]);
ylim([-2 2]);
xlabel('In-Phase');
ylabel('Quadrature');
title('Equalized Data Constellation');

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

if ~ishandle(plotState.fig)
    plotState = initPlots();
end

subplot(2,1,1);
nShow = min(numel(eqSym), 2000);
set(plotState.hConst, 'XData', real(eqSym(1:nShow)), 'YData', imag(eqSym(1:nShow)));

subplot(2,1,2);
if ~isempty(stats.detHist)
    set(plotState.hBer, 'XData', stats.detHist, 'YData', max(stats.berHist, 1e-5));
end

drawnow limitrate;

end

%% =========================================================
function shortPre = buildShortPreamble(L)

a = buildKnownBPSK(L, 11);
shortPre = [a; a];
shortPre = shortPre / sqrt(mean(abs(shortPre).^2));

end

function longPre = buildOneOFDMSymbol(actVals, cfg)

grid = complex(zeros(cfg.Nfft,1));
grid(cfg.usedIdx) = actVals;

td = ifft(ifftshift(grid));
tdcp = [td(end-cfg.CP+1:end); td];

longPre = tdcp / sqrt(mean(abs(tdcp).^2));

end

function bits = buildReferenceBits(N, seed)
old = rng;
rng(seed, 'twister');
bits = randi([0 1], N, 1, 'uint8');
rng(old);
end

function sym = buildKnownBPSK(N, seed)
old = rng;
rng(seed, 'twister');
bits = randi([0 1], N, 1, 'uint8');
rng(old);
sym = bitsToBPSK(bits);
end

function sym = bitsToBPSK(bits)
bits = uint8(bits(:));
sym = 1 - 2*double(bits);
sym = complex(sym, 0);
end