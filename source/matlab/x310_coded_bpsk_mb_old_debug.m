function result = x310_coded_bpsk_mb_old_debug(mode, overrides)
% x310_coded_bpsk_mb_old_debug('tx')
% x310_coded_bpsk_mb_old_debug('rx')
%
% 功能：
%   - 基于 x310_uncoded_bpsk_debug 的 USRP BPSK+OFDM 调试链路
%   - 将 MB-OLD 中的 LDPC 编码 + BP/OLD 译码接入到实时收发流程
%   - 保留前导、导频、同步、CFO 估计与单拍均衡逻辑
%
% 默认码型：
%   - CCSDS LDPC (N=128, K=64), 每帧 15 个码块，刚好填满 1920 个数据比特




if nargin < 2
    overrides = struct();
end

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

cfg.projectRoot = fileparts(mfilename('fullpath'));
cfg.mbOldRoot = fullfile(cfg.projectRoot, 'MB-OLD');

% ---------------- USRP 参数 ----------------
cfg.txIP = '192.168.40.2';
cfg.rxIP = '192.168.50.2';

cfg.fc   = 2.45e9;
cfg.mcr  = 184.32e6;
cfg.fs   = 7.68e6;
cfg.interp = round(cfg.mcr / cfg.fs);
cfg.decim  = cfg.interp;

cfg.txGain = 10;
cfg.rxGain = 15;



% ---------------- OFDM 参数 ----------------
cfg.Nfft     = 512;
cfg.CP       = 36;
cfg.activeSC = 240;
cfg.numSym   = 16;

cfg.pilotSym = 1:2:cfg.numSym;
cfg.dataSym  = 2:2:cfg.numSym;
cfg.numDataSym = numel(cfg.dataSym);

negIdx = (cfg.Nfft/2 - cfg.activeSC/2 + 1) : (cfg.Nfft/2);
posIdx = (cfg.Nfft/2 + 2) : (cfg.Nfft/2 + 1 + cfg.activeSC/2);
cfg.usedIdx = [negIdx, posIdx];

% ---------------- 前导参数 ----------------
cfg.preHalfLen = 256;
cfg.preamble   = buildPreamble(cfg.preHalfLen);

% ---------------- MB-OLD 编码参数 ----------------
cfg.codecProfile = "ccsds128_64";
cfg.bpMaxIter = 20;
cfg.oldSegments = 2;

% 对应 main_LDPC_BP_OSD.m 中 CCSDS(128,64) 的经验权重
cfg.osdWeights = [0.3241 0.3347 0.3413];
cfg.alphaOsd = 2;

% ---------------- CUDA BP-OSD 参数 ----------------
cfg.useCudaDecoder = true;
cfg.cudaProjectRoot = fullfile(fileparts(cfg.projectRoot), 'cuda_osd_project');
cfg.cudaDllPath = '';
cfg.cudaMaxBatchSize = 4096;
cfg.cudaMinBatchSize = 30;
cfg.cudaMaxLatencyUs = 2000;
cfg.cudaArrivalEwmaAlpha = 0.15;
cfg.cudaDefaultInterarrivalUs = 150;
cfg.cudaBpMaxIter = 8;
cfg.cudaBpNormalization = 0.8;
cfg.cudaBpOffset = 0.15;
cfg.cudaBpDamping = 0.15;
cfg.cudaBpAcceptLlr = 1.5;
cfg.cudaOsdThreads = 256;
cfg.cudaPollMaxCodewords = 256;

% ---------------- RX 参数 ----------------
cfg.frameLen = numel(cfg.preamble) + cfg.numSym * (cfg.Nfft + cfg.CP);
cfg.rxSamplesPerFrame = 2 * cfg.frameLen;

cfg.resetOnOverrun  = true;
cfg.maxBufferedFrames = 4;
cfg.syncPeakThresh  = 0.45;
cfg.pilotCorrThresh = 0.88;

cfg.skipFrames  = 10;
cfg.reportEvery = 20;
cfg.logEvery    = 10;
cfg.plotEvery   = 2;
cfg.maxRunTimeSec = 30;
cfg.maxValidatedFrames = inf;
cfg.maxDetectedFrames = inf;
cfg.printFinalStats = true;
cfg.showPlot = true;

cfg = applyOverrides(cfg, overrides);
cfg = initCodec(cfg);
cfg.cudaMinBatchSize = max(cfg.cudaMinBatchSize, cfg.codec.numBlocksPerFrame);
cfg.cudaPollMaxCodewords = max(cfg.cudaPollMaxCodewords, cfg.codec.numBlocksPerFrame);

fprintf('Coded BPSK MB-OLD debug mode\n');
fprintf('codec = %s | codeword = (%d,%d) | blocks/frame = %d\n', ...
    cfg.codec.profile, cfg.codec.N, cfg.codec.K, cfg.codec.numBlocksPerFrame);
fprintf('info bits/frame = %d | coded bits/frame = %d | OFDM symbols = %d (pilot=%d, data=%d)\n', ...
    cfg.infoBitsPerFrame, cfg.codedBitsPerFrame, cfg.numSym, numel(cfg.pilotSym), numel(cfg.dataSym));
if cfg.useCudaDecoder
    fprintf('CUDA BP-OSD stream | minBatch=%d | maxBatch=%d | maxLatency=%d us\n', ...
        cfg.cudaMinBatchSize, cfg.cudaMaxBatchSize, cfg.cudaMaxLatencyUs);
end

end

function cfg = initCodec(cfg)

setupMbOldPaths(cfg.mbOldRoot);

switch string(cfg.codecProfile)
    case "ccsds128_64"
        H = CCSDS_ldpc_n128_k64_H();
    otherwise
        error('Unsupported codec profile: %s', cfg.codecProfile);
end

[~, M, N, K, vnDegree, cnDegree, P, HRowAbsIdx, HColRelIdx, vnDist, cnDist] = H_matrix_process(H);
G = [eye(K), P'];

payloadBitsPerFrame = cfg.activeSC * numel(cfg.dataSym);
if mod(payloadBitsPerFrame, N) ~= 0
    error('Payload size %d is not divisible by codeword length %d.', payloadBitsPerFrame, N);
end

codec = struct();
codec.profile = char(string(cfg.codecProfile));
codec.H = H;
codec.G = G;
codec.M = M;
codec.N = N;
codec.K = K;
codec.P = P;
codec.vnDegree = vnDegree;
codec.cnDegree = cnDegree;
codec.HRowAbsIdx = HRowAbsIdx;
codec.HColRelIdx = HColRelIdx;
codec.vnDistribution = vnDist;
codec.cnDistribution = cnDist;
codec.numBlocksPerFrame = payloadBitsPerFrame / N;
codec.infoBitsPerFrame = codec.numBlocksPerFrame * K;
codec.codedBitsPerFrame = codec.numBlocksPerFrame * N;
codec.bpMaxIter = cfg.bpMaxIter;
codec.oldSegments = cfg.oldSegments;
codec.osdWeights = cfg.osdWeights(:).';
codec.alphaOsd = cfg.alphaOsd;

cfg.codec = codec;
cfg.infoBitsPerFrame = codec.infoBitsPerFrame;
cfg.codedBitsPerFrame = codec.codedBitsPerFrame;
cfg.bitsPerFrame = codec.infoBitsPerFrame;
cfg.refInfoBits = buildReferenceBits(cfg.infoBitsPerFrame);
cfg.pilotMat = buildPilotMatrix(cfg.activeSC, numel(cfg.pilotSym));

end

function setupMbOldPaths(rootPath)

persistent configuredRoots
if isempty(configuredRoots)
    configuredRoots = strings(0, 1);
end

if any(configuredRoots == string(rootPath))
    return;
end

addpath(rootPath, '-begin');
addpath(fullfile(rootPath, 'Channels'), '-begin');
addpath(fullfile(rootPath, 'Decoders'), '-begin');
addpath(fullfile(rootPath, 'load_H_matrix'), '-begin');
addpath(fullfile(rootPath, 'Modulation'), '-begin');
addpath(fullfile(rootPath, 'My_LDPC'), '-begin');
addpath(fullfile(rootPath, 'Tools'), '-begin');
addpath(fullfile(rootPath, 'Tools', 'OLD'), '-begin');
addpath(fullfile(rootPath, 'Tools', 'Integer_partitions'), '-begin');

configuredRoots(end+1, 1) = string(rootPath);
end

function setupCudaDecoderPaths(rootPath)

persistent configuredRoots
if isempty(configuredRoots)
    configuredRoots = strings(0, 1);
end

if any(configuredRoots == string(rootPath))
    return;
end

matlabRoot = fullfile(rootPath, 'matlab');
if ~isfolder(matlabRoot)
    error('CUDA decoder MATLAB path not found: %s', matlabRoot);
end

addpath(matlabRoot, '-begin');
configuredRoots(end+1, 1) = string(rootPath);
end

function decoder = initCudaDecoder(cfg)

setupCudaDecoderPaths(cfg.cudaProjectRoot);

decoder = CudaBpOsdDecoder( ...
    'dllPath', cfg.cudaDllPath, ...
    'n', cfg.codec.N, ...
    'k', cfg.codec.K, ...
    'm', cfg.codec.M, ...
    'codewordsPerFrame', cfg.codec.numBlocksPerFrame, ...
    'maxBatchSize', cfg.cudaMaxBatchSize, ...
    'minBatchSize', cfg.cudaMinBatchSize, ...
    'maxLatencyUs', cfg.cudaMaxLatencyUs, ...
    'arrivalEwmaAlpha', cfg.cudaArrivalEwmaAlpha, ...
    'defaultInterarrivalUs', cfg.cudaDefaultInterarrivalUs, ...
    'bpMaxIterations', cfg.cudaBpMaxIter, ...
    'bpNormalization', cfg.cudaBpNormalization, ...
    'bpOffset', cfg.cudaBpOffset, ...
    'bpDamping', cfg.cudaBpDamping, ...
    'bpMinAbsLlrAccept', cfg.cudaBpAcceptLlr, ...
    'osdThreads', cfg.cudaOsdThreads);
end

%% =========================================================
function result = runTx(cfg)

tx = comm.SDRuTransmitter( ...
    'Platform',            'X310', ...
    'IPAddress',           cfg.txIP, ...
    'CenterFrequency',     cfg.fc, ...
    'MasterClockRate',     cfg.mcr, ...
    'InterpolationFactor', cfg.interp, ...
    'Gain',                cfg.txGain);

cleanupObj = onCleanup(@() release(tx)); 

fprintf('TX start | IP=%s | fc=%.3f GHz | fs=%.2f Msps | txGain=%g dB\n', ...
    cfg.txIP, cfg.fc/1e9, cfg.fs/1e6, cfg.txGain);

frameCnt = 0;
t0 = tic;
txFrame = buildTxFrame(cfg.refInfoBits, cfg);

while true
    underrun = tx(txFrame);

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

rx = comm.SDRuReceiver( ...
    'Platform',         'X310', ...
    'IPAddress',        cfg.rxIP, ...
    'CenterFrequency',  cfg.fc, ...
    'MasterClockRate',  cfg.mcr, ...
    'DecimationFactor', cfg.decim, ...
    'Gain',             cfg.rxGain, ...
    'SamplesPerFrame',  cfg.rxSamplesPerFrame, ...
    'OutputDataType',   'double');

cleanupObj = onCleanup(@() release(rx)); 

fprintf('RX start | IP=%s | fc=%.3f GHz | fs=%.2f Msps | rxGain=%g dB\n', ...
    cfg.rxIP, cfg.fc/1e9, cfg.fs/1e6, cfg.rxGain);

stats = initStats(cfg);
buf = complex(zeros(0,1));
cudaDecoder = [];
cudaCleanup = [];

if cfg.useCudaDecoder
    cudaDecoder = initCudaDecoder(cfg);
    cudaCleanup = onCleanup(@() delete(cudaDecoder)); %#ok<NASGU>
    fprintf('RX CUDA decoder ready | dll=%s\n', char(string(cudaDecoder.DllPath)));
end

if cfg.showPlot
    plotState = initPlots();
else
    plotState = [];
end

while true
    [y, ~, overrun] = rx();

    if overrun ~= 0
        stats.overrunCount = stats.overrunCount + 1;
        fprintf('[RX] overrun detected -> flush buffer and resync\n');
        if cfg.resetOnOverrun
            buf = complex(zeros(0,1));
            if cfg.useCudaDecoder
                [stats, plotState, stopNow] = maybeFlushCudaFrames(cudaDecoder, stats, cfg, plotState);
                if stopNow
                    result = finishRxRun(stats, cfg, cudaDecoder);
                    return;
                end
            end
            continue;
        end
    end

    buf = [buf; y]; 

    maxBufLen = cfg.maxBufferedFrames * cfg.frameLen;
    if numel(buf) > maxBufLen
        buf = buf(end-maxBufLen+1:end);
    end

    while numel(buf) >= cfg.frameLen
        [startIdx, peak, cfoHz] = findFrame(buf, cfg);

        if isempty(startIdx)
            buf = buf(max(1, numel(buf)-cfg.frameLen+1):end);
            if cfg.useCudaDecoder
                [stats, plotState, stopNow] = maybeFlushCudaFrames(cudaDecoder, stats, cfg, plotState);
                if stopNow
                    result = finishRxRun(stats, cfg, cudaDecoder);
                    return;
                end
            end
            break;
        end

        if startIdx + cfg.frameLen - 1 > numel(buf)
            buf = buf(startIdx:end);
            break;
        end

        oneFrame = buf(startIdx : startIdx + cfg.frameLen - 1);
        frameIndex = stats.detectedFrames + 1;
        stats.detectedFrames = frameIndex;

        if cfg.useCudaDecoder
            [llrFrame, eqSym, diagInfo] = prepareFrameForDecode(oneFrame, cfg, cfoHz);
            validated = (peak >= cfg.syncPeakThresh) && (diagInfo.pilotCorr >= cfg.pilotCorrThresh);
            submitTimestampUs = uint64(round(toc(stats.tStart) * 1e6));

            userData = struct();
            userData.detectedFrameIndex = frameIndex;
            userData.peak = peak;
            userData.validated = validated;
            userData.diagInfo = diagInfo;
            userData.eqSym = eqSym;

            cudaDecoder.submitFrame(uint64(frameIndex), ...
                                    reshape(single(llrFrame(:)), cfg.codec.N, []), ...
                                    submitTimestampUs, ...
                                    userData);
            [stats, plotState, stopNow] = processReadyCudaFrames(cudaDecoder, stats, cfg, plotState, false);
        else
            [infoBitsHat, eqSym, diagInfo] = decodeOneFrame(oneFrame, cfg, cfoHz);
            validated = (peak >= cfg.syncPeakThresh) && (diagInfo.pilotCorr >= cfg.pilotCorrThresh);
            [stats, plotState, stopNow] = handleDecodedFrameResult(stats, ...
                                                                   cfg, ...
                                                                   plotState, ...
                                                                   frameIndex, ...
                                                                   peak, ...
                                                                   validated, ...
                                                                   infoBitsHat, ...
                                                                   eqSym, ...
                                                                   diagInfo);
        end

        buf = buf(startIdx + cfg.frameLen : end);

        if stopNow || stats.detectedFrames >= cfg.maxDetectedFrames
            result = finishRxRun(stats, cfg, cudaDecoder);
            return;
        end
    end

    if cfg.useCudaDecoder
        [stats, plotState, stopNow] = maybeFlushCudaFrames(cudaDecoder, stats, cfg, plotState);
        if stopNow
            result = finishRxRun(stats, cfg, cudaDecoder);
            return;
        end
    end

    if toc(stats.tStart) >= cfg.maxRunTimeSec
        result = finishRxRun(stats, cfg, cudaDecoder);
        return;
    end
end

end

%% =========================================================
function result = finishRxRun(stats, cfg, cudaDecoder)

decoderStats = [];
if cfg.useCudaDecoder && ~isempty(cudaDecoder)
    [stats, ~, ~] = processReadyCudaFrames(cudaDecoder, stats, cfg, [], true);
    decoderStats = cudaDecoder.stats();
end

if cfg.printFinalStats
    printStats(stats, cfg, decoderStats);
end
result = finalizeResult(stats, cfg, decoderStats);
end

function [stats, plotState, stopNow] = maybeFlushCudaFrames(cudaDecoder, stats, cfg, plotState)

stopNow = false;
if isempty(cudaDecoder)
    return;
end

decoderStats = cudaDecoder.stats();
if decoderStats.currentPendingCodewords == 0
    return;
end

nowUs = uint64(round(toc(stats.tStart) * 1e6));
if nowUs < decoderStats.lastSubmitTimestampUs
    return;
end

if double(nowUs - decoderStats.lastSubmitTimestampUs) >= double(cfg.cudaMaxLatencyUs)
    [stats, plotState, stopNow] = processReadyCudaFrames(cudaDecoder, stats, cfg, plotState, true);
end
end

function [stats, plotState, stopNow] = processReadyCudaFrames(cudaDecoder, stats, cfg, plotState, flushPending)

stopNow = false;
if isempty(cudaDecoder)
    return;
end

if nargin < 5
    flushPending = false;
end

if flushPending
    [~, readyFrames] = cudaDecoder.flush();
else
    [~, readyFrames] = cudaDecoder.poll(cfg.cudaPollMaxCodewords);
end

for idx = 1:numel(readyFrames)
    frame = readyFrames(idx);
    frameMeta = frame.userData;
    diagInfo = frameMeta.diagInfo;
    diagInfo.bpPassBlocks = cfg.codec.numBlocksPerFrame - double(frame.usedOsdCount);
    diagInfo.oldFallbackBlocks = double(frame.usedOsdCount);
    diagInfo.avgBpIter = NaN;
    diagInfo.avgOldTests = NaN;

    [stats, plotState, frameStop] = handleDecodedFrameResult(stats, ...
                                                             cfg, ...
                                                             plotState, ...
                                                             frameMeta.detectedFrameIndex, ...
                                                             frameMeta.peak, ...
                                                             frameMeta.validated, ...
                                                             uint8(frame.infoBits(:)), ...
                                                             frameMeta.eqSym, ...
                                                             diagInfo);
    stopNow = stopNow || frameStop;
end
end

function [stats, plotState, stopNow] = handleDecodedFrameResult(stats, ...
                                                                cfg, ...
                                                                plotState, ...
                                                                frameIndex, ...
                                                                peak, ...
                                                                validated, ...
                                                                infoBitsHat, ...
                                                                eqSym, ...
                                                                diagInfo)

bitErr = sum(uint8(infoBitsHat(:)) ~= uint8(cfg.refInfoBits(:)));
ber = bitErr / cfg.infoBitsPerFrame;

stats = updateStats(stats, cfg, frameIndex, bitErr, ber, validated, diagInfo);
printDecodedFrameLog(cfg, frameIndex, peak, ber, validated, diagInfo);

if cfg.showPlot && ~isempty(plotState) && mod(frameIndex, cfg.plotEvery) == 0
    plotState = updatePlots(plotState, eqSym, stats);
end

if frameIndex > cfg.skipFrames && mod(frameIndex, cfg.reportEvery) == 0
    printStats(stats, cfg);
end

stopNow = stats.validatedFrames >= cfg.maxValidatedFrames;
end

function printDecodedFrameLog(cfg, frameIndex, peak, ber, validated, diagInfo)

doPrint = validated || mod(frameIndex, cfg.logEvery) == 0;
if ~doPrint
    return;
end

logPrefix = '[RX]';
if cfg.useCudaDecoder
    logPrefix = '[RX][CUDA]';
end

if isfinite(diagInfo.avgBpIter)
    fprintf([logPrefix ' det=%5d | peak=%.3f | CFO=%8.1f Hz | SNR=%6.2f dB | pilotCorr=%.3f | ' ...
             'BER=%8.3e | BPpass=%2d/%2d | OLD=%2d | avgIter=%.2f | valid=%d\n'], ...
        frameIndex, peak, diagInfo.cfoHz, diagInfo.snrEst, diagInfo.pilotCorr, ...
        ber, diagInfo.bpPassBlocks, cfg.codec.numBlocksPerFrame, diagInfo.oldFallbackBlocks, ...
        diagInfo.avgBpIter, validated);
else
    fprintf([logPrefix ' det=%5d | peak=%.3f | CFO=%8.1f Hz | SNR=%6.2f dB | pilotCorr=%.3f | ' ...
             'BER=%8.3e | BPpass=%2d/%2d | OSD=%2d | valid=%d\n'], ...
        frameIndex, peak, diagInfo.cfoHz, diagInfo.snrEst, diagInfo.pilotCorr, ...
        ber, diagInfo.bpPassBlocks, cfg.codec.numBlocksPerFrame, diagInfo.oldFallbackBlocks, ...
        validated);
end
end

%% =========================================================
function txFrame = buildTxFrame(infoBits, cfg)

codedBits = encodeFrameBits(infoBits, cfg.codec);
dataSym = bitsToBPSK(codedBits);
dataGrid = reshape(dataSym, cfg.activeSC, cfg.numDataSym);

grid = complex(zeros(cfg.Nfft, cfg.numSym));
grid(cfg.usedIdx, cfg.pilotSym) = cfg.pilotMat;
grid(cfg.usedIdx, cfg.dataSym)  = dataGrid;

td = ifft(ifftshift(grid, 1), cfg.Nfft, 1);
tdcp = [td(end-cfg.CP+1:end,:); td];
payloadWave = tdcp(:);

txFrame = [cfg.preamble; payloadWave];
txFrame = 0.5 * txFrame / max(abs(txFrame) + 1e-12);

end

function codedBits = encodeFrameBits(infoBits, codec)

infoBits = uint8(infoBits(:));
if numel(infoBits) ~= codec.infoBitsPerFrame
    error('Expected %d information bits per frame, got %d.', codec.infoBitsPerFrame, numel(infoBits));
end

codedBits = zeros(codec.codedBitsPerFrame, 1, 'uint8');

for blk = 1:codec.numBlocksPerFrame
    infoIdx = (blk - 1) * codec.K + (1:codec.K);
    codeIdx = (blk - 1) * codec.N + (1:codec.N);

    infoBlock = double(infoBits(infoIdx)).';
    parityBits = mod(infoBlock * codec.P', 2);
    codeword = uint8([infoBlock, parityBits]);
    codedBits(codeIdx) = codeword(:);
end

end

%% =========================================================
function [llr, eqSym, diagInfo] = prepareFrameForDecode(rxFrame, cfg, cfoHz)

n = (0:numel(rxFrame)-1).';
rxFrame = rxFrame .* exp(-1j * 2*pi * cfoHz / cfg.fs * n);

r = rxFrame(numel(cfg.preamble)+1:end);
rMat = reshape(r, cfg.Nfft + cfg.CP, cfg.numSym);
rMat = rMat(cfg.CP+1:end, :);

R = fftshift(fft(rMat, cfg.Nfft, 1), 1);
Y = R(cfg.usedIdx, :);

Hp = Y(:, cfg.pilotSym) ./ cfg.pilotMat;

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

Hest = complex(zeros(cfg.activeSC, cfg.numSym));
symAxis = 1:cfg.numSym;
for k = 1:cfg.activeSC
    Hest(k,:) = interp1(cfg.pilotSym, Hp(k,:), symAxis, 'linear', 'extrap');
end

Yd = Y(:, cfg.dataSym);
Hd = Hest(:, cfg.dataSym);

eq0 = Yd ./ max(Hd, 1e-9);
hard0 = bpskHardSlice(eq0(:));
resid = Yd(:) - Hd(:) .* hard0;
noiseVar = max(mean(abs(resid).^2), 1e-8);

eq = conj(Hd) .* Yd ./ (abs(Hd).^2 + noiseVar);
eqSym = eq(:);

llr = 2 * real(eqSym) / noiseVar;

diagInfo.cfoHz = cfoHz;
diagInfo.snrEst = 10 * log10(mean(abs(Hd(:)).^2) / noiseVar);
diagInfo.pilotCorr = pilotCorr;
end

function [infoBitsHat, eqSym, diagInfo] = decodeOneFrame(rxFrame, cfg, cfoHz)

[llr, eqSym, diagInfo] = prepareFrameForDecode(rxFrame, cfg, cfoHz);
[infoBitsHat, decInfo] = decodeFrameLlrs(llr, cfg.codec);

diagInfo.bpPassBlocks = decInfo.bpPassBlocks;
diagInfo.oldFallbackBlocks = decInfo.oldFallbackBlocks;
diagInfo.avgBpIter = decInfo.avgBpIter;
diagInfo.avgOldTests = decInfo.avgOldTests;

end

function [infoBitsHat, info] = decodeFrameLlrs(llr, codec)

llr = double(llr(:)).';
if numel(llr) ~= codec.codedBitsPerFrame
    error('Expected %d coded LLRs per frame, got %d.', codec.codedBitsPerFrame, numel(llr));
end

infoBitsHat = zeros(codec.infoBitsPerFrame, 1, 'uint8');
bpPassBlocks = 0;
oldFallbackBlocks = 0;
bpIterTotal = 0;
oldTestsTotal = 0;

for blk = 1:codec.numBlocksPerFrame
    codeIdx = (blk - 1) * codec.N + (1:codec.N);
    infoIdx = (blk - 1) * codec.K + (1:codec.K);
    llrBlock = llr(codeIdx);

    [~, checkFlag, cHat, iterThisTime, outputModifiedLlr, ~] = ...
        OperationNum_modified_LDPC_Flooding_BP_decoder( ...
            llrBlock, codec.HRowAbsIdx, codec.HColRelIdx, codec.N, codec.M, ...
            codec.vnDegree, codec.cnDegree, codec.bpMaxIter);

    bpIterTotal = bpIterTotal + iterThisTime;

    if checkFlag == 1
        bpPassBlocks = bpPassBlocks + 1;
    else
        alpha = min(codec.alphaOsd, size(outputModifiedLlr, 1));
        weightVec = codec.osdWeights(1:(alpha + 1));
        modifiedLlr = weightVec * [llrBlock; outputModifiedLlr(1:alpha,:)];
        [~, testNum, cHat] = OperationNum_modified_OLDn_decoder(llrBlock, modifiedLlr, codec.G, codec.oldSegments);
        oldFallbackBlocks = oldFallbackBlocks + 1;
        oldTestsTotal = oldTestsTotal + testNum;
    end

    infoBitsHat(infoIdx) = uint8(cHat(1:codec.K)).';
end

info = struct();
info.bpPassBlocks = bpPassBlocks;
info.oldFallbackBlocks = oldFallbackBlocks;
info.avgBpIter = bpIterTotal / codec.numBlocksPerFrame;
if oldFallbackBlocks > 0
    info.avgOldTests = oldTestsTotal / oldFallbackBlocks;
else
    info.avgOldTests = 0;
end
end

%% =========================================================
function [startIdx, peak, cfoHz] = findFrame(buf, cfg)

pre = cfg.preamble(:);
Lp  = numel(pre);

corrVal = conv(buf, flipud(conj(pre)), 'valid');
engVal  = conv(abs(buf).^2, ones(Lp,1), 'valid');

metric = abs(corrVal).^2 ./ (engVal * sum(abs(pre).^2) + eps);
[peak, startIdx] = max(metric);

if isempty(peak) || peak < cfg.syncPeakThresh
    startIdx = [];
    cfoHz = 0;
    return;
end

preRx = buf(startIdx : startIdx + 2*cfg.preHalfLen - 1);
r1 = preRx(1:cfg.preHalfLen);
r2 = preRx(cfg.preHalfLen+1:end);
cfoHz = angle(sum(conj(r1).*r2)) * cfg.fs / (2*pi*cfg.preHalfLen);

end

%% =========================================================
function stats = initStats(cfg)

stats.tStart = tic;

stats.detectedFrames  = 0;
stats.completedFrames = 0;
stats.validatedFrames = 0;
stats.goodFrames      = 0;
stats.errFrames       = 0;

stats.totalBits = 0;
stats.errBits   = 0;

stats.sumSnr = 0;
stats.sumCfo = 0;
stats.sumBpPassBlocks = 0;
stats.sumOldFallbackBlocks = 0;
stats.sumBpIter = 0;
stats.sumOldTests = 0;

stats.overrunCount = 0;
stats.skipFrames = cfg.skipFrames;

stats.berHist = [];
stats.detHist = [];

end

function stats = updateStats(stats, cfg, frameIndex, bitErr, ber, validated, diagInfo)

stats.completedFrames = stats.completedFrames + 1;

if frameIndex <= stats.skipFrames
    return;
end

if validated
    stats.validatedFrames = stats.validatedFrames + 1;
    stats.totalBits = stats.totalBits + cfg.infoBitsPerFrame;
    stats.errBits   = stats.errBits + bitErr;

    stats.sumSnr = stats.sumSnr + diagInfo.snrEst;
    stats.sumCfo = stats.sumCfo + abs(diagInfo.cfoHz);
    stats.sumBpPassBlocks = stats.sumBpPassBlocks + diagInfo.bpPassBlocks;
    stats.sumOldFallbackBlocks = stats.sumOldFallbackBlocks + diagInfo.oldFallbackBlocks;
    if isfinite(diagInfo.avgBpIter)
        stats.sumBpIter = stats.sumBpIter + diagInfo.avgBpIter;
    end
    if isfinite(diagInfo.avgOldTests)
        stats.sumOldTests = stats.sumOldTests + diagInfo.avgOldTests;
    end

    stats.berHist(end+1) = ber; %#ok<AGROW>
    stats.detHist(end+1) = frameIndex; %#ok<AGROW>

    if bitErr == 0
        stats.goodFrames = stats.goodFrames + 1;
    else
        stats.errFrames = stats.errFrames + 1;
    end
end

end

function printStats(stats, cfg, decoderStats)

if nargin < 3
    decoderStats = [];
end

elapsed = toc(stats.tStart);
validated = stats.validatedFrames;

fprintf('\n========== Coded BPSK MB-OLD Statistics ==========\n');
fprintf('Elapsed time          : %.2f s\n', elapsed);
fprintf('Codec profile         : %s\n', cfg.codec.profile);
fprintf('Skipped frames        : %d\n', stats.skipFrames);
fprintf('Detected frames       : %d\n', max(stats.detectedFrames - stats.skipFrames, 0));
fprintf('Completed decodes     : %d\n', max(stats.completedFrames - stats.skipFrames, 0));
fprintf('Validated frames      : %d\n', validated);
fprintf('Good frames           : %d\n', stats.goodFrames);
fprintf('Error frames          : %d\n', stats.errFrames);
fprintf('Overruns              : %d\n', stats.overrunCount);
if cfg.useCudaDecoder && ~isempty(decoderStats)
    fprintf('CUDA decoded batches  : %d\n', decoderStats.decodedBatches);
    fprintf('CUDA pending words    : %d\n', decoderStats.currentPendingCodewords);
    fprintf('CUDA average batch    : %.2f\n', decoderStats.averageBatchSize);
end

if validated > 0
    ber = stats.errBits / max(stats.totalBits, 1);
    fer = stats.errFrames / validated;
    avgSnr = stats.sumSnr / validated;
    avgCfo = stats.sumCfo / validated;
    avgBpPass = stats.sumBpPassBlocks / validated;
    avgOldFallback = stats.sumOldFallbackBlocks / validated;
    goodput = stats.goodFrames * cfg.infoBitsPerFrame / elapsed / 1e6;

    fprintf('Info BER (validated)  : %.3e\n', ber);
    fprintf('FER (validated)       : %.3e\n', fer);
    fprintf('Average SNR est       : %.2f dB\n', avgSnr);
    fprintf('Average |CFO|         : %.2f Hz\n', avgCfo);
    fprintf('Avg BP-pass blocks    : %.2f / %d\n', avgBpPass, cfg.codec.numBlocksPerFrame);
    fprintf('Avg OLD fallback      : %.2f blocks/frame\n', avgOldFallback);
    if cfg.useCudaDecoder
        fprintf('Average BP iterations : n/a (CUDA stream path)\n');
        fprintf('Average OLD tests     : n/a (CUDA stream path)\n');
    else
        avgBpIter = stats.sumBpIter / validated;
        avgOldTests = stats.sumOldTests / validated;
        fprintf('Average BP iterations : %.2f\n', avgBpIter);
        fprintf('Average OLD tests     : %.2f\n', avgOldTests);
    end
    fprintf('Zero-error goodput    : %.3f Mb/s\n', goodput);
end

fprintf('=================================================\n\n');

end

%% =========================================================
function plotState = initPlots()

plotState.fig = figure('Name', 'Coded BPSK MB-OLD RX Debug', 'NumberTitle', 'off');

subplot(2,1,1);
plotState.hConst = plot(nan, nan, '.', 'MarkerSize', 8);
grid on;
xlim([-2 2]);
ylim([-2 2]);
xlabel('In-Phase');
ylabel('Quadrature');
title('Equalized BPSK Constellation');

subplot(2,1,2);
plotState.hBer = plot(nan, nan, '-o', 'LineWidth', 1);
grid on;
set(gca, 'YScale', 'log');
ylim([1e-6 1]);
xlabel('Detected frame index');
ylabel('Info BER');
title('Validated Frame BER');

drawnow;

end

function plotState = updatePlots(plotState, eqSym, stats)

if ~ishandle(plotState.fig)
    plotState = initPlots();
end

subplot(2,1,1);
nShow = min(numel(eqSym), 1500);
set(plotState.hConst, 'XData', real(eqSym(1:nShow)), 'YData', imag(eqSym(1:nShow)));

subplot(2,1,2);
if ~isempty(stats.detHist)
    set(plotState.hBer, 'XData', stats.detHist, 'YData', max(stats.berHist, 1e-6));
end

drawnow limitrate;

end

%% =========================================================
function pre = buildPreamble(L)
old = rng;
rng(11, 'twister');
bits = randi([0 1], L, 1, 'uint8');
rng(old);

half = bitsToBPSK(bits);
pre = [half; half];
pre = pre / sqrt(mean(abs(pre).^2));
end

function bits = buildReferenceBits(N)
old = rng;
rng(20260420, 'twister');
bits = randi([0 1], N, 1, 'uint8');
rng(old);
end

function P = buildPilotMatrix(nSC, nPilot)
old = rng;
rng(73, 'twister');
bits = randi([0 1], nSC * nPilot, 1, 'uint8');
rng(old);

P = reshape(bitsToBPSK(bits), nSC, nPilot);
end

function sym = bitsToBPSK(bits)
bits = uint8(bits(:));
sym = 1 - 2*double(bits);
sym = complex(sym, 0);
end

function sym = bpskHardSlice(x)
sym = complex(ones(size(x)), 0);
sym(real(x) < 0) = -1;
end

function result = finalizeResult(stats, cfg, decoderStats)

if nargin < 3
    decoderStats = [];
end

elapsed = toc(stats.tStart);
validated = stats.validatedFrames;

result = struct();
result.elapsedSec = elapsed;
result.codecProfile = cfg.codec.profile;
result.skippedFrames = stats.skipFrames;
result.detectedFrames = max(stats.detectedFrames - stats.skipFrames, 0);
result.completedFrames = max(stats.completedFrames - stats.skipFrames, 0);
result.validatedFrames = validated;
result.goodFrames = stats.goodFrames;
result.errorFrames = stats.errFrames;
result.overruns = stats.overrunCount;
result.infoBitsPerFrame = cfg.infoBitsPerFrame;
result.codedBitsPerFrame = cfg.codedBitsPerFrame;
result.txGain = cfg.txGain;
result.rxGain = cfg.rxGain;

if validated > 0
    result.ber = stats.errBits / max(stats.totalBits, 1);
    result.fer = stats.errFrames / validated;
    result.avgSnrDb = stats.sumSnr / validated;
    result.avgAbsCfoHz = stats.sumCfo / validated;
    result.avgBpPassBlocks = stats.sumBpPassBlocks / validated;
    result.avgOldFallbackBlocks = stats.sumOldFallbackBlocks / validated;
    if cfg.useCudaDecoder
        result.avgBpIter = NaN;
        result.avgOldTests = NaN;
    else
        result.avgBpIter = stats.sumBpIter / validated;
        result.avgOldTests = stats.sumOldTests / validated;
    end
    result.goodputMbps = stats.goodFrames * cfg.infoBitsPerFrame / max(elapsed, eps) / 1e6;
else
    result.ber = NaN;
    result.fer = NaN;
    result.avgSnrDb = NaN;
    result.avgAbsCfoHz = NaN;
    result.avgBpPassBlocks = NaN;
    result.avgOldFallbackBlocks = NaN;
    result.avgBpIter = NaN;
    result.avgOldTests = NaN;
    result.goodputMbps = 0;
end

if cfg.useCudaDecoder && ~isempty(decoderStats)
    result.cudaDecodedBatches = decoderStats.decodedBatches;
    result.cudaCurrentPendingCodewords = decoderStats.currentPendingCodewords;
    result.cudaAverageBatchSize = decoderStats.averageBatchSize;
    result.cudaSubmittedCodewords = decoderStats.submittedCodewords;
    result.cudaDecodedCodewords = decoderStats.decodedCodewords;
    result.cudaBpSuccessCodewords = decoderStats.bpSuccessCodewords;
    result.cudaOsdFallbackCodewords = decoderStats.osdFallbackCodewords;
else
    result.cudaDecodedBatches = NaN;
    result.cudaCurrentPendingCodewords = NaN;
    result.cudaAverageBatchSize = NaN;
    result.cudaSubmittedCodewords = NaN;
    result.cudaDecodedCodewords = NaN;
    result.cudaBpSuccessCodewords = NaN;
    result.cudaOsdFallbackCodewords = NaN;
end

end

function cfg = applyOverrides(cfg, overrides)

if isempty(overrides)
    return;
end

overrideFields = fieldnames(overrides);
for idx = 1:numel(overrideFields)
    fieldName = overrideFields{idx};
    cfg.(fieldName) = overrides.(fieldName);
end

end
