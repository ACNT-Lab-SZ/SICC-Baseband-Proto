function x310_video_phy_link(mode)
% x310_video_phy_link("tx")  % 在发射端 MATLAB 会话运行
% x310_video_phy_link("rx")  % 在接收端 MATLAB 会话运行
%
% 工程骨架：
%   - NR-like OFDM: 15 kHz SCS, 1024 FFT, CP=72
%   - DVB-S2 LDPC: rate=1/2, normal frame 64800 bits
%   - 45个数据OFDM符号 * 720个子载波 * 2 bit/QPSK = 64800 bits
%   - 3个全频导频OFDM符号用于信道估计
%
% 依赖：
%   - Wireless Testbench Support Package for NI USRP Radios
%   - Communications Toolbox
%
% 建议先同轴+衰减器连通，再上空口。

cfg = localCfg();

mode = lower(string(mode));
switch mode
    case "tx"
        runTx(cfg);
    case "rx"
        runRx(cfg);
    otherwise
        error('mode 必须是 "tx" 或 "rx"');
end

end

%% ========================= 配置 =========================
function cfg = localCfg()

% ---------------- SDR / USRP 参数 ----------------
cfg.txIP   = "192.168.40.2";
cfg.rxIP   = "192.168.50.2";
cfg.fc     = 2.45e9;          % 射频中心频率
cfg.mcr    = 184.32e6;        % X310 主时钟
cfg.fs     = 15.36e6;         % 15 kHz * 1024 = 15.36 Msps
cfg.interp = round(cfg.mcr/cfg.fs);   % = 12
cfg.decim  = cfg.interp;

cfg.txGain = 20;               % 先保守设置
cfg.rxGain = 20;

% ---------------- OFDM 参数 ----------------
cfg.Nfft       = 1024;
cfg.CP         = 72;
cfg.numSym     = 48;          % 总OFDM符号数
cfg.pilotSym   = [1 17 33];   % 3个块导频符号
cfg.dataSym    = setdiff(1:cfg.numSym, cfg.pilotSym); % 共45个数据符号
cfg.activeSC   = 720;         % 不含DC，左右各360个子载波
cfg.bitsPerSym = 2;           % QPSK

% 720个有效子载波，跳过DC
negIdx = (cfg.Nfft/2 - cfg.activeSC/2 + 1) : (cfg.Nfft/2);
posIdx = (cfg.Nfft/2 + 2) : (cfg.Nfft/2 + 1 + cfg.activeSC/2);
cfg.usedIdx = [negIdx, posIdx];

% ---------------- 前导参数 ----------------
cfg.preHalfLen = 256;         % 重复半段长度
cfg.preamble   = buildPreamble(cfg.preHalfLen);

% ---------------- LDPC 参数 ----------------
% DVB-S2 normal frame, rate = 1/2
H = dvbs2ldpc(1/2);
cfg.encCfg = ldpcEncoderConfig(H);
cfg.decCfg = ldpcDecoderConfig(H, 'layered-bp');
cfg.maxIter = 25;

cfg.K = cfg.encCfg.NumInformationBits;   % 32400
cfg.N = cfg.encCfg.BlockLength;          % 64800

assert(cfg.K == 32400, 'DVB-S2 R=1/2 信息比特应为32400');
assert(cfg.N == 64800, 'DVB-S2 R=1/2 码长应为64800');
assert(numel(cfg.dataSym) * cfg.activeSC * cfg.bitsPerSym == cfg.N, ...
    'OFDM资源与LDPC码长不匹配');

% 每帧信息区：4字节序号 + 用户数据
cfg.seqBytes  = 4;
cfg.infoBytes = cfg.K / 8;              % 4050 bytes
cfg.userBytes = cfg.infoBytes - cfg.seqBytes; % 4046 bytes

% ---------------- 导频 ----------------
cfg.pilotMat = buildPilotMatrix(cfg.activeSC, numel(cfg.pilotSym));

% ---------------- 文件接口 ----------------
cfg.payloadFile = 'video_payload.bin';  % 可替换成视频码流文件
cfg.outputFile  = 'rx_payload.bin';

% ---------------- 帧长 ----------------
cfg.symLen   = cfg.Nfft + cfg.CP;
cfg.frameLen = numel(cfg.preamble) + cfg.numSym * cfg.symLen;

% RX 每次多抓一点，便于搜帧
cfg.rxSamplesPerFrame = 2 * cfg.frameLen;

% ---------------- 业务模式与统计 ----------------
cfg.trafficMode = "test";   % "test" or "video"
% test: 发送确定性伪随机载荷，可统计BER/FER/吞吐量
% video: 发送真实文件/视频载荷，只统计吞吐量，BER无法逐位比较

cfg.testSeed    = 20260418; % test模式下按 frameID 派生参考比特
cfg.reportEvery = 50;       % 每50帧打印一次统计

end

%% ========================= 发射端 =========================
function runTx(cfg)

tx = comm.SDRuTransmitter( ...
    Platform            = "X310", ...
    IPAddress           = cfg.txIP, ...
    CenterFrequency     = cfg.fc, ...
    MasterClockRate     = cfg.mcr, ...
    InterpolationFactor = cfg.interp, ...
    Gain                = cfg.txGain);

c = onCleanup(@() release(tx)); 

fprintf('TX启动: IP=%s, fc=%.3f GHz, fs=%.2f Msps\n', ...
    cfg.txIP, cfg.fc/1e9, cfg.fs/1e6);

frameID = uint32(0);
t0 = tic;

while true
    infoBits = buildInfoBits(cfg, frameID);
    txFrame  = buildTxFrame(infoBits, cfg);

    underrun = tx(txFrame);
    if underrun ~= 0
        fprintf('[TX] underrun @ frame=%u\n', frameID);
    end

    if mod(double(frameID), 50) == 0
        fprintf('[TX] frame=%u, elapsed=%.2f s\n', frameID, toc(t0));
    end

    frameID = frameID + 1;
end

end

%% ========================= 接收端 =========================
function runRx(cfg)

rx = comm.SDRuReceiver( ...
    Platform         = "X310", ...
    IPAddress        = cfg.rxIP, ...
    CenterFrequency  = cfg.fc, ...
    MasterClockRate  = cfg.mcr, ...
    DecimationFactor = cfg.decim, ...
    Gain             = cfg.rxGain, ...
    SamplesPerFrame  = cfg.rxSamplesPerFrame, ...
    OutputDataType   = "double");

c1 = onCleanup(@() release(rx)); %#ok<NASGU>

fidOut = fopen(cfg.outputFile, 'ab');
if fidOut < 0
    error('无法打开输出文件 %s', cfg.outputFile);
end
c2 = onCleanup(@() fclose(fidOut)); %#ok<NASGU>

stats = initLinkStats(cfg);

fprintf('RX启动: IP=%s, fc=%.3f GHz, fs=%.2f Msps, mode=%s\n', ...
    cfg.rxIP, cfg.fc/1e9, cfg.fs/1e6, cfg.trafficMode);

buf = complex(zeros(0,1));

while true
    [y, ~, overrun] = rx();
    if overrun ~= 0
        fprintf('[RX] overrun detected\n');
    end

    buf = [buf; y]; %#ok<AGROW>

    while numel(buf) >= cfg.frameLen
        [startIdx, peak, cfoHz] = findFrame(buf, cfg);

        if isempty(startIdx)
            buf = buf(max(1, numel(buf)-cfg.frameLen+1):end);
            break;
        end

        if startIdx + cfg.frameLen - 1 > numel(buf)
            buf = buf(startIdx:end);
            break;
        end

        oneFrame = buf(startIdx : startIdx + cfg.frameLen - 1);
        [infoBitsHat, codedBitsHard, diagInfo] = decodeOneFrame(oneFrame, cfg, cfoHz);

        seqNum = bitsToUint32(infoBitsHat(1:32));
        payloadBytes = bitsToBytes(infoBitsHat(33:end));

        % ---------- 统计 ----------
        switch lower(string(cfg.trafficMode))
            case "test"
                infoBitsRef  = buildInfoBitsFromFrameID(cfg, uint32(seqNum));
                codedBitsRef = ldpcEncode(infoBitsRef, cfg.encCfg);

                preErrBits  = sum(uint8(codedBitsRef(:)) ~= uint8(codedBitsHard(:)));
                postErrBits = sum(uint8(infoBitsRef(:))  ~= uint8(infoBitsHat(:)));

                frameErr = (postErrBits > 0) || (~diagInfo.parityOK);
                payloadOK = ~frameErr;

            case "video"
                preErrBits  = NaN;
                postErrBits = NaN;

                % video模式没有参考比特，无法统计BER
                % 这里只用LDPC parity是否通过来近似判断一帧是否成功
                frameErr = ~diagInfo.parityOK;
                payloadOK = ~frameErr;

            otherwise
                error('未知 trafficMode');
        end

        if payloadOK
            fwrite(fidOut, payloadBytes, 'uint8');
        end

        stats = updateLinkStats(stats, cfg, ...
            preErrBits, postErrBits, frameErr, payloadOK, diagInfo);

        if mod(stats.totalFrames, cfg.reportEvery) == 0
            printLinkStats(stats, cfg);
        end

        % 单帧日志
        if lower(string(cfg.trafficMode)) == "test"
            fprintf(['[RX] seq=%10u | peak=%.3f | CFO=%8.1f Hz | SNR=%5.1f dB | ' ...
                     'iter=%2d | preBER=%8.3e | postBER=%8.3e | %s\n'], ...
                seqNum, peak, diagInfo.cfoHz, diagInfo.snrEst, diagInfo.iter, ...
                stats.lastPreBER, stats.lastPostBER, ternary(payloadOK,'OK','FAIL'));
        else
            fprintf(['[RX] seq=%10u | peak=%.3f | CFO=%8.1f Hz | SNR=%5.1f dB | ' ...
                     'iter=%2d | %s\n'], ...
                seqNum, peak, diagInfo.cfoHz, diagInfo.snrEst, diagInfo.iter, ...
                ternary(payloadOK,'OK','FAIL'));
        end

        buf = buf(startIdx + cfg.frameLen : end);
    end
end

end

%% ========================= 发送帧构造 =========================
function infoBits = buildInfoBits(cfg, frameID)

switch lower(string(cfg.trafficMode))
    case "test"
        infoBits = buildInfoBitsFromFrameID(cfg, frameID);

    case "video"
        infoBits = buildInfoBitsFromFile(cfg, frameID);

    otherwise
        error('cfg.trafficMode 必须是 "test" 或 "video"');
end

end

function infoBits = buildInfoBitsFromFrameID(cfg, frameID)
% test模式：根据 frameID 生成可重构的确定性载荷
seqBits = uint32ToBits(uint32(frameID));

old = rng;
rng(double(cfg.testSeed) + double(frameID), "twister");
payloadBytes = randi([0 255], cfg.userBytes, 1, 'uint8');
rng(old);

infoBits = int8([seqBits; bytesToBits(payloadBytes)]);
end

function infoBits = buildInfoBitsFromFile(cfg, frameID)
% video模式：真实文件/视频载荷，只能统计goodput/FER，不能逐位BER

persistent fidIn
if isempty(fidIn)
    if exist(cfg.payloadFile, 'file') == 2
        fidIn = fopen(cfg.payloadFile, 'rb');
        if fidIn < 0
            error('无法打开输入文件 %s', cfg.payloadFile);
        end
    else
        error('video模式下找不到输入文件 %s', cfg.payloadFile);
    end
end

seqBits = uint32ToBits(uint32(frameID));

[d, cnt] = fread(fidIn, cfg.userBytes, 'uint8=>uint8');
if cnt < cfg.userBytes
    fseek(fidIn, 0, 'bof');
    [d2, cnt2] = fread(fidIn, cfg.userBytes - cnt, 'uint8=>uint8');
    d = [d; d2];
    cnt = cnt + cnt2;
end
if cnt < cfg.userBytes
    d = [d; zeros(cfg.userBytes-cnt,1,'uint8')];
end

infoBits = int8([seqBits; bytesToBits(d(1:cfg.userBytes))]);
end

function txFrame = buildTxFrame(infoBits, cfg)

% LDPC编码
codedBits = ldpcEncode(infoBits, cfg.encCfg);

% QPSK映射，长度 = 32400 symbols
dataSym = bitsToQPSK(codedBits);

% 资源映射：720 x 45
dataGrid = reshape(dataSym, cfg.activeSC, numel(cfg.dataSym));

% 频域资源网格（fftshift后索引）
grid = complex(zeros(cfg.Nfft, cfg.numSym));
grid(cfg.usedIdx, cfg.pilotSym) = cfg.pilotMat;
grid(cfg.usedIdx, cfg.dataSym)  = dataGrid;

% OFDM调制
td = ifft(ifftshift(grid, 1), cfg.Nfft, 1);

% 加CP
tdcp = [td(end-cfg.CP+1:end,:); td];
payloadWave = tdcp(:);

% 前导 + 负载
txFrame = [cfg.preamble; payloadWave];

% 归一化
txFrame = 0.7 * txFrame / max(abs(txFrame) + 1e-12);

end

%% ========================= 接收解调 =========================
function [infoBitsHat, codedBitsHard, diagInfo] = decodeOneFrame(rxFrame, cfg, cfoHz)

% CFO校正
n = (0:numel(rxFrame)-1).';
rxFrame = rxFrame .* exp(-1j * 2*pi * cfoHz / cfg.fs * n);

% 去前导
r = rxFrame(numel(cfg.preamble)+1:end);

% OFDM按符号重组
rMat = reshape(r, cfg.symLen, cfg.numSym);
rMat = rMat(cfg.CP+1:end, :);

% FFT
R = fftshift(fft(rMat, cfg.Nfft, 1), 1);
Y = R(cfg.usedIdx, :);   % 720 x 48

% 导频信道估计
Hp = Y(:, cfg.pilotSym) ./ cfg.pilotMat;  % 720 x 3
symAxis = 1:cfg.numSym;
Hest = complex(zeros(cfg.activeSC, cfg.numSym));
for k = 1:cfg.activeSC
    Hest(k,:) = interp1(cfg.pilotSym, Hp(k,:), symAxis, 'linear', 'extrap');
end

% 数据抽取
Yd = Y(:, cfg.dataSym);
Hd = Hest(:, cfg.dataSym);

% 先做一次粗估计噪声
eq0 = Yd ./ max(Hd, 1e-9);
hard0sym = qpskHardSlice(eq0(:));
resid = Yd(:) - Hd(:).*hard0sym;
noiseVar = max(mean(abs(resid).^2), 1e-8);

% MMSE均衡
eq = conj(Hd).*Yd ./ (abs(Hd).^2 + noiseVar);

% 软解调
sigma2eq = noiseVar ./ max(abs(Hd(:)).^2, 1e-9);
llr = qpskSoftDemap(eq(:), sigma2eq);

% LDPC前硬判决，用于 pre-FEC BER
codedBitsHard = int8(llr < 0);   % LLR>0 -> bit 0, LLR<0 -> bit 1

% LDPC译码
[infoBitsHat, actIter, finalParityChecks] = ldpcDecode( ...
    llr, cfg.decCfg, cfg.maxIter, ...
    DecisionType='hard', Termination='early');

diagInfo.parityOK = all(finalParityChecks(:) == 0);
diagInfo.iter     = actIter(1);
diagInfo.cfoHz    = cfoHz;
diagInfo.snrEst   = 10*log10(mean(abs(Hd(:)).^2) / noiseVar);

end

%% ========================= 搜帧与CFO =========================
function [startIdx, peak, cfoHz] = findFrame(buf, cfg)

pre = cfg.preamble(:);
Lp  = numel(pre);

corrVal = conv(buf, flipud(conj(pre)), 'valid');
engVal  = conv(abs(buf).^2, ones(Lp,1), 'valid');

metric = abs(corrVal).^2 ./ (engVal * sum(abs(pre).^2) + eps);
[peak, startIdx] = max(metric);

% 阈值可根据现场环境调整
if isempty(peak) || peak < 0.55
    startIdx = [];
    cfoHz = 0;
    return;
end

% 利用重复半前导估计CFO
preRx = buf(startIdx : startIdx + 2*cfg.preHalfLen - 1);
r1 = preRx(1:cfg.preHalfLen);
r2 = preRx(cfg.preHalfLen+1:end);
cfoHz = angle(sum(conj(r1).*r2)) * cfg.fs / (2*pi*cfg.preHalfLen);

end

%% ========================= 底层辅助函数 =========================
function pre = buildPreamble(L)
old = rng;
rng(11);
bits = randi([0 1], 2*L, 1, 'uint8');
rng(old);

half = bitsToQPSK(bits);
pre = [half; half];
pre = pre / sqrt(mean(abs(pre).^2));
end

function P = buildPilotMatrix(nSC, nPilot)
old = rng;
rng(73);
bits = randi([0 1], 2*nSC*nPilot, 1, 'uint8');
rng(old);

P = reshape(bitsToQPSK(bits), nSC, nPilot);
end

function sym = bitsToQPSK(bits)
bits = uint8(bits(:));
assert(mod(numel(bits),2)==0, 'QPSK映射输入比特数必须为偶数');

b = reshape(bits, 2, []).';
iPart = 1 - 2*double(b(:,1));
qPart = 1 - 2*double(b(:,2));
sym = complex(iPart, qPart) / sqrt(2);
end

function llr = qpskSoftDemap(sym, sigma2)
sym = sym(:);
sigma2 = double(sigma2(:));
sigma2 = max(sigma2, 1e-12);

llrI = 2*sqrt(2) * real(sym) ./ sigma2;
llrQ = 2*sqrt(2) * imag(sym) ./ sigma2;

llr = zeros(2*numel(sym),1);
llr(1:2:end) = llrI;
llr(2:2:end) = llrQ;
end

function sym = qpskHardSlice(x)
xr = real(x);
xi = imag(x);
sym = complex(sign0(xr), sign0(xi)) / sqrt(2);
end

function y = sign0(x)
y = ones(size(x));
y(x < 0) = -1;
end

function bits = bytesToBits(bytes)
bytes = uint8(bytes(:));
N = numel(bytes);
bitsMat = zeros(N,8,'uint8');
for k = 1:8
    bitsMat(:,k) = bitget(bytes, 9-k);
end
bits = reshape(bitsMat.', [], 1);
end

function bytes = bitsToBytes(bits)
bits = uint8(bits(:));
assert(mod(numel(bits),8)==0, 'bitsToBytes输入长度必须是8的整数倍');

B = reshape(bits, 8, []).';
w = uint8(2.^(7:-1:0));
bytes = uint8(sum(double(B).*double(w), 2));
end

function bits = uint32ToBits(x)
x = uint32(x);
bits = zeros(32,1,'uint8');
for k = 1:32
    bits(k) = bitget(x, 33-k);
end
end

function x = bitsToUint32(bits)
bits = uint8(bits(:));
assert(numel(bits) >= 32, '需要至少32比特');
x = uint32(0);
for k = 1:32
    x = bitor(bitshift(x,1), uint32(bits(k)~=0));
end
end

function stats = initLinkStats(cfg)

stats.tStart = tic;

stats.totalFrames = 0;
stats.okFrames    = 0;
stats.errFrames   = 0;

stats.totalPreBits  = 0;
stats.totalPostBits = 0;

stats.preErrBits    = 0;
stats.postErrBits   = 0;

stats.totalInfoBits = 0;   % 成功接收的信息比特（含seq）
stats.totalUserBits = 0;   % 成功接收的业务比特（不含seq）

stats.lastPreBER  = NaN;
stats.lastPostBER = NaN;

stats.sumIter = 0;
stats.sumSnr  = 0;
stats.sumCfo  = 0;

stats.cfgK        = cfg.K;
stats.cfgUserBits = cfg.userBytes * 8;

end

function stats = updateLinkStats(stats, cfg, ...
    preErrBits, postErrBits, frameErr, payloadOK, diagInfo)

stats.totalFrames = stats.totalFrames + 1;
stats.sumIter = stats.sumIter + double(diagInfo.iter);
stats.sumSnr  = stats.sumSnr  + double(diagInfo.snrEst);
stats.sumCfo  = stats.sumCfo  + abs(double(diagInfo.cfoHz));

if ~isnan(preErrBits)
    stats.preErrBits   = stats.preErrBits + double(preErrBits);
    stats.totalPreBits = stats.totalPreBits + double(cfg.N);
    stats.lastPreBER   = double(preErrBits) / double(cfg.N);
end

if ~isnan(postErrBits)
    stats.postErrBits    = stats.postErrBits + double(postErrBits);
    stats.totalPostBits  = stats.totalPostBits + double(cfg.K);
    stats.lastPostBER    = double(postErrBits) / double(cfg.K);
end

if frameErr
    stats.errFrames = stats.errFrames + 1;
else
    stats.okFrames = stats.okFrames + 1;
end

if payloadOK
    stats.totalInfoBits = stats.totalInfoBits + double(cfg.K);
    stats.totalUserBits = stats.totalUserBits + double(cfg.userBytes * 8);
end

end

function printLinkStats(stats, cfg)

elapsed = toc(stats.tStart);

fer = stats.errFrames / max(stats.totalFrames, 1);

if stats.totalPreBits > 0
    preBER = stats.preErrBits / stats.totalPreBits;
else
    preBER = NaN;
end

if stats.totalPostBits > 0
    postBER = stats.postErrBits / stats.totalPostBits;
else
    postBER = NaN;
end

% 吞吐量定义
grossInfoMbps = stats.totalFrames * cfg.K / elapsed / 1e6;
goodputMbps   = stats.totalUserBits / elapsed / 1e6;
frameRate     = stats.totalFrames / elapsed;

avgIter = stats.sumIter / max(stats.totalFrames, 1);
avgSnr  = stats.sumSnr  / max(stats.totalFrames, 1);
avgCfo  = stats.sumCfo  / max(stats.totalFrames, 1);

fprintf('\n========== Link Statistics ==========\n');
fprintf('Elapsed time          : %.2f s\n', elapsed);
fprintf('Total frames          : %d\n', stats.totalFrames);
fprintf('Successful frames     : %d\n', stats.okFrames);
fprintf('Error frames          : %d\n', stats.errFrames);
fprintf('Frame rate            : %.2f frame/s\n', frameRate);
fprintf('Gross info throughput : %.3f Mb/s\n', grossInfoMbps);
fprintf('User goodput          : %.3f Mb/s\n', goodputMbps);
fprintf('FER                   : %.3e\n', fer);

if lower(string(cfg.trafficMode)) == "test"
    fprintf('Pre-FEC BER           : %.3e\n', preBER);
    fprintf('Post-FEC BER          : %.3e\n', postBER);
else
    fprintf('Pre-FEC BER           : N/A (video mode)\n');
    fprintf('Post-FEC BER          : N/A (video mode)\n');
end

fprintf('Average LDPC iter     : %.2f\n', avgIter);
fprintf('Average SNR est       : %.2f dB\n', avgSnr);
fprintf('Average |CFO|         : %.2f Hz\n', avgCfo);
fprintf('=====================================\n\n');

end

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end