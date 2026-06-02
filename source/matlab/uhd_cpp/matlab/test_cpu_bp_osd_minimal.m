% Random-codeword information BER/FER sweep for CPU BP-OSD on CCSDS LDPC (128,64).
% This script does not call the CUDA DLL.

clear;
clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fullfile(thisDir, '..', '..', 'MB-OLD', 'My_LDPC'));

rng(7);

H = CCSDS_ldpc_n128_k64_H();
[m, n] = size(H);
k = n - m;
rate = k / n;
[G, infoCols] = cpu_ldpc_system_from_h(H); 

framesPerPoint = 1000;
ebn0DbList = 3.0;

opts = struct();
opts.maxIter = 20;
opts.normalization = 0.80;
opts.damping = 0.15;

% OSD 参数
opts.osdOrder = 3;
opts.osdMaxFlipPositions = k;

% 对于 k=64, 完整 order-3 OSD 测试数为：
% 1 + C(64,1) + C(64,2) + C(64,3) = 43745
opts.osdMaxTests = 43745;

% 论文中的 likelihood accumulation 参数
opts.alpha = 1.0;
opts.useLlrAccumulation = true;
opts.osdLlrSource = 'accumulated';

results = table('Size', [numel(ebn0DbList), 10], ...
    'VariableTypes', {'double', 'double', 'double', 'double', 'double', ...
                      'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'EbN0dB', 'Frames', 'InfoBitErrors', ...
                      'InfoBlockErrors', 'InfoBER', 'InfoFER', ...
                      'CodeBitErrors', 'ParityOkRate', 'OsdUseRate', ...
                      'AvgBpIter'});

fprintf('CPU BP-OSD random-codeword info BER sweep, CCSDS LDPC (%d,%d), frames/SNR=%d\n', ...
    n, k, framesPerPoint);
fprintf('%8s %8s %14s %14s %12s %12s %14s %10s %10s\n', ...
    'EbN0dB', 'Frames', 'InfoBitErr', 'InfoBlkErr', ...
    'InfoBER', 'InfoFER', 'CodeBitErr', 'OSDRate', 'AvgIter');

for snrIdx = 1:numel(ebn0DbList)
    ebn0Db = ebn0DbList(snrIdx);
    sigma = sqrt(1 / (2 * rate * 10^(ebn0Db / 10)));

    txInfoBits = uint8(randi([0, 1], k, framesPerPoint));
    txBits = uint8(mod(double(G.') * double(txInfoBits), 2));
    assert(nnz(mod(H * double(txBits), 2)) == 0, ...
        'Generated random codewords failed parity checks.');

    noise = sigma * randn(n, framesPerPoint);
    rx = 1 - 2 * double(txBits) + noise;
    llr = 2 * rx / (sigma^2);

    [rxInfoBits, codeBits, diag] = cpu_bp_osd_decode_1(llr, H, opts);

    infoBitErrors = nnz(rxInfoBits ~= txInfoBits);
    infoBlockErrors = nnz(any(rxInfoBits ~= txInfoBits, 1));
    infoBer = infoBitErrors / numel(txInfoBits);
    infoFer = infoBlockErrors / framesPerPoint;
    codeBitErrors = nnz(codeBits ~= txBits);
    parityOkRate = nnz(diag.parityOk) / framesPerPoint;
    osdUseRate = nnz(diag.usedOsd) / framesPerPoint;
    avgBpIter = mean(diag.bpIterations);

    results{snrIdx, :} = [ebn0Db, framesPerPoint, infoBitErrors, ...
        infoBlockErrors, infoBer, infoFer, codeBitErrors, parityOkRate, ...
        osdUseRate, avgBpIter];

    fprintf('%8.2f %8d %14d %14d %12.4e %12.4e %14d %10.3f %10.2f\n', ...
        ebn0Db, framesPerPoint, infoBitErrors, infoBlockErrors, ...
        infoBer, infoFer, codeBitErrors, osdUseRate, avgBpIter);
end

disp(results);
