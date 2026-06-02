function [infoBits, codeBits, diag] = cpu_bp_osd_decode_1(llr, H, opts)
%CPU_BP_OSD_DECODE CPU-only BP-OSD decoder for LDPC codewords.
%
%   [infoBits, codeBits, diag] = cpu_bp_osd_decode(llr, H, opts)
%
%   llr convention:
%       positive LLR -> bit 0 is more likely
%       negative LLR -> bit 1 is more likely
%
%   This decoder first runs normalized min-sum BP. If BP does not satisfy
%   the parity checks, it falls back to order-N OSD.
%
%   Compared with a conventional serial BP-OSD implementation, this version
%   supports the likelihood-accumulation reliability proposed in
%       M. Jiang et al., "Reliability-Based Iterative Decoding of LDPC Codes
%       Using Likelihood Accumulation," IEEE Communications Letters, 2007.
%
%   The accumulated BP output is updated as
%       L_acc = L_app^(k) + alpha * L_acc,
%   with initialization L_acc = L_ch.  The OSD reliability is then |L_acc|.
%
%   Main options:
%       opts.maxIter              BP maximum iterations. Default: 20
%       opts.normalization        normalized-min-sum factor. Default: 0.80
%       opts.damping              VN-message damping factor. Default: 0.15
%       opts.osdOrder             OSD order. Default: 2
%       opts.osdMaxTests          maximum OSD candidates. Default: 4096
%       opts.osdMaxFlipPositions  weakest information positions to test.
%                                 Default: 32
%       opts.alpha                LLR accumulation factor. Default: 1.0
%       opts.useLlrAccumulation   enable accumulated-LLR OSD. Default: true
%       opts.useBpReliability     if accumulation is disabled, use BP APP
%                                 LLRs instead of channel LLRs. Default: true
%       opts.osdLlrSource         'auto', 'accumulated', 'final', 'best',
%                                 or 'channel'. Default: 'auto'
%
%   Notes:
%       - opts.alpha = 0 with opts.osdLlrSource = 'accumulated' reduces the
%         OSD input to the current/final BP APP LLR, matching conventional
%         serial BP-OSD reliability.
%       - opts.alpha = 1 performs direct LLR accumulation.
%       - This file intentionally does not load or call the CUDA DLL.

if nargin < 2 || isempty(H)
    H = CCSDS_ldpc_n128_k64_H();
end
if nargin < 3
    opts = struct();
end

opts = fill_defaults(opts);
H = logical(mod(H, 2));
[m, n] = size(H); %#ok<ASGLU>

if isvector(llr)
    llrMat = double(llr(:));
elseif size(llr, 1) == n
    llrMat = double(llr);
elseif size(llr, 2) == n
    llrMat = double(llr.');
else
    error('cpu_bp_osd_decode:BadLlrShape', ...
        'LLR must be N x frames, frames x N, or a length-N vector.');
end

[G, infoCols, pivotCols, rankH] = cpu_ldpc_system_from_h(H);
k = size(G, 1);
numFrames = size(llrMat, 2);

codeBits = false(n, numFrames);
infoBits = false(k, numFrames);

diag.parityOk = false(1, numFrames);
diag.usedOsd = false(1, numFrames);
diag.bpIterations = zeros(1, numFrames);
diag.bpSyndromeWeight = zeros(1, numFrames);
diag.osdTests = zeros(1, numFrames);
diag.osdLlrSource = cell(1, numFrames);
diag.infoCols = infoCols;
diag.pivotCols = pivotCols;
diag.rankH = rankH;
diag.options = opts;

for frame = 1:numFrames
    frameLlr = llrMat(:, frame);
    bp = normalized_min_sum_bp(frameLlr, H, opts);

    candidate = bp.hardBits;
    usedOsd = false;
    osdTests = 0;
    osdSource = '';

    if ~bp.parityOk
        usedOsd = true;
        [osdLlr, osdSource] = select_osd_llr(bp, frameLlr, opts);
        osd = osd_decode(osdLlr, G, H, opts);
        candidate = osd.codeBits;
        osdTests = osd.tests;
    end

    codeBits(:, frame) = candidate(:);
    infoBits(:, frame) = candidate(infoCols);
    diag.parityOk(frame) = all(mod(H * double(candidate(:)), 2) == 0);
    diag.usedOsd(frame) = usedOsd;
    diag.bpIterations(frame) = bp.iterations;
    diag.bpSyndromeWeight(frame) = bp.bestSyndromeWeight;
    diag.osdTests(frame) = osdTests;
    diag.osdLlrSource{frame} = osdSource;
end

infoBits = uint8(infoBits);
codeBits = uint8(codeBits);
end

function opts = fill_defaults(opts)
defaults.maxIter = 20;
defaults.normalization = 0.80;
defaults.damping = 0.15;
defaults.osdOrder = 2;
defaults.osdMaxTests = 4096;
defaults.osdMaxFlipPositions = 32;

% Likelihood-accumulation options.
defaults.alpha = 1.0;
defaults.useLlrAccumulation = true;
defaults.useBpReliability = true;
defaults.osdLlrSource = 'auto';

names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = defaults.(name);
    end
end

if opts.maxIter < 0 || fix(opts.maxIter) ~= opts.maxIter
    error('cpu_bp_osd_decode:BadOption', 'opts.maxIter must be a nonnegative integer.');
end
if opts.osdOrder < 0 || fix(opts.osdOrder) ~= opts.osdOrder
    error('cpu_bp_osd_decode:BadOption', 'opts.osdOrder must be a nonnegative integer.');
end
if opts.osdMaxTests < 1 || fix(opts.osdMaxTests) ~= opts.osdMaxTests
    error('cpu_bp_osd_decode:BadOption', 'opts.osdMaxTests must be a positive integer.');
end
if opts.osdMaxFlipPositions < 1 || fix(opts.osdMaxFlipPositions) ~= opts.osdMaxFlipPositions
    error('cpu_bp_osd_decode:BadOption', 'opts.osdMaxFlipPositions must be a positive integer.');
end
end

function bp = normalized_min_sum_bp(llr, H, opts)
[m, n] = size(H);
[checkIdx, varIdx] = find(H);
numEdges = numel(varIdx);

rowEdges = cell(m, 1);
colEdges = cell(n, 1);
for e = 1:numEdges
    rowEdges{checkIdx(e)}(end + 1) = e; %#ok<AGROW>
    colEdges{varIdx(e)}(end + 1) = e; %#ok<AGROW>
end

q = llr(varIdx);
rmsg = zeros(numEdges, 1);
appLlr = llr(:);
finalAppLlr = appLlr;

% L_acc = L^0 at initialization, corresponding to Eq. (4) in the paper.
accumulatedLlr = appLlr;

hardBits = appLlr < 0;
bestAppLlr = appLlr;
bestHardBits = hardBits;
bestSyndromeWeight = sum(mod(H * double(hardBits), 2));
parityOk = bestSyndromeWeight == 0;
iterations = 0;

for iter = 1:opts.maxIter
    iterations = iter;

    % Check-node update: normalized min-sum.
    for c = 1:m
        edges = rowEdges{c};
        vals = q(edges);
        signs = sign(vals);
        signs(signs == 0) = 1;
        absVals = abs(vals);

        [min1, idx1] = min(absVals);
        if numel(absVals) == 1
            min2 = min1;
        else
            tmp = absVals;
            tmp(idx1) = inf;
            min2 = min(tmp);
        end

        totalSign = prod(signs);
        for ii = 1:numel(edges)
            if ii == idx1
                mag = min2;
            else
                mag = min1;
            end
            rmsg(edges(ii)) = opts.normalization * totalSign * signs(ii) * mag;
        end
    end

    % A-posteriori LLR output of the current BP iteration.
    appLlr = llr(:);
    for v = 1:n
        appLlr(v) = appLlr(v) + sum(rmsg(colEdges{v}));
    end
    finalAppLlr = appLlr;

    % Likelihood accumulation: L_acc = L_app^(k) + alpha * L_acc.
    % If alpha = 1, this is direct LLR accumulation; if alpha = 0, this
    % reduces to using the current BP APP LLR.
    accumulatedLlr = appLlr + opts.alpha * accumulatedLlr;

    hardBits = appLlr < 0;
    syndromeWeight = sum(mod(H * double(hardBits), 2));
    if syndromeWeight < bestSyndromeWeight
        bestSyndromeWeight = syndromeWeight;
        bestAppLlr = appLlr;
        bestHardBits = hardBits;
    end

    if syndromeWeight == 0
        parityOk = true;
        bestAppLlr = appLlr;
        bestHardBits = hardBits;
        bestSyndromeWeight = 0;
        break;
    end

    % Variable-node extrinsic-message update with damping.
    for e = 1:numEdges
        fresh = appLlr(varIdx(e)) - rmsg(e);
        q(e) = (1 - opts.damping) * fresh + opts.damping * q(e);
    end
end

bp.parityOk = parityOk;
bp.iterations = iterations;
bp.hardBits = bestHardBits;
bp.bestAppLlr = bestAppLlr;
bp.finalAppLlr = finalAppLlr;
bp.accumulatedLlr = accumulatedLlr;
bp.bestSyndromeWeight = bestSyndromeWeight;
end

function [osdLlr, source] = select_osd_llr(bp, channelLlr, opts)
source = lower(strtrim(char(opts.osdLlrSource)));

if strcmp(source, 'auto')
    if opts.useLlrAccumulation
        source = 'accumulated';
    elseif opts.useBpReliability
        source = 'final';
    else
        source = 'channel';
    end
end

switch source
    case {'acc', 'accum', 'accumulated', 'llraccumulation'}
        osdLlr = bp.accumulatedLlr;
        source = 'accumulated';
    case {'final', 'last', 'bp', 'bpapp'}
        osdLlr = bp.finalAppLlr;
        source = 'final';
    case {'best', 'bestsyndrome', 'bestapp'}
        osdLlr = bp.bestAppLlr;
        source = 'best';
    case {'channel', 'ch', 'input'}
        osdLlr = channelLlr(:);
        source = 'channel';
    otherwise
        error('cpu_bp_osd_decode:BadOption', ...
            'Unknown opts.osdLlrSource: %s', char(opts.osdLlrSource));
end
end

function osd = osd_decode(llr, G, H, opts)
n = numel(llr);
k = size(G, 1);

reliability = abs(llr(:));
hard = llr(:) < 0;

[~, reliabilityPerm] = sort(reliability, 'descend');
Grel = G(:, reliabilityPerm);
hardRel = hard(reliabilityPerm);
llrRel = llr(reliabilityPerm);

[Gsys, sysPerm] = generator_to_systematic(Grel);
hardSys = hardRel(sysPerm).';
llrSys = llrRel(sysPerm).';
relSys = abs(llrSys);

baseMsg = hardSys(1:k);
[bestSys, bestMetric] = encode_and_metric(baseMsg, Gsys, hardSys, relSys);
tests = 1;

[~, weakInfoOrder] = sort(relSys(1:k), 'ascend');
flipPool = weakInfoOrder(1:min([numel(weakInfoOrder), opts.osdMaxFlipPositions]));

for order = 1:opts.osdOrder
    if numel(flipPool) < order || tests >= opts.osdMaxTests
        break;
    end

    combos = nchoosek(flipPool, order);
    for row = 1:size(combos, 1)
        msg = baseMsg;
        msg(combos(row, :)) = ~msg(combos(row, :));

        [candSys, metric] = encode_and_metric(msg, Gsys, hardSys, relSys);
        tests = tests + 1;
        if metric < bestMetric
            bestMetric = metric;
            bestSys = candSys;
        end
        if tests >= opts.osdMaxTests
            break;
        end
    end
end

codeRel = false(n, 1);
codeRel(sysPerm) = bestSys(:);

codeBits = false(n, 1);
codeBits(reliabilityPerm) = codeRel;

if any(mod(H * double(codeBits), 2) ~= 0)
    error('cpu_bp_osd_decode:OsdInternalError', ...
        'OSD produced a codeword that does not satisfy H.');
end

osd.codeBits = codeBits;
osd.tests = tests;
end

function [codeSys, metric] = encode_and_metric(msg, Gsys, hardSys, relSys)
codeSys = mod(double(msg) * double(Gsys), 2) > 0;
metric = sum(relSys(codeSys ~= hardSys));
end

function [G, infoCols, pivotCols, rankH] = cpu_ldpc_system_from_h(H)
% Local wrapper kept for compatibility with the original main function.
[G, infoCols, pivotCols, rankH] = nullspace_generator_from_h(H);
end

function [G, infoCols, pivotCols, rankH] = nullspace_generator_from_h(H)
[R, pivotCols, rankH] = gf2_rref(H);
n = size(H, 2);
infoCols = setdiff(1:n, pivotCols, 'stable');
k = numel(infoCols);

G = false(k, n);
for i = 1:k
    freeCol = infoCols(i);
    G(i, freeCol) = true;
    for row = 1:rankH
        if R(row, freeCol)
            G(i, pivotCols(row)) = true;
        end
    end
end
end

function [Gsys, perm] = generator_to_systematic(Gin)
Gsys = logical(mod(Gin, 2));
[k, n] = size(Gsys);
perm = 1:n;

for target = 1:k
    pivotRow = [];
    pivotCol = [];
    for col = target:n
        relRow = find(Gsys(target:k, col), 1, 'first');
        if ~isempty(relRow)
            pivotRow = target + relRow - 1;
            pivotCol = col;
            break;
        end
    end
    if isempty(pivotRow)
        error('cpu_bp_osd_decode:RankDeficientGenerator', ...
            'Generator matrix is rank deficient.');
    end

    if pivotRow ~= target
        Gsys([target, pivotRow], :) = Gsys([pivotRow, target], :);
    end
    if pivotCol ~= target
        Gsys(:, [target, pivotCol]) = Gsys(:, [pivotCol, target]);
        perm([target, pivotCol]) = perm([pivotCol, target]);
    end

    for row = 1:k
        if row ~= target && Gsys(row, target)
            Gsys(row, :) = xor(Gsys(row, :), Gsys(target, :));
        end
    end
end
end

function [R, pivotCols, rankA] = gf2_rref(A)
R = logical(mod(A, 2));
[m, n] = size(R);
pivotCols = [];
row = 1;

for col = 1:n
    if row > m
        break;
    end

    relPivot = find(R(row:m, col), 1, 'first');
    if isempty(relPivot)
        continue;
    end
    pivot = row + relPivot - 1;

    if pivot ~= row
        R([row, pivot], :) = R([pivot, row], :);
    end

    rowsToClear = find(R(:, col)).';
    rowsToClear(rowsToClear == row) = [];
    for r = rowsToClear
        R(r, :) = xor(R(r, :), R(row, :));
    end

    pivotCols(end + 1) = col; %#ok<AGROW>
    row = row + 1;
end

rankA = row - 1;
end
