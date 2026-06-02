function result = run_x310_uncoded_bpsk_case(mode, txGain, rxGain, maxRunTimeSec, maxValidatedFrames, resultPath)
%RUN_X310_UNCODED_BPSK_CASE Run one bounded TX or RX debug case.

if nargin < 4 || isempty(maxRunTimeSec)
    maxRunTimeSec = 12;
end
if nargin < 5 || isempty(maxValidatedFrames)
    maxValidatedFrames = 40;
end
if nargin < 6
    resultPath = "";
end

overrides = struct();
overrides.txGain = txGain;
overrides.rxGain = rxGain;
overrides.showPlot = false;
overrides.maxRunTimeSec = maxRunTimeSec;
overrides.maxValidatedFrames = maxValidatedFrames;
overrides.maxDetectedFrames = maxValidatedFrames + 20;
overrides.skipFrames = 5;
overrides.reportEvery = maxValidatedFrames;
overrides.logEvery = maxValidatedFrames;
overrides.plotEvery = maxValidatedFrames;

result = x310_uncoded_bpsk_debug(mode, overrides);

if strlength(string(resultPath)) > 0
    payload = jsonencode(result);
    fid = fopen(resultPath, 'w');
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s', payload);
end

if nargout == 0
    disp(result);
end
end
