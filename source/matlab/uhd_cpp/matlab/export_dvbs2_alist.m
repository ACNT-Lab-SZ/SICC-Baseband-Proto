function export_dvbs2_alist(rate, outputPath)
%EXPORT_DVBS2_ALIST Export MATLAB dvbs2ldpc parity-check matrix to alist.
%
%   export_dvbs2_alist(1/2, 'DVB_S2_N64800_R12.alist')

if nargin < 1 || isempty(rate)
    rate = 1/2;
end
if nargin < 2 || isempty(outputPath)
    outputPath = fullfile(pwd, 'DVB_S2_N64800_R12.alist');
end

H = dvbs2ldpc(rate, 'sparse');
[m, n] = size(H);
[rowIdx, colIdx] = find(H);

colRows = accumarray(colIdx, rowIdx, [n, 1], @(x) {sort(x(:).')});
rowCols = accumarray(rowIdx, colIdx, [m, 1], @(x) {sort(x(:).')});

colWeights = cellfun(@numel, colRows).';
rowWeights = cellfun(@numel, rowCols).';
maxColWeight = max(colWeights);
maxRowWeight = max(rowWeights);

outDir = fileparts(outputPath);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end

fid = fopen(outputPath, 'w');
if fid < 0
    error('Could not open output alist file: %s', outputPath);
end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '%d %d\n', n, m);
fprintf(fid, '%d %d\n', maxColWeight, maxRowWeight);
fprintf(fid, '%d ', colWeights);
fprintf(fid, '\n');
fprintf(fid, '%d ', rowWeights);
fprintf(fid, '\n');

for c = 1:n
    vals = colRows{c};
    vals(end + 1:maxColWeight) = 0; %#ok<AGROW>
    fprintf(fid, '%d ', vals);
    fprintf(fid, '\n');
end

for r = 1:m
    vals = rowCols{r};
    vals(end + 1:maxRowWeight) = 0; %#ok<AGROW>
    fprintf(fid, '%d ', vals);
    fprintf(fid, '\n');
end

fprintf('Exported DVB-S2 LDPC alist: %s\n', outputPath);
fprintf('N=%d M=%d K=%d nnz=%d maxCol=%d maxRow=%d\n', ...
    n, m, n - m, numel(rowIdx), maxColWeight, maxRowWeight);
end
