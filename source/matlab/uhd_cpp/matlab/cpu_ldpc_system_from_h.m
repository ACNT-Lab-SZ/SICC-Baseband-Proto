function [G, infoCols, pivotCols, rankH] = cpu_ldpc_system_from_h(H)
%CPU_LDPC_SYSTEM_FROM_H Build a systematic-compatible GF(2) generator from H.
%
%   [G, infoCols, pivotCols, rankH] = cpu_ldpc_system_from_h(H)
%
%   H is an M x N parity-check matrix. G is K x N and satisfies G*H' = 0
%   over GF(2). The information bits are placed at infoCols.

H = logical(mod(H, 2));
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
