% Make random linear parity code matrix without duplicate rows or all zero
% rows.

function [G,H] = make_RLC(k,n,p)

    P = make_parity_check(k,n,p);
    % If there are duplicate rows, re-randomise 
    while (size(unique(P','rows'),1)<n-k)
        P = make_parity_check(k,n,p);
    end
    G = [eye(k) P];
    H = [transpose(P) eye(n-k)];

end

function P = make_parity_check(k,n,p)
    P = binornd(1,p,k,n-k); %以参数1，p生成二项分布
    % If a row is all zeros, then the associated bit has no protection so
    % pick again.
    tf=ismember(P,zeros(1,n-k),'rows');
    while (sum(tf)>0)
        P(tf,:) = binornd(1,p,sum(tf),n-k);
        tf=ismember(P,zeros(1,n-k),'rows');
    end
end