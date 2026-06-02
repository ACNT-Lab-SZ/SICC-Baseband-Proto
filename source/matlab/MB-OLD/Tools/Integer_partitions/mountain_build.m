function u = mountain_build(u,k,w,W1,n1)
u(k+1:w) = u(k)*ones(1,w-k);
W2 = W1-sum(u);
q = floor(W2/(n1-u(k)));
r = W2-q*(n1-u(k));
if(~isnan(q))
    if q ~= 0
        u(w-q+1:w)=n1*ones(1,q);
    end
    if w-q>0
        u(w-q)=u(w-q)+r;
    end
end
end