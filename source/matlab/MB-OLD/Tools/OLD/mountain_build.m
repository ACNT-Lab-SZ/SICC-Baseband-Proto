function u = mountain_build(u,k,w,W1,n1)
u(k+1:w) = u(k)*ones(1,w-k);%将k+1:w的u赋值与u(k)相同
W2 = W1-sum(u);%减去已分配的值
q = floor(W2/(n1-u(k)));%剩余值可以填满几列
r = W2-q*(n1-u(k));%得到未分配的值
if(~isnan(q))
    if q ~= 0
        u(w-q+1:w)=n1*ones(1,q);%若q非零，则将最后面q列填满
    end
    if w-q>0
        u(w-q)=u(w-q)+r;
    end
end
end