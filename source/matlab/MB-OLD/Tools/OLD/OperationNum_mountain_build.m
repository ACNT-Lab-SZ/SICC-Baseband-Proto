function [operations_for_mountain_build , u] = OperationNum_mountain_build(u,k,w,W1,n1)
u(k+1:w) = u(k)*ones(1,w-k);%将k+1:w的u赋值与u(k)相同
W2 = W1-sum(u);%减去已分配的值
operations_for_mountain_build = length(u);%********
q = floor(W2/(n1-u(k)));%剩余值可以填满几列
operations_for_mountain_build = operations_for_mountain_build+2;
r = W2-q*(n1-u(k));%得到未分配的值
operations_for_mountain_build = operations_for_mountain_build+2;%***********n1-u(k)上面已算出
operations_for_mountain_build = operations_for_mountain_build+1;%***********下述if语句
if(~isnan(q))
    operations_for_mountain_build = operations_for_mountain_build+1;%***********下述if语句
    if q ~= 0
        u(w-q+1:w)=n1*ones(1,q);%若q非零，则将最后面q列填满
    end
    operations_for_mountain_build = operations_for_mountain_build+1;%***********下述if语句
    if w-q>0
        u(w-q)=u(w-q)+r;
        operations_for_mountain_build = operations_for_mountain_build+1;%***********
    end
end
end