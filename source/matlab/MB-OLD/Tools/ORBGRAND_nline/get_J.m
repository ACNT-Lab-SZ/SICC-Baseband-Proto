function [J0, J] = get_J(L,I,Q,beta,m)
J0 = round(L(1)/Q)-beta(1);%第一条直线的截距
for i=1:m-1%剩余直线的初始值
    J(i) = round(L(I(i))/Q);
end
end
