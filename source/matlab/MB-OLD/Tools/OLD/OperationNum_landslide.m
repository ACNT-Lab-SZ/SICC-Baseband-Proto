function [operations, z] = OperationNum_landslide(W,w,n)
W1=W-w*(w+1)/2;
operations = 4;%********
n1=n-w;
operations = operations+1;%********
% Create the first integer partition
jj=1;
% Start with empty vector and breaking at first index
u = zeros(1,w);
k=1;
[operations_for_mountain_build, u] = OperationNum_mountain_build(u,k,w,W1,n1);
operations = operations+operations_for_mountain_build;%********
z(jj,:)=u;
% Evaluate drops
d=circshift(u,-1)-u;    %将初始分割的整数从右向左循环移位，再减去初始分割得每位对应的“drop”
d(w)=0; %最后一位“drop”始终是0
operations = operations+1;%********
% Evaluate accumuated drops
D = cumsum(d,'reverse');
operations = operations+w-1;%********
% Each loop generates a new integer partition
while D(1)>=2
    % Find the last index with an accumulated drop >=2
    k=find(D>=2,1,'last');
    operations = operations+k;%********
    % Increase its index by one.
    u(k)=u(k)+1;
    operations = operations+1;%********
    [operations_for_mountain_build, u] = OperationNum_mountain_build(u,k,w,W1,n1);
    operations = operations+operations_for_mountain_build;%********
    % Record the partition
    jj=jj+1;
    z(jj,:)=u;
    % Evaluate drops
    d=circshift(u,-1)-u;
    operations = operations+2;%********
    d(w)=0;
    % Evaluate accumulated drops
    D = cumsum(d,'reverse');
    operations = operations+w-1;%********
end
z = z + repmat([1:w],size(z,1),1);
end