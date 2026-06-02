%功能说明：获得量化后各直线斜率
%输入：
%   L - 排序后的可靠度
%   I - 拟合曲线的分段下标
%   Q - 量化参数
%   D - 拟合曲线段数
%输出：
%   s - 各直线斜率
function s = get_slope(L,J,Q,D)
s = zeros(1,D);
s(1) = round((L(J(1))-L(1))/((J(1)-1)*Q));
for i=2:D
    s(i) = round((L(J(i))-L(J(i-1)))/((J(i)-J(i-1))*Q));
end
end