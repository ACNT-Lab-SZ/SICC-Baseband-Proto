%功能说明：得到各拟合直线的截距，边界点属于右边直线
%输入：
%   L - 排序后的可靠度
%   I - 拟合曲线的分段下标
%   Q - 量化参数
%   beta - 各直线斜率
%   m - 拟合曲线段数
%输出：
%   I - 直线截距
function I = get_intercept(L,J,Q,s,D)
I = zeros(1,D);
for i=1:D
    I(i) = round(L(J(i))/Q)-s(i)*J(i);
end
end
