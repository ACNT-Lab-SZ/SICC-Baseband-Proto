%功能说明：得到量化参数Q
%输入：
%   L - 排序后的可靠度
%   J - 拟合曲线的分段下标
%   m - 拟合曲线段数
%输出：
%   Q - 量化参数，本质是未量化前各拟合直线中斜率绝对值最小的
function Q = get_quan_paraQ(L,J,D)
Q = abs((L(J(1))-L(1))/(J(1)-1));%第一条直线
for i = 2:D  
    temp = abs((L(J(i))-L(J(i-1)))/(J(i)-J(i-1)));
    if temp < Q
        Q = temp;
    end
end
end