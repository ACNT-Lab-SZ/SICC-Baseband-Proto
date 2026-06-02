function z = get_beta(L,I,Q,m)
n = length(L);
z(1) = round((L(I(1))-L(1))/((I(1)-1)*Q));%beta(1)单独计算
for i=2:m-1
    z(i) = round((L(I(i))-L(I(i-1)))/((I(i)-I(i-1))*Q));
end
z(m) = round((L(round(n/2))-L(I(m-1)))/((round(n/2)-I(m-1))*Q));%最后一段（高可靠区域）斜率，由过中点的直线拟合
end