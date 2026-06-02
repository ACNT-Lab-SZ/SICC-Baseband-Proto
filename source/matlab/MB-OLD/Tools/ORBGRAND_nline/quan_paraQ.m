function Q = quan_paraQ(L,I,m)
n = length(L);
Q = (L(I(1))-L(1))/(I(1)-1);%第一段斜率
for i = 2:m-1         %从第二段开始
    temp = (L(I(i))-L(I(i-1)))/(I(i)-I(i-1));
    if temp < Q
        Q = temp;
    end
end
temp = (L(round(n/2))-L(I(m-1)))/(round(n/2)-I(m-1));%最后一段（高可靠区域）斜率用过中点直线拟合
if temp < Q
    Q = temp;
end
end