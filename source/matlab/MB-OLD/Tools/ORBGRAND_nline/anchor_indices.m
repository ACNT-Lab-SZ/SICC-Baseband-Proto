%功能说明：获得各分段的端点下标
%输入：
%   L - 升序排列后的可靠度
%   m - 拟合曲线段数
%输出：
%   I - 拟合曲线的分段下标，共m个元素
function I = anchor_indices(L,m)
n = length(L);
if m > 4
    error(['unsupported segmentation m = ' num2str(m)]);
end

%% 分两段时(m=2)
for i = 1:round(n/2)%在下标1到round(n/2)之间遍历，比较垂直距离
    vertical_distance1(i) = abs(L(i)-((L(round(n/2))-L(1))/(round(n/2)-1)*(i-1)+ L(1)));%求垂直距离
end

I(1) = find(max(vertical_distance1)==vertical_distance1);%找到垂直距离最大的点对应的横坐标
I(2) = n;%必有的

%若只分两段，则可以返回
if m==2
    %将拟合曲线画出
%     i = 1:n/2;
%     plot(i,((L(round(n/2))-L(1))/(round(n/2)-1)*(i-1)+ L(1)));
%     hold on;
%     i = 1:I(1);
%     plot(i,((L(I(1))-L(1))/(I(1)-1)*(i-1)+ L(1)));
%     i = I(1)+1:I(2);
%     plot(i,((L(round(n/2))-L(I(1)))/(round(n/2)-I(1))*(i-I(1))+ L(I(1))));
    
    return;
end


%% 分三段时(m=3)
for i = 1:I(1)%在下标1到I(1)之间遍历，比较垂直距离
    vertical_distance2(i) = abs(L(i)-((L(I(1))-L(1))/(I(1)-1)*(i-1)+ L(1)));%求垂直距离
end
for i = I(1)+1:round(n/2)%在下标I(1)+1到round(n/2)之间遍历，比较垂直距离
    vertical_distance2(i) = abs(L(i)-((L(round(n/2))-L(I(1)))/(round(n/2)-I(1))*(i-I(1))+ L(I(1))));%求垂直距离
end
I(3) = find(max(vertical_distance2)==vertical_distance2);%找到垂直距离最大的点对应的横坐标
%将坐标顺序放置
I = sort(I,'ascend');
%若只分三段，则可以返回
if m==3
    %将拟合曲线画出
%     i = 1:I(1);
%     plot(i,((L(I(1))-L(1))/(I(1)-1)*(i-1)+ L(1)));
%     hold on;
%     i = I(1)+1:I(2);
%     plot(i,((L(I(2))-L(I(1)))/(I(2)-I(1))*(i-I(1))+ L(I(1))));
%     i = I(2)+1:I(3);
%     plot(i,((L(round(n/2))-L(I(2)))/(round(n/2)-I(2))*(i-I(2))+ L(I(2))));
    
    return;
end



%% 分四段时(m=4)
for i = 1:I(1)%在下标1到n/2之间遍历，比较垂直距离
    vertical_distance3(i) = L(i)-((L(I(1))-L(1))/(I(1)-1)*(i-1)+ L(1));%求垂直距离
end
for i = I(1)+1:I(2)%在下标I(1)+1到I(2)之间遍历，比较垂直距离
    vertical_distance3(i) = abs(L(i)-((L(I(2))-L(I(1)))/(I(2)-I(1))*(i-I(1))+ L(I(1))));%求垂直距离
end
for i = I(2)+1:n/2%在下标I(2)+1到round(n/2)之间遍历，比较垂直距离
    vertical_distance3(i) = abs(L(i)-((L(round(n/2))-L(I(2)))/(round(n/2)-I(2))*(i-I(2))+ L(I(2))));%求垂直距离
end
I(4) = find(max(vertical_distance3)==vertical_distance3);
%将坐标顺序放置
I = sort(I,'ascend');
%将拟合曲线画出
% i = 1:I(1);
% plot(i,((L(I(1))-L(1))/(I(1)-1)*(i-1)+ L(1)));
% hold on;
% i = I(1):I(2);
% plot(i,((L(I(2))-L(I(1)))/(I(2)-I(1))*(i-I(1))+ L(I(1))));
% i = I(2):I(3);
% plot(i,((L(I(3))-L(I(2)))/(I(3)-I(2))*(i-I(2))+ L(I(2))));
% i = I(3):I(4);
% plot(i,((L(n/2)-L(I(3)))/(n/2-I(3))*(i-I(3))+ L(I(3))));

end