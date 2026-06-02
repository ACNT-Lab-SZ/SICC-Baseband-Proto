%功能说明：获得各分段的端点下标
%输入：
%   L - 降序排列后的可靠度，
%   D - 拟合曲线段数，只支持D=2/3/4
%输出：
%   J - 拟合曲线的分段下标，共m个元素

function J = get_seg_indices(L, D)
n = length(L);
if D > 4
    error(['unsupported segmentation D = ' num2str(D)]);
end

%% 分两段时(D=2)
vertical_distance1 = zeros(1,n);
for i = 1:n %在下标1到n之间遍历，比较垂直距离
    vertical_distance1(i) = abs(L(i)-((L(n)-L(1))/(n-1)*(i-1)+ L(1)));%求垂直距离
end

J(1) = find(max(vertical_distance1)==vertical_distance1, 1, 'last');%找到垂直距离最大的点对应的横坐标
J(2) = n;%必有的

%若只分两段，则可以返回
if D==2
%将拟合曲线画出
%     i = 1:n;
%     plot(i,((L(n)-L(1))/(n-1)*(i-1)+ L(1)));%连接首尾两个端点的直线
%     hold on;
%     i = 1:J(1);
%     plot(i,((L(J(1))-L(1))/(J(1)-1)*(i-1)+ L(1)));
%     i = J(1):J(2);
%     plot(i,((L(J(2))-L(J(1)))/(J(2)-J(1))*(i-J(1))+ L(J(1))));
    return;
end

%% 分三段时(D=3)
vertical_distance2 = zeros(1,n);
for i = 1:J(1)%在下标1到J(1)之间遍历，比较垂直距离
    vertical_distance2(i) = abs(L(i)-((L(J(1))-L(1))/(J(1)-1)*(i-1)+ L(1)));%求垂直距离
end
for i = J(1)+1:J(2)%在下标J(1)+1到J(2)之间遍历，比较垂直距离
    vertical_distance2(i) = abs(L(i)-((L(J(2))-L(J(1)))/(J(2)-J(1))*(i-J(1))+ L(J(1))));%求垂直距离
end
J(3) = find(max(vertical_distance2)==vertical_distance2);%找到垂直距离最大的点对应的横坐标
%将坐标顺序放置
J = sort(J,'ascend');
%若只分三段，则可以返回
if D==3
%将拟合曲线画出
%     i = 1:J(1);
%     plot(i,((L(J(1))-L(1))/(J(1)-1)*(i-1)+ L(1)));
%     i = J(1):J(2);
%     plot(i,((L(J(2))-L(J(1)))/(J(2)-J(1))*(i-J(1))+ L(J(1))));
%     i = J(2):J(3);
%     plot(i,((L(J(3))-L(J(2)))/(J(3)-J(2))*(i-J(2))+ L(J(2)))); 
    return;
end

%% 分四段时(D=4)
vertical_distance3 = zeros(1,n);
for i = 1:J(1)%在下标1到J(1)之间遍历，比较垂直距离
    vertical_distance3(i) = abs(L(i)-((L(J(1))-L(1))/(J(1)-1)*(i-1)+ L(1)));%求垂直距离
end
for i = J(1)+1:J(2)%在下标J(1)+1到J(2)之间遍历，比较垂直距离
    vertical_distance3(i) = abs(L(i)-((L(J(2))-L(J(1)))/(J(2)-J(1))*(i-J(1))+ L(J(1))));%求垂直距离
end
for i = J(2)+1:J(3)%在下标J(2)+1到J(3)之间遍历，比较垂直距离
    vertical_distance3(i) = abs(L(i)-((L(J(3))-L(J(2)))/(J(3)-J(2))*(i-J(2))+ L(J(2))));%求垂直距离
end
J(4) = find(max(vertical_distance3)==vertical_distance3);
%将坐标顺序放置
J = sort(J,'ascend');
%将拟合曲线画出
% i = 1:J(1);
% plot(i,((L(J(1))-L(1))/(J(1)-1)*(i-1)+ L(1)));
% i = J(1):J(2);
% plot(i,((L(J(2))-L(J(1)))/(J(2)-J(1))*(i-J(1))+ L(J(1))));
% i = J(2):J(3);
% plot(i,((L(J(3))-L(J(2)))/(J(3)-J(2))*(i-J(2))+ L(J(2)))); 
% i = J(3):J(4);
% plot(i,((L(J(4))-L(J(3)))/(J(4)-J(3))*(i-J(3))+ L(J(3))));

end