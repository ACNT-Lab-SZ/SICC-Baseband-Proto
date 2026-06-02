%逻辑重量W，汉明重量w_H
n = 64;%码长
W = 1:128;%在基本模型中，可靠度是从1开始的
%验证(-sqrt(1+8*W)-1)/2 =<w_H<=(sqrt(1+8*W)-1)/2

w_H11 = (-sqrt(1+8*W)-1)/2;
w_H12 = (sqrt(1+8*W)-1)/2;%上界
w_H12_floor = floor((sqrt(1+8*W)-1)/2);%上界
figure(1)
plot(W,w_H11,W,w_H12,W,w_H12_floor,'linewidth',2);
hold on;
grid on;

%验证(1+2*n-sqrt((2*n+1)^2-8*W))/2 =<w_H<=(1+2*n+sqrt((2*n+1)^2-8*W))/2
w_H21 = (1+2*n-sqrt((2*n+1)^2-8*W))/2;%下界
w_H21_ceil = ceil((1+2*n-sqrt((2*n+1)^2-8*W))/2);%下界
w_H22 = (1+2*n+sqrt((2*n+1)^2-8*W))/2;
plot(W,w_H21,W,w_H21_ceil,W,w_H22,'linewidth',2);

legend('$\frac{-\sqrt{1+8W}-1}{2}$',...
'$\frac{\sqrt{1+8W}-1}{2}$, upper bound',...
'$\left\lfloor \frac{\sqrt{1+8W}-1}{2} \right\rfloor$, upper bound',...
'$\frac{1+2n-\sqrt{(1+2n)^2-8W}}{2}$, lower bound',...
'$\left\lceil \frac{1+2n-\sqrt{(1+2n)^2-8W}}{2}\right\rceil$, lower bound',...
'$\frac{1+2n+\sqrt{(1+2n)^2-8W}}{2}$',...
'interpreter','latex','fontsize',14);
xlabel('W');
ylabel('$w_H$','interpreter','latex');