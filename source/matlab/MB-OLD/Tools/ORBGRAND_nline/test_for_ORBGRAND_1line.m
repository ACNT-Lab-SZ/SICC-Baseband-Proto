n = 64;%码长
J0 = -1;
W = 0:128;

%验证(-1-2*J0-sqrt((1+2*J0)^2+8*W))/2 =<w<=(-1-2*J0+sqrt((1+2*J0)^2+8*W))/2
w_H11 = (-1-2*J0-sqrt((1+2*J0)^2+8*W))/2;
w_H12 = (-1-2*J0+sqrt((1+2*J0)^2+8*W))/2;%上界
w_H12_floor = floor((-1-2*J0+sqrt((1+2*J0)^2+8*W))/2);%上界
figure(1)
plot(W,w_H11,W,w_H12,W,w_H12_floor,'linewidth',2);
hold on;
grid on;

%验证(1+2*(n+J0)-sqrt((1+2*(n+J0))^2-8*W))/2 =<w<=(1+2*(n+J0)+sqrt((1+2*(n+J0))^2-8*W))/2
w_H21 = (1+2*(n+J0)-sqrt((1+2*(n+J0))^2-8*W))/2;%下界
w_H21_ceil = ceil((1+2*(n+J0)-sqrt((1+2*(n+J0))^2-8*W))/2);%下界
w_H22 = (1+2*(n+J0)+sqrt((1+2*(n+J0))^2-8*W))/2;
plot(W,w_H21,W,w_H21_ceil,W,w_H22,'linewidth',2);
legend('$\frac{-1-2J_0-\sqrt{(1+2J_0)^2+8W}}{2}$',...
'$\frac{-1-2J_0+\sqrt{(1+2J_0)^2+8W}}{2} $, upper bound',...
'$\left\lfloor \frac{-1-2J_0+\sqrt{(1+2J_0)^2+8W}}{2} \right\rfloor $, upper bound',...
'$\frac{1+2(n+J_0)-\sqrt{(1+2(n+J_0))^2-8W }}{2} $, lower bound',...
'$\left\lceil \frac{1+2(n+J_0)-\sqrt{(1+2(n+J_0))^2-8W }}{2} \right\rceil$, lower bound',...
'$\frac{1+2(n+J_0)+\sqrt{(1+2(n+J_0))^2-8W }}{2} $',...
'interpreter','latex','fontsize',14);
xlabel('W');
ylabel('$w_H$','interpreter','latex');