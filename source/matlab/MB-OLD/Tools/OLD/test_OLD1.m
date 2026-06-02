clc;clear;
I = 121;
R = 57:200;
k = 64;

w11 = (-2*I+2*k+1-sqrt((2*I-2*k-1)^2)+8*R)/2;
w12 = (-2*I+2*k+1+sqrt((2*I-2*k-1)^2)+8*R)/2;%上界
w12_floor = floor((-2*I+2*k+1+sqrt((2*I-2*k-1)^2)+8*R)/2);%上界
figure(1)
plot(R,w11,R,w12,R,w12_floor,'linewidth',2);
hold on;
grid on;


w21 = (2*I-1-sqrt((1-2*I)^2-8*R))/2;
w21_ceil = ceil((2*I-1-sqrt((1-2*I)^2-8*R))/2);
w22 = (2*I-1+sqrt((1-2*I)^2-8*R))/2;


plot(R,w21,R,w21_ceil,R,w22,'linewidth',2);
legend('$\frac{-2I+2k+1-\sqrt{(2I-2k-1)^2+8R}}{2}$',...
'$\frac{-2I+2k+1+\sqrt{(2I-2k-1)^2+8R}}{2}$, upper bound',...
'$\left\lfloor  \frac{-2I+2k+1+\sqrt{(2I-2k-1)^2+8R}}{2} \right\rfloor$, upper bound',...
'$\frac{2I-1-\sqrt{(1-2I)^2}-8R}{2}$, lower bound',...
'$\left\lceil \frac{2I-1-\sqrt{(1-2I)^2}-8R}{2} \right\rceil $, lower bound',...
'$\frac{2I-1+\sqrt{(1-2I)^2}-8R}{2} $',...
'interpreter','latex','fontsize',14);

xlabel('$R$','interpreter','latex');
ylabel('$w$','interpreter','latex');






