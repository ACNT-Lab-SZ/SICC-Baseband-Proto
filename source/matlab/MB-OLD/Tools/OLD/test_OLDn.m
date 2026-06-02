clc;clear;
I_d = 237;
J_d = 19;
J_d1 = 0;
R = 1:500;
k = 64;
s_d = -1;

w11 = (2*s_d*J_d+2*I_d+s_d+sqrt(-8*s_d*R+(2*s_d*J_d+2*I_d+s_d)^2))/(2*s_d);
w12 = (2*s_d*J_d+2*I_d+s_d-sqrt(-8*s_d*R+(2*s_d*J_d+2*I_d)^2))/(2*s_d);%上界
w12_floor = floor((2*s_d*J_d+2*I_d+s_d-sqrt(-8*s_d*R+(2*s_d*J_d+2*I_d)^2))/(2*s_d));%上界
figure(1)
plot(R,w11,R,w12,R,w12_floor,'linewidth',2);
hold on;
grid on;


w21 = (-(2*s_d*J_d1+2*I_d+s_d)+sqrt(8*s_d*R+(2*s_d*J_d1+2*I_d+s_d)^2))/(2*s_d);
w21_ceil = ceil((-(2*s_d*J_d1+2*I_d+s_d)+sqrt(8*s_d*R+(2*s_d*J_d1+2*I_d+s_d)^2))/(2*s_d));
w22 = (-(2*s_d*J_d1+2*I_d+s_d)-sqrt(8*s_d*R+(2*s_d*J_d1+2*I_d+s_d)^2))/(2*s_d);

w21temp = -sqrt(2*R/s_d+(J_d1+I_d/s_d+0.5)^2)-(J_d1+I_d/s_d+0.5);
w21temp_ceil = ceil(-sqrt(2*R/s_d+(J_d1+I_d/s_d+0.5)^2)-(J_d1+I_d/s_d+0.5));


plot(R,w21,R,w21_ceil,R,w22,'linewidth',2);
plot(R,w21temp,R,w21temp_ceil,'linewidth',2);

legend('$\frac{2s_dJ_d+2I_d+s_d+\sqrt{-8s_dR+(2s_dJ_d+2I_d+s_d)^2}}{2s_d} $',...
'$\frac{2s_dJ_d+2I_d+s_d-\sqrt{-8s_dR+(2s_dJ_d+2I_d+s_d)^2}}{2s_d} $, upper bound',...
'$\left\lfloor \frac{2s_dJ_d+2I_d+s_d-\sqrt{-8s_dR+(2s_dJ_d+2I_d+s_d)^2}}{2s_d} \right\rfloor$, upper bound',...
'$\frac{-(2s_dJ_{d-1}+2I_d+s_d)+\sqrt{8s_dR+(2s_dJ_{d-1}+2I_d+s_d)^2}}{2s_d} $, lower bound',...
'$\left\lceil \frac{-(2s_dJ_{d-1}+2I_d+s_d)+\sqrt{8s_dR+(2s_dJ_{d-1}+2I_d+s_d)^2}}{2s_d} \right\rceil$, lower bound',...
'$\frac{-(2s_dJ_{d-1}+2I_d+s_d)-\sqrt{8s_dR+(2s_dJ_{d-1}+2I_d+s_d)^2}}{2s_d} $',...
'1',...
'2',...
'interpreter','latex','fontsize',14);

xlabel('$R$','interpreter','latex');
ylabel('$w$','interpreter','latex');






