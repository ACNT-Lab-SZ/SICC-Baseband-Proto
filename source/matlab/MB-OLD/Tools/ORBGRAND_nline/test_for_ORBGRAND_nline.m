Wi = 4:256;%可靠度重量从0开始
beta = 1;
J_i1 = -4;
I_i = 64;
I_i1 = 0;

w_H11 = (-beta-2*J_i1-sqrt(8*beta*Wi+(beta+2*J_i1)^2))/(2*beta);
w_H12 = (-beta-2*J_i1+sqrt(8*beta*Wi+(beta+2*J_i1)^2))/(2*beta);%上界 
w_H12_floor = floor((-beta-2*J_i1+sqrt(8*beta*Wi+(beta+2*J_i1)^2))/(2*beta));%上界
figure(1)
plot(Wi,w_H11,Wi,w_H12,Wi,w_H12_floor,'linewidth',2);
hold on;
grid on;

w2 = (sqrt(1+8*Wi)-1)/2;
w2_floor = floor((sqrt(1+8*Wi)-1)/2);
plot(Wi,w2,Wi,w2_floor,'linewidth',2);


w_H21 = (beta+2*J_i1+2*beta*(I_i-I_i1)-sqrt(-8*beta*Wi+(2*beta*(I_i1-I_i)-beta-2*J_i1)^2 )  )/(2*beta);%下界
w_H21_ceil = ceil((beta+2*J_i1+2*beta*(I_i-I_i1)-sqrt(-8*beta*Wi+(2*beta*(I_i1-I_i)-beta-2*J_i1)^2 ) )/(2*beta));%下界
w_H22 = (beta+2*J_i1+2*beta*(I_i-I_i1)+sqrt(-8*beta*Wi+(2*beta*(I_i1-I_i)-beta-2*J_i1)^2 )  )/(2*beta);
plot(Wi,w_H21,Wi,w_H21_ceil,Wi,w_H22,'linewidth',2);
legend('$\frac{-\beta-2J_{i-1}-\sqrt{8\beta W_i + (\beta+2J_{i-1})^2}}{2\beta}$',...
    '$ \frac{-\beta-2J_{i-1}+\sqrt{8\beta W_i + (\beta+2J_{i-1})^2}}{2\beta} $, upper bound',...
    '$\left\lfloor \frac{-\beta-2J_{i-1}+\sqrt{8\beta W_i + (\beta+2J_{i-1})^2}}{2\beta} \right\rfloor$, upper bound',...
    '$\frac{\sqrt{1+8W_i}-1}{2} $',...
    '$\left\lfloor \frac{\sqrt{1+8W_i}-1}{2} \right\rfloor$',...
'$ \frac{\beta+2J_{i-1}+2\beta(I_i-I_{i-1})-\sqrt{-8\beta W_i+(2\beta(I_{i-1}-I_i)-2J_{i-1}-\beta)^2}}{2\beta} $, lower bound',...
'$ \left\lceil \frac{\beta+2J_{i-1}+2\beta(I_i-I_{i-1})-\sqrt{-8\beta W_i+(2\beta(I_{i-1}-I_i)-2J_{i-1}-\beta)^2}}{2\beta}\right\rceil $, lower bound',...
'$ \frac{\beta+2J_{i-1}+2\beta(I_i-I_{i-1})+\sqrt{-8\beta W_i+(2\beta(I_{i-1}-I_i)-2J_{i-1}-\beta)^2}}{2\beta} $',...
'interpreter','latex','fontsize',14);
xlabel('$W$','interpreter','latex');
ylabel('$w_H$','interpreter','latex');









