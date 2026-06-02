function  wi = GRAND_valid_partial_Hamming_Weights(Wi,J_iminus1,I_i,I_iminus1,beta)
%找到该段的汉明重量
k = 0;
wi = [];
if Wi==0||Wi==-1%若可靠度为0，若是第一段可能翻转（），若是其他段则一定不翻转，但后续分段仍有可能翻转，因此先赋一个无效值，使集合不为空
    wi(1) = 0;
end
if((I_i-I_iminus1)*(2*J_iminus1+(I_i-I_iminus1+1)*beta)/2>=Wi)%当前可靠度不能超出该区间最大值，在basic和1line中并未加此限制，是因为它们不可能达到该限制，因为最大可靠度是翻转所有比特
    w = max(1,ceil((beta+2*J_iminus1+2*beta*(I_i-I_iminus1)-sqrt(-8*beta*Wi+(beta+2*J_iminus1+2*beta*(I_i-I_iminus1))^2))/(2*beta)));%汉明重量下界ceil((beta+2*J_iminus1+2*beta*(I_i-I_iminus1)-sqrt(-8*beta*Wi+(beta+2*J_iminus1+2*beta*(I_i-I_iminus1))^2))/(2*beta))
    while w <= floor((-beta-2*J_iminus1+sqrt(8*beta*Wi+(beta+2*J_iminus1)^2))/(2*beta))%汉明重量上界，floor(-0.5-J_iminus1/beta+sqrt(2*Wi/beta+(0.5+J_iminus1/beta)^2))+1%floor((sqrt(1+8*Wi)-1)/2)%%%%%%%%% %%%%%%%%%floor((sqrt(1+8*Wi)-1)/2)%
        if(mod(Wi - w*J_iminus1,beta)==0)%能够被整除即可，其余两个条件在汉明重量上下界中已经包含了%(Wi - w*J_iminus1)/beta>=(1+w)*w/2)&&((Wi - w*J_iminus1)/beta<=(I_i-I_iminus1+1)*w-(1+w)*w/2)&&
            k = k+1;
            wi(k) = w;
        end
        w = w+1;
    end
end
end

