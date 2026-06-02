function [M_right_up, M_right_down] = index_Matrix(N)
n = log2(N);
M_right_up = zeros(N/2, n);
M_right_down = zeros(N/2, n);
for i = 1:n
    temp = 0;
    for p = 0:2^(n-i)-1%可分为多少部分
        for k = 1:2^(i-1)%每部分有多少个元素
            temp= temp+1;
            M_right_up(temp,i) = k+p*2^i;
            M_right_down(temp,i) = k+p*2^i+2^(i-1); 
        end
    end
    
end



                