% Create a Polar-assisted convolutional code, following the instruction
% in E. Arikan, "From sequential decoding to channel polarization
% and back again", arXiv:1908.09594, 2019.

function [G_sys, H] = make_PAC_code(N, k, c, type)

%Inputs: codeword length as integer N,
%        number of data bits as integer K,
%        convolution polynomial c as array of ints,
%        type - 0:RM rate-profiling  1:polar rate-profiling
%Outputs: K by N binary generator matrix G
%         N-K by N binary parity check matrix H

%generate rate-profiling vector A using score metric


if(type==0)
    [~, indices] = sort(sum(de2bi(0:(N-1)),2), 'descend');
    
    %rate profiling matrix given vector indices
    rate_prof_mat = zeros(k,N);
    for i = 1:k
        rate_prof_mat(i, indices(i)) = 1;
    end
else
    [~, ~, ~, frozen_pattern] = Para_5GConstruction(k, N, 0);
    A = find(frozen_pattern==0);
    %rate profiling matrix given vector A of indices
    rate_prof_mat = zeros(k,N);
    for i = 1:k
        rate_prof_mat(i, A(i)) = 1;
    end
end


%Generate convolutional precoding matrix given polynomial c,

%T is convolution matrix
T = zeros(N,N);
for i = 1:length(c)
    k = 0;
    for j = 1:(N+1-i)
        T(j,i+k) = c(1,i);
        k = k+1;
    end
end

%polar coding matrix
G_polar_full = G_matrix(N);

%generator matrix G is product of these matrices, modulo 2
G = mod(rate_prof_mat*T*G_polar_full,2);

%systematic
[G_sys,~] = my_Gauss_Elimination(G);
H = [G_sys(:,k+1:end)', eye(N-k)];

end
