%功能说明：可统计操作数，并输出每次迭代的软信息
%输入：
%输出：
%   modified_llr - 每轮迭代后的软信息
%   modified_llr_pass_most_check - 通过校验方程最多的一次迭代后的软信息
%   operation_num - 操作数
function [operation_num, check_flag, x_hat, iter_this_time, modified_llr, modified_llr_pass_most_check] = OperationNum_modified_LDPC_Flooding_BP_decoder(llr, H_row_one_absolute_index, H_column_one_relative_index, N, M, vn_degree, cn_degree, max_iter)
operation_num = 0;%操作数：加、乘、比较
modified_llr = zeros(max_iter,N);%存储每次循环后的LLR
syndrome_error_num = M;%当前循环后校验方程错误个数
modified_llr_pass_most_check = llr;%通过最多校验方程个数的中间LLR

check_flag = 0;
VN_array = zeros(max(vn_degree), N);
iter_this_time = max_iter;


for v = 1 : N%初始化
    for v_neighbor = 1 : vn_degree(v)
        VN_array(v_neighbor, v) = llr(v);%Belief Propagation Rule. The initial 2/sigma^2*y is automatically incorporateed here.
    end
end

for t = 1 : max_iter
    %CN update
    for c = 1 : M
        %精准计算
        product = ones(max(cn_degree),1);
        for c_neighbor = 1 : cn_degree(c)%read data from VNs, and then store in CNs memory.
            for i = 1:cn_degree(c)%针对每个变量节点
                if(i~=c_neighbor)%剔除该变量节点
                    %5G移动通信中的信道编码 公式（3.38）
                    CN_tanh_tmp = 1 - 2/(1 + exp(VN_array(H_column_one_relative_index(c, i), H_row_one_absolute_index(c, i))));%Exact decoding.
                    product(c_neighbor) = product(c_neighbor) * CN_tanh_tmp;
                end
            end
        end
        for c_neighbor = 1 : cn_degree(c)
            x = product(c_neighbor);
            x = sign(x) * min(abs(x), 1 - 1e-15);%Numerical Stability.
            VN_array(H_column_one_relative_index(c, c_neighbor), H_row_one_absolute_index(c, c_neighbor)) = log((1 + x)/(1 - x));%Exact decoding.
        end
        operation_num = operation_num+2*(cn_degree(c)-1)+cn_degree(c);%*************乘和查表
    end
    
    sum_VN = sum(VN_array);%VN update
    for v = 1 : N
        for v_neighbor = 1 : vn_degree(v)
            VN_array(v_neighbor, v) = sum_VN(v) - VN_array(v_neighbor, v) + llr(v);%Belief Propagation Rule. The initial 2/sigma^2*y is automatically incorporateed here.
        end
        operation_num = operation_num + vn_degree(v)-1;%*************变量节点更新
    end

    x_hat = (sum_VN + llr) < 0;%Belief propagation Decision.
    operation_num = operation_num + N;%**************总LLR，基于变量更新计算
    
    operation_num = operation_num + N;%**************硬判决
    operation_num = operation_num + M*N;%**************矩阵校验
    operation_num = operation_num + M;%**************判断是否为0
    
    %当前循环的LLR
    modified_llr(t,:) = (sum_VN + llr);

    parity_check = zeros(M, 1);
    for m = 1 : M
        for k = 1 : 1 : cn_degree(m)
            parity_check(m) = parity_check(m) + x_hat(H_row_one_absolute_index(m, k));
        end
    end
    
    %保存通过校验方程数最多的LLR
    temp = sum(mod(parity_check, 2));%未通过校验方程个数
    if temp < syndrome_error_num
        syndrome_error_num = temp;
        modified_llr_pass_most_check = (sum_VN' + llr);
    end
    
    if ~sum(mod(parity_check, 2))%early stop, to see whether Hx = 0.
        iter_this_time = t;
        check_flag = 1;
        break;
    end
end
end
