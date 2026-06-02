clc;clear
warning off;
addpath('load_H_matrix');
addpath('Tools/Count-4-6-8-10-cycles');
addpath('Tools/Integer_partitions');
addpath('Tools/OLD');
addpath('Modulation');
addpath('Channels');
addpath('Decoders');
addpath('My_LDPC');
addpath('Tools');
%% 获得H矩阵，并处理
H = CCSDS_ldpc_n128_k64_H();
% H = CCSDS_ldpc_n256_k128_H();
% H = CCSDS_ldpc_n512_k256_H();
% H = Gallager_n8000_k4000_dv3dc6_H();
% H = N999M111();
% H = N1998M222();
% H = N273M82();
% H = N96M48();
% H = N504M252();
% H = N1008M504();
% H = N1057M244();
% H = N2048M1030();
% H = N16383M2131();
% H = EG255();
% H = EG1023();
% H = EG4095();
% H = PG273();
% H = PG1057();
% H = IEEE80211n(648, 5/6);%N takes values in {648, 1296, 1944}. R takes values in {1/2, 2/3, 3/4, 5/6}.
% H = IEEE80216e(2304, '1/2');
% H = LDPC_GF256();
[H_column_permuted,M, N, K, vn_degree, cn_degree, P, H_row_one_absolute_index, H_comlumn_one_relative_index, vn_distribution, cn_distribution] = H_matrix_process(H);
G = [eye(K) P'];%生成矩阵
%% 找到最小的环，即围长
num_4_cycles = count_4_cycles(H_row_one_absolute_index, cn_degree);
if num_4_cycles>0
    girth = 4;
else
    num_6_cycles = count_6_cycles(H_row_one_absolute_index, cn_degree);
    if num_6_cycles>0
        girth = 6;
    else
        num_8_cycles = count_8_cycles(H_row_one_absolute_index, cn_degree);
        if num_8_cycles>0
            girth = 8;
        else
            num_10_cycles = count_10_cycles(H_row_one_absolute_index, cn_degree);
            if num_10_cycles>0
                girth = 10;
            else
                error('girth is larger than 10, not supported');
            end
        end
    end
end
%Note that Counting 10-cycles will take several minutes. For codes of length 1000, the number of 10-cycles usually exceeds 10^7.
%% 设置参数
R = K/N;
max_iter = 20;%原始BP最大迭代次数
order = 2;%OSD阶数
max_test_num = 2081;
max_runs = 1e12;
display_interval = 10;
max_err = 300;
%******信噪比设置*****
vec = 0:0.5:3.5;
vec_string = 'Eb/N0';
%vec_string = 'SNR';
%vec_string = 'Es/N0';

%******Fast_OSD参数******
T_fastOSD = 10;%psc门限（校验子汉明重量）
lambda_fastOSD = 0.5;%pnc规则

%******OLDn参数******
D = 2;

%*******调制方式********
conste_name = 'BPSK';
[constellation, rho_inv, base_vec, demod_indices, is2D, m, num_conste_points] = get_constellation(conste_name, N);%星座映射
%Supported constellation types:
%BPSK, QPSK, 16QAM, 64QAM, 256QAM, 4ASK, 8ASK, 16ASK. All in Gray Lableing.
%Naive Modulation, e.g., x1,x2,...,xm are mapped into one symbol and then transmitted
%BICM-style demodulation, i.e., independent bit level, one received symbol ←→ m LLRs

%% 开始仿真
num_runs = zeros(length(vec), 1);
num_block_err = zeros(length(vec), 1);
num_bit_err = zeros(length(vec), 1);
bler = zeros(length(vec),1);
ber = zeros(length(vec),1);
num_error_bits = zeros(length(vec),1);
num_error_blocks = zeros(length(vec),1);
%迭代次数
num_total_iter = zeros(length(vec),1);
num_ave_iter = zeros(length(vec),1);
%测试TEP数量
num_total_test = zeros(length(vec),1);
num_ave_test = zeros(length(vec),1);
%操作数
total_operation_num = zeros(1,length(vec));
ave_operation_num = zeros(1,length(vec));
name = ['.\Results\LDPC_N' num2str(N) '_K' num2str(K) '_BP OSD' '.txt'];
filename = fopen(name,'a+');
fprintf(filename,'\n\n');
fprintf(filename,'程序开始时间：%s \n',datestr(now));
fprintf(filename,'N = %d,  ',N);
fprintf(filename,'K = %d,  ',K);
fprintf(filename,'conste_name = %s  ',conste_name);fprintf(filename,'\n');
fprintf(filename,'max_iter = %d,  ', max_iter);
%传统OSD
fprintf(filename,'order = %d  ', order);
fprintf(filename,'max_test_num = %d  ', max_test_num);
fprintf(filename,'\n');
fprintf(filename,'max_runs = %1.2e,  ',max_runs);
fprintf(filename,'max_err = %d,  ',max_err);
fprintf(filename,'display_interval = %d  ',display_interval);fprintf(filename,'\n');
fprintf(filename,[vec_string '         BER          BLER        ave_iter_num      ave_test_num    ave_operation_num  total_blocks']);fprintf(filename,'\n');


tic
% profile on

for i_vec = 1 : length(vec)
    
    sigma = 1/sqrt(2 * R * m) * 10^(-vec(i_vec)/20);%EbN0;
    %sigma = 1/10^(vec(i_vec)/20);%SNR
    %sigma = 1/sqrt(2) * 10^(-vec(i_vec)/20);%E_s/N_0
    for i_run = 1 : max_runs
        info = round(rand(1, K));%生成信源
        parity_check_bits = mod(info*P', 2);%编码
        c = [info  parity_check_bits];%完整码字
        symbol = modulation(constellation, m, rho_inv, c', base_vec, N);%调制
        if is2D
            noise = 1/sqrt(2)*(randn(N/m, 1) + randn(N/m, 1) * 1j);%噪声
        else
            noise = randn(N/m, 1);%噪声
        end
        y = symbol + sigma * noise;

        %LLR
        llr = demodulation(y, constellation, m, num_conste_points, demod_indices, sigma, N)';
        
        %BP，可统计操作数
        [operation_num, check_flag, c_hat, iter_this_time, output_modified_llr, output_modified_llr_pass_most_check] = OperationNum_modified_LDPC_Flooding_BP_decoder(llr, H_row_one_absolute_index, H_comlumn_one_relative_index, N, M, vn_degree, cn_degree, max_iter);
        total_operation_num(i_vec) = total_operation_num(i_vec) + operation_num;
        
        %% -------------------------proposed BP-OSD,根据信道接收信息计算WHD-------------------------
        %******proposed BP-OSD参数******
        alpha_OSD = girth/2-1;
        if check_flag == 0
            %通常的系数
%             if alpha_OSD==2
%             temp = [0.325  0.33  0.345 ];   
%             else
%                 if alpha_OSD==1
%             temp = [0.48 0.52 ];   
%                 end
%             end
            %CCSDS 128 64 3db
            temp = [ 0.3241  0.3347  0.3413]; 
            
            %获得修正后的LLR
            modified_llr = temp*[llr; output_modified_llr(1:alpha_OSD,:)];

            %统计操作数
            [operation_num, test_num, c_hat] = OperationNum_modified_OLDn_decoder(llr, modified_llr, G, D);
            
            total_operation_num(i_vec) = total_operation_num(i_vec) + operation_num; 
            num_total_test(i_vec) = num_total_test(i_vec) + test_num;

        end
        %-------------------------proposed BP-OSD,根据信道接收信息计算WHD-------------------------

        info_esti = c_hat(1 : K);
        if any(info_esti ~= info)
            num_block_err(i_vec) = num_block_err(i_vec) + 1;
            num_bit_err(i_vec) = num_bit_err(i_vec) + sum(info ~= info_esti);
        end
        
       
        
        %总仿真帧数加一
        num_runs(i_vec) = num_runs(i_vec) + 1;
        
        %总迭代次数加
        num_total_iter(i_vec) = num_total_iter(i_vec) + iter_this_time;
        %BER
        ber(i_vec) = num_bit_err(i_vec)/(num_runs(i_vec)*K);
        %BLER
        bler(i_vec) = num_block_err(i_vec)/num_runs(i_vec);
        %平均迭代次数
        num_ave_iter(i_vec) = num_total_iter(i_vec)/num_runs(i_vec);
        %平均测试TEPs次数
        num_ave_test(i_vec) = num_total_test(i_vec)/num_runs(i_vec);
        %平均操作数
        ave_operation_num(i_vec) = total_operation_num(i_vec)/(num_runs(i_vec)*K);

        %若达到最大仿真帧数则跳出
        if num_block_err(i_vec) == max_err
            break;
        end
        
        %每仿真display_interval帧，在命令行窗口输出一次
        if  mod(i_run, display_interval) == 0
            disp(' ')
            disp([conste_name ' Simualtion Running = ' num2str(i_run)])
            disp(['N = ' num2str(N) ', K = ' num2str(K) ', max_iter = ' num2str(max_iter) '. ' 'H density = ' num2str(100 * sum(vn_degree)/M/N) '%.']);
            disp(' ');
            disp('VN Degree Distribution: ');
            disp(vn_distribution);
            disp(' ');
            disp('CN Degree Distribution: ');
            disp(cn_distribution);
            disp(' ');
            disp([vec_string '      BER           BLER       num_ave_iter  num_ave_test   ave_operation_num  num_block_err']);
            disp(num2str([vec(1:i_vec)', ber(1:i_vec), bler(1:i_vec), num_ave_iter(1:i_vec), num_ave_test(1:i_vec), ave_operation_num(1:i_vec)', num_block_err(1:i_vec)]));
            disp(' ');
        end
    end
    fprintf(filename,'%f    %1.9f    %1.9f     %f      %f     %f       %d\n',vec(i_vec), ber(i_vec), bler(i_vec), num_ave_iter(i_vec), num_ave_test(i_vec), ave_operation_num(i_vec), num_runs(i_vec));
end
% profile viewer
toc
disp('BLER simulation is finished.')

