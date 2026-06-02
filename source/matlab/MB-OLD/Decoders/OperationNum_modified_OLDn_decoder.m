%功能说明：OLD算法，多段线性拟合+统计操作数，根据modified_llr排序，选择MRB，根据llr计算WHD
%输入：
%   llr - 信道LLR
%   modified_llr - 修正后的LLR
%   G - 生成矩阵
%   D - 分段数，只支持D=2/3/4
%输出：
%   test_num - 重编码次数
%   c_esti - 输出码字
%   operation_num - 操作数（加、乘、比较）
function  [operation_num, test_num, c_esti] = OperationNum_modified_OLDn_decoder(llr, modified_llr, G, D)
[k,n]= size(G);%码的维数
test_num = 0;
operation_num = 0;%操作数：加、乘、比较
operation_num = operation_num + n;%********根据LLR获得可靠度需要n个比较
%------------------------可靠度降序排列,第一次置换------------------------
[~,perm_index1] = sort(abs(modified_llr),'descend');
operation_num = operation_num + n*log(n)/log(2);%********比较操作
G1 = G(:,perm_index1);
modified_llr1 = modified_llr(perm_index1);
llr1 = llr(perm_index1);
max_test_num = 300;
%---------------------------生成系统阵,第二次置换-------------------------
[G_sys,perm_index2] = my_Gauss_Elimination(G1);
operation_num = operation_num + n*min(k,n-k)^2;%********高斯消元的操作
modified_llr2 = modified_llr1(perm_index2);
%MRB
MRB = modified_llr2(1:k)<0;
operation_num = operation_num + k;%********硬判决包括n个比较操作
%MRB LLR
llr_MRB = modified_llr2(1:k);

llr2 = llr1(perm_index2);
%置换后的原始硬判决序列
llr2_hat = llr2<0;
operation_num = operation_num + n;%********硬判决包括n个比较操作


%---------------------------------停止条件-----------------------------------
Pe_LRB = 1./(1+exp(abs(llr2(k+1:end))));%排序后LRB每个比特错误概率序列
EWHD_LRB = abs(llr2(k+1:end))*Pe_LRB';%LRB的WHD均值
operation_num = operation_num + 2*(n-k)-1+ n-k + n-k;%********

%-----------------------order 0----------------------------------
test_num = test_num+1;
c_temp = mod(MRB*G_sys,2);%重编码
operation_num = operation_num + k*(n-k);%********重编码操作数
WHD_opt = mod(c_temp+llr2_hat,2)*abs(llr2');%加权汉明距离 weighted Hamming distance，WHD，越小表明该码字更能是正确的
operation_num = operation_num + n-k-1;%********计算WHD
c_temp(perm_index2) = c_temp;%恢复第二次置换
c_temp(perm_index1) = c_temp;%恢复第一次置换
c_esti = c_temp;



WHD_thre = EWHD_LRB;%初始门限，此时MRB的WHD为0
operation_num = operation_num + 1;%********下述比较操作
if(WHD_opt<=WHD_thre)
    return;
end

if test_num >= max_test_num
    return;
end

%-----------------------order 0----------------------------------

[L, ind_order] = sort(abs(llr_MRB),'descend');%可靠度降序排列
%画出llr_MRB降序排列
% plot(1:k,L);
% hold on;

J = get_seg_indices(L,D);%各段直线的端点下标，其中，J(m)=k是固定的
for d=1:D-1
    operation_num = operation_num + 3*d + 5*n;%********找到分段下标，近似直线斜率需要3个操作数，每个点的垂直距离需要3个操作数(3n)
end
Q = get_quan_paraQ(L,J,D);%量化参数
operation_num = operation_num + 4*D + D-1;%********找到量化参数，近似直线斜率需要4个操作数(上述找分段下标已计算)，只需做D-1次比较
s = get_slope(L,J,Q,D);%斜率
operation_num = operation_num + 2*D;%********找到量化参数，各近似直线斜率除以Q并取整，需要2D次操作
I = get_intercept(L,J,Q,s,D);%
operation_num = operation_num + 4*D;%********找到各直线截距，每个截距计算需4次操作


%最大可靠度
R_max = (s(1)*(0+1)+I(1)+s(1)*J(1)+I(1))*(J(1)-1)/2;
for i = 2:D
    R_max = R_max+(s(i)*(J(i-1)+1)+I(i)+s(i)*J(i)+I(i))*(J(i)-J(i-1))/2;
end
operation_num = operation_num + 7*D;%********计算R的最大值，右端点值（一个操作数），左端点值需要从直线表达式中解出（2操作数）
R = round(L(k)/Q);%初始可靠度，上式已计算出
%计算汉明重量上下界，其中部分值计算一次即可
operation_num = operation_num + 4*D + 5*D;%************上界4个操作数，下界5个操作数



while R <= R_max
    
    Rd = OLD_integer_splitting(R,D);   %将R分割为D个数，每一行是一种可能
    operation_num = operation_num + size(Rd,1);%********查表，一种分割等价于一次查表

    
    for i = 1:size(Rd,1)     %共有size(Rd,1)种分割情况
        %-------------对当前分割寻找合法的汉明重量------------------------
        w = [];         %在下一次计算汉明重量前置空
        flag = 1;       %flag = 1标志该分段是有效的
        for j = 1:D     %每种分割共由m个数组成，遍历m个数，找到合法的汉明重量
            if j == 1   %对第一段做特别处理
                [operations, valid_Hamm] = OperationNum_OLD_valid_partial_Hamming_Weights(Rd(i,j),0,J(j),s(j),I(j));%是一个行向量
            else
                [operations, valid_Hamm] = OperationNum_OLD_valid_partial_Hamming_Weights(Rd(i,j),J(j-1),J(j),s(j),I(j));%是一个行向量
            end
            operation_num = operation_num + operations;%********上述寻找有效汉明重量操作数
            operation_num = operation_num + 1;%********下述if语句
            if isempty(valid_Hamm) %检查当前分段是否存在合理的汉明重量，若没有，则该种分割需要抛弃，flag = 0
                flag = 0;
                break;
            end
            w{j} = valid_Hamm;%将每段的汉明重量存储在相应的元胞中
        end 
        %-------------对当前分割寻找合法的汉明重量------------------------
        
        if flag == 1%该分段是有效的
            %对需要进行比特翻转的段，依次存储其翻转位置
            noise_locations1 = [];
            noise_locations2 = [];
            noise_locations3 = [];
            noise_locations4 = [];
            flip_seg_num = 0;
            for h = 1:D         %遍历每段
                if ~(w{h}==0)   %汉明重量非零时需要翻转，w{k}会出现只有一个元素且为0的情况，此时不翻转
                    flip_seg_num = flip_seg_num+1;
                    for t = 1:size(w{h},2)  %每段可能的汉明重量有size(w{k},2)个 
                        if h==1     %对第一段做特殊处理
                            W = (Rd(i,h)-w{h}(1,t)*I(h))/s(h);
                            operation_num = operation_num + 3;%********
                            [operations, location_temp] = OperationNum_landslide(W,w{h}(1,t),J(h));%获得所有满足W和w的所有可能的位置组合，按行存储
                            operation_num = operation_num + operations;%********
                        else
                            W = (Rd(i,h)-w{h}(1,t)*I(h))/s(h)-J(h-1)*w{h}(1,t);
                            operation_num = operation_num + 5;%********
                            [operations, location_temp] = OperationNum_landslide(W,w{h}(1,t),J(h)-J(h-1));%获得所有满足W和w的所有可能的位置组合，按行存储
                            operation_num = operation_num + operations;%********
                        end 
                         if h > 1    %只对第二段及之后的段的翻转位置进行补偿
                            location_temp = location_temp+J(h-1);
                            operation_num = operation_num + length(location_temp);%********
                        end
                        
                        switch flip_seg_num  %将翻转位置存在对应的集合中
                            case 0
                                % 所有段都不翻转，此情况已被 order-0 覆盖
                                continue;
                            case 1
                                noise_locations1{t} = location_temp;
                            case 2
                                noise_locations2{t} = location_temp;
                            case 3
                                noise_locations3{t} = location_temp;
                            case 4
                                noise_locations4{t} = location_temp;
                            otherwise
                                error('unsupported segmentation');
                        end
                    end
                end
            end
            
            %翻转
            switch flip_seg_num
                case 1
                    for y =1:size(noise_locations1,2)   %遍历该段每种汉明重量
                        for z =1:size(noise_locations1{y},1)    %遍历该汉明重量下所有的位置组合
                            test_num = test_num+1;
                            TEP = zeros(1,k);
                            TEP(ind_order(noise_locations1{y}(z,:))) = 1;
                            MRB_temp = mod(TEP+MRB,2);%翻转
                            c_temp = mod(MRB_temp*G_sys,2);%得到完整码字
                            operation_num = operation_num +k*(n-k+1);%********重编码操作数（包含MRB翻转）
                            WHD = mod(c_temp+llr2_hat,2)*abs(llr2');%计算当前WHD
                            operation_num = operation_num + n-1;%********计算WHD
                            operation_num = operation_num + 1;%***********下述if语句
                            if(WHD < WHD_opt)%若当前WHD更小，则进行替换
                                WHD_opt = WHD;%更新最优WHD
                                c_temp(perm_index2) = c_temp;%恢复第二次置换
                                c_temp(perm_index1) = c_temp;%恢复第一次置换
                                c_esti = c_temp;
                            end
                            WHD_thre = EWHD_LRB+TEP*abs(llr2(1:k)');%门限
                            operation_num = operation_num + 1;%***********
                            operation_num = operation_num + 1;%***********
                            if(WHD_opt<=WHD_thre)
                                return;
                            end
                            if test_num >= max_test_num
                                return;
                            end
                        end
                    end
                    
                case 2
                    for w =1:size(noise_locations1,2)
                        for x =1:size(noise_locations1{w},1)
                            for y =1:size(noise_locations2,2)
                                for z =1:size(noise_locations2{y},1)
                                    test_num = test_num+1;
                                    TEP = zeros(1,k);
                                    TEP(ind_order(noise_locations1{w}(x,:))) = 1;
                                    TEP(ind_order(noise_locations2{y}(z,:))) = 1;
                                    MRB_temp = mod(TEP+MRB,2);%翻转
                                    c_temp = mod(MRB_temp*G_sys,2);%得到完整码字
                                    operation_num = operation_num +k*(n-k+1);%********重编码操作数（包含MRB翻转） 
                                    WHD = mod(c_temp+llr2_hat,2)*abs(llr2');%计算当前WHD
                                    operation_num = operation_num + n-1;%********计算WHD
                                    operation_num = operation_num + 1;%***********下述if语句
                                    if(WHD < WHD_opt)%若当前WHD更小，则进行替换
                                        WHD_opt = WHD;%更新最优WHD
                                        c_temp(perm_index2) = c_temp;%恢复第二次置换
                                        c_temp(perm_index1) = c_temp;%恢复第一次置换
                                        c_esti = c_temp;
                                    end
                                    WHD_thre = EWHD_LRB+TEP*abs(llr2(1:k)');%门限
                                    operation_num = operation_num + 1;%***********
                                    operation_num = operation_num + 1;%***********
                                    if(WHD_opt<=WHD_thre)
                                        return;
                                    end
                                    if test_num >= max_test_num
                                        return;
                                    end
                                end
                            end
                        end
                    end
                    
                case 3
                    for w =1:size(noise_locations1,2)
                        for x =1:size(noise_locations1{w},1)
                            for y =1:size(noise_locations2,2)
                                for z =1:size(noise_locations2{y},1)
                                    for u =1:size(noise_locations3,2)
                                        for v =1:size(noise_locations3{u},1)
                                            test_num = test_num+1;
                                            TEP = zeros(1,k);
                                            TEP(ind_order(noise_locations1{w}(x,:))) = 1;
                                            TEP(ind_order(noise_locations2{y}(z,:))) = 1;
                                            TEP(ind_order(noise_locations3{u}(v,:))) = 1;
                                            MRB_temp = mod(TEP+MRB,2);%翻转 
                                            c_temp = mod(MRB_temp*G_sys,2);%得到完整码字
                                            operation_num = operation_num +k*(n-k+1);%********重编码操作数（包含MRB翻转）  
                                            WHD = mod(c_temp+llr2_hat,2)*abs(llr2');%计算当前WHD 
                                            operation_num = operation_num + n-1;%********计算WHD
                                            operation_num = operation_num + 1;%***********
                                            if(WHD < WHD_opt)%若当前WHD更小，则进行替换
                                                WHD_opt = WHD;%更新最优WHD
                                                c_temp(perm_index2) = c_temp;%恢复第二次置换
                                                c_temp(perm_index1) = c_temp;%恢复第一次置换
                                                c_esti = c_temp;
                                            end
                                            WHD_thre = EWHD_LRB+TEP*abs(llr2(1:k)');%门限
                                            operation_num = operation_num + 1;%***********
                                            operation_num = operation_num + 1;%***********
                                            if(WHD_opt<=WHD_thre)
                                                return;
                                            end
                                            if test_num >= max_test_num
                                                return;
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                    
                case 4
                    
                    for w =1:size(noise_locations1,2)
                        for x =1:size(noise_locations1{w},1)
                            for y =1:size(noise_locations2,2)
                                for z =1:size(noise_locations2{y},1)
                                    for u =1:size(noise_locations3,2)
                                        for v =1:size(noise_locations3{u},1)
                                            for p =1:size(noise_locations4,2)
                                                for q =1:size(noise_locations4{p},1)
                                                    test_num = test_num+1;
                                                    TEP = zeros(1,k);
                                                    TEP(ind_order(noise_locations1{w}(x,:))) = 1;
                                                    TEP(ind_order(noise_locations2{y}(z,:))) = 1;
                                                    TEP(ind_order(noise_locations3{u}(v,:))) = 1;
                                                    TEP(ind_order(noise_locations4{p}(q,:))) = 1;
                                                    MRB_temp = mod(TEP+MRB,2);%翻转
                                                    c_temp = mod(MRB_temp*G_sys,2);%得到完整码字
                                                    operation_num = operation_num +k*(n-k+1);%********重编码操作数（包含MRB翻转）
                                                    WHD = mod(c_temp+llr2_hat,2)*abs(llr2');%计算当前WHD
                                                    operation_num = operation_num + n-1;%********计算WHD
                                                    operation_num = operation_num + 1;%***********
                                                    if(WHD < WHD_opt)%若当前WHD更小，则进行替换
                                                        WHD_opt = WHD;%更新最优WHD
                                                        c_temp(perm_index2) = c_temp;%恢复第二次置换
                                                        c_temp(perm_index1) = c_temp;%恢复第一次置换
                                                        c_esti = c_temp;
                                                    end
                                                    WHD_thre = EWHD_LRB+TEP*abs(llr2(1:k)');%门限
                                                    operation_num = operation_num + 1;%***********
                                                    operation_num = operation_num + 1;%***********
                                                    if(WHD_opt<=WHD_thre)
                                                        return;
                                                    end
                                                    if test_num >= max_test_num
                                                        return;
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                    
                otherwise
                    error('unsupported segmentation');
                    
            end
            
        end
        
    end
    R=R+1;
end
end




