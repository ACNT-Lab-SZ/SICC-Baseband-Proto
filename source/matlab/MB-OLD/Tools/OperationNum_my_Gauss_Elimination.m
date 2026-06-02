
function [operations, H, exchangej] = OperationNum_my_Gauss_Elimination(H)
operations = 0;
[m,n] = size(H);
exchangej = 1:n;
for i=1:m % 逐一检查H的左边待单位化的矩阵的主对角线元素，若为0，则优先行交换，再列交换，使得该元素非零
    operations = operations+1;%**************下述if语句
    if H(i,i)==0 % 若H(i,i)==0
        j=i+find(H(i+1:m,i), 1,'first');% 优先行交换，寻找该列中的第一个非零元，记录该非零元的行（ H(i+1:m,i)只对该列H(i,i)后面的元素寻找，因为在i上面的行已经交换好不可再改变）
        if ~isempty(j)%若不为空
            operations = operations+(j-i);%**********find等价于做了(j-i)次比较操作
        end
        H([j i],:)=H([i,j],:);% 将H的该行与H的第i行交换
        operations = operations+1;%************下述if语句
        if isempty(j)
            j=i+find(H(i,i+1:n), 1);%寻找该行中第一个非零元，记录该非零元的列 ,如果也为空,那就go die吧
            operations = operations+(j-i);%************find等价于做了(j-i)次比较操作
            if isempty(j)
                error('Matrix is not full rank, systematic matrix can not be made');
            end
            H(:,[j i])=H(:,[i,j]);  %交换列
            exchangej([i j])=exchangej([j i]);%置换后的下标
            
            %确保LRB部分是降序排列的-根据exchangej中LRB部分的元素是递增排列的
            if(j>m)%表明第j列是在LRB中的，将此时exchangej中第j个元素和左边直到m+1个元素依次比较
                for k=j-1:-1:m+1
                    if(exchangej(k+1)<exchangej(k))
                        H(:,[k+1 k])=H(:,[k,k+1]);  %交换列
                        exchangej([k+1 k])=exchangej([k k+1]);%置换后的下标
                    end
                end
            end
        end
        
        for k=i+1:m % 将H(i,i)元素以下的非零元采用行叠加变零
            operations = operations+1;%*********下述if语句
            if H(k,i)==1 % 有，则将H的第i行叠加到该行
                H(k,:)=H(i,:)+H(k,:);
                H(k,:)=mod(H(k,:),2); %%%%进行叠加处！！！！！
                operations = operations+(n-i);%***********两行之间模二加,对剩余n-i个元素相加即可   
            end % 无,则检查H的下一行
        end
    else % 若H(i,i)==1，不需要交换行，只需要检查H(i,i)元素以下的第i+1行到m行是否还有非零元
        for k=i+1:m %将H(i,i)元素以下的非零元采用行叠加变零
            operations = operations+1;%*************下述if语句
            if H(k,i)==1 % 有，则将H的第i行叠加到该行
                H(k,:)=H(i,:)+H(k,:);
                H(k,:)=mod(H(k,:),2);  %%%%进行叠加处！！！！！
                operations = operations+(n-i);%*********两行之间模二加
            end
        end
    end  % 无,则检查H的下一行
end
     
for i=m:-1:1 % 自第m行向第j行叠加,j=m-1:-1:1 ，将H左边待单位化的矩阵的主对角线上面的元素通过行叠加变零
    for k=i-1:-1:1  
        operations = operations+1;%*************下述if语句
        if H(k,i)==1
            H(k,:)=H(i,:)+H(k,:);
            H(k,:)=mod(H(k,:),2);   %%%%进行叠加处！！！！！
            operations = operations+(n-i);%*********两行之间模二加
        end
    end % 循环之后得到左半为单位阵的矩阵H = [I|P]
end
end


