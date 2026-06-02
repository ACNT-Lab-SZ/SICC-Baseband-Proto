function num_4_cycles = count_4_cycles(H, cn)
disp('We are now counting the number of length-4-cycles in the target H.');
M = size(H, 1);
num_4_cycles = 0;
for m1 = 1 : M - 1%遍历第一行到倒数第二行
    for i1 = 1 : cn(m1)%遍历每一行的非零元素位置
        index_1 = H(m1, i1);
        for m2 = m1 + 1 : M - 1%遍历下一行到最后一行
            for i2 = 1 : cn(m2)%遍历每一行的非零元素位置
                index_2 = H(m2, i2);
                if index_1 == index_2
                  for i3 = i2+1 : cn(m2)
                    index_3 = H(m2, i3);
                    for i4 = i1+1 : cn(m1)
                        index_4 = H(m1, i4);
                        if index_3 == index_4
                           num_4_cycles = num_4_cycles + 1;
                        end
                    end
                  end
                end
            end
        end
    end
end
disp(' ');
disp(['The number of 4-cycles in the target H is ' num2str(num_4_cycles) '.'])

