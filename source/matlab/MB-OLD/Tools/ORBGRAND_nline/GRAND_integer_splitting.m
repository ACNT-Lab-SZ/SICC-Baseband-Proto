function Wm = GRAND_integer_splitting(W,m,flag)
%将整数W分为m个整数的和，m个数可以相同，存储在Wm中，共m列，每一行是一种分割
Wm = [];
switch m
    case 2
        k = 0;
        if(flag==0)    %2023.4.30,若flag==0，即可靠度Wi=0对第一段仍需要翻转的，此时应从-1开始分段，对应第一段不翻转；否则，可靠度Wi=0不需翻转
            for W1 = -1:W
                k = k+1;
                W2 = W-W1;
                Wm(k,:) = [W1 W2];
            end
        else
            for W1 = 0:W
                k = k+1;
                W2 = W-W1;
                Wm(k,:) = [W1 W2];
            end
        end
        
    case 3
        k = 0;
        if(flag==0)
            for W1 = -1:W
                for W2 = 0:W-W1
                    k = k+1;
                    W3 = W-W1-W2;
                    Wm(k,:) = [W1 W2 W3];
                end
            end
        else
            for W1 = 0:W
                for W2 = 0:W-W1
                    k = k+1;
                    W3 = W-W1-W2;
                    Wm(k,:) = [W1 W2 W3];
                end
            end
        end
        
    case 4
        k = 0;
        if(flag==0)
            for W1 = -1:W
                for W2 = 0:W-W1
                    for W3 = 0:W-W1-W2
                        k = k+1;
                        W4 = W-W1-W2-W3;
                        Wm(k,:) = [W1 W2 W3 W4];
                    end
                end
            end
        else
            for W1 = 0:W
                for W2 = 0:W-W1
                    for W3 = 0:W-W1-W2
                        k = k+1;
                        W4 = W-W1-W2-W3;
                        Wm(k,:) = [W1 W2 W3 W4];
                    end
                end
            end
            
            
        end
    otherwise
        error(['unsupported integer splitting: W = ' num2str(W)  '  m = '  num2str(m)]);
        
end

end