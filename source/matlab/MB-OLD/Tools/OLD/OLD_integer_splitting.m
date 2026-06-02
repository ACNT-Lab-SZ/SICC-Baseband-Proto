function Wm = OLD_integer_splitting(W,m)
%将整数W分为m个整数的和，m个数可以相同，存储在Wm中，共m列，每一行是一种分割
Wm = [];
switch m
    case 2
        k = 0;
        for W1 = 0:W
            k = k+1;
            W2 = W-W1;
            Wm(k,:) = [W1 W2];
        end
        
    case 3
        k = 0;
        
        for W1 = 0:W
            for W2 = 0:W-W1
                k = k+1;
                W3 = W-W1-W2;
                Wm(k,:) = [W1 W2 W3];
            end
        end
        
        
    case 4
        k = 0;
        
        for W1 = 0:W
            for W2 = 0:W-W1
                for W3 = 0:W-W1-W2
                    k = k+1;
                    W4 = W-W1-W2-W3;
                    Wm(k,:) = [W1 W2 W3 W4];
                end
            end
        end
        
        
        
        
    otherwise
        error(['unsupported integer splitting: W = ' num2str(W)  '  m = '  num2str(m)]);
        
end

end