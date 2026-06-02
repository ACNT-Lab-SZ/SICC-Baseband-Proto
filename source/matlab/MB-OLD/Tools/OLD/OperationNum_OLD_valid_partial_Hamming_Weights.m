%功能说明：寻找该段合法的汉明重量
function  [operations, wi] = OperationNum_OLD_valid_partial_Hamming_Weights(R,J1,J2,s,I)
%找到该段的汉明重量
k = 0;
wi = [];

operations = 1;%*******下述判断

if R==0||R==-1%%若该段直线对应的值为0，表明该段不翻转，但其他段可能翻转，因此先赋值0
    wi(1) = 0;
    return;
end

operations = operations+1;%*******下述判断，每段最大值前面已算出

if(R<=(s*(J1+1)+I+s*J2+I)*(J2-J1)/2)
    w = ceil((sqrt(8*s*R+(2*s*J1+2*I+s)^2)-(2*s*J1+2*I+s))/(2*s));
    
    operations = operations+7;%*******上述计算汉明重量下界
    operations = operations+4;%*******上述计算汉明重量上界
    
    while w <= floor( (2*s*J2+2*I+s-sqrt(-8*s*R+(2*s*J2+2*I+s)^2))/(2*s) )
        
        operations = operations+4;%*******下述判断整除
        if(mod(R - w*I,s)==0)%能够被整除即可
            k = k+1;
            wi(k) = w;
        end
        w = w+1;
    end
end
end

