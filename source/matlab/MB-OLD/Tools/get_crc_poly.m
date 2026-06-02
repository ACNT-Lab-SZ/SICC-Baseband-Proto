function g = get_crc_poly(crc_length)
switch crc_length
    case 0
        g = [1];
    case 4
        g = [1 0 0 1 1];
    case 6
        g = [1 1 0 0 0 0 1];%5G
    case 7
        g = [1 1 1 0 0 1 0 1];%文献CRC Codes as Error Correction Codes
    case 8
%         g = [1 0 1 0 0 1 1 0 1];
        g = [1 1 1 1 1 1 0 0 1];
    case 10
        g = [1 1 0 0 1 0 0 1 1 1 1];
    case 11
        g = [1 1 1 0 0 0 1 0 0 0 0 1];%5G
    case 12
        g = [1 1 0 0 0 0 0 0 0 1 1 0 1];
    case 16
        %g = [1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1];
        g = [1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1];%5G
    case 24
        g = [1 1 0 1 1 0 0 1 0 1 0 1 1 0 0 0 1 0 0 0 1 0 1 1 1];%5G CRC24C 
        %g = [1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1];%5G CRC24B
        %g = [1 1 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 1 1 1 1 1 0 1 1];%5G CRC24A
    otherwise
        disp('Unsupported CRC length. Program terminates')
end