%根据速率匹配模式，对LLR做相应处理
function modified_llr = llr_rate_matching(llr, frozen_pattern, rate_matching_pattern, rate_matching_mode)
E = length(llr);
N = length(frozen_pattern);

if strcmp(rate_matching_mode,'repetition')
    if E < N
        error('rate_matching_mode is not compatible with E');
    end
    
    modified_llr = zeros(1,N);
    for i=1:E
        modified_llr(rate_matching_pattern(i)) = modified_llr(rate_matching_pattern(i)) + llr(i);
    end
else
    if strcmp(rate_matching_mode,'puncturing')
        % Zero valued LLRs are used for punctured bits, because the decoder
        % doesn't know if they have values of 0 or 1.
        if E >= N
            error('rate_matching_mode is not compatible with E');
        end
        modified_llr = zeros(1,N);
    else
        if strcmp(rate_matching_mode,'shortening')
            % Infinite valued LLRs are used for shortened bits, because the
            % decoder knows that they have values of 0.
            if E >= N
                error('rate_matching_mode is not compatible with E');
            end
            modified_llr = inf(1,N);
        else
            error('Unknown rate_matching_mode');
        end
    end
    modified_llr(rate_matching_pattern) = llr;
end
end
