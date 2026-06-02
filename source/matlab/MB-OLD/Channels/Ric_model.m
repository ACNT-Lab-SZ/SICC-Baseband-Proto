function H=Ric_model(K,L)
% Rician Channel Model
%   Input:
%       K   : K factor,lnear
%       L      : # of channel realization
%   Output:
%       h      : channel vector

H = sqrt(K/(K+1)) + sqrt(1/(K+1))*Ray_model(L);