function [NameIDi] = calculate_yearly_diff(NID,N,lim)
if nargin < 3
    lim = size(NID,1);
end
if nargin < 2 || sum(N ~= 1) || sum(N ~= 2)
    N = [1,2];
end

NameIDi = zeros(lim,2,size(NID,3));

for k = 2:size(NID,3)
    for i = 1:lim
        for j = N
            MIDi = NID(i,j,k);
            ii = 0;
            isFound = 0;
            while ~isFound && ii < size(NID,1)
                ii = ii + 1;
                isFound = (MIDi == NID(ii,j,k-1));
            end
            if isFound
                NameIDi(i,j,k) = i-ii;
            else 
                NameIDi(i,j,k) = -1;
            end
        end
    end
    k
end
