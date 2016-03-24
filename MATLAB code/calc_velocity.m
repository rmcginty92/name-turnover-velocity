function [velocity_out] = calc_velocity(nid,N,lim)
%   This computes a metric quantifying how much change occurs in the
%   rankings of most popular names in a given year.
%   nid:    3 dimensional matrix of numerical name id values. First 
%           dimension - list of unique ids, second dimension - gender,
%           and third dimension - year (index 1 = 1880)
%
%   N:      Specifies which column in 2nd dim to compute (default=all)
N = 1:size(nid,2);
%{
lambda = 8;
factorial_approx = @(n) sqrt(2*pi.*n).*(n/exp(1)).^n;
f_poisson = @(x,lambda) (lambda.^x).*exp(-lambda)./factorial_approx(x);
f_poisson = @(x,lambda) (lambda.^x).*exp(-lambda)./factorial(x);
integrate = @(f,a,b) discrete_integral(f,a,b);
%out = integral(@(x)f_poisson(x,lambda),0,10);
%}
% Consider just squaring difference of place in top 20
%{
output1 = zeros(size(nid));
output2 = zeros(size(nid));
output3 = zeros(size(nid));
%}
velocity_out = zeros(size(nid));

for k = 2:size(nid,3)
    for j = N
        for i = 1:size(nid,1)
            if nid(i,j,k) > 0
                i_prev = find(nid(:,j,k) == nid(i,j,k-1));
                if isempty(i_prev) && nid(end,j,k) < 1; i_prev = find(nid(:,j,k) < 1); 
                elseif isempty(i_prev); i_prev = size(nid,1);end
                velocity_out(i,j,k) = i - i_prev(1);
            end
        end
        disp(['Completed k = ',num2str(k)]);
        %fflush(stdout);
    end
end


%velocity_out = cat(4,output1,output2,output3)

end