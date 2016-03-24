function [out] = poisson_calc(vo, lambda)

if nargin <  2
    lambda = 8;
end
f_poisson = @(x,lambda) (lambda.^x).*exp(-lambda)./factorial(x);
%factorial_approx = @(n) sqrt(2*pi.*n).*(n/exp(1)).^n;
%f_poisson = @(x,lambda) (lambda.^x).*exp(-lambda)./factorial_approx(x);
integrate = @(f,a,b) discrete_integral(f,a,b);
out = zeros(size(vo));

for k = 1:size(vo,3)
    for j = 1:size(vo,2)
        for i = 1:size(vo,1)
            a = min(i,i-vo(i,j,k));
            b = max(i,i-vo(i,j,k));
            %{
            if sign(vo(i,j,k)) == sign(-1)
                a = i + vo(i,j,k);
                b = i;
            else
                a = i + vo(i,j,k);
                b = i;
            %}
            out(i,j,k) = integrate(@(x)f_poisson(x,lambda),a,b);
            
        end
        
    end
    display_progress(k)
end
