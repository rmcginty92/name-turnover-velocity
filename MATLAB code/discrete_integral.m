function [result] = discrete_integral(func,a,b);
% Discretized integration on function handle a = min, b = max
result = 0;
for i =a:b
    result = result + func(i);
end
end