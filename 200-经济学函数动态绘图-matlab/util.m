function u=util(c)

global gamma;
u=-inf*ones(size(c)); % initialize into -inf, change only those c positive
cpos=find(c>0);       % find those positive c

if gamma==1
    u(cpos)=log(c(cpos));
else
    u(cpos)=c(cpos).^(1-gamma)./(1-gamma);
end

