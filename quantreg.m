function [coef,vel]=quantreg(x,y,tau,intc)

% Input:
% x: n*p matrix, covariate
% y: n*1 vector, response
% tau: quantile level (0<tau<1)
% intc: 1 or 0 (include intercept?)

% Output:
% coef: intercept and slope (p-by-1 or (p+1)-by-1 vector)

n=length(y);
if intc==1
    x=[ones(n,1),x];
end
p=size(x,2);
f=[zeros(p,1);tau*ones(n,1);(1-tau)*ones(n,1)];
A=[zeros(2*n,p),-eye(2*n)];
b=zeros(2*n,1);
Aeq=[x,eye(n),-eye(n)];
[coef,vel]=linprog(f,A,b,Aeq,y);
coef=coef(1:p);
end