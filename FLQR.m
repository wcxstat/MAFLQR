function [intep,slope,coefmat,valvec]=FLQR(t,u,y,tau,p,candmodel,xx)

% Input:
% t,u: covarate u_{il}=X_i(t_{il})+epsilon_{il}
% y: response
% tau: quantile level
% p: set option
% candmodel: candidate models (cutoff level of FPCA)
% xx: FPCA object
% MODEL: Q_tau(Y|X)=a+\int_T b(t)(X(t)-mu(t))dt

% Output:
% intep: intercept (1-by-length(candmodel) vector)
% slope: slope function (newt-by-length(candmodel) matrix)
% coefmat: coefficient matrix (including intercept)

if nargin==6
    xx = FPCA(u,t,p);
end
eigfunx = getVal(xx,'phi');
scorex = getVal(xx,'xi_est');
maxN=max(candmodel);

numcand=length(candmodel); % the number of candidate models
intep=zeros(1,numcand);
slope=zeros(size(eigfunx,1),numcand);
coefmat=zeros(maxN+1,numcand);
valvec=zeros(1,numcand);
for J=1:numcand
    xi=scorex(:,1:candmodel(J));
    [coef,val]=quantreg(xi,y,tau,1);
    intep(J)=coef(1);
    if candmodel(J)==0
        slope(:,J)=zeros(size(eigfunx,1),1);
    else
        slope(:,J)=eigfunx(:,1:candmodel(J)) * coef(2:end);
    end
    coefmat(1:(candmodel(J)+1),J)=coef;
    valvec(J)=val;
end