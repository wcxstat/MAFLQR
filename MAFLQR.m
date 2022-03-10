function [Qpred,w,MAintep,MAslope]=MAFLQR(t,u,y,tau,p,candmodel,K,t_new,u_new,xx)

% Input:
% t,u: covarate u_{il}=X_i(t_{il})+epsilon_{il}
% y: response
% tau: quantile level
% p: set option
% candmodel: candidate models (cutoff level of FPCA)
% K: K-fold cross validation to choose weights
% t_new,u_new: prediction covariate

% Output:
% Qpred: MA quantile prediction from (t_new,u_new) (row vector)
% w: selected weights (length(candmodel) by 1 vector)
% MAintep: averaged interpect (a number)
% MAslope: averaged slope function (row vector)


n0=length(t_new); % sample size for prediction data
maxN=max(candmodel);
if nargin==9
    xx = FPCA(u,t,p);
end

[intep,slope,coefmat]=FLQR(t,u,y,tau,p,candmodel,xx);
[~,newpcx]=FPCApred(xx,u_new,t_new);
pred_candmodel=[ones(n0,1),newpcx(:,1:maxN)]*coefmat;
if length(candmodel)==1
    w=1;
else
    w=MACV(t,u,y,tau,p,candmodel,K,xx);
end
Qpred=(pred_candmodel*w)';
MAintep=intep*w;
MAslope=(slope*w)';