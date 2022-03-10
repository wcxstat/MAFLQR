function [w]=MACV(t,u,y,tau,p,candmodel,K,xx)

% Input:
% t,u: covarate u_{il}=X_i(t_{il})+epsilon_{il}
% y: response
% tau: quantile level
% p: set option
% candmodel: candidate models (cutoff level of FPCA)
% K: K-fold cross validation
% xx: FPCA object

% Output:
% w: selected weights (length(candmodel) by 1 vector)

n=length(y); % sample size
M=floor(n/K); % sample size for each group
if nargin==7
    xx = FPCA(u,t,p);
end
eigfunx = getVal(xx,'phi');
scorex = getVal(xx,'xi_est');
newt=p.newdata;
numcand=length(candmodel); % the number of candidate models
Qhat=zeros(n,numcand);

for k=1:K
    if k<K
        index=(k-1)*M+(1:M);
    else
        index=(K*M-M+1):n;
    end
    index1=setdiff(1:n,index);
    t_k=t(index1);
    u_k=u(index1);
    y_k=y(index1);
    [intep,slope]=FLQR(t_k,u_k,y_k,tau,p,candmodel);
    for J=1:numcand
        Xc=scorex(index,1:candmodel(J))*(eigfunx(:,1:candmodel(J)))';
        Qhat(index,J)=intep(J)+trapz(newt,repmat(slope(:,J),1,length(index)).*Xc')';
    end
end

f=[zeros(numcand,1);tau*ones(n,1);(1-tau)*ones(n,1)];
Aeq=[[Qhat,eye(n),-eye(n)];[ones(1,numcand),zeros(1,n),zeros(1,n)]];
slt=linprog(f,[],[],Aeq,[y';1],zeros(numcand+2*n,1),[]);
w=slt(1:numcand);