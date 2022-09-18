function [phi,flag_phi] = BICtune_phi(nlambda1,lambda1_min_ratio,n,N,r,K,...
    y,ntps,age,X2,B1X1,H,Sigma,D,Q,Delta,sum_age,sum_age2,det_A,sum_XX2_inv,...
    Sigma_inv,log_det_Sigma,maxit,tol)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tune penalty factor lambda1 for phi by BIC = -2/N * marginal LL + log(n)/N * (dB + dG))
% require findLam1max.m
% require compDev.m
% require solveB.m
%
% Input
%   nlambda1: number of candidate values for lambda1
%   lambda1_min_ratio:  smallest value for lambda1, as a fraction of lambda1_max.
%       If nobs > nvaribles, 0.0001 is recommended; otherwise 0.01 is suggested.
%   n: number of subjects
%   N: total number of observations, N = sum(T_i)
%   r: number of outcomes
%   K: number of columns in Q
%   y: max_tps x r x n array of continuous responses 
%   ntps: n x 1 vector of number of time points for each subject 
%   age: n x max_tps matrix, standardized age
%   X2: max_tps x (1+p) x n array, (standardized) design matrix (g_{it},u_i*g_{it})
%   B1X1: max_tps x r x n array of mmx('mult',X1,rho)
%   H: max_tps x r x n array of Z * zeta
%   Sigma: r x 1 vector of variances
%   D: 2r x 1 vector of initial D
%   Q: 2r x K matrix, row norms < 1 
%   Delta: 2r x 1 vector, Delta = 1 - sum(Q.^2,2)
%   sum_age: n x 1 vector of sum(age, 2, 'omitnan');
%   sum_age2: n x 1 vector of sum(age.^2, 2, 'omitnan') 
%   det_A: n x 1 vector of (ntps.* sum_age2 - sum_age.^2)
%   sum_XX2_inv: (1+p) x (1+p) matrix, inv( sum(X2_i'*X2_i) )
%   Sigma_inv: r x 1 vector of 1./Sigma
%   log_det_Sigma: sum(log(Sigma))
%   maxit: maximum iterations
%   tol: threshold of relative change in parameter estimate
%
% Output
%   phi: estimate of phi under optimal lambda1
%   flag_phi: indicator of convergence (1 converge, 0 not)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute (y - X1 * rho - H)
H = y - B1X1 - H; % max_tps x r x n
% set NaN to 0
H(isnan(H)) = 0;

% compute OLS of phi
phi_bar = sum_XX2_inv * sum(mmx('mult',X2,H,'tn'),3); % (1+p) x r

% Find lambda1_max that shrinks all the entries of phi to zero
lambda1_max = findLam1max(N,H,X2,phi_bar,abs(phi_bar),Sigma_inv);

% compute BIC at lambda1_max ------------------
BIC_set = zeros(1,nlambda1);

% compute residuals
BX = y - B1X1; % mmx('mult',X2,phi) is zero
% compute sum of squared residuals
resid2 = squeeze( sum(sum(BX.^2,1,'omitnan'),3) )'; % r x 1

% compute -2/N * marginal LL
dev = compDev(n,r,N,K,ntps,age,BX,Sigma,D,Q,Delta,sum_age,sum_age2,det_A,...
    resid2,Sigma_inv,log_det_Sigma);

% add degree of freedom
BIC_set(nlambda1) = dev; % df = nnz(phi) = 0; nnz(D) * (K+1) is same across lambda1

% compute BIC for other lambda1 values ----------------
% set lambda1 sequence
lambda1_max = log(lambda1_max);
lambda1_seq = exp( ...
    linspace( (log(lambda1_min_ratio) + lambda1_max), lambda1_max, nlambda1 ) ...
    ); % 1 x nlambda1

for j = 1:nlambda1-1
    lambda1 = lambda1_seq(j);
    [phi,flag_phi] = solveB(N,H,X2,phi_bar,Sigma,lambda1,abs(phi_bar),maxit,tol);
    
    if ~flag_phi
        return
    end
    
    % compute residuals
    BX = y - B1X1 - mmx('mult',X2,phi); 
    % compute sum of squared residuals
    resid2 = squeeze( sum(sum(BX.^2,1,'omitnan'),3) )'; % r x 1
    
    % compute -2/N * marginal LL
    dev = compDev(n,r,N,K,ntps,age,BX,Sigma,D,Q,Delta,sum_age,sum_age2,det_A,...
        resid2,Sigma_inv,log_det_Sigma);
    
    % add degree of freedom
    df = nnz(phi);
    BIC_set(j) = dev + log(n)/N * df; % nnz(D) * (K+1) is same across lambda1 values
end

% choose lambda corresponding to minimum BIC
lambda1 = max(lambda1_seq(BIC_set <= min(BIC_set)));
phi = solveB(N,H,X2,phi_bar,Sigma,lambda1,abs(phi_bar),maxit,tol);

