function D2 = BICtune_D(nlambda2,lambda2_min_ratio,n,N,r,K,ntps,age,...
    BX,Sigma,D,Q,Delta,sum_age,sum_age2,det_A,resid2,Sigma_inv,log_det_Sigma,...
    ind_y,gM2,gM3,H2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tune penalty factor lambda2 for D(2:2:2r) by BIC = -2/N * marginal LL + log(n)/N * (dB + dG))
%
% Input
%   nlambda2: number of candidate values for lambda2
%   lambda2_min_ratio:  smallest value for lambda2, as a fraction of lambda2_max.
%   n: number of subjects
%   N: total number of observations, N = sum(T_i)
%   r: number of outcomes
%   K: number of columns in Q
%   ntps: n x 1 vector of number of time points for each subject 
%   age: n x max_tps matrix, standardized age
%   BX: max_tps x r x n array of residuals, BX = y - BX
%   Sigma: r x 1 vector of variances
%   D: 2r x 1 vector, {d_j:j=1,...,2r}
%   Q: 2r x K matrix, row norms < 1 
%   Delta: 2r x 1 vector, Delta = 1 - sum(Q.^2,2)
%   sum_age: n x 1 vector of sum(age, 2, 'omitnan');
%   sum_age2: n x 1 vector of sum(age.^2, 2, 'omitnan') 
%   det_A: n x 1 vector of (ntps.* sum_age2 - sum_age.^2)
%   resid2: r x 1 vector, sum of squared residuals
%   Sigma_inv: r x 1 vector of 1./Sigma
%   log_det_Sigma: sum(log(Sigma))
%   ind_y: r x 1 vector, indicator of outcomes with d_{2j}=0 (cannot be a vector of all 1's)
%   gM2: r x 1 vector of sum( M_i(2j-1,2j) * sum(g_{it}) )
%   gM3: r x 1 vector of sum( M_i(2j,2j) * sum(g_{it}^2) )
%   H2: r x 1 vector of sum(m_{i,2j} * sum(g_{it} * (y-BX)))
%
% Output
%   D2: estimate of D(2:2:2r) under optimal lambda2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute c_j
H2 = H2 - D(1:2:end).* gM2; % r x 1
% compute OLS of d_{2j} and take absolute value
abs_D2_bar = abs(H2(~ind_y))./ gM3(~ind_y); % sum(~ind_y) x 1

% Find lambda2_max that shrinks all the d_2j's to zero
lambda2_max = max( abs(H2(~ind_y)).* abs_D2_bar.* Sigma_inv(~ind_y) * 2/N );

% compute BIC at lambda2_max ------------------
BIC_set = zeros(1,nlambda2);

% compute -2/N * marginal LL
D(2:2:end) = 0;
dev = compDev(n,r,N,K,ntps,age,BX,Sigma,D,Q,Delta,sum_age,sum_age2,det_A,...
    resid2,Sigma_inv,log_det_Sigma);

% add degree of freedom
BIC_set(nlambda2) = dev; % nnz(D(2:2:end)) = 0; nnz(D(1:2:end)) & nnz(phi) are same across lambda2

% compute BIC for other lambda2 values ----------------
% set lambda2 sequence
lambda2_max = log(lambda2_max);
lambda2_seq = exp( ...
    linspace( (log(lambda2_min_ratio) + lambda2_max), lambda2_max, nlambda2 ) ...
    ); % 1 x nlambda2

D2 = H2; % r x 1
for j = 1:nlambda2-1
    lambda2 = lambda2_seq(j);   
    D2(~ind_y) = sign(H2(~ind_y)).* ...
        max( abs(H2(~ind_y)) - lambda2./abs_D2_bar.* Sigma(~ind_y) * N/2, 0)./ gM3(~ind_y);
    D(2:2:end) = D2; % r x 1
    
    % compute -2/N * marginal LL
    dev = compDev(n,r,N,K,ntps,age,BX,Sigma,D,Q,Delta,sum_age,sum_age2,det_A,...
        resid2,Sigma_inv,log_det_Sigma);
    
    % add degree of freedom
    df = nnz(D(2:2:end)) * (K+1);
    BIC_set(j) = dev + log(n)/N * df; % nnz(D(1:2:end)) & nnz(phi) are same across lambda2 values
end

% choose lambda corresponding to minimum BIC
lambda2 = max(lambda2_seq(BIC_set <= min(BIC_set)));
D2(~ind_y) = sign(H2(~ind_y)).* ...
        max( abs(H2(~ind_y)) - lambda2./abs_D2_bar.* Sigma(~ind_y) * N/2, 0)./ gM3(~ind_y);
    