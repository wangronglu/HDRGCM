function dev = compDev(n,r,N,K,ntps,age,BX,Sigma,D,Q,Delta,sum_age,sum_age2,...
    det_A,resid2,Sigma_inv,log_det_Sigma)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute BIC given parameter estimates
%
% Input
%   n: number of subjects
%   r: number of outcomes
%   N: total number of observations, N = sum(T_i)
%   K: number of columns in Q
%   ntps: n x 1 vector of number of time points for each subject 
%   age: n x max_tps matrix, standardized age
%   BX: max_tps x r x n array of residuals, BX = y - BX
%   Sigma: r x 1 vector of variances
%   D: 2r x 1 vector of initial D
%   Q: 2r x K matrix, row norms < 1 
%   Delta: 2r x 1 vector, Delta = 1 - sum(Q.^2,2)
%   sum_age: n x 1 vector of sum(age, 2, 'omitnan');
%   sum_age2: n x 1 vector of sum(age.^2, 2, 'omitnan') 
%   det_A: n x 1 vector of (ntps.* sum_age2 - sum_age.^2)
%   resid2: r x 1 vector, sum of squared residuals
%   Sigma_inv: r x 1 vector of 1./Sigma
%   log_det_Sigma: sum(log(Sigma))
%
% Output
%   dev: -2/N * marginal LL 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute indicator of outcomes with d_{2j}=0
ind_y = (D(2:2:2*r) == 0); % r x 1
% compute indicator of d_{2j-1} with d_{2j}=0
ind_D1 = find(D==0) - 1; % sum(ind_y) x 1
% set rows corresponding to d_{2j}=0 to zero
Q(D==0,:) = 0;
Delta(D==0) = 0;

% compute D * sum(I \kron t(1,g_{it}) * inv(Sigma) * (y - BX)); 
dzsr = zeros(2*r,n);
dzsr(1:2:2*r-1,:) = squeeze( sum(BX, 1,'omitnan') ).* repmat(D(1:2:end).* Sigma_inv,[1,n]); % r x n
dzsr(2:2:2*r,:) = squeeze( sum( permute(repmat(age,[1,1,r]),[2,3,1]).* BX, 1,'omitnan') ).* ...
    repmat(D(2:2:end).* Sigma_inv,[1,n]); % r x n

% compute marginal log-likelihood
dev = (1 - 2*n/N)* log_det_Sigma + n/N * sum(log(Sigma(ind_y))) + ...
        (r - sum(ind_y))/N * sum(log(det_A)) + sum(ind_y)/N * sum(log(ntps)) + ...
        sum(resid2.* Sigma_inv)/N;
    
meta = zeros(2*r,n); % (m_1,...,m_n)

for i = 1:n
    % compute C_i
    Ci1 = Sigma.* ntps(i)/det_A(i) + D(2:2:2*r).^2.* Delta(2:2:2*r); % r x 1
    Ci2 = Sigma.* sum_age(i)/det_A(i); % r x 1
    Ci3 = Sigma.* sum_age2(i)/det_A(i) + D(1:2:2*r-1).^2.* Delta(1:2:2*r-1); % r x 1
    det_Ci_inv = Ci1.* Ci3 - Ci2.^2; % r x 1
    Ci1 = Ci1./det_Ci_inv;
    Ci2 = Ci2./det_Ci_inv;
    Ci3 = Ci3./det_Ci_inv;
    
    % modify and add zero row and zero column to C_i if some d_{2j}=0
    det_Ci_inv(ind_y) = Sigma(ind_y)./ntps(i) + D(ind_D1).^2.* Delta(ind_D1); % sum(ind_y) x 1
    Ci1(ind_y) = 1./det_Ci_inv(ind_y);
    Ci2(ind_y) = 0;
    Ci3(ind_y) = 0;
    
    % compute D * C_i * D
    Ci1 = D(1:2:end).^2.* Ci1; % r x 1
    Ci2 = D(1:2:end).* D(2:2:end).* Ci2; % r x 1
    Ci3 = D(2:2:end).^2.* Ci3; % r x 1
    
    % compute D * C_i * D * Q
    tmp = zeros(2*r,K);
    tmp(1:2:2*r-1,:) = repmat(Ci1,[1,K]).* Q(1:2:end,:) + ...
        repmat(Ci2,[1,K]).* Q(2:2:end,:);
    tmp(2:2:2*r,:) = repmat(Ci2,[1,K]).* Q(1:2:end,:) + ...
        repmat(Ci3,[1,K]).* Q(2:2:end,:);
    
    % compute R * D * Ci * D * Q
    H = Q * (Q'* tmp) + repmat(Delta,[1,K]).* tmp; % 2r x K
    
    % compute F_i ----
    [Evec,Eval] = eig(Q' * tmp, 'vector'); % K x K
    log_det_F_inv = sum(log(1 + Eval));
    F_i = Evec * ( repmat(1./(1 + Eval),[1,K]).* Evec'); % K x K
        
    % compute m_i -------------------------
    % (Delta - Delta * D * Ci * D * Delta) * dzsr
    meta(1:2:2*r-1,i) = ( Delta(1:2:end) - Delta(1:2:end).^2.* Ci1 ).* dzsr(1:2:end,i) - ...
        Delta(1:2:end).* Delta(2:2:end).* Ci2.* dzsr(2:2:end,i); % r x 1
    meta(2:2:2*r,i) = ( Delta(2:2:end) - Delta(2:2:end).^2.* Ci3 ).* dzsr(2:2:end,i) - ...
        Delta(1:2:end).* Delta(2:2:end).* Ci2.* dzsr(1:2:end,i); % r x 1
    
    % add other terms
    meta(:,i) = meta(:,i) + H * ( F_i * (H'* dzsr(:,i)) ) + ...
        ( Q - H )*( Q'* dzsr(:,i) ) - Q * ( tmp' * (Delta.* dzsr(:,i)) );
    
    dev = dev + sum(log(det_Ci_inv))/N + log_det_F_inv/N - ...
        sum(dzsr(:,i).* meta(:,i))/N;
end
