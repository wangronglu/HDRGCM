function [mu0,alpha0,mu1,alpha1,gamma,Sigma,D,Q,meta,flag,dev,ntry_Q,ntry_BD,...
    flag_Q,flag_phi,flag_BDS] = ...
    HDRGCMrap(y,ntps,age,u_pred,w_pred,K,nlambda1,lambda1_min_ratio,nlambda2,...
    lambda2_min_ratio,maxit,tol,ss,mu0,alpha0,mu1,alpha1,gamma,Sigma,D,Q)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate reparameterized HDRGCM with adaptive L1 penalty on mu1,alpha1 and D.
% R = Q * Q' + I - diag(Q * Q') is a correlation matrix. 
% G = diag(D) * R * diag(D) is the covariance matrix of random effects.
% Covariates will be standardized so that each column mean is 0 and each column
%   norm is sqrt(N) in the design matrix.
% Estimated coefficients will be transformed to original scales except for age.
% Q is updated by projected gradient descent (require updateQ.m & projectGD_Q.m).
% Stop inner/outer iterations when relative change of parameter estimate < tol.
% The penalty factors are tuned at each inner cycle of M step by BIC (log(n)).
% Require BICtune_phi.m, BICtune_D.m, solveB.m, findLam1max.m and compDev.m
% 
% Input:
%   y: max_tps x r x n array of continuous responses 
%   ntps: n x 1 vector of number of time points for each subject ( ntps(i)>=3 )
%   age: n x max_tps matrix, age(i,1:ntps(i)) contains the age for each time point of subject i 
%   u_pred: n x p matrix of time-invariant covariates or []
%   w_pred: n x max_tps x q array of time-varying covariates or []
%   K: low rank in factor model 
%   nlambda1: number of candidate values for lambda1 in BIC tuning
%   lambda1_min_ratio:  smallest value for lambda1, as a fraction of lambda1_max.
%           If nobs > nvaribles, 0.0001 is recommended; otherwise 0.01 is suggested.
%   nlambda2: number of candidate values for lambda2 in BIC tuning
%   lambda2_min_ratio:  smallest value for lambda2, as a fraction of lambda2_max.
%   maxit: maximum iterations
%   tol: threshold of relative change in parameter estimate
%   ss: stepsize in projected gradient descent
%   mu0: 1 x r vector of initial fixed intercepts 
%   alpha0: p x r matrix of initial coefficients for u_pred
%   mu1: 1 x r vector of initial fixed slopes for standardized age
%   alpha1: p x r matrix of initial coefficients for interaction terms
%           of u_pred and standardized age
%   gamma: q x r matrix of initial coefficients for w_pred 
%   Sigma: r x 1 vector of initial variances (entries should all be positive) 
%   D: 2r x 1 vector of initial D
%   Q: 2r x K matrix of initial Q (row-wise norms < 1)
%
% Output:
%   mu0: 1 x r vector of fixed intercepts
%   alpha0: p x r matrix of coefficients for u_pred
%   mu1: 1 x r vector of fixed slopes for standardized age
%   alpha1: p x r matrix of coefficients for interaction terms of u_pred and standardized age 
%   gamma: q x r matrix of coefficients for w_pred
%   Sigma: r x 1 vector of variances
%   D: 2r x 1 vector
%   Q: 2r x K matrix 
%   meta: 2r x n matrix of expected random effects eta_i,i=1,...,n.
%   flag: indicator of convergence; 1: converge, 0: not converge.
%   dev: -2/N * marginal log-likelihood over iterations
%   ntry_Q: number of inner iterations for updating Q at each M step
%   ntry_BD: number of inner iterations for updating B,D and Sigma at each M step
%   flag_Q: indicator of convergence for Q; 1: converge, 0: not converge.
%   flag_phi: indicator of convergence for phi; 1: converge, 0: not converge.
%   flag_BDS: indicator of convergence for B,D,Sigma; 1: converge, 0: not converge.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% process initial values
if sum(Sigma <= 0)
    error('Initial entries of Sigma must all be positive.')
end

Delta = 1 - sum(Q.^2,2); % 2r x 1
if sum(Delta <= 0)
    error('Row norms of initial Q must all < 1.')
end

%% record dimensions
[max_tps,r,n] = size(y);
    
if isempty(u_pred)
    p = 0;
    u_pred = zeros(n,p);
else
    p = size(u_pred,2);
end

if isempty(w_pred)
    q = 0;
    w_pred = zeros(n,max_tps,q);
else
    q = size(w_pred,3);
end

N = sum(ntps); % number of observations for each outcome

flag = 0;
flag_Q = 0;
flag_phi = 0;
flag_BDS = 0;
dev = zeros(maxit,1); % -2/N * marginal log-likelihood 

% number of inner iterations per outer iteration
ntry_BD = zeros(maxit,1); 
ntry_Q = zeros(maxit,1);

maxQ = sqrt(0.99); % maximum row norm of Q in PGD

%% standardize covariates
% standardize age
mean_age = mean(age(:),'omitnan');
std_age = std(age(:),1,'omitnan'); % normalize by N
age = (age - mean_age)./std_age; % column norm in design matrix = sqrt(N)

% create and standardize time interaction terms
uage_pred = repmat(age,[1,1,p]).* permute(repmat(u_pred,[1,1,max_tps]),[1,3,2]); % n x mat_tps x p
uage_pred = reshape(uage_pred,[n * max_tps,p]);
mean_uage = mean(uage_pred,1,'omitnan'); % 1 x p
std_uage = std(uage_pred,1,1,'omitnan'); % 1 x p, normalize by N
uage_pred = (uage_pred - repmat(mean_uage,[n * max_tps,1]))./ repmat(std_uage,[n * max_tps,1]);
uage_pred = reshape(uage_pred,[n,max_tps,p]);

% standardize u_pred (weighted by # time points)
mean_u = sum(u_pred.* repmat(ntps,[1,p]),1)./sum(ntps); % 1 x p
u_pred = u_pred - repmat(mean_u,[n,1]);
std_u = std(u_pred, ntps, 1); % 1 x p, normalize by N
u_pred = u_pred./repmat(std_u,[n,1]); % column norm in design matrix = sqrt(N)

% standardize w_pred 
w_pred = reshape(w_pred,[n * max_tps,q]);
mean_w = mean(w_pred,1,'omitnan'); % 1 x q
std_w = std(w_pred,1,1,'omitnan'); % 1 x q, normalize by N
w_pred = (w_pred - repmat(mean_w,[n * max_tps,1]) )./ repmat(std_w,[n * max_tps,1]);
w_pred = reshape(w_pred,[n,max_tps,q]); % column norm in design matrix = sqrt(N)

% transform parameters correspondingly ----------------------------
mu0 = mu0 + mean_u * alpha0 + mean_uage * alpha1 + mean_w * gamma;
alpha0 = alpha0.* repmat(std_u',[1,r]);
alpha1 = alpha1.* repmat(std_uage',[1,r]);
gamma = gamma.* repmat(std_w',[1,r]);

rho = [mu0; alpha0; gamma]; % (1+p+q) x r
phi = [mu1; alpha1]; % (1+p) x r

%% Intermediate variables
% functions of age 
sum_age = sum(age, 2, 'omitnan'); % n x 1, sum(sum_age) = 0
sum_age2 = sum(age.^2, 2, 'omitnan'); % n x 1, sum(sum_age2) = N
% determinant of A_i
det_A = ntps.* sum_age2 - sum_age.^2; % n x 1 

% design matrix of random effects
Z = ones(n,max_tps,2);
Z(:,:,2) = age;
Z = permute(Z,[2,3,1]); % max_tps x 2 x n

% design matrix of fixed effects ------------------
X1 = zeros(max_tps,1+p+q,n); 
sum_XX1_inv = zeros(1+p+q,1+p+q);
X2 = zeros(max_tps,1+p,n);
sum_XX2_inv = zeros(1+p,1+p);

for i = 1:n
    % X1 = (1,u_i,w_{it})
    tmp = [repmat([1,u_pred(i,:)],[ntps(i),1]), reshape(w_pred(i,1:ntps(i),:),[ntps(i),q])]; % ntps(i) x (1+p+q)
    X1(1:ntps(i),:,i) = tmp;
    sum_XX1_inv = sum_XX1_inv + tmp'* tmp;
    
    % X2 = (g_{it},u_i*g_{it})
    tmp = [age(i,1:ntps(i))',reshape(uage_pred(i,1:ntps(i),:),[ntps(i),p])]; % ntps(i) x (1+p)
    X2(1:ntps(i),:,i) = tmp;
    sum_XX2_inv = sum_XX2_inv + tmp'* tmp;
end
% compute inverse
sum_XX1_inv = sum_XX1_inv \ eye(1+p+q); 
sum_XX2_inv = sum_XX2_inv \ eye(1+p); 

clear u_pred w_pred uage_pred

% compute BX
BX = mmx('mult',X1,rho) + mmx('mult',X2,phi); % max_tps x r x n
% compute residuals
BX = y - BX; % max_tps x r x n
% compute sum of squared residuals
resid2 = squeeze( sum(sum(BX.^2,1,'omitnan'),3) )'; % r x 1

% compute sum(log(Sigma)) and Sigma_inv
tmp = log(Sigma); % r x 1
log_det_Sigma = sum(tmp);
Sigma_inv = exp(-tmp); % 1./Sigma, r x 1

%% E step given initial parameters 
% indicator of outcomes with d_{2j}=0
ind_y = (D(2:2:2*r) == 0); % r x 1
% indicator of d_{2j-1} with d_{2j}=0
ind_D1 = find(D==0) - 1; % sum(ind_y) x 1
% set rows corresponding to d_{2j}=0 to zero
Q(D==0,:) = 0;
Delta(D==0) = 0;

% compute D * sum(I \kron t(1,g_{it}) * inv(Sigma) * (y - BX)); 
dzsr = zeros(2*r,n);
dzsr(1:2:2*r-1,:) = squeeze( sum(BX, 1,'omitnan') ).* repmat(D(1:2:end).* Sigma_inv,[1,n]); % r x n
dzsr(2:2:2*r,:) = squeeze( sum( permute(repmat(age,[1,1,r]),[2,3,1]).* BX, 1,'omitnan') ).* ...
    repmat(D(2:2:end).* Sigma_inv,[1,n]); % r x n

% intermediate variables
S = zeros(2*r,2*r); % S = sum(M_i)/n
gM1 = zeros(r,1); % sum( T_i * M_i(2j-1,2j-1) )
gM2 = zeros(r,1); % sum( M_i(2j-1,2j) * sum(g_{it}) )
gM3 = zeros(r,1); % sum( M_i(2j,2j) * sum(g_{it}^2) )

% compute Omega_i, m_i and initial marginal log-likelihood --------------
meta = zeros(2*r,n); % (m_1,...,m_n)
dev(1) = (1 - 2*n/N)* log_det_Sigma + n/N * sum(log(Sigma(ind_y))) + ...
    (r - sum(ind_y))/N * sum(log(det_A)) + sum(ind_y)/N * sum(log(ntps)) + ...
    sum(resid2.* Sigma_inv)/N;

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
    
    % compute Omega_i ---------------------
    M_i = zeros(2*r,2*r);
    
    % Delta - Delta * D * Ci * D * Delta
    M_i(1:2*2*r+2:end) = Delta(1:2:end) - Delta(1:2:end).^2.* Ci1;
    M_i(2*r+2:2*2*r+2:end) = Delta(2:2:end) - Delta(2:2:end).^2.* Ci3;
    M_i(2*r+1:2*2*r+2:end) = - Delta(1:2:end).* Delta(2:2:end).* Ci2;
    M_i(2:2*2*r+2:end) = - Delta(1:2:end).* Delta(2:2:end).* Ci2;
    
    % add other terms
    M_i = M_i + H * ( F_i * H'- Q') + Q * ( Q'- (repmat(Delta,[1,K]).* tmp)');
    
    % M_i = Omega_i + m_i * t(m_i) ----
    M_i = M_i + meta(:,i) * meta(:,i)';
    
    S = S + M_i;
    
    gM1 = gM1 + M_i(1:2*2*r+2:end)'.* ntps(i); % r x 1
    gM2 = gM2 + M_i(2*r+1:2*2*r+2:end)'.* sum_age(i); % r x 1
    gM3 = gM3 + M_i(2*r+2:2*2*r+2:end)'.* sum_age2(i); % r x 1
    
    dev(1) = dev(1) + sum(log(det_Ci_inv))/N + log_det_F_inv/N - ...
        sum(dzsr(:,i).* meta(:,i))/N;
end
clear M_i
S = S/n;

if isnan(dev(1))
    error('Initial log-likelihood is NaN.')
end

%%
for iter = 2:maxit
    % store parameter values at previous step -----------------------------
    rho_old = rho; % (1+p+q) x r
    phi_old = phi; % (1+p) x r
    D_old = D; % 2r x 1
    Sigma_old = Sigma; % r x 1
    Q_old = Q; % 2r x K
    
    % M step --------------------------------------------------------------
    % update Q and Delta --------------------------------------
    % remove zero rows and columns from S
    S = S(D~=0,D~=0); % (2r - sum(ind_y)) x (2r - sum(ind_y))
    % remove entries of Q and Delta corresponding to d_{2j}=0
    Q = Q(D~=0,:); % (2r-sum(ind_y)) x K
    Delta = Delta(D~=0); % (2r - sum(ind_y)) x 1
    
    [Q,Delta,ntry_Q(iter),flag_Q] = updateQ(Q,Delta,S,ss,maxQ,maxit,tol);
    
    if ~flag_Q % Q does not converge in PGD
        return
    end
    
    clear S
    
    % add zero rows to Q and zero entries to Delta corresponding to d_{2j}=0
    tmp = Q;
    Q = zeros(2*r,K);
    Q(D~=0,:) = tmp;
    
    tmp = Delta;
    Delta = zeros(2*r,1);
    Delta(D~=0) = tmp;
    
    % update B,D and Sigma ---------------------------------------
    flag_BDS = 0; 
    
    for itry = 1:maxit
        % store parameter values at previous inner step ---------
        rho_old_in = rho;  % (1+p+q) x r
        phi_old_in = phi; % (1+p) x r 
        D_old_in = D; % 2r x 1
        Sigma_old_in = Sigma; % r x 1
        
        % update B ----------------------------------------------
        % compute zeta
        tmp = repmat(D,[1,n]).* meta; % 2r x n
        
        % H = Z * zeta
        H = mmx('mult',Z,reshape(tmp,[2,r,n]) ); % max_tps x r x n
        
        % update rho -----------------------------------
        % compute X2 * phi
        BX = mmx('mult',X2,phi); % max_tps x r x n
        % compute (y - X2 * phi - H)
        BX = (y - BX - H);  % max_tps x r x n
        % set NaN to 0
        BX(isnan(BX)) = 0;
        
        rho = sum_XX1_inv * sum(mmx('mult',X1,BX,'tn'),3); % (1+p+q) x r
        
        % update phi -----------------------------------
        % compute X1 * rho
        BX = mmx('mult',X1,rho); % max_tps x r x n     
        
        [phi,flag_phi] = BICtune_phi(nlambda1,lambda1_min_ratio,n,N,r,K,y,ntps,age,X2,...
            BX,H,Sigma,D,Q,Delta,sum_age,sum_age2,det_A,sum_XX2_inv,Sigma_inv,...
            log_det_Sigma,maxit,tol);   
        
        if ~flag_phi
            return
        end
               
        % update D ----------------------------------------------
        % update BX
        BX = BX + mmx('mult',X2,phi); % max_tps x r x n
        BX = y - BX; % max_tps x r x n
        
        % update sum of squared residuals
        resid2 = squeeze( sum(sum(BX.^2,1,'omitnan'),3) )'; % r x 1
        
        % compute sum(I \kron t(1,g_{it}) * (y - BX))
        dzsr(1:2:2*r-1,:) = squeeze(sum(BX,1,'omitnan')); % r x n
        dzsr(2:2:2*r,:) = squeeze(sum(permute(repmat(age,[1,1,r]),[2,3,1]).* BX,1,'omitnan')); % r x n
        
        % compute sum(m_i * (I \kron t(1,g_{it})) * (y - BX))
        H = sum(meta.* dzsr,2); % 2r x 1
        
        % update d_{2j-1} ------------------------
        D(1:2:2*r-1) = ( H(1:2:end) - D(2:2:end).* gM2 )./ gM1; % r x 1
        
        % update d_{2j} --------------------------            
        if sum(~ind_y)>0
            D(2:2:2*r) = BICtune_D(nlambda2,lambda2_min_ratio,n,N,r,K,ntps,age,...
                BX,Sigma,D,Q,Delta,sum_age,sum_age2,det_A,resid2,Sigma_inv,...
                log_det_Sigma,ind_y,gM2,gM3,H(2:2:end));
        end
               
        % update Sigma ----------------------------------------------------               
        Sigma = ( D(1:2:end).^2.* gM1 + 2 * D(1:2:end).* D(2:2:end).* gM2 + ...
            D(2:2:end).^2.* gM3 - 2 * D(1:2:end).* H(1:2:end) - ...
            2 * D(2:2:end).* H(2:2:end) + resid2 )/N; % r x 1
        
        % compute sum(log(Sigma)) and Sigma_inv
        tmp = log(Sigma); % r x 1
        log_det_Sigma = sum(tmp);
        Sigma_inv = exp(-tmp); % 1./Sigma, r x 1        
        
        % stopping rule ---------------------------------------------------
        % Frobenius norm 
        beta_diff = sqrt( ...
            sum( (rho(:) - rho_old_in(:)).^2 ) + sum( (phi(:) - phi_old_in(:)).^2 ) );
        D_diff = sqrt( sum( (D - D_old_in).^2 ) );
        Sigma_diff = sqrt( sum( (Sigma - Sigma_old_in).^2 ) );
        
        % relative change of parameter estimates
        if ( max( [beta_diff / sqrt( sum(rho_old_in(:).^2) + sum(phi_old_in(:).^2) ),...
                D_diff / sqrt( sum(D_old_in.^2) ),...
                Sigma_diff / sqrt( sum(Sigma_old_in.^2) )] ) < tol )
            flag_BDS = 1;
            break
        end
    end
    ntry_BD(iter) = itry;
    
    if ~flag_BDS
        return
    end
    
    % E step --------------------------------------------------------------
    % update indicator of outcomes with d_{2j}=0
    ind_y = (D(2:2:2*r) == 0); % r x 1
    % indicator of d_{2j-1} with d_{2j}=0
    ind_D1 = find(D==0) - 1; % sum(ind_y) x 1
    % set rows corresponding to d_{2j}=0 to zero
    Q(D==0,:) = 0;
    Delta(D==0) = 0;
    
    % compute D * sum(I \kron t(1,g_{it}) * inv(Sigma) * (y - BX))
    dzsr(1:2:2*r-1,:) = repmat(D(1:2:end).* Sigma_inv,[1,n]).* dzsr(1:2:end,:);
    dzsr(2:2:2*r,:) = repmat(D(2:2:end).* Sigma_inv,[1,n]).* dzsr(2:2:end,:);
    
    % update intermediate variables
    S = zeros(2*r,2*r); % S = sum(M_i)/n
    gM1 = zeros(r,1); % sum( T_i * M_i(2j-1,2j-1) )
    gM2 = zeros(r,1); % sum( M_i(2j-1,2j) * sum(g_{it}) )
    gM3 = zeros(r,1); % sum( M_i(2j,2j) * sum(g_{it}^2) )
    
    % compute deviance = -2/N * marginal log-likelihood and update Omega_i and m_i
    dev(iter) = (1 - 2*n/N)* log_det_Sigma + n/N * sum(log(Sigma(ind_y))) + ...
        (r - sum(ind_y))/N * sum(log(det_A)) + sum(ind_y)/N * sum(log(ntps)) + ...
        sum(resid2.* Sigma_inv)/N;
   
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
        
        % compute Omega_i ---------------------
        M_i = zeros(2*r,2*r);
        
        % Delta - Delta * D * Ci * D * Delta
        M_i(1:2*2*r+2:end) = Delta(1:2:end) - Delta(1:2:end).^2.* Ci1;
        M_i(2*r+2:2*2*r+2:end) = Delta(2:2:end) - Delta(2:2:end).^2.* Ci3;
        M_i(2*r+1:2*2*r+2:end) = - Delta(1:2:end).* Delta(2:2:end).* Ci2;
        M_i(2:2*2*r+2:end) = - Delta(1:2:end).* Delta(2:2:end).* Ci2;
        
        % add other terms
        M_i = M_i + H * ( F_i * H'- Q') + Q * ( Q'- (repmat(Delta,[1,K]).* tmp)');
        
        % M_i = Omega_i + m_i * t(m_i) ----
        M_i = M_i + meta(:,i) * meta(:,i)';
        
        S = S + M_i;
        
        gM1 = gM1 + M_i(1:2*2*r+2:end)'.* ntps(i); % r x 1
        gM2 = gM2 + M_i(2*r+1:2*2*r+2:end)'.* sum_age(i); % r x 1
        gM3 = gM3 + M_i(2*r+2:2*2*r+2:end)'.* sum_age2(i); % r x 1
        
        dev(iter) = dev(iter) + sum(log(det_Ci_inv))/N + log_det_F_inv/N - ...
            sum(dzsr(:,i).* meta(:,i))/N;
    end
    clear M_i
    S = S/n;
    
    % stopping rule -------------------------------------------------------
    disp(iter)
    if isnan(dev(iter))
        error('Log-likelihood becomes NaN.')
    end
    
    % Frobenius norm
    beta_diff = sqrt( ...
        sum( (rho(:) - rho_old(:)).^2 ) + sum( (phi(:) - phi_old(:)).^2 ) );
    D_diff = sqrt( sum( (D - D_old).^2 ) );
    Sigma_diff = sqrt( sum( (Sigma - Sigma_old).^2 ) );
    Q_diff = sqrt( sum( (Q(:) - Q_old(:)).^2 ) );
    
    % relative change of parameter estimates
    if ( max( [beta_diff / sqrt( sum(rho_old(:).^2) + sum(phi_old(:).^2) ),...
            D_diff / sqrt( sum(D_old.^2) ),...
            Sigma_diff / sqrt( sum(Sigma_old.^2) ),...
            Q_diff / sqrt( sum(Q_old(:).^2) )] ) < tol )
        flag = 1;
        break
    end
end

dev = dev(1:iter);
ntry_BD = ntry_BD(2:iter);
ntry_Q = ntry_Q(2:iter);

mu0 = rho(1,:); % 1 x r
alpha0 = rho(2:p+1,:); % p x r
gamma = rho(p+2:end,:); % q x r
mu1 = phi(1,:); % 1 x r
alpha1 = phi(2:end,:); % p x r

% transform to original scales
alpha0 = alpha0./ repmat(std_u',[1,r]);
alpha1 = alpha1./ repmat(std_uage',[1,r]);
gamma = gamma./ repmat(std_w',[1,r]);
mu0 = mu0 - mean_u * alpha0 - mean_uage * alpha1 - mean_w * gamma;
