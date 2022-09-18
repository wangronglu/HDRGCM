function [mu0,alpha0,mu1,alpha1,gamma,Sigma,Q,Delta,mzeta,LF,flag,ntry] = ...
    HDRGCM0(y,ntps,age,u_pred,w_pred,K,maxit,tol)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate HDRGCM model
% G = Q * Q' + diag(Delta) is the covariance matrix of random effects.
% Covariates will be standardized so that each column mean is 0 and each column
% norm is sqrt(N) in the design matrix.
% The estimated coefficients will be transformed to original scales except for age.
% Stop inner/outer iterations when relative change of parameter estimate < tol.
% Require updateQDelta.m
% Initialize mu0 at sample mean of y and other coefficients of B at zero.
% Initialize Sigma at sample variance of y.
% Initialize G at I.
%
% Input:
%   y: max_tps x r x n array of continuous responses 
%   ntps: n x 1 vector of number of time points for each subject ( ntps(i)>=3 )
%   age: n x max_tps matrix, age(i,1:ntps(i)) contains the age for each time point of subject i 
%   u_pred: n x p matrix of time-invariant covariates or []
%   w_pred: n x max_tps x q array of time-varying covariates or []
%   K: low rank in factor model 
%   maxit: maximum iterations, e.g. 1000
%   tol: threshold of relative change in parameter estimate, e.g. 0.001
%
% Output:
%   mu0: 1 x r vector of fixed intercepts
%   alpha0: p x r matrix of coefficients for u_pred
%   mu1: 1 x r vector of fixed slopes for standardized age
%   alpha1: p x r matrix of coefficients for interaction terms of u_pred and standardized age 
%   gamma: q x r matrix of coefficients for w_pred
%   Sigma: r x 1 vector of variances
%   Q: 2r x K matrix 
%   Delta: 2r x 1 vector
%   mzeta: 2r x n matrix of expected random effects zeta_i,i=1,...,n.
%   LF: -2/N * marginal log-likelihood over iterations
%   flag: indicator of convergence; 1: converge, 0: not converge.
%   ntry: number of inner iterations at each M step

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

LF = zeros(maxit,1); % loss function
flag = 0;
ntry = zeros(maxit,1); % number of inner iterations at each M step

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

% design matrix of fixed effects -----------
X = zeros(max_tps,2+2*p+q,n); 
sum_XX_inv = zeros(2+2*p+q,2+2*p+q);

for i = 1:n    
    % X = (1,u_i,w_{it},g_{it},u_i*g_{it})
    tmp = [repmat([1,u_pred(i,:)],[ntps(i),1]), reshape(w_pred(i,1:ntps(i),:),[ntps(i),q]),...
            age(i,1:ntps(i))',reshape(uage_pred(i,1:ntps(i),:),[ntps(i),p])]; % ntps(i) x (2+2p+q)
    X(1:ntps(i),:,i) = tmp;
    sum_XX_inv = sum_XX_inv + tmp'* tmp;
end
sum_XX_inv = sum_XX_inv \ eye(2+2*p+q); % compute inverse

clear u_pred w_pred uage_pred

%% Initialization
mu1 = zeros(1,r);
alpha0 = zeros(p,r);
alpha1 = zeros(p,r);
gamma = zeros(q,r);

% initialize mu_{0j} at sample mean of y_{ijt} ---------------
mu0 = squeeze(sum(sum(y,1,'omitnan'),3))/N; % 1 x r

beta = [mu0;alpha0;gamma;mu1;alpha1]; % (2+2p+q) x r

% initialize Sigma at sample variance --------------------
BX = repmat(mu0,[max_tps,1,n]); % max_tps x r x n
BX = y - BX; % max_tps x r x n
resid2 = squeeze(sum(sum(BX.^2, 1,'omitnan'),3))'; % r x 1
Sigma = resid2 /(N-1); % r x 1

% compute sum(log(Sigma)) and Sigma_inv
tmp = log(Sigma); % r x 1
log_det_Sigma = sum(tmp);
Sigma_inv = exp(-tmp); % 1./Sigma, r x 1

% initialize G at I -------------------
Delta = ones(2*r,1);
Q = zeros(2*r,K);

%% compute initial marginal log-likelihood

% compute sum(I \kron t(1,g_{it}) * inv(Sigma) * (y - BX))
zsr = zeros(2*r,n);
zsr(1:2:2*r-1,:) = squeeze( sum(BX, 1,'omitnan') ).* repmat(Sigma_inv,[1,n]); % r x n
zsr(2:2:2*r,:) = squeeze( sum( permute(repmat(age,[1,1,r]),[2,3,1]).* BX, 1,'omitnan') ).* ...
                    repmat(Sigma_inv,[1,n]); % r x n

% intermediate variables
S = zeros(2*r,2*r); % S = sum(M_i)/n
gM1 = zeros(r,1); % sum( T_i * M_i(2j-1,2j-1) )
gM2 = zeros(r,1); % sum( M_i(2j-1,2j) * sum(g_{it}) )
gM3 = zeros(r,1); % sum( M_i(2j,2j) * sum(g_{it}^2) )

% compute Omega_i, m_i and marginal log-likelihood * (-2/N) ----------
mzeta = zeros(2*r,n); % (m_1,...,m_n)
LF(1) = (1 - 2*n/N)* log_det_Sigma + r/N * sum(log(det_A)) + sum(resid2.* Sigma_inv)/N;

for i = 1:n
    % compute C_i
    Ci1 = Sigma.* ntps(i)/det_A(i) + Delta(2:2:2*r); % r x 1
    Ci2 = Sigma.* sum_age(i)/det_A(i); % r x 1
    Ci3 = Sigma.* sum_age2(i)/det_A(i) + Delta(1:2:2*r-1); % r x 1
    det_Ci_inv = Ci1.* Ci3 - Ci2.^2; % r x 1
    Ci1 = Ci1./det_Ci_inv;
    Ci2 = Ci2./det_Ci_inv;
    Ci3 = Ci3./det_Ci_inv;
    
    % compute m_i -------------------------
    % (Delta - Delta * Ci * Delta) * zsr
    mzeta(1:2:2*r-1,i) = ( Delta(1:2:end) - Delta(1:2:end).^2.* Ci1 ).* zsr(1:2:end,i) - ...
        Delta(1:2:end).* Delta(2:2:end).* Ci2.* zsr(2:2:end,i); % r x 1
    mzeta(2:2:2*r,i) = ( Delta(2:2:end) - Delta(2:2:end).^2.* Ci3 ).* zsr(2:2:end,i) - ...
        Delta(1:2:end).* Delta(2:2:end).* Ci2.* zsr(1:2:end,i); % r x 1
    
    % compute Omega_i ---------------------
    M_i = zeros(2*r,2*r);
    
    % Delta - Delta * Ci * Delta
    M_i(1:2*2*r+2:end) = Delta(1:2:end) - Delta(1:2:end).^2.* Ci1;
    M_i(2*r+2:2*2*r+2:end) = Delta(2:2:end) - Delta(2:2:end).^2.* Ci3;
    M_i(2*r+1:2*2*r+2:end) = - Delta(1:2:end).* Delta(2:2:end).* Ci2;
    M_i(2:2*2*r+2:end) = - Delta(1:2:end).* Delta(2:2:end).* Ci2;
    
    % M_i = Omega_i + m_i * t(m_i) ----
    M_i = M_i + mzeta(:,i) * mzeta(:,i)';
    
    S = S + M_i;
    gM1 = gM1 + M_i(1:2*2*r+2:end)'.* ntps(i); % r x 1
    gM2 = gM2 + M_i(2*r+1:2*2*r+2:end)'.* sum_age(i); % r x 1
    gM3 = gM3 + M_i(2*r+2:2*2*r+2:end)'.* sum_age2(i); % r x 1
    
    LF(1) = LF(1) + sum(log(det_Ci_inv))/N - sum(zsr(:,i).* mzeta(:,i))/N;
end
clear M_i
S = S/n;
    
if isnan(LF(1))   
    error('Initial log-likelihood is NaN.')
end

%%
for iter = 2:maxit
    % store parameter values at previous outer step -----------------------
    beta_old = beta;  % (2+2p+q) x r
    Sigma_old = Sigma; % r x 1
    Q_old = Q; % 2r x K
    Delta_old = Delta; % 2r x 1
    
    %% M step
    % update B -----------------------------------------------------------
    % H = Z * zeta
    H = mmx('mult',Z,reshape(mzeta,[2,r,n]) ); % max_tps x r x n
        
    % update beta -----------------------------------
    % compute (y - H)
    BX = (y - H);  % max_tps x r x n
    % set NaN to 0
    BX(isnan(BX)) = 0;
    beta = sum_XX_inv * sum(mmx('mult',X,BX,'tn'),3); % (2+2p+q) x r    
        
    % update Sigma ------------------------------------------------------- 
    % update BX
    BX = mmx('mult',X,beta); % max_tps x r x n
    BX = y - BX; % max_tps x r x n
    
    % compute sum( H.*(y - BX) )
    H = squeeze(sum(sum(H.* BX, 1,'omitnan'),3))'; % r x 1
           
    % update sum of squared residuals
    resid2 = squeeze(sum(sum(BX.^2, 1,'omitnan'),3))'; % r x 1
    
    Sigma = ( resid2 - 2 * H + gM1 + 2 * gM2 + gM3 )/N; % r x 1
    
    % compute sum(log(Sigma)) and Sigma_inv
    tmp = log(Sigma); % r x 1
    log_det_Sigma = sum(tmp);
    Sigma_inv = exp(-tmp); % 1./Sigma, r x 1
    
    % update Q and Delta --------------------------------------------------
    [Q,Delta,ntry(iter)] = updateQDelta(Q,Delta,S,K,r,maxit,tol);
    
    %% E step
    % update Omega_i and m_i --------------------------------------------
    % compute sum(I \kron t(1,g_{it}) * inv(Sigma) * (y - BX))
    zsr(1:2:2*r-1,:) = squeeze( sum(BX, 1,'omitnan') ).* repmat(Sigma_inv,[1,n]); % r x n
    zsr(2:2:2*r,:) = squeeze( sum( permute(repmat(age,[1,1,r]),[2,3,1]).* BX, 1,'omitnan') ).* ...
        repmat(Sigma_inv,[1,n]); % r x n
    
    % update intermediate variables
    S = zeros(2*r,2*r); % S = sum(M_i)/n
    gM1 = zeros(r,1); % sum( T_i * M_i(2j-1,2j-1) )
    gM2 = zeros(r,1); % sum( M_i(2j-1,2j) * sum(g_{it}) )
    gM3 = zeros(r,1); % sum( M_i(2j,2j) * sum(g_{it}^2) )
    
    % compute marginal log-likelihood and update Omega_i and m_i --------
    LF(iter) = (1 - 2*n/N)* log_det_Sigma + r/N * sum(log(det_A)) + ...
        sum(resid2.* Sigma_inv)/N;
    
    for i = 1:n
        % compute C_i
        Ci1 = Sigma.* ntps(i)/det_A(i) + Delta(2:2:2*r); % r x 1
        Ci2 = Sigma.* sum_age(i)/det_A(i); % r x 1
        Ci3 = Sigma.* sum_age2(i)/det_A(i) + Delta(1:2:2*r-1); % r x 1
        det_Ci_inv = Ci1.* Ci3 - Ci2.^2; % r x 1
        Ci1 = Ci1./det_Ci_inv;
        Ci2 = Ci2./det_Ci_inv;
        Ci3 = Ci3./det_Ci_inv;
        
        % compute C_i * Q
        tmp = zeros(2*r,K);
        tmp(1:2:2*r-1,:) = repmat(Ci1,[1,K]).* Q(1:2:end,:) + ...
                           repmat(Ci2,[1,K]).* Q(2:2:end,:);
        tmp(2:2:2*r,:) = repmat(Ci2,[1,K]).* Q(1:2:end,:) + ...
                         repmat(Ci3,[1,K]).* Q(2:2:end,:);
                     
        % compute G * Ci * Q
        H = Q * (Q'* tmp) + repmat(Delta,[1,K]).* tmp; % 2r x K
        
        % compute F_i ----
        [Evec,Eval] = eig(Q' * tmp, 'vector'); % K x K
        log_det_F_inv = sum(log(1 + Eval));
        F_i = Evec * ( repmat(1./(1 + Eval),[1,K]).* Evec'); % K x K
        
        % compute m_i -------------------------
        % (Delta - Delta * Ci * Delta) * zsr
        mzeta(1:2:2*r-1,i) = ( Delta(1:2:end) - Delta(1:2:end).^2.* Ci1 ).* zsr(1:2:end,i) - ...
            Delta(1:2:end).* Delta(2:2:end).* Ci2.* zsr(2:2:end,i); % r x 1
        mzeta(2:2:2*r,i) = ( Delta(2:2:end) - Delta(2:2:end).^2.* Ci3 ).* zsr(2:2:end,i) - ...
            Delta(1:2:end).* Delta(2:2:end).* Ci2.* zsr(1:2:end,i); % r x 1
        
        % add other terms
        mzeta(:,i) = mzeta(:,i) + H * ( F_i * (H'* zsr(:,i)) ) + ...
            (Q - H)*(Q'* zsr(:,i)) - Q * ( tmp' * (Delta.* zsr(:,i)) );
        
        % compute Omega_i ---------------------
        M_i = zeros(2*r,2*r);
        
        % Delta - Delta * Ci * Delta
        M_i(1:2*2*r+2:end) = Delta(1:2:end) - Delta(1:2:end).^2.* Ci1;         
        M_i(2*r+2:2*2*r+2:end) = Delta(2:2:end) - Delta(2:2:end).^2.* Ci3;
        M_i(2*r+1:2*2*r+2:end) = - Delta(1:2:end).* Delta(2:2:end).* Ci2;       
        M_i(2:2*2*r+2:end) = - Delta(1:2:end).* Delta(2:2:end).* Ci2;
        
        % add other terms
        M_i = M_i + H * ( F_i * H'- Q') + Q * ( Q'- (repmat(Delta,[1,K]).* tmp)');
        
        % M_i = Omega_i + m_i * t(m_i) ----
        M_i = M_i + mzeta(:,i) * mzeta(:,i)';
        
        S = S + M_i;
        gM1 = gM1 + M_i(1:2*2*r+2:end)'.* ntps(i); % r x 1
        gM2 = gM2 + M_i(2*r+1:2*2*r+2:end)'.* sum_age(i); % r x 1
        gM3 = gM3 + M_i(2*r+2:2*2*r+2:end)'.* sum_age2(i); % r x 1        
                
        LF(iter) = LF(iter) + sum(log(det_Ci_inv))/N + log_det_F_inv/N - ... 
            sum(zsr(:,i).* mzeta(:,i))/N;
    end     
    clear M_i
    S = S/n;
    
    % stopping rule -------------------------------------------------------
    disp(iter)    
    if isnan(LF(iter))
        error('Log-likelihood becomes NaN.')
    end  
    
    % Frobenius norm of (beta - beta_old)
    beta_diff = sqrt(sum((beta(:) - beta_old(:)).^2 )); 
    Sigma_diff = sqrt(sum((Sigma - Sigma_old).^2));
    Q_diff = sqrt(sum((Q(:) - Q_old(:)).^2)); 
    Delta_diff = sqrt(sum((Delta - Delta_old).^2));
    
    % relative change of parameter estimates
    if ( max([beta_diff/sqrt(sum(beta_old(:).^2)),...
            Sigma_diff/sqrt(sum(Sigma_old.^2)),...
            Q_diff/sqrt(sum(Q_old(:).^2)),...
            Delta_diff/sqrt(sum(Delta_old.^2))]) < tol )
        flag = 1;
        break
    end          
end

LF = LF(1:iter);
ntry = ntry(2:iter);

mu0 = beta(1,:); % 1 x r
alpha0 = beta(2:p+1,:); % p x r
gamma = beta(p+2:1+p+q,:); % q x r
mu1 = beta(2+p+q,:); % 1 x r
alpha1 = beta(3+p+q:end,:); % p x r

% transform to original scales
alpha0 = alpha0./ repmat(std_u',[1,r]);
alpha1 = alpha1./ repmat(std_uage',[1,r]);
gamma = gamma./ repmat(std_w',[1,r]);
mu0 = mu0 - mean_u * alpha0 - mean_uage * alpha1 - mean_w * gamma;