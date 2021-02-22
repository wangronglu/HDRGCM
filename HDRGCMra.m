function [mu0,alpha0,mu1,alpha1,gamma,Sigma,D,Q,meta,LF,flag,dev] = ...
    HDRGCMra(y,ntps,age,u_pred,w_pred,K,lambda1,lambda2,maxit,tol,...
    mu0,alpha0,mu1,alpha1,gamma,Sigma,D,Q)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate reparameterized HDRGCM with adaptive L1-regularization on mu1,alpha1 and D.
% R = Q*t(Q) + I - diag(Q*t(Q)) is a correlation matrix. 
% G = diag(D) * R * diag(D) is the covariance matrix of random effects.
% Covariates will be standardized so that each column mean is 0 and each column
% norm is sqrt(N) in the design matrix.
% Estimated coefficients will be transformed to original scales except for age.
% use gradient descent to update Q 
% require Loss_Q_func.m
%
% Input:
%   y: max_tps x r x n array of continuous responses 
%   ntps: n x 1 vector of number of time points for each subject ( ntps(i)>=3 )
%   age: n x max_tps matrix, age(i,1:ntps(i)) contains the age for each time point of subject i 
%   u_pred: n x p matrix of time-invariant (demeaned) covariates or []
%   w_pred: n x max_tps x q array of time-varying (demeaned) covariates or []
%   K: dimension of latent features 
%   lambda1: penalty factor for fixed effects
%   lambda2: penalty factor for variances of random slopes
%   maxit: maximum iterations, e.g. 1000
%   tol: threshold of relative change in objective function,e.g. tol=1e-5.
%   mu0: 1 x r vector of initial fixed intercepts 
%   alpha0: p x r matrix of initial coefficients for u_pred
%   mu1: 1 x r vector of initial fixed slopes for standardized age
%   alpha1: p x r matrix of initial coefficients for interaction terms
%               of u_pred and standardized age 
%   gamma: q x r matrix of initial coefficients for w_pred 
%   Sigma: r x 1 vector of initial variances; entries should all be positive 
%   D: 2r x 1 vector of initial D; zero entries indicate no random effects
%   Q: 2r x K matrix of initial Q; do not use zero matrix; row norms < 1
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
%   LF: loss function values over iterations
%   flag: indicator of convergence; 1: converge, 0: not converge.
%   dev: -2/N * marginal log-likelihood

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

% parameters in backtracking line search
maxtry = 10; % maximum # trials 
ss0 = 0.1; % initial step size
a = 0.1; % scale parameter of gradient
b = 0.1; % shrinkage parameter of step size

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

% design matrix of fixed effects
X1 = zeros(max_tps,1+p+q,n); 
sum_XX1_inv = zeros(1+p+q,1+p+q);
X2 = zeros(max_tps,1+p,n);

for i = 1:n
    % X1 = (1,u_i,w_{it})
    tmp = [repmat([1,u_pred(i,:)],[ntps(i),1]), reshape(w_pred(i,1:ntps(i),:),[ntps(i),q])];
    X1(1:ntps(i),:,i) = tmp;
    sum_XX1_inv = sum_XX1_inv + tmp'* tmp;
    
    % X2 = (g_{it},u_i*g_{it})
    X2(1:ntps(i),:,i) = [age(i,1:ntps(i))',reshape(uage_pred(i,1:ntps(i),:),[ntps(i),p])];
end

clear u_pred w_pred uage_pred
sum_XX1_inv = sum_XX1_inv \ eye(1+p+q); % compute inverse

%% process initial values
mu0 = mu0 + mean_u * alpha0 + mean_uage * alpha1 + mean_w * gamma;
alpha0 = alpha0.* repmat(std_u',[1,r]);
alpha1 = alpha1.* repmat(std_uage',[1,r]);
gamma = gamma.* repmat(std_w',[1,r]);

if (sum(Sigma <= 0) > 0)
    error('Initial entries of Sigma must all be positive.')
end

% compute sum(log(Sigma)) and Sigma_inv
tmp = log(Sigma); % r x 1
log_det_Sigma = sum(tmp);
Sigma_inv = exp(-tmp); % 1./Sigma, r x 1

% indicator of outcomes with d_{2j}=0
ind_y = (D(2:2:2*r) == 0); % r x 1
% indicator of d_{2j-1} with d_{2j}=0
ind_D1 = find(D==0) - 1; % sum(ind_y) x 1

Delta = 1 - sum(Q.^2,2); % 2r x 1
if (sum(Delta <= 0) > 0)
    error('Row norms of initial Q must all be < 1.')
end

%% compute initial marginal log-likelihood
% compute BX
rho = [mu0; alpha0; gamma]; % (1+p+q) x r
phi = [mu1; alpha1]; % (1+p) x r
BX = mmx('mult',X1,rho) + mmx('mult',X2,phi); % max_tps x r x n

% compute sum of squared residuals
resid2 = squeeze( sum(sum((y - BX).^2,1,'omitnan'),3) )'; % r x 1
    
% compute Omega_i and m_i --------------------------------------------
meta = zeros(2*r,n); % (m_1,...,m_n)

% compute D * sum(I \kron t(1,g_{it}) * inv(Sigma) * (y - BX)); 
dzsr = zeros(2*r,n);
dzsr(1:2:2*r-1,:) = squeeze( sum(y - BX, 1,'omitnan') ).* ...
    repmat(D(1:2:end).* Sigma_inv,[1,n]); % r x n
dzsr(2:2:2*r,:) = squeeze( sum( permute(repmat(age,[1,1,r]),[2,3,1]).* (y - BX), 1,'omitnan') ).* ...
    repmat(D(2:2:end).* Sigma_inv,[1,n]); % r x n

% intermediate variables
S = zeros(2*r,2*r); % S = sum(M_i)/n
gM1 = zeros(r,1); % sum( T_i * M_i(2j-1,2j-1) )
gM2 = zeros(r,1); % sum( M_i(2j-1,2j) * sum(g_{it}) )
gM3 = zeros(r,1); % sum( M_i(2j,2j) * sum(g_{it}^2) )

% compute initial marginal log-likelihood
LF(1) = (1 - 2*n/N)* log_det_Sigma + n/N * sum(log(Sigma(ind_y))) + ...
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
    clear M_i
    
    LF(1) = LF(1) + sum(log(det_Ci_inv))/N + log_det_F_inv/N - ...
        sum(dzsr(:,i).* meta(:,i))/N;
end
S = S/n;

% record deviance
dev = LF(1);

% weighted penalty factor
lambda1 = lambda1./abs(phi); % (1+p) x r
lambda2 = lambda2./abs(D(2:2:2*r)); % r x 1
% add penalty terms
LF(1) = LF(1) + sum(sum(lambda1.* abs(phi))) + sum(lambda2.* abs(D(2:2:2*r)));

if isnan(LF(1))
    LF = LF(1);
    
    % transform to original scales
    alpha0 = alpha0./ repmat(std_u',[1,r]);
    alpha1 = alpha1./ repmat(std_uage',[1,r]);
    gamma = gamma./ repmat(std_w',[1,r]);
    mu0 = mu0 - mean_u * alpha0 - mean_uage * alpha1 - mean_w * gamma;
    return
end

%%
for iter = 2:maxit   
    % update B -----------------------------------------------------------
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
    
    % compute (y - X1 * rho - H)
    H = y - BX - H; % max_tps x r x n
    % set NaN to 0
    H(isnan(H)) = 0;
    
    for k=1:p+1
        ind = ones(1,p+1);
        ind(k) = 0;
        ind = logical(ind);
        
        % compute (y - X1 * rho - H - X2(:,-k) * phi(-k,:))
        tmp = H - mmx('mult',X2(:,ind,:),phi(ind,:)); % max_tps x r x n
        
        % compute c_k for phi
        tmp = sum(mmx('mult',permute(X2(:,k,:),[2,1,3]),tmp),3); % 1 x r
        
        phi(k,:) = sign(tmp).* max( abs(tmp) - lambda1(k,:).* N/2.* Sigma', 0)./N;        
    end
    
    % update D ---------------------------------------------------------- 
    % update BX
    BX = BX + mmx('mult',X2,phi); % max_tps x r x n
    
    % compute sum(I \kron t(1,g_{it}) * (y - BX))
    dzsr(1:2:2*r-1,:) = squeeze(sum(y - BX,1,'omitnan')); % r x n
    dzsr(2:2:2*r,:) = squeeze(sum(permute(repmat(age,[1,1,r]),[2,3,1]).* (y - BX),1,'omitnan')); % r x n
    
    % compute sum(m_i * (I \kron t(1,g_{it})) * (y - BX))
    H = sum(meta.* dzsr,2); % 2r x 1
        
    % update d_{2j-1} ----
    D(1:2:2*r-1) = ( H(1:2:end) - D(2:2:end).* gM2 )./ gM1; % r x 1
    
    % update d_{2j} ----    
    tmp = H(2:2:end) - D(1:2:end).* gM2; % r x 1    
    tmp(~ind_y) = sign(tmp(~ind_y)).* max( abs(tmp(~ind_y)) - lambda2(~ind_y).* N/2.* Sigma(~ind_y), 0)./ gM3(~ind_y);
    D(2:2:2*r) = tmp; % r x 1
    
    % update indicator of outcomes with d_{2j}=0
    ind_y = (D(2:2:2*r) == 0); % r x 1    
    % indicator of d_{2j-1} with d_{2j}=0
    ind_D1 = find(D==0) - 1; % sum(ind_y) x 1
    
    % update Sigma -------------------------------------------------------    
    % update sum of squared residuals
    resid2 = squeeze( sum(sum((y - BX).^2,1,'omitnan'),3) )'; % r x 1
        
    Sigma = ( D(1:2:2*r-1).^2.* gM1 + 2 * D(1:2:2*r-1).* D(2:2:2*r).* gM2 + ...
        D(2:2:2*r).^2.* gM3 - 2 * D(1:2:end).* H(1:2:end) - ...
        2 * D(2:2:end).* H(2:2:end) + resid2 )/N; % r x 1
    
    % compute sum(log(Sigma)) and Sigma_inv
    tmp = log(Sigma); % r x 1
    log_det_Sigma = sum(tmp);
    Sigma_inv = exp(-tmp); % 1./Sigma, r x 1
    
    % update Q and Delta -------------------------------------------------
    % remove zero rows and columns from S
    S = S(D~=0,D~=0); % (2r - sum(ind_y)) x (2r - sum(ind_y))
    % remove entries of Q and Delta corresponding to d_{2j}=0
    Q = Q(D~=0,:); % (2r-sum(ind_y)) x K
    Delta = Delta(D~=0); % (2r - sum(ind_y)) x 1
    
    % compute current log|Q*t(Q)+Delta|+tr( inv(Q*t(Q)+Delta) * S )
    [Loss_Q_old,Lambda_inv] = Loss_Q_func(Q,S);
    
    % compute gradient of Q ------------------
    % compute inv(Delta) * Q
    Delta = 1./ Delta; % (2r - sum(ind_y)) x 1
    gM1 = repmat(Delta,[1,K]).* Q; % (2r-sum(ind_y)) x K
    
    % compute inv(Delta) * Q * Lambda_inv
    gM2 = gM1 * Lambda_inv; % (2r-sum(ind_y)) x K
    
    % compute S * inv(Delta) * Q
    gM3 = S * gM1; % (2r-sum(ind_y)) x K
    
    % compute ( inv(R) - inv(R) * S * inv(R) )* Q
    H = gM2 - repmat(Delta,[1,K]).* (gM3 * Lambda_inv) + gM2 * (gM3' * gM2); % (2r-sum(ind_y)) x K
    
    % compute diag( inv(R) - inv(R) * S * inv(R) )
    tmp = Delta - sum(gM2.* gM1,2) - Delta.^2.* diag(S) + 2 * sum(gM2.* gM3,2).* Delta ...
        - sum( (gM2 * (gM3' * gM1)).* gM2,2); % (2r - sum(ind_y)) x 1
    
    % compute gradient of Q
    H = 2* H - 2* repmat(tmp,[1,K]).* Q; % (2r-sum(ind_y)) x K
    
    % squared gradient norm
    grad_norm2 = sum(sum(H.^2));
    
    % backtracking line search -------------------
    if (grad_norm2 > 0)
        flag_Q = 0; % indicator of sufficient decrease in Loss_Q_new
        
        % initial step size
        ss = ss0;
        tmp = Q - ss.* H; % Q candidate
        itry = 0;
        
        while (itry < maxtry) && (sum( sum(tmp.^2,2) >= 1 ) > 0) % ensure row norms of Q < 1
            ss = ss * b;
            tmp = Q - ss.* H; % Q candidate
            itry = itry + 1;
        end
        
        while (itry < maxtry) && (flag_Q~=1)
            [Loss_Q_new,~] = Loss_Q_func(tmp,S);
            if ( Loss_Q_new < (Loss_Q_old - a * ss * grad_norm2) )
                flag_Q = 1;
                Q = tmp;
            else
                ss = ss * b;
                tmp = Q - ss.* H; % Q candidate
                itry = itry + 1;
            end
        end
    end
    
    % compute Delta
    Delta = 1 - sum(Q.^2,2); % (2r - sum(ind_y)) x 1
    
    % add zero rows to Q and zero entries to Delta corresponding to d_{2j}=0 
    tmp = Q;
    Q = zeros(2*r,K);
    Q(D~=0,:) = tmp;
    
    tmp = Delta;
    Delta = zeros(2*r,1);
    Delta(D~=0) = tmp;
    
    % update Omega_i and m_i --------------------------------------------     
    % compute D * sum(I \kron t(1,g_{it}) * inv(Sigma) * (y - BX))
    dzsr(1:2:2*r-1,:) = repmat(D(1:2:end).* Sigma_inv,[1,n]).* dzsr(1:2:end,:);
    dzsr(2:2:2*r,:) = repmat(D(2:2:end).* Sigma_inv,[1,n]).* dzsr(2:2:end,:);
    
    % update intermediate variables
    S = zeros(2*r,2*r); % S = sum(M_i)/n
    gM1 = zeros(r,1); % sum( T_i * M_i(2j-1,2j-1) )
    gM2 = zeros(r,1); % sum( M_i(2j-1,2j) * sum(g_{it}) )
    gM3 = zeros(r,1); % sum( M_i(2j,2j) * sum(g_{it}^2) )
    
    % compute marginal log-likelihood
    LF(iter) = (1 - 2*n/N)* log_det_Sigma + n/N * sum(log(Sigma(ind_y))) + ...
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
        clear M_i
                
        LF(iter) = LF(iter) + sum(log(det_Ci_inv))/N + log_det_F_inv/N - ... 
            sum(dzsr(:,i).* meta(:,i))/N;
    end
    S = S/n;
    
    % record deviance
    dev = LF(iter);
    
    % add penalty terms
    LF(iter) = LF(iter) + sum(sum(lambda1.* abs(phi))) + sum(lambda2.* abs(D(2:2:2*r)));
    
    % stopping rule -------------------------------------------------------
    disp(iter)
    if ( abs(LF(iter-1) - LF(iter)) < tol * abs(LF(iter-1)) )
        flag = 1;
        break
    end
    if isnan(LF(iter))
        break
    end      
end

LF = LF(1:iter);

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