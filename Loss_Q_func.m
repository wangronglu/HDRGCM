function [res,Lambda_inv] = Loss_Q_func(Q,S)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute res = log|Q*t(Q)+Delta| + tr( inv(Q*t(Q)+Delta) * S )
% where Delta = I - diag(Q*t(Q))
%
% Input
%   Q: d x K matrix
%   S: d x d matrix
%
% Output
%   res: scalar
%   Lambda_inv: K x K matrix, inv( I_K + t(Q) * inv(Delta) * Q )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = size(Q,2);

% compute Delta
Delta = 1 - sum(Q.^2,2); % d x 1

% compute inv(Delta) * Q
tmp = repmat(1./Delta,[1,K]).* Q;

% compute log-determinant and inverse of Lambda = I_K + t(Q) * inv(Delta) * Q
[Evec,Eval] = eig(Q' * tmp, 'vector'); % K x K
log_det_Lambda = sum(log(1 + Eval));
% inv(Lambda)
Lambda_inv = Evec * ( repmat(1./(1 + Eval),[1,K]).* Evec'); % K x K

% compute tr( inv(Lambda)*t(Q)*inv(Delta)*S*inv(Delta)*Q )
tmp = sum(sum(Lambda_inv.* (tmp'* S * tmp)));
res = sum(log(Delta)) + log_det_Lambda + sum(diag(S)./Delta) - tmp;

