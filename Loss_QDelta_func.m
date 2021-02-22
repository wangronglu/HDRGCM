function res = Loss_QDelta_func(Q,Delta,S)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute res = log|Q*t(Q)+Delta| + tr( inv(Q*t(Q)+Delta) * S )
%
% Input
%   Q: d x K matrix
%   Delta: d x 1 vector
%   S: d x d matrix
%
% Output
%   res: scalar

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = size(Q,2);

% compute inv(Delta) * Q
tmp = repmat(1./Delta,[1,K]).* Q;

% compute log-determinant and inverse of Lambda = I_K + t(Q) * inv(Delta) * Q
[Evec,Eval] = eig(Q' * tmp, 'vector'); % K x K
log_det_Lambda = sum(log(1 + Eval));
% inv(Lambda)
Lambda_inv = Evec * ( repmat(1./(1 + Eval),[1,K]).* Evec'); % K x K

% compute tr( inv(Lambda)*t(Q)*inv(Delta)*S*inv(Delta)*Q )
tmp = sum(sum(Lambda_inv.* (tmp'* S * tmp)));
% compute res
res = sum(log(Delta)) + log_det_Lambda + sum(diag(S)./Delta) - tmp;
