function [Q,Delta] = projectGD_Q(Q,Delta,S,ss,maxQ)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute one-step projected gradient descent of Q
% R = Q * Q' + I - diag(Q * Q')
% Delta = 1 - sum(Q.^2,2);
%
% Input
%   Q: d x K matrix of initial Q (row-wise norms < 1)
%   Delta: d x 1 vector of initial Delta
%   S: d x d matrix
%   ss: stepsize in gradient descent
%   maxQ: maximum projected row norm of Q
%
% Output
%   Q: d x K matrix of updated Q (row-wise norms < 1)
%   Delta: d x 1 vector of updated Delta 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = size(Q,2);

% compute inv(Delta) * Q
Delta = 1./ Delta; % d x 1
gM1 = repmat(Delta,[1,K]).* Q; % d x K

% compute inverse of Lambda = I_K + t(Q) * inv(Delta) * Q
[Evec,Eval] = eig(Q' * gM1, 'vector'); % K x K
Lambda_inv = Evec * ( repmat(1./(1 + Eval),[1,K]).* Evec'); % K x K

% compute inv(Delta) * Q * Lambda_inv
gM2 = gM1 * Lambda_inv; % d x K

% compute S * inv(Delta) * Q
gM3 = S * gM1; % d x K

% compute ( inv(R) - inv(R) * S * inv(R) )* Q
H = gM2 - repmat(Delta,[1,K]).* (gM3 * Lambda_inv) + gM2 * (gM3' * gM2); % d x K

% compute diag( inv(R) - inv(R) * S * inv(R) )
tmp = Delta - sum(gM2.* gM1,2) - Delta.^2.* diag(S) + 2 * sum(gM2.* gM3,2).* Delta ...
    - sum( (gM2 * (gM3' * gM1)).* gM2,2); % d x 1

% compute gradient of Q 
% H = 2[(inv(R)-inv(R)*S*inv(R))-diag(inv(R)-inv(R)*S*inv(R))] * Q; d x K matrix
H = 2* H - 2* repmat(tmp,[1,K]).* Q; % d x K

% gradient descent update of Q 
Q = Q - ss.* H; % Q candidate, d x K
    
% project onto the feasible set (row-wise norms <= maxQ)
H = sum(Q.^2,2); % squared row norms of Q, d x 1
Q(H>=1,:) = Q(H>=1,:)./repmat(sqrt(H(H>=1)),[1,K]).* maxQ;

% update Delta
H(H>=1) = maxQ * maxQ;
Delta = 1 - H; % d x 1