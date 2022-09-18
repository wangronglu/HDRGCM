function [Q,Delta,ntry] = updateQDelta(Q,Delta,S,K,r,maxit,tol)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% alternatively compute Q and Delta until minor update 
% G = Q * Q' + diag(Delta)
%
% Input
%   Q: 2r x K matrix of initial Q 
%   Delta: 2r x 1 vector of initial Delta
%   S: 2r x 2r matrix
%   K: number of columns in Q
%   r: number of outcomes
%   maxit: maximum iterations
%   tol: threshold of relative change in parameter estimate
%
% Output
%   Q: d x K matrix of updated Q 
%   Delta: d x 1 vector of updated Delta 
%   ntry: number of iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for itry = 1:maxit
    % store old parameter values 
    Q_old = Q; % 2r x K
    Delta_old = Delta; % 2r x 1
    
    % update Q --------------------------------------------------------
    % compute Delta^{-1/2}
    tmp = 1./sqrt(Delta); % 2r x 1
    
    % compute K eigenvectors of S* = Delta^{-1/2} * S * Delta^{-1/2}
    % corresponding to the K largest eigenvalues
    [Q,tmp] = eigs((tmp * tmp').* S,K);
    Q(:,Q(1,:)<0) = - Q(:,Q(1,:)<0); % require first entry positive for uniqueness
    tmp = diag(tmp) - ones(K,1); % K x 1
    
    % check if all eigenvalues are greater than 1
    if ( sum(tmp > 0) == K )
        % compute Q = sqrt(Delta) * U * sqrt(Lambda - I)
        Q = repmat(sqrt(Delta),[1,K]).* Q.* repmat(sqrt(tmp)',[2*r,1]);
    else
        error('Some eigenvalues of S* <1. Consider to use a smaller K.')
    end
    
    % update Delta ----------------------------------------------------
    % compute diag( S - Q * t(Q) )
    Delta = diag(S) - sum(Q.^2,2); % 2r x 1
    
    % check if entries of Delta are all positive
    if ( sum(Delta <= 0) > 0 )
        error('Some entries of Delta <= 0. Consider to use a smaller K.')
    end
    
    % stopping rule ---------------------------------------------------
    if ( sqrt(sum((Q(:) - Q_old(:)).^2)) /sqrt(sum(Q_old(:).^2)) < tol ) && ...
            ( sqrt(sum((Delta - Delta_old).^2)) / sqrt(sum(Delta_old.^2)) < tol )
        break
    end
end
ntry = itry;