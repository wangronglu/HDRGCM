function [Q,Delta,ntry,flag_Q] = updateQ(Q,Delta,S,ss,maxQ,maxit,tol)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute Q and Delta by iterative projected gradient descent until minor update
% R = Q * Q' + I - diag(Q * Q')
% Delta = 1 - sum(Q.^2,2)
% require projectGD_Q.m
%
% Input
%   Q: d x K matrix of initial Q (row-wise norms < 1)
%   Delta: d x 1 vector of initial Delta
%   S: d x d matrix
%   ss: stepsize in gradient descent
%   maxQ: maximum projected row norm of Q
%   maxit: maximum iterations
%   tol: threshold of relative change in parameter estimate
%
% Output
%   Q: d x K matrix of updated Q (row-wise norms < 1)
%   Delta: d x 1 vector of updated Delta 
%   ntry: number of iterations
%   flag_Q: indicator of convergence (1 converge, 0 not)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
flag_Q = 0;

for itry = 1:maxit
    if ( sum(Q(:).^2)==0 ) % if Q is a zero matrix
        flag_Q = 1;
        break % Q remains zero matrix as gradient is zero
    else
        Q_old = Q; % d x K
        
        % do one-step projected gradient descent
        [Q,Delta] = projectGD_Q(Q_old,Delta,S,ss,maxQ);
        
        % check relative change of Q
        if (sqrt(sum((Q(:) - Q_old(:)).^2)) / sqrt(sum(Q_old(:).^2)) < tol)
            flag_Q = 1;
            break
        end
    end
end
ntry = itry;