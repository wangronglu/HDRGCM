function [B,flag_B] = solveB(N,y,X,B,Sigma,lambda,abs_Bbar,maxit,tol)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update columns of coefficient matrix B in parallel for standardized covariates (adaptive L1)
%
% Input
%   N: total number of observations, N = sum(T_i)
%   y: T x r x n array of outcomes (do not include NaN values)
%   X: T x p x n array of standardized covariates, sum(X(:,k,i)'* X(:,k,i))= N, for k=1,...,p
%   B: p x r matrix of initial B
%   Sigma: r x 1 vector of variances
%   lambda: penalty factor
%   abs_Bbar: p x r matrix, absolute value of some nonzero estimate for B
%   maxit: maximum iterations
%   tol: threshold of relative change in parameter estimate
%
% Output
%   B: p x r matrix of updated B
%   flag_B: indicator of convergence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = size(X,2); % number of variables in X
flag_B = 0;

for s = 1:maxit
    B_old = B;
    
    % sequentially update each row of B
    for k=1:p
        % compute (y - X(:,-k) * B(-k,:))
        ind = ((1:p) ~= k); % 1 x p logical
        tmp = y - mmx('mult',X(:,ind,:),B(ind,:)); % T x r x n
        
        % compute c_k for B
        tmp = sum(mmx('mult',permute(X(:,k,:),[2,1,3]),tmp),3)/N; % 1 x r
        
        B(k,:) = sign(tmp).* max( abs(tmp) - lambda/2./abs_Bbar(k,:).* Sigma', 0);
    end
    
    % check relative change of B ----------------------
    B_old_norm2 = sum(B_old(:).^2); 
    
    if ( B_old_norm2 > 0 ) 
        if ( sqrt( sum((B(:) - B_old(:)).^2) ) / sqrt(B_old_norm2) < tol )
            flag_B = 1;
            break
        end
    else % B_old is a zero matrix
        if ( sum(B(:).^2) == 0 )
            flag_B = 1;
            break
        end
    end
end