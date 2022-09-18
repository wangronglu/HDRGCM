function lambda1_max = findLam1max(N,y,X,B,abs_Bbar,Sigma_inv)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find lambda1_max that shrinks all the entries of B to zero
%
% Input
%   N: total number of observations, N = sum(T_i)
%   y: T x r x n array of outcomes (do not include NaN values)
%   X: T x p x n array of standardized covariates, sum(X(:,k,i)'* X(:,k,i))= N, for k=1,...,p
%   B: p x r matrix of initial B
%   abs_Bbar: p x r matrix, absolute value of some nonzero estimate for B
%   Sigma_inv: r x 1 vector of 1./Sigma
%
% Output
%   lambda1_max: value of lambda1_max
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = size(X,2); % number of variables in X
lambda1_max = 0; % initial lambda1_max

if nnz(B) % initial B is not a zero matrix
    % sequentially update each row of B and meanwhile update lambda1max
    for k=1:p
        % compute (y - X(:,-k) * B(-k,:))
        ind = ((1:p) ~= k); % 1 x p logical
        tmp = y - mmx('mult',X(:,ind,:),B(ind,:)); % T x r x n
        
        % compute c_k for B
        tmp = sum(mmx('mult',permute(X(:,k,:),[2,1,3]),tmp),3)/N; % 1 x r
        
        % compute lambda1_max candidate
        tmp = max( 2 * abs_Bbar(k,:).* abs(tmp).* Sigma_inv' );
        
        % update lambda1_max
        lambda1_max = max(lambda1_max,tmp);
        
        % update B
        B(k,:) = 0;
    end
end

% B is now a zero matrix.
% sequentially go through each row of B and update lambda1max
for k=1:p
    % (y - X(:,-k) * B(-k,:)) is just y
    % compute c_k for B
    tmp = sum(mmx('mult',permute(X(:,k,:),[2,1,3]),y),3)/N; % 1 x r
    
    % compute lambda1_max candidate
    tmp = max(2 * abs_Bbar(k,:).* abs(tmp).* Sigma_inv');
    
    % update lambda1_max
    lambda1_max = max(lambda1_max,tmp);
end
        