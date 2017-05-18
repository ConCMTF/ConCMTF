function [x] = cvx_solve(X, A, B, epsilon_sparsity, epsilon_orthogonality, i, mode)
%% solves a convex optimzation
% input X*A - B, only solving for the ith column of X
%   Usage: [x] = cvx_solve(X, A, B, epsilon_sparsity, epsilon_orthogonality, i, mode)
%  Output: x which is the ith column of X
%   if mode == 'factors', it solves the following program
%    X * A - B
%   if mode == 'core', it solves the following program
%   A * X - B
%   This code depends on CVX Solver which is available at:
%          http://cvxr.com/cvx/
%   by Sanaz Bahargam
%       http://cs-people.b.edu/bahargam/
%       E-mail:bahargam@gmail.com or bahargam@bu.edu
%       Last updated: February-14, 2017

x = [];

if strcmp(mode, 'Factors') 
    C = X; C(:,i) = []; %removing the column we are solving for
    pinvA = pinv(A);
    len = size(X,1);
    cvx_begin quiet
        variable x(len,1)  nonnegative
        minimize norm( x - B * pinvA(:,i) , 2)
        subject to
        transpose(x) * C  <= epsilon_orthogonality
        norm(x, 1) <= epsilon_sparsity
    cvx_end
end
if strcmp(mode, 'Core')
    len = size(A,2);
    cvx_begin quiet
        variable x(len) nonnegative
        minimize ( norm(A * x  - B , 2))
        subject to
        norm(x, 1)  <= epsilon_sparsity
    cvx_end
end



