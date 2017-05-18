function [FacA, FacB, FacC, FacD, core] = Constrained_CMTF(tensor, matrix, dimensions,  opts)
%% Constrained coupled_Matrix Tensor Factorization
%   Usage: [FacA, FacB, FacC, FacD, core] = Con_CMTF(tensor, dimensions, matrix, opt);
%   Output: FacA, FacB, FacC, FacD are the factor matrices and Core is the core tensor
%   opts.
%         mode: a string determining the decomposiiton 'PARAFAC' or 'Tucker'
%         maxiter: max number of iterations
%         convergence_delta: the convergence condition
%         sparsity: a vector determining sparsity for each column of
%         Factors, FacA, FacB, FacC, FacD
%         orthogonality: a vector determining orthogonality for each column of
%         Factors, FacA, FacB, FacC, FacD
%         core_sparsity: a single value determining sparsity of the core
%         tensor
%   This code depends on the TensorToolbox which is available at:
%           http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.5.html
%       and Tensor lab at:
%           URL: http://www.tensorlab.net
%
%   by Sanaz Bahargam
%       http://cs-people.bu.edu/bahargam/
%       E-mail:bahargam@gmail.com or bahargam@bu.edu
%       Last updated: February-14, 2017

%initialize the factors and core
core = rand(dimensions(:, 1:3));
FacA = rand(size(tensor, 1), dimensions(1, 1));
FacB = rand(size(tensor, 2), dimensions(1, 2));
FacC = rand(size(tensor, 3), dimensions(1, 3));
FacD = rand(size(matrix, 2), dimensions(1, 1)) ;
if opts.mode == 'PARAFAC' %creates a digonal tensor
    core = zeros(dimensions(:, 1:3));
    for i = 1:dimensions(1, 1)
        core(i, i, i) = 1;
    end;
end
iter = 0; fit_error =100000; error= 100000;
while fit_error > opts.convergence_delta &&  iter <  opts.maxiter
    iter =  iter + 1;
    %% solve for FacA, first mode
    tensor_FacA = tens2mat(tensor,1);
    N =  tens2mat(core,1) * transpose(kron(FacC, FacB));
    for i = 1: size(FacA, 2)  %solve for each column
        A = N; B = tensor_FacA; X = FacA;
        A = horzcat(A, transpose(FacD));
        B = horzcat(B, matrix);
        x = cvx_solve(X, A, B,  opts.sparsity(1), opts.orthogonality(1), i, 'Factors');
        FacA(:, i) = x;
    end
    FacA = normc(FacA);
    %% Solve for FacB, second mode
    tensor_FacB = tens2mat(tensor, 2);
    N =  tens2mat(core, 2) * transpose(kron(FacC, FacA));
    for i = 1: size(FacB,2)   %solve for each column
        B = tensor_FacB; A = N; X = FacB;
        x = cvx_solve(X, A, B, opts.sparsity(2), opts.orthogonality(2), i, 'Factors');
        FacB(:, i) = x;
    end
    FacB = normc(FacB);
    %% solve for FacC, third mode
    tensor_R = tens2mat(tensor, 3);
    N =  tens2mat(core, 3) * transpose(kron(FacB, FacA));
    for i = 1: size(FacC, 2)   %solve for each column
        B = tensor_R; A = N; X = FacC;
        x = cvx_solve(X, A, B, opts.sparsity(3), opts.orthogonality(3), i, 'Factors');
        FacC(:, i) = x;
    end
    FacC = normc(FacC);
    %% solve for D, the side matrix
    for i = 1: size(FacD,2)     %solve for each column
        epsilon = 0.5;
        A = transpose(FacA); B = transpose(matrix); X = FacD;
        x = cvx_solve(X, A, B, opts.sparsity(4), opts.orthogonality(4), i, 'Factors');
        FacD(:, i) = x;
    end
    FacD = normc(FacD);
    %% solve for core
    if strcmp(opts.mode, 'Tucker') 
        Vec_tensor =  tens2vec(tensor, 1);
        N = kron(FacC, FacB, FacA) ;
        x = cvx_solve(X, N, Vec_tensor, opts.core_sparsity, NaN, NaN, 'Core');
        core = reshape(x, size(core));
    end
    
    tensor_FacA = tens2mat(tensor, 1);
    N =  FacA * tens2mat(core, 1) * transpose(kron(FacC, FacB));
    fit_error = error - norm(tensor_FacA - N, 2);
    error = norm(tensor_FacA - N, 2);
end
end
