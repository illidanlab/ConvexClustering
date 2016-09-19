function [P_output, fv_primal, fv_dual ] = cvxclus_dual( X, lambda, gamma, backtrack)
% This is a function to call cvx slover and initialize some variables
% ----------------------- Input ------------------------------
%X: data matrix, d*m
%gamma: parameter to control weights 
%lambda: regularization parameter
% ----------------------- Output ------------------------------
%P_output: d*n matrix
%fv_primal: primal values
%fv_dual: dual function values

index1 = [];
index2 = [];
[d,m] = size(X);
for ii = 1:m-1
    index1 = [index1,ii*ones(1,m-1-ii+1)]; %#ok
    index2 = [index2,ii+1:m]; %#ok
end
weight=calweight(X,gamma);
maxiter=1e6;
tol=1e-8;
Theta0 = zeros(d, m*(m-1)/2);
[P_output,~,~,fv_dual,fv_primal] = ProxClusn(X,lambda,index1,index2,Theta0,tol,maxiter, weight, backtrack);
end

