function [ P_output, primal ] = cvxclus_eric( X, lambda ,gamma)
% This is a function to call ADMM slover and initialize some variables
% ----------------------- Input ------------------------------
%X: data matrix, d*m
%gamma: parameter to control weights 
%lambda: regularization parameter
% ----------------------- Output ------------------------------
%P_output: d*n matrix
%primal: primal values

[p,n] = size(X);
w =calweight_eric(X,gamma);
nK = length(w);
%compactify_edges input arguements
sizes1 =zeros(n,1);
sizes2 =zeros(n,1);
index1 = [];
index2 = [];
for ii = 1:n-1
    index1 = [index1,ii*ones(1,n-1-ii+1)]; %#ok
    index2 = [index2,ii+1:n]; %#ok
end

P=[index1;index2]';
M1 = zeros(size(P,1),n);
M2 = zeros(size(P,1),n);

for j = 1:n
    group1 =find(P(:,1) == j);
    sizes1(j,1) = length(group1);
    if (sizes1(j,1) > 0)
        M1(1:sizes1(j,1),j) = group1;
    end
    group2 = find(P(:,2) == j);
    sizes2(j,1) = length(group2);
    if (sizes2(j,1) > 0)
        M2(1:sizes2(j,1),j) = group2;
    end
end

M1 = M1(1:max(sizes1),:);
M2 = M2(1:max(sizes2),:);

ix=P-ones(size(P));%c index starts from 0
M1=M1-ones(size(M1));
M2=M2-ones(size(M2));
s1=sizes1';
s2=sizes2';

mix1=size(M1,1);
mix2=size(M2,1);

%initialize input arguments

Lambda = zeros(p,nK);
nu = 1.999/n;
max_iter = 10000;
tol = 1e-8;
primal=zeros(max_iter,1)';
dual=zeros(max_iter,1)';
type=2;% 2 norm
U = zeros(p,n);
V = zeros(p,nK);
iter=0;
[P_output,~]=cvxclustr(X, Lambda, U, V, p, n, nK, ix, w, lambda, nu,  type,s1, s2, M1, M2, mix1, mix2, primal, dual, max_iter, iter, tol);



end

