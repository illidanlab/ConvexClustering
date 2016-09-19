clear;
clc;
addpath(genpath('../../Algorithms/RobustCvxSolver/'))
%create a random problem
n = 500;
d = 50;
rnggn = 2;
c = 2;
b = 4;
[X,~]=syndata_c(d,c,n,rnggn,b);
%calculate weights
gamma=0.1;
weight=calweight(X,gamma);

maxiter=10000;
tol=1e-7;
lambda=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1];
z=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1];
for i=1:length(lambda)
    for j=1:length(z);
        [P_Rcvx{i,j},Q_Rcvx{i,j},fun_Rcvx{i,j},iter(i,j)]=RClust(X,lambda(i),z(j),maxiter,tol,0);
    end
end




  