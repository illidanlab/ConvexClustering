clear;
clc;
addpath(genpath('../../Algorithms/CvxSolver/'))
addpath(genpath('../../Algorithms/ADMMSolver/'))
%create a random problem
n = 500;
d = 50;
rnggn = 2;
c = 2;
b = 4;
[X,~]=syndata_c(d,c,n,rnggn,b);
%calculate weights
gamma=0.1;
lambda=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1];
for i=1:length(lambda)
    tic;
   [P_output{i}, fv_primal{i}, fv_dual{i}] = cvxclus_dual(X, lambda(i), gamma, 20);
   disp(num2str(toc));
   tic;
   [ P_output{i}, primal{i} ] = cvxclus_eric( X, lambda(i) ,gamma);
   disp(num2str(toc));
end




  