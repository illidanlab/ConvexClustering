% This is the example of compare efficiency of AMA and our method. 
% n is sample size, d is feature dimension, c is clusters number
% b+mod((d-b),c) is outlier feature dimension 
% rnggn is seed 
clear;
clc;
addpath(genpath('../../Algorithms/CvxSolver/'))
addpath(genpath('../../Algorithms/AMASolver/'))
%create a random problem
n = 500;
d = 100 : 100 : 1500;
rnggn = 2;
c = 2;
b = 4;
gamma=0;
lambda=0.1;
for i=1:length(d)
    %calculate weights
   [X,~]=syndata_c(d(i),c,n,rnggn,b);
   disp(strcat('Dimension: ', int2str(d(i))));
   tic;
   %our method
   [P_output{i}, fv_primal{i}, fv_dual{i}] = cvxclus_dual(X, lambda, gamma, 20);
   disp(strcat('Time cost of Our Method: ', num2str(toc)));
   tic;
   %AMA
   [ P_output{i}, primal{i} ] = cvxclus_eric( X, lambda,gamma);
   disp(strcat('Time cost of AMA: ', num2str(toc)));
end




  