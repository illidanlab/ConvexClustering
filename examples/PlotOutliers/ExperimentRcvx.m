% This is the example of using Robust COnvex Clustering to find outliers.
% n is sample size, d is feature dimension, c is clusters number
% b+mod((d-b),c) is outlier feature dimension 
% rnggn is seed 
clear;
clc;
addpath(genpath('../../Algorithms/RobustCvxSolver/'))
%create a random problem
n = 20;
d = 100;
rnggn = 2;
c = 2;
b = 10;
[X,~]=syndata_c(d,c,n,rnggn,b);
%calculate weights
gamma=0.1;
weight=calweight(X,gamma);

maxiter=10000;
tol=1e-7;
lambda=[0.02, 0.03, 0.04];
z=[0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5]
for i=1:length(lambda)
    for j=1:length(z);
        [P_Rcvx{i,j},Q_Rcvx{i,j},fun_Rcvx{i,j},iter(i,j)]=RClust(X,lambda(i),z(j),maxiter,tol,0);
    end
end
%plot
figure;
for i = 1 : size(Q_Rcvx, 1);
   for j = 1 : size(Q_Rcvx, 2)
   weight_Q{i,j}=sqrt(sum(Q_Rcvx{i,j}.^2,2));
   end
end

for i=1:100
    for j=1:41
    weight_b(i,j)=weight_Q{5,j}(i,1);
    end
end
 x=[0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5]
for i=1:10
    plot(x,weight_b(i,:),'r--');
    hold on;
end

for i=11:100
    plot(x,weight_b(i,:),'k:');
    hold on;
end
set(gca,'Position',[.1 .1 .4 0.8]);
xlabel('\beta');
ylabel('feature weight');
title('\alpha=0.040');
set(gca,'FontSize',12);