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
z=0.1 : 0.01 : 0.5;
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
    for j=1:length(z)
    weight_b(i,j)=weight_Q{1,j}(i,1);
    end
end
 
for i=1:10
    plot(z,weight_b(i,:),'r--');
    hold on;
end

for i=11:100
    plot(z,weight_b(i,:),'k:');
    hold on;
end
set(gca,'Position',[.1 .1 .4 0.8]);
xlabel('\beta');
ylabel('feature weight');
title('\alpha=0.020');
set(gca,'FontSize',12);