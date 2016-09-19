function [weight]=calweight_eric(X,gamma)
%Summary of this function goes here:
%This function is used to calculate the weight of convex clustering. It is
%the w_ij in 0.5\|X-P\| + \lambda \sum w_ij\|P_i-P_j\|, w_ij = exp(-gamma\|x_i-x_j\|^2)
% ----------------------- Input ------------------------------
%X: data matrix
%gamma: parameter to control weights 
% ----------------------- Output ------------------------------
%w_ij: 1:(m*(m-1)/2)
index1 = [];
index2 = [];
[~,m] = size(X);
for ii = 1:m-1
   index1 = [index1,ii*ones(1,m-1-ii+1)];
   index2 = [index2,ii+1:m];
end
for ii=1:length(index1)
   weight(ii)=exp(-gamma*(norm(X(:,index1(ii))-X(:,index2(ii)),2))^2);   
end
end