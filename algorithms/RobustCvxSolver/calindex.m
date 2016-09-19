function [index1,index2]=calindex(X)
% ----------------------- Input ------------------------------
% X: d*m
% ----------------------- Output ------------------------------
% index1: m*(m-1)/2 dimensional index vector: [ones(1,m-1), 2*ones(1,m-2), .... , m-1]
% index2: m*(m-1)/2 dimensional index vector: [2:m, 3:m, ... , m]
index1 = [];
index2 = [];
[d,m] = size(X);
for ii = 1:m-1
    index1 = [index1,ii*ones(1,m-1-ii+1)];
    index2 = [index2,ii+1:m];
end
end