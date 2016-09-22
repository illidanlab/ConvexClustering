function [w,w_c]=syndata_c(d,c,n,rnggn,b)
% This is the function to create a ramdom matrix with c clusters and n
% samples.
% ----------------------- Input ------------------------------
%d is the feature dimeansion 
%c is the number of clusters
%n is the number of total points
%b the dimension of feature outliers is b+mod((d-b),c)
%rnggn: seed 
% ----------------------- Output ------------------------------
%w: d*n
%w_c: centroid matrix d*c
rng(rnggn);
c_n=n/c;  
w_c=zeros(d,c);
w=zeros(d,n);
e=mod((d-b),c); 
w(1:(e+b),:)=4 * [randn((e+b),n/2+2),randn((e+b),n/2-2)];
en=(d-b-e)/c;
begin=b+e+1;
for i=1:c
    w_c(begin:(begin+en-1),i)=30 * randn(en,1);
    for j=1:c_n
        col=(i-1)*c_n+j;
        w(begin:(begin+en-1),col)=w_c(begin:(begin+en-1),i)+4*randn(en,1);
    end
    begin=begin+en;
end
end


  
 


 


    
