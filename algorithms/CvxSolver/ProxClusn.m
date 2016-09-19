function [P,Theta,iter,fun,primal] = ProxClusn(U,lambda,index1,index2,Theta0,tol,maxiter,weight, backtrack_mem)
% This is the solver for min_P 0.5*\|P-U\|_F^2 + lambda*sum_{i<j}w_ij \|p_i - p_j\|
% ----------------------- Input ------------------------------
% U: data matrix. Size: d*n, here d is the feature dimension, n is the sample
%    size.
% lambda: regularization parameter
% index1: m*(m-1)/2 dimensional index vector: [ones(1,m-1), 2*ones(1,m-2), .... , m-1]
% index2: m*(m-1)/2 dimensional index vector: [2:m, 3:m, ... , m]
% Theta0: d*(n*(n-1)/2)
% tol: tolerence
% maxiter: maximum iteration
% weight: weight matrix. Size: d*(n*(n-1)/2)  
Theta = Theta0;
[d,m] = size(U);
fun = zeros(maxiter+1,1);

A = ThetaMulC(weight.*Theta,m);
P = U - A;
Pmin=P(:,index1) - P(:,index2);
grad = weight.*Pmin;
fun(1) = 0.5*norm(A,'fro')^2 + sum(sum(Theta.* grad));
primal(1)=0.5*norm(P-U,'fro')^2+lambda*sum(sqrt(sum(grad.^2,1)));
alpha = 1; beta = 0.5; sigma = 1e-4; 

for iter = 1:maxiter 
    Theta_old = Theta; grad_old = grad;
    alpha = min(max(alpha,1e-10),1e10); 
    P_old=P;
    % line search
    for inneriter = 1:200
        Theta = ProjTheta(Theta_old + alpha*grad_old,lambda);
        A = ThetaMulC(weight.*Theta,m);
        P = U - A;
        Pmin=P(:,index1) - P(:,index2);
        grad = weight.*Pmin;
        fun(iter+1) = 0.5*norm(A,'fro')^2 + sum(sum(Theta.* grad));
        primal(iter+1)=0.5*norm(P-U,'fro')^2+lambda*sum(sqrt(sum(grad.^2,1)));
        if (iter+1)>backtrack_mem
            fun_old=[fun_old(2:end),fun(iter)];
        else
            fun_old(iter)=fun(iter);
        end
        
            
        if -fun(iter+1) <= -min(fun_old) - sigma/(2*alpha)*norm(Theta-Theta_old,'fro')^2
            break;
        else
            alpha = alpha*beta;
        end
    end   
    
    % stopping condition
    if abs(primal(iter)- primal(iter+1))/abs(primal(iter+1)) < tol
        break;
    end
    st = Theta - Theta_old;
    rt = grad_old - grad;
    alpha = sum(sum(st.*st))/sum(sum(st.*rt));
     
end
fun = fun(1: min(maxiter,iter)+1);


function [Theta] = ProjTheta(Theta0,lambda)
% min_Theta 0.5*\|Theta-Theta0\|_F^2
% s.t. \|Theta_i\| <= lambda

Theta = Theta0./repmat( max(1,sqrt(sum(Theta0.^2))/lambda), size(Theta0,1),1 );

