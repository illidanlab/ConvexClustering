function [P,Q,fun,iter,primal_P]=RClust(X,lambda,z,maxiter,tol,gamma)
%min 0.5*\|X-P-Q\|^2 + lambda sum_{i<j}(weight*\|P_i-P_j\|_2)
%           s.t \|Q\|_1,2<z;
%----------------------- Input ------------------------------
%     X:d*m matrix, d is dimension of feature, m is sample size
%     gamma:weight parameter wij=exp(-gamma\|X_i-X_j\|)
%----------------------- Output ------------------------------
%     P:d*m matrix
%     Q:d*m matrix
%     fun:function value
%     iter:iteration number
Q=zeros(size(X));
P=zeros(size(X));
[index1,index2]=calindex(X);
weight=calweight(X,gamma);
func_val_eval = @(P, Q) 0.5*norm(X-P-Q,'fro')^2+lambda*sum(sqrt(sum((weight.*(P(:,index1) - P(:,index2))).^2,1)))+z*sum(sqrt(sum(Q.^2,2)));
fun(1)=func_val_eval(P, Q);

 for iter=1:maxiter
    
    [P,~,primal_P]=cvxclus_dual((X-Q), lambda, gamma, 20);
    Q=FGLasso_projection((X-P),z);
    fun(iter+1)=func_val_eval(P, Q);
    
    if abs(fun(iter) - fun(iter+1))/abs(fun(iter+1)) < tol
        break;
    end
 end
end
function [X] = FGLasso_projection (D, lambda )
    % l2.1 norm projection.
        X = repmat(max(0, 1 - lambda./sqrt(sum(D.^2,2))),1,size(D,2)).*D;
end