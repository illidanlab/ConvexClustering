//#include <R.h> 
//#include <Rmath.h> 
//#include <R_ext/BLAS.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mex.h"
#include <stdlib.h>
/*#include "blas.h"*/

double dnrm2_(int *n, double *x, int *incx)
{
	long int ix, nn, iincx;
	double norm, scale, absxi, ssq, temp;

	/*  DNRM2 returns the euclidean norm of a vector via the function
	name, so that
	DNRM2 := sqrt( x'*x )
	-- This version written on 25-October-1982.
	Modified on 14-October-1993 to inline the call to SLASSQ.
	Sven Hammarling, Nag Ltd.   */

	/* Dereference inputs */
	nn = *n;
	iincx = *incx;

	if (nn > 0 && iincx > 0)
	{
		if (nn == 1)
		{
			norm = fabs(x[0]);
		}
		else
		{
			scale = 0.0;
			ssq = 1.0;

			/* The following loop is equivalent to this call to the LAPACK
			auxiliary routine:   CALL SLASSQ( N, X, INCX, SCALE, SSQ ) */

			for (ix = (nn - 1)*iincx; ix >= 0; ix -= iincx)
			{
				if (x[ix] != 0.0)
				{
					absxi = fabs(x[ix]);
					if (scale < absxi)
					{
						temp = scale / absxi;
						ssq = ssq * (temp * temp) + 1.0;
						scale = absxi;
					}
					else
					{
						temp = absxi / scale;
						ssq += temp * temp;
					}
				}
			}
			norm = scale * sqrt(ssq);
		}
	}
	else
		norm = 0.0;

	return norm;

} /* dnrm2_ */

void kernel_weights(double *X, int *p, int *n, double *phi, double *w) {
  int i, j, k, l;
  double sos;
  k = 0;
  for (i=0; i<*n-1; i++)
    for (j=i+1; j<*n; j++) {
      sos = 0.;
      for (l=0; l<*p; l++)
	sos += pow(X[l + (*p)*i]-X[l + (*p)*j],2.);
      w[k] = exp(-(*phi)*sos);
      k += 1;
    }
}

void loss_primal_L2(double *X, double *U, double *gamma, int *ix, int *n, int *p, int *nK,
		   double *w, double *output) {  
  const int one = 1;
  int j,k;
  double *dU = calloc(*p, sizeof(double));
  double penalty = 0.;
  double temp;
  
  for (k = 0; k < *nK; k++) {
    for (j = 0; j < *p; j++)
      dU[j] = U[(*p)*ix[k]+j] - U[(*p)*ix[*nK+k]+j] ;
    penalty += w[k]*dnrm2_(p,dU,&one);
  }
  
  temp = 0.;
  for (j = 0; j < *p; j++)
    for (k = 0; k < *n; k++)
      temp += pow(X[k*(*p)+j] - U[k*(*p)+j],2.);
  *output = 0.5*temp + (*gamma)*penalty;
  free(dU);
}

void loss_primal_L1(double *X, double *U, double *gamma, int *ix, int *n, int *p, int *nK,
		 double *w, double *output) {
  int j,k;
  double penalty = 0.;
  double temp = 0.;

  for (k = 0; k < *nK; k++)
    for (j = 0; j < *p; j++)
      penalty += fabs(U[(*p)*ix[k]+j] - U[(*p)*ix[*nK+k]+j]);

  for (j = 0; j < *p; j++)
    for (k = 0; k < *n; k++)
      temp += pow(X[k*(*p)+j] - U[k*(*p)+j],2.);
  *output = 0.5*temp + (*gamma)*penalty;

}

void loss_dual(double *X, double *Lambda, int *ix, int *n, int *p, int *nK,
	       int *s1, int *s2, int *M1, int *M2, int *mix1, int *mix2,
	       double *output) {
  int ii, jj, kk;
  double first_term, second_term;
  double l1_ij, l2_ij;

  first_term = 0.;
  for (ii=0; ii<*n; ii++) {
    for (jj=0; jj<*p; jj++) {
      l1_ij = 0.;
      if (s1[ii] > 0)
	for (kk=0; kk<s1[ii]; kk++)
          l1_ij += Lambda[jj + M1[ii*(*mix1)+kk]*(*p)];
      l2_ij = 0.;
      if (s2[ii] > 0)
	for (kk=0; kk<s2[ii]; kk++)
	  l2_ij += Lambda[jj + M2[ii*(*mix2)+kk]*(*p)];
      first_term += pow(l1_ij-l2_ij,2.);
    }
  }
  second_term = 0.;
  for (ii=0; ii<*nK; ii++)
    for (jj=0; jj<*p; jj++)
      second_term += (X[jj + ix[ii]*(*p)] - X[jj + ix[*nK + ii]*(*p)])*Lambda[jj + ii*(*p)];
  *output = -0.5*first_term - second_term;
}

void prox_L1(double *x, int n, double *px, double tau) {
  int i;
  double y;
  
  for (i=0; i<n; i++) {
    y = x[i];
    px[i] = 0.0;
    if (y > tau)
      px[i] = y - tau;
    else if (y < -tau)
      px[i] = y + tau;
  }
}

void prox_L2(double *x, int n, double *px, double tau) {
  int i;
  double lv;
  
  lv = 0.0;
  for (i=0; i<n; i++)
    lv += pow(x[i],2.0);
  lv = sqrt(lv);
  if (lv == 0.)
    for (i=0; i<n; i++)
      px[i] = x[i];
  else
    for (i=0; i<n; i++)
      px[i] = fmax(0.,1.-(tau/lv))*x[i];  
}

void update_U_ama(double *X, double *Lambda, double *U, int *M1, int* M2, int *s1, int *s2,
	      int *mix1, int *mix2, int *n, int *p, int *nK) {
  int ii, jj, kk;
  double u_temp;
  
  for (ii=0; ii<*n; ii++) {
    for (jj=0; jj<*p; jj++) {
      u_temp = X[jj + ii*(*p)];
      if (s1[ii] > 0)
	for (kk=0; kk<s1[ii]; kk++)
	  u_temp += Lambda[jj + M1[ii*(*mix1)+kk]*(*p)];
      if (s2[ii] > 0)
	for (kk=0; kk<s2[ii]; kk++)
	  u_temp -= Lambda[jj + M2[ii*(*mix2)+kk]*(*p)];
      U[jj + ii*(*p)] = u_temp;
    }
  }
}

void update_V_L2(double *U, double *Lambda, double *V, double *w,
	      double *gamma, double *nu, int *ix, int *p, int *nK) {
  
  int i, j, k, kk;
  double *z = calloc(*p, sizeof(double));
  double *zz = calloc(*p, sizeof(double));
  
  for (kk=0; kk<*nK; kk++) {
    i = ix[kk];
    j = ix[*nK+kk];
    for (k=0; k<*p; k++)
      z[k] = U[k + i*(*p)] - U[k + j*(*p)] - (1.0/(*nu))*Lambda[k + kk*(*p)];
    prox_L2(z,*p,zz,w[kk]*(*gamma)/(*nu));
    for (k=0; k<*p; k++)
      V[k + kk*(*p)] = zz[k];
  }
  free(z);
  free(zz);
}

void update_V_L1(double *U, double *Lambda, double *V, double *w,
	      double *gamma, double *nu, int *ix, int *p, int *nK) {
  
  int i, j, k, kk;
  double *z = calloc(*p, sizeof(double));
  double *zz = calloc(*p, sizeof(double));
  
  for (kk=0; kk<*nK; kk++) {
    i = ix[kk];
    j = ix[*nK+kk];
    for (k=0; k<*p; k++)
      z[k] = U[k + i*(*p)] - U[k + j*(*p)] - (1.0/(*nu))*Lambda[k + kk*(*p)];
    prox_L1(z,*p,zz,w[kk]*(*gamma)/(*nu));
    for (k=0; k<*p; k++)
      V[k + kk*(*p)] = zz[k];
  }
  free(z);
  free(zz);
}

void proj_L2(double *x, int n, double *proj_x, double tau) {
  const int one = 1;
  int i;
  double norm_x;
  norm_x = dnrm2_(&n,x,&one);
  if (norm_x > tau)
    for (i=0; i<n; i++)
      proj_x[i] = (tau/norm_x)*x[i];
  else
    for (i=0; i<n; i++)
      proj_x[i] = x[i];
}


void proj_Linf(double *x, int n, double *proj_x, double tau) {
  int i;

  for (i=0; i<n; i++)
    proj_x[i] = fmin(fmax(x[i],-tau),tau);
}

void update_Lambda_ama_L2(double *Lambda, double *U, double *nu, double *gamma,
		   int *ix, int *p, int *nK, double *w) {
  int i, j;
  double *x = calloc(*p, sizeof(double));
  double *y = calloc(*p, sizeof(double));
  for (j=0; j<*nK; j++) {
    for (i=0; i<*p; i++)
      x[i] = Lambda[i + j*(*p)] - (*nu)*(U[i + ix[j]*(*p)] - U[i + ix[j + (*nK)]*(*p)]);
    proj_L2(x,*p,y,(*gamma)*w[j]);
    for (i=0; i<*p; i++)
      Lambda[i + j*(*p)] = y[i];
  }
  free(x);
  free(y);
}

void update_Lambda_ama_Linf(double *Lambda, double *U, double *nu, double *gamma,
			  int *ix, int *p, int *nK, double *w) {
  int i, j;
  double *x = calloc(*p, sizeof(double));
  double *y = calloc(*p, sizeof(double));
  for (j=0; j<*nK; j++) {
    for (i=0; i<*p; i++)
      x[i] = Lambda[i + j*(*p)] - (*nu)*(U[i + ix[j]*(*p)] - U[i + ix[j + (*nK)]*(*p)]);
    proj_Linf(x,*p,y,(*gamma)*w[j]);
    for (i=0; i<*p; i++)
      Lambda[i + j*(*p)] = y[i];
  }
  free(x);
  free(y);
}

void convex_cluster_ama(double *X, double *Lambda, double *U, double *V,
			int *p, int *n, int *nK, int *ix, double *w,
			double *gamma, double *nu, int *type,
			int *s1, int *s2, int *M1, int *M2,
			int *mix1, int *mix2, double *primal, double *dual,
			int *max_iter, int *iter, double *tol) {
  int ii, its;
  double *Lambda_old = calloc((*p)*(*nK), sizeof(double));
  double fp, fd;
  void (*update_Lambda)(double*,double*,double*,double*,int*,int*,int*,double*);
  void (*update_V)(double*,double*,double*,double*,double*,double*,int*,int*,int*);
  void (*loss_primal)(double*,double*,double*,int*,int*,int*,int*,double*,double*);

  if (*type == 1) {
    update_Lambda = &update_Lambda_ama_Linf;
    loss_primal = &loss_primal_L1;
    update_V = &update_V_L1;
  }
  else {
    update_Lambda = &update_Lambda_ama_L2;
    loss_primal = &loss_primal_L2;
    update_V = &update_V_L2;
  }
  
  for (its=0; its<*max_iter; its++) {
    for (ii=0; ii<(*p)*(*nK); ii++)
      Lambda_old[ii] = Lambda[ii];
    update_U_ama(X,Lambda_old,U,M1,M2,s1,s2,mix1,mix2,n,p,nK);
    update_Lambda(Lambda,U,nu,gamma,ix,p,nK,w);
    loss_primal(X,U,gamma,ix,n,p,nK,w,&fp);
    primal[its] = fp;
    loss_dual(X,Lambda,ix,n,p,nK,s1,s2,M1,M2,mix1,mix2,&fd);
    dual[its] = fd;
    if (fp-fd < *tol)
      break;
  }
  *iter = its;
  update_V(U,Lambda,V,w,gamma,nu,ix,p,nK);
  free(Lambda_old);
}

void convex_cluster_ama_acc(double *X, double *Lambda, double *U, double *V,
			      int *p, int *n, int *nK, int *ix, double *w, double *gamma,
			    double *nu, int* type, int *s1, int *s2, int *M1, int*M2,
			      int *mix1, int *mix2, double *primal, double *dual, int *max_iter,
			      int *iter, double *tol) {

  int its,i;
  double *Lambda_old = calloc((*p)*(*nK),sizeof(double));
  double *S = calloc((*p)*(*nK),sizeof(double));
  double fp, fd;
  void (*update_Lambda)(double*,double*,double*,double*,int*,int*,int*,double*);
  void (*update_V)(double*,double*,double*,double*,double*,double*,int*,int*,int*);
  void (*loss_primal)(double*,double*,double*,int*,int*,int*,int*,double*,double*);

  if (*type == 1) {
    update_Lambda = &update_Lambda_ama_Linf;
    loss_primal = &loss_primal_L1;
    update_V = &update_V_L1;
  }
  else {
    update_Lambda = &update_Lambda_ama_L2;
    loss_primal = &loss_primal_L2;
    update_V = &update_V_L2;
  }

  for (its=0; its<2; its++) {
    for (i=0; i<(*p)*(*nK); i++)
      Lambda_old[i] = Lambda[i];
    update_U_ama(X,Lambda_old,U,M1,M2,s1,s2,mix1,mix2,n,p,nK);
    update_Lambda(Lambda,U,nu,gamma,ix,p,nK,w);
    loss_primal(X,U,gamma,ix,n,p,nK,w,&fp);
    primal[its] = fp;
    loss_dual(X,Lambda,ix,n,p,nK,s1,s2,M1,M2,mix1,mix2,&fd);
    dual[its] = fd;
  }
  for (its=2; its<*max_iter; its++) {
    for (i=0; i<(*p)*(*nK); i++)
      S[i] = Lambda[i] + ((double)(its-1))/((double)(its+2))*(Lambda[i] - Lambda_old[i]);
    update_U_ama(X,S,U,M1,M2,s1,s2,mix1,mix2,n,p,nK);
    update_Lambda(S,U,nu,gamma,ix,p,nK,w);
    for (i=0; i<(*p)*(*nK); i++) {
      Lambda_old[i] = Lambda[i];
      Lambda[i] = S[i];
    }
    loss_primal(X,U,gamma,ix,n,p,nK,w,&fp);
    primal[its] = fp;
    loss_dual(X,Lambda,ix,n,p,nK,s1,s2,M1,M2,mix1,mix2,&fd);
    dual[its] = fd;
    //    if (fp-fd < (*tol)*(1.+0.5*(fp+fd))) break;
    if (fp-fd < *tol) break;
  }
  *iter = its;
  update_V(U,Lambda,V,w,gamma,nu,ix,p,nK);
  free(Lambda_old);
  free(S);
}

/********************** ADMM **********************/
void tolerance_primal(double *U, double *V, double *eps_abs, double *eps_rel, int *ix, int *p, int *nK, double *output) {
  int j,k;
  double du = 0.0;
  double dv = 0.0;

  *output = sqrt( (double) (*p)*(*nK))*(*eps_abs);
  for (k=0; k<*nK; k++) {
    for (j=0; j<*p; j++) {
      du += pow(U[j + (*p)*ix[k]] - U[j + (*p)*ix[k + *nK]], 2.0);
      dv += pow(V[j + (*p)*k],2.0);
    }
  }
  du = sqrt(du);
  dv = sqrt(dv);
  *output += (*eps_rel)*fmax(du,dv);
}

void tolerance_dual(double *Lambda, double *eps_abs, double *eps_rel, int *n, int *p, int *nK,
		    int *s1, int *s2, int *M1, int *M2, int *mix1, int *mix2, double *output) {
  int ii, jj, kk;
  int index;
  double sos, u;

  *output = sqrt( (double) (*p)*(*n))*(*eps_abs);
  sos = 0.0;
  for (ii=0; ii<*n; ii++) {
    for (jj=0; jj<*p; jj++) {
      u = 0.0;
      if (s1[ii] > 0)
        for (kk=0; kk<s1[ii]; kk++) {
          index = jj + (*p)*M1[kk + (*mix1)*ii];
          u += Lambda[index];
        }
      if (s2[ii] > 0)
        for (kk=0; kk<s2[ii]; kk++) {
          index = jj + (*p)*M2[kk + (*mix2)*ii];
          u -= Lambda[index];
        }
      sos += pow(u,2.0);
    }
  }
  *output += sqrt(sos)*(*eps_rel);
}

// This function computes the mean of the data matrix X.
void get_xbar(double *X, double *xbar, int *n, int *p) {
  int i, j;
  double rs;
  for (i=0; i<(*p); i++) {
    rs = 0.;
    for (j=0; j<(*n); j++) {
      rs += X[i + (*p)*j];
    }
    xbar[i] = rs/ (double) (*n);
  }
}

// This function computes the L2 norm of the primal residual.
void residual_primal(double *U, double *V, int *ix,
		     int *p, int *nK, double *residual) {
  int i, j;
  int Ix, Jx;
  double sos;
  sos = 0.;
  for (i=0; i<(*nK); i++) {
    Ix = ix[i];
    Jx = ix[i + (*nK)];
    for (j=0; j<(*p); j++)
      sos += pow(U[j + (*p)*Ix] - U[j + (*p)*Jx] - V[j + (*p)*i], 2.0);
  }
  *residual = sqrt(sos);
}

/* This function maps the index pair (i,j) to its dictionary ordering index k.
   This version of the mapping takes a vector of first indices i and a single
   second index j. The indices i and j are between 0 and n-1. The index k 
   takes on values between 0 and n*(n-1)/2 - 1.
 */ 
void tri2vecA(int *i, int j, int p, int *k, int n) {
  int l;
  for (l=0; l<n; l++)
    k[l] = p*i[l] - (i[l]+1)*i[l]/2 + j - i[l] - 1;
}

/* This function maps the index pair (i,j) to its dictionary ordering index k.
   This version of the mapping takes a vector of second indices j and a single
   first index i. The indices i and j are between 0 and n-1. The index k 
   takes on values between 0 and n*(n-1)/2 - 1.
 */ 
void tri2vecB(int i, int *j, int p, int *k, int n) {
  int l;
  for (l=0; l<n; l++)
    k[l] = p*i - (i+1)*i/2 + j[l] - i - 1;
}

// This function computes the L2 norm of the dual residual.
void residual_dual(double *V, double *V_old, int *n, int *p, int *nK,
		   int *s1, int *s2, int *M1, int *M2, int *mix1, int *mix2,
		   double *nu, double *residual) {

  int ii, jj, kk;
  int index1;
  double s, sos;

  // Initialize sos
  sos = 0.0;

  for (ii=0; ii<*n; ii++) {
    for (jj=0; jj<*p; jj++) {
      s = 0.0;
      if (s1[ii] > 0)
	for (kk=0; kk<s1[ii]; kk++) {
	  index1 = jj + (*p)*M1[kk + (*mix1)*ii];
	  s += V[index1] - V_old[index1];
	}
      if (s2[ii] >0)
	for (kk=0; kk<s2[ii]; kk++) {
	  index1 = jj + (*p)*M2[kk + (*mix2)*ii];
	  s -= V[index1] - V_old[index1];
	}
      sos += pow(s,2.0);
    }
  }
  *residual = (*nu)*sqrt(sos);
}

// update_U for ADMM
void update_U_admm(double *X, double *Lambda, double *U, double *V,
		  double *xbar, int *n, int *p, 
		  int *s1, int *s2, int *M1, int *M2, int *mix1, int *mix2,		  
		  double *nu) {
  int i, j, k;
  int index0, index1;
  double omega, u;
   
  omega = 1.0/(1.0 + ((double) *n)*(*nu));

  for (j=0; j<*n; j++) {
    for (i=0; i<*p; i++) {
      index0 = i + (*p)*j;
      u = X[index0];
      if (s1[j] > 0)
        for (k=0; k<s1[j]; k++) {
	  index1 = i + (*p)*M1[k + (*mix1)*j];
	  u += Lambda[index1] + (*nu)*V[index1];
	}
      if (s2[j] > 0)
	for (k=0; k<s2[j]; k++) {
	  index1 = i + (*p)*M2[k + (*mix2)*j];
	  u -= Lambda[index1] + (*nu)*V[index1];
	}
      U[index0] = omega*u + (1.0-omega)*xbar[i];
    }
  }
}

void update_Lambda_admm(double *Lambda, double *U,
		       double *V, double *nu,
		       int *ix, int *p, int *nK) {
  int i,j;
  double lambda;
  for (j=0; j<*nK; j++)
    for (i=0; i<*p; i++) {
      lambda = Lambda[i + (*p)*j];
      lambda -= (*nu)*(U[i + (*p)*ix[j]] - U[i + (*p)*ix[j+(*nK)]] - V[i + (*p)*j]);
      Lambda[i + (*p)*j] = lambda;
    }
}

void convex_cluster_admm(double *X, double *Lambda, double *U, double *V,
			 int *p, int *n, int *nK, int *ix, double *w, double *gamma, double *nu, int *type,
			 int *s1, int *s2, int *M1, int *M2, int *mix1, int *mix2,
			 double *primal, double *dual, double *tols_primal, double *tols_dual, int *max_iter, int *iter,
			 double *eps_abs, double *eps_rel) {
  int its, jj;
  double *V_old = calloc((*p)*(*nK),sizeof(double));
  double *xbar = calloc(*p,sizeof(double));
  double rp, rd, upsilon, LambdaNorm;
  double tp, td;
  void (*update_V)(double*,double*,double*,double*,double*,double*,int*,int*,int*);

  if (*type == 1) {
    update_V = &update_V_L1;
  }
  else {
    update_V = &update_V_L2;
  }

  get_xbar(X,xbar,n,p);
  upsilon = 0.0;
  for (its=0; its<*n; its++)
    for (jj=0; jj<*p; jj++)      
      upsilon += pow(X[jj + (*p)*its] - xbar[jj],2.0);
  upsilon = sqrt(upsilon);
  for (its=0; its<*max_iter; its++) {
    for (jj=0; jj<(*p)*(*nK); jj++)
      V_old[jj]= V[jj];
    update_U_admm(X,Lambda,U,V,xbar,n,p,s1,s2,M1,M2,mix1,mix2,nu);
    update_V(U,Lambda,V,w,gamma,nu,ix,p,nK);
    update_Lambda_admm(Lambda,U,V,nu,ix,p,nK);
    LambdaNorm = 0.0;
    for (jj=0; jj<(*p)*(*nK); jj++)
      LambdaNorm += pow(Lambda[jj],2.0);
    LambdaNorm = sqrt(LambdaNorm);
    residual_primal(U,V,ix,p,nK,&rp);
    primal[its] = rp;
    tolerance_primal(U, V, eps_abs, eps_rel, ix, p, nK, &tp);
    tols_primal[its] = tp;
    residual_dual(V,V_old,n,p,nK,s1,s2,M1,M2,mix1,mix2,nu,&rd);
    dual[its] = rd;
    tolerance_dual(Lambda, eps_abs, eps_rel, n, p, nK, s1, s2, M1, M2, mix1, mix2, &td);
    tols_dual[its] = td;
    if ((rp <= tp) & (rd <= td)) break;
  }
  if (its > *max_iter)
    *iter = *max_iter;
  else
    *iter = its;
  free(V_old);
  free(xbar);
}

void convex_cluster_admm_acc(double *X, double *Lambda, double *U, double *V,
			     int *p, int *n, int *nK, int *ix, double *w, double *gamma, double *nu, int *type,
			 int *s1, int *s2, int *M1, int *M2, int *mix1, int *mix2,
			 double *primal, double *dual, double *tols_primal, double *tols_dual, int *max_iter, int *iter,
			 double *eps_abs, double *eps_rel) {
  int its, jj;
  double *V_old = calloc((*p)*(*nK),sizeof(double));
  double *Lambda_old = calloc((*p)*(*nK),sizeof(double));
  double *xbar = calloc(*p,sizeof(double));
  double rp, rd, upsilon, LambdaNorm;
  double tp, td;
  double r_last;
  double alpha, alpha_old;
  void (*update_V)(double*,double*,double*,double*,double*,double*,int*,int*,int*);

  if (*type == 1) {
    update_V = &update_V_L1;
  }
  else {
    update_V = &update_V_L2;
  }

  r_last = HUGE_VAL;
  alpha_old = 1.;

  get_xbar(X,xbar,n,p);
  upsilon = 0.0;
  for (its=0; its<*n; its++)
    for (jj=0; jj<*p; jj++)      
      upsilon += pow(X[jj + (*p)*its] - xbar[jj],2.0);
  upsilon = sqrt(upsilon);
  for (its=0; its<*max_iter; its++) {
    for (jj=0; jj<(*p)*(*nK); jj++) {
      V_old[jj]= V[jj];
      Lambda_old[jj] = Lambda[jj];
    }
    update_U_admm(X,Lambda,U,V,xbar,n,p,s1,s2,M1,M2,mix1,mix2,nu);
    update_V(U,Lambda,V,w,gamma,nu,ix,p,nK);
    update_Lambda_admm(Lambda,U,V,nu,ix,p,nK);
    LambdaNorm = 0.0;
    for (jj=0; jj<(*p)*(*nK); jj++)
      LambdaNorm += pow(Lambda[jj],2.0);
    LambdaNorm = sqrt(LambdaNorm);
    residual_primal(U,V,ix,p,nK,&rp);
    primal[its] = rp;
    tolerance_primal(U, V, eps_abs, eps_rel, ix, p, nK, &tp);
    tols_primal[its] = tp;
    residual_dual(V,V_old,n,p,nK,s1,s2,M1,M2,mix1,mix2,nu,&rd);
    dual[its] = rd;
    tolerance_dual(Lambda, eps_abs, eps_rel, n, p, nK, s1, s2, M1, M2, mix1, mix2, &td);
    tols_dual[its] = td;
    if ((rp <= tp) & (rd <= td)) break;
    if ((rp > r_last) | (rd > r_last)) {
      alpha_old = 1.;
      r_last = HUGE_VAL;
    } else {
      alpha = 0.5*(1.0 + sqrt(1.0 + 4.0*pow(alpha_old,2.0)));
      for (jj=0; jj<(*p)*(*nK); jj++) {
	V[jj] = V[jj] + ((alpha_old-1.)/alpha)*(V[jj] - V_old[jj]);
	Lambda[jj] = Lambda[jj] + ((alpha_old-1.)/alpha)*(Lambda[jj] - Lambda_old[jj]);
      }
      r_last = rp;
      if (r_last < rd)
	r_last = rd;
      alpha_old = alpha;
    }    
  }
  if (its > *max_iter)
    *iter = *max_iter;
  else
    *iter = its;
  free(V_old);
  free(Lambda_old);
  free(xbar);
}
void mexFunction
(
int nargot,
mxArray *pargout[],
int nargin,
const mxArray *pargin[]
)
{
	double *X, *Lambda, *U, *V,  *w, *gamma, *nu,  *primal, *dual,  *tol, *output1,*output2;

    double iteer;
	int i, j, idx;
    int a,ss;
	size_t row;
	size_t col;
    int iter,mix1,mix2,type,nK,n;
    long int max_iter;


    int p;
		if (nargin != 23) {
		mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
			"23 inputs required.");
		}




    
	X = mxGetPr(pargin[0]);
    Lambda = mxGetPr(pargin[1]);
    U = mxGetPr(pargin[2]);
    V = mxGetPr(pargin[3]);
    p = mxGetScalar(pargin[4]);
    n=mxGetScalar(pargin[5]);
    nK=mxGetScalar(pargin[6]);

	
   
    
    col=mxGetN(pargin[7]);
    row=mxGetM(pargin[7]);
    ss=col*row;
    int *ix=(int*) malloc(sizeof(int)*ss);
    for(i=0;i<row;i++)
      for(j=0;j<col;j++){
            idx=i*col+j;
            a=mxGetPr(pargin[7])[idx];
            ix[idx]=a;
 
      }
   
    
    w = mxGetPr(pargin[8]);
    gamma = mxGetPr(pargin[9]);
	nu = mxGetPr(pargin[10]);
    type=mxGetScalar(pargin[11]);
    
    
     col=mxGetN(pargin[12]);
     row=mxGetM(pargin[12]);
     ss=col*row;
     int *s1=(int*) malloc(sizeof(int)*ss);
     for(i=0;i<row;i++)
      for(j=0;j<col;j++){
           idx=i*col+j;
            a=mxGetPr(pargin[12])[idx];
            s1[idx]=a;
 
      }
   
       col=mxGetN(pargin[13]);
     row=mxGetM(pargin[13]);
     ss=col*row;
     int *s2=(int*) malloc(sizeof(int)*ss);
     for(i=0;i<row;i++)
      for(j=0;j<col;j++){
           idx=i*col+j;
            a=mxGetPr(pargin[13])[idx];
            s2[idx]=a;
 
      }
     
   
     

    col=mxGetN(pargin[14]);
    row=mxGetM(pargin[14]);
    ss=col*row;
    int *M1=(int*) malloc(sizeof(int)*ss);
    for(i=0;i<row;i++)
      for(j=0;j<col;j++){
           idx=i*col+j;
            a=mxGetPr(pargin[14])[idx];
            M1[idx]=a;
 
      }
    
    
    
    col=mxGetN(pargin[15]);
    row=mxGetM(pargin[15]);
    ss=col*row;
    int *M2=(int*) malloc(sizeof(int)*ss);
    for(i=0;i<row;i++)
      for(j=0;j<col;j++){
           idx=i*col+j;
            a=mxGetPr(pargin[15])[idx];
            M2[idx]=a;
 
      }
    
    
    mix1=mxGetScalar(pargin[16]);
    mix2=mxGetScalar(pargin[17]);
    
    primal = mxGetPr(pargin[18]);
    dual = mxGetPr(pargin[19]);

    
    max_iter = mxGetScalar(pargin[20]);
    iter=mxGetScalar(pargin[21]);

	tol = mxGetPr(pargin[22]);

	//output
	col = mxGetN(pargin[2]);
	row = mxGetM(pargin[2]);

	pargout[0] = mxCreateDoubleMatrix(row, col, mxREAL);
	pargout[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
 	output1 = mxGetPr(pargout[0]);
 	output2 = mxGetPr(pargout[1]);
   
 convex_cluster_ama_acc(X, Lambda, U, V, &p, &n, &nK, ix, w, gamma, nu, &type, s1, s2, M1, M2, &mix1, &mix2, primal, dual, &max_iter, &iter, tol);
 
   for (i = 0; i<row; i++)
		for (j = 0; j<col; j++){
		idx = i*col + j;
		output1[idx] = U[idx];

		}
    iteer=iter;
    *output2=iteer;
    free (ix);
    free (M1);
    free (M2);
	free(s2);
    free(s1);
}