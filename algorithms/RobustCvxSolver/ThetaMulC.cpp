#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include <string.h>
#include "matrix.h"

/*
-------------------------- Function ThetaMulC -----------------------------

 sum_{i<j} theta_{ij}*(e_i - e_j)^T,
 where 
 * theta_{ij} is a d x 1 vector
 * e_i: standard basis (only the i-th entry is 1)
 * i, j \in {1,...,m}    

 Usage (in matlab):
 [x] = ThetaMulC(Theta,m),
 *where
 * Theta: d x m*(m-1)/2 matrix
 * x: d x m matrix
-------------------------- Function ThetaMulC -----------------------------
*/


void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*set up input arguments */
    double*  Theta  =              mxGetPr(prhs[0]);
    size_t   d      =              mxGetM(prhs[0]);
    long     m      =            (long)mxGetScalar(prhs[1]);
 
    double *x;


    /* set up output arguments */
    plhs[0] = mxCreateDoubleMatrix(d,m,mxREAL);
    
    x = mxGetPr(plhs[0]);
    
    long N = 0;
    long i, j, k;
    
    //printf("d=%d,m=%d\n",d,m);
    
    memset(x,0,d*m*sizeof(double));
     for (i=0; i<m-1; i++)
     {
         for (j=i+1; j<m; j++)
         {
            for (k=0; k<d; k++)
            {
                x[i * d + k] = x[i * d + k] + Theta[N * d + k];
                x[j * d + k] = x[j * d + k] - Theta[N * d + k];
            }
             N++;
         }
     }
    //printf("N=%d\n",N);
}


