/**********************************************
Copyright (c) 2018 pcubre. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation and/or 
other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software must 
display the following acknowledgement: 
This product includes software developed by pcubre.
4. Neither the name of pcubre nor the names of its contributors may be used to endorse or 
promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY PCUBRE "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PCUBRE
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************/

/* Basic API header files qsub -I -lselect=1:ncpus=1:mem=1gb:ngpus=1
nvcc -O3 -arch=sm_35 -G -lcublas -lcurand -lcusolver --shared -Xcompiler -fPIC -o cuHorse.so cuHorse.cu
*/
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* CUDA API header files */
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cublas_v2.h"
#include "curand.h"
#include "curand_kernel.h"
#include "math_functions.h"

/* Gibbs Sampler from GPU accelerated */
__global__ void init(unsigned int seed, curandState_t *d_states) {
  int idx = threadIdx.x +blockDim.x * blockIdx.x;
  curand_init(seed, idx, 0, &d_states[idx]);
}

__global__ void sample_z(curandState_t *d_states, int *d_Y, double *d_mu, double *d_z) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int d_y_star = 2 * d_Y[idx] - 1;
  double d_mu_star = d_mu[idx] * d_y_star;
  double d_s;
  double d_u_1;
  double d_u_2;
  //Efficient sampling methods, Li and Ghosh
  //make boundary (a,infty) from (0,infty)
  double a = - d_mu_star;
  double a_0 = 0.2570;
  double lambda_star;
  // if a <= 0 use normal rejection sampling
  if( a <= 0){
    d_s = curand_normal_double(&d_states[idx]);
    while( d_s < a){
      d_s = curand_normal_double(&d_states[idx]);
    }
  } else if( a < a_0) { //use half-normal rejection sampling
    d_s = curand_normal_double(&d_states[idx]);
    while( -d_s < a && d_s < a){
      d_s = curand_normal_double(&d_states[idx]);
    }
    if (d_s < 0) { 
      d_s = - d_s;
    } 
  } else { //use the one sided translated-exponential
    lambda_star = .5 * (a + pow( a * a + 4, .5));
    d_u_1 = curand_uniform_double(&d_states[idx]);
    d_u_2 = curand_uniform_double(&d_states[idx]);
    d_s = - log(d_u_2) / lambda_star + a;
    while(d_u_1 > exp(-0.5 * (d_s-lambda_star) * (d_s-lambda_star))){
      d_u_1 = curand_uniform_double(&d_states[idx]);
      d_u_2 = curand_uniform_double(&d_states[idx]);
      d_s = - log(d_u_2) / lambda_star + a;  
    } 
  }
  d_z[idx] = d_y_star * (d_s + d_mu_star);  
}


__global__ void sample_others(double *d_uniform, double *d_lambda_square_inverse, double *d_v_inverse, double *d_e_inverse, double *d_beta_square, double *d_tau_square_inverse, int *p) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  d_lambda_square_inverse[idx] = -log(d_uniform[idx]) / (d_v_inverse[idx] + d_tau_square_inverse[0] * d_beta_square[idx]  / 2);
  d_v_inverse[idx] = -log(d_uniform[idx + p[0]]) / (1 + d_lambda_square_inverse[idx]);
  if (idx == 0) {
    d_e_inverse[0] = -log(d_uniform[2 * p[0]]) / (1 + d_tau_square_inverse[0]);
  }
}

__global__ void sample_tau(curandState_t *d_states, double *d_shape, double *d_scale, double *d_tau_square_inverse) {
  __shared__ int local_array[32];
  __shared__ double d_tau_X[32];
  __shared__ int flag;

  double d_u_1;
  double d_u_2;
  double d_V;
  double a = pow(2 * d_shape[0] - 1, -.5);
  double b = d_shape[0] - log(4.0);
  double c = d_shape[0] + 1 / a;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx == 0) {
    flag = 1;
  }
  while (flag) {
    d_u_1 = curand_uniform_double(&d_states[idx]);
    d_u_2 = curand_uniform_double(&d_states[idx]);
    d_V = a * log(d_u_1 / (1 - d_u_1));
    d_tau_X[idx] = d_shape[0] * exp(d_V);
    if (b + c*d_V - d_tau_X[idx] >= log(d_u_1*d_u_1*d_u_2)) {
      local_array[idx] = 0;
    } else {
      local_array[idx] = 1;
    }
    __syncthreads();
    if (idx == 0) {
      for (int i = 0; i < 32; i++) {
        if (local_array[i] == 1) {
          d_tau_square_inverse[0] =  d_scale[0] / d_tau_X[i];
          flag = 0;
          break;
        }
      }
    }
  }
}

__global__ void beta_square(double *d_beta, double *d_beta_square) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  d_beta_square[idx] = d_beta[idx] * d_beta[idx];
}
__global__ void prep_ab(double *d_shape, double *d_scale, int *d_p, double *d_e_inverse) {
  d_shape[0] = (d_p[0] + 1) / 2;
  d_scale[0] = d_e_inverse[0] + d_scale[0] / 2;
}
extern "C" /*__declspec(dllexport)*/
void gibbs_sampler(int *n_in, int *p_in, double *X, int *Y, double *store, int *iter) {
  int n = n_in[0];
  int p = p_in[0];

  cublasHandle_t handle;
  //cublasStatus_t status;
  //cudaError_t cudaStat;
  cusolverDnHandle_t solver_handle;
  //cusolverStatus_t solver_status;
  curandGenerator_t gen;
  //curandStatus_t curand_status;
  curandState_t *d_states;
  
  cudaStream_t stream1;
  cudaStream_t stream2;
  
  /* Device Storage */
  double *d_X;
  int *d_Y;
  double *d_XTX;
  double *d_Sigma_inverse;
  //double *d_Lambda_square_inverse;
  double *d_lambda_square_inverse;
  double *d_Workspace;
  double *d_s;
  double *d_XTz;
  double *d_z;
  double *d_uniform;
  double *d_beta_square;
  double *d_scale;
  double *d_shape;
  int *d_p;
  double *d_tau_square_inverse;
  double *d_v_inverse;
  double *d_e_inverse;
  int *d_info;
  double *d_mu;
  double *d_zero;
  double *d_one;


  /*Host side variables */
  int Lwork;
  double tau_square_inverse = 1.0;
  double e_inverse = 1.0;
  double *lambda_square = (double *)malloc(p * sizeof(double));
  double *v_inverse = (double *)malloc(p * sizeof(double));
  for (int i = 0; i < p; i++) {
    v_inverse[i] = 1.0;
    lambda_square[i] = 1.0;
  }

  /* Set up handles */
  cublasCreate(&handle);
  cusolverDnCreate(&solver_handle);
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL); //set to time
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  /* Move X and Y to GPU */
  cudaMalloc((void **)&d_X, n * p * sizeof(double));
  cudaMalloc((void **)&d_Y, n * sizeof(int));

  cublasSetMatrixAsync(n, p, sizeof(double), X, n, d_X, n,stream1);
  cublasSetVectorAsync(n, sizeof(int), Y, 1, d_Y, 1,stream2);

  /* Allocate space for XTX */
  cudaMalloc((void **)&d_XTX, p * p * sizeof(double));
  /* Allocate Space for Sigma inverse, R inverse */
  cudaMalloc((void **)&d_Sigma_inverse, p * p * sizeof(double));
  /* Allocate Space for Lambda square inverse */
  cudaMalloc((void **)&d_lambda_square_inverse, p * sizeof(double));
  cublasSetVectorAsync(p, sizeof(double), lambda_square, 1, d_lambda_square_inverse, 1,stream1);

  /* Allocate Space for s, v */
  cudaMalloc((void **)&d_s, p * sizeof(double));
  /* Allocate Space for XTz, mu */
  cudaMalloc((void **)&d_XTz, p * sizeof(double));
  /* Allocate Space for z */
  cudaMalloc((void **)&d_z, n * sizeof(double));
  cudaMemsetAsync(d_z,0,n * sizeof(double),stream2);
  /* Allocate space for the random states */
  cudaMalloc((void **)&d_states, n * sizeof(curandState_t));
  /* Allocate space for the random uniform */
  cudaMalloc((void **)&d_uniform, (p + p + 1) * sizeof(double));
  /* Allocate space for beta_square */
  cudaMalloc((void **)&d_beta_square, p * sizeof(double));
  /* Allocated space for rate */
  cudaMalloc((void **)&d_scale, sizeof(double));
  /* Allocated space for shape */
  cudaMalloc((void **)&d_shape, sizeof(double));
  /* Allocate space for p */
  cudaMalloc((void **)&d_p, sizeof(int));
  cudaMemcpy(d_p, &p, sizeof(int), cudaMemcpyHostToDevice);
  /* Allocate space for tau_square_inverse */
  cudaMalloc((void **)&d_tau_square_inverse, sizeof(double));
  cudaMemcpyAsync(d_tau_square_inverse, &tau_square_inverse, sizeof(double), cudaMemcpyHostToDevice,stream1);
  /* Allocate space for v_inverse */
  cudaMalloc((void **)&d_v_inverse, p * sizeof(double));
  cublasSetVectorAsync(p, sizeof(double), v_inverse, 1, d_v_inverse, 1,stream2);
  /* Allocate space for e_inverse */
  cudaMalloc((void **)&d_e_inverse, sizeof(double));
  cudaMemcpyAsync(d_e_inverse, &e_inverse, sizeof(double), cudaMemcpyHostToDevice,stream1);
  /* Allocate space for info */
  cudaMalloc((void **)&d_info, sizeof(double));
  /* Allocate space for mu */
  cudaMalloc((void **)&d_mu, n * sizeof(double));
  /* Allocate space for one and zero */
  cudaMalloc((void **)&d_one, sizeof(double));
  cudaMalloc((void **)&d_zero, sizeof(double));
  double one = 1.0;
  double zero = 0.0;
  cublasSetVectorAsync(1, sizeof(double), &one, 1, d_one, 1, stream2);
  cublasSetVectorAsync(1, sizeof(double), &zero, 1, d_zero, 1, stream1);
  init <<<n/32, 32 >>>(time(0), d_states);
  /*Step i: Prevompute X^TX before starting the algorithm*/
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, d_one, d_X, n, d_X, n, d_zero, d_XTX, p);

  /*Step ii: Calculate SigmaInv = X^TX + tau^-2Lambda^-2*/
  cublasDcopy(handle, p * p, d_XTX, 1, d_Sigma_inverse, 1);
  cublasDaxpy(handle, p, d_tau_square_inverse, d_lambda_square_inverse, 1, d_Sigma_inverse, p + 1);

  /*Step iii: Perform a Cholesky decomposition to find R_inverse = Chol(Sigma_inverse)*/
  cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, p, d_Sigma_inverse, p, &Lwork);
  cudaMalloc((void **)&d_Workspace, Lwork * sizeof(double));

  for (int t = 0; t < iter[0]; t++) {

    /*Step ii: Calculate SigmaInv = X^TX + tau^-2Lambda^-2*/
    cublasDcopy(handle, p * p, d_XTX, 1, d_Sigma_inverse, 1);
    cublasDaxpy(handle, p, d_tau_square_inverse, d_lambda_square_inverse, 1, d_Sigma_inverse, p + 1);
    cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, p, d_Sigma_inverse, p, d_Workspace, Lwork, d_info);
    
    /*Step iv: Draw s, a vector of IID standard normals */
    curandGenerateNormalDouble(gen, d_s, p, 0.0, 1.0);
    
    /*Step v: Compute R * s by solving the trangular system R^-1 * v = s for v */
    cublasDtrsv(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, p, d_Sigma_inverse, p, d_s, 1);
    
    /*Step vi:(checked) Compute XTz */
    cublasDgemv(handle, CUBLAS_OP_T, n, p, d_one, d_X, n, d_z, 1, d_zero, d_XTz, 1);

    /*Step vii:(checked) Compute mu = Sigma * XT * z by solving the linear system Sigma_inv * mu = XT * z */
    cusolverDnDpotrs(solver_handle, CUBLAS_FILL_MODE_LOWER, p, 1, d_Sigma_inverse, p, d_XTz, p, d_info);

    /*Step viii: Compute Beta = Rs + mu */
    cublasDaxpy(handle, p, d_one, d_s, 1, d_XTz, 1);
    cublasGetVectorAsync(p, sizeof(double), d_XTz, 1, store + t * p , 1, stream1);
    beta_square <<<1, p, 0, stream2 >>>(d_XTz, d_beta_square);

    /* Sample z */
    cublasDgemv(handle, CUBLAS_OP_N, n, p, d_one, d_X, n, d_XTz, 1, d_zero, d_mu, 1);
    sample_z <<<n/8, 8>>>(d_states, d_Y, d_mu, d_z);

    /* Sample lambda^2 v^-1 and e^-1*/
    curandGenerateUniformDouble(gen, d_uniform, p + p + 1);
    sample_others <<<1, p>>>(d_uniform, d_lambda_square_inverse, d_v_inverse, d_e_inverse, d_beta_square, d_tau_square_inverse, d_p);

    /* Sample tau^-2 */
    cublasDdot(handle, p, d_beta_square, 1, d_lambda_square_inverse, 1, d_scale);
    prep_ab <<<1, 1 >>>(d_shape, d_scale, d_p, d_e_inverse);
    sample_tau <<<1, 32 >>>(d_states, d_shape, d_scale, d_tau_square_inverse);
  }


  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_XTX);
  cudaFree(d_Sigma_inverse);
  cudaFree(d_lambda_square_inverse);
  cudaFree(d_Workspace);
  cudaFree(d_s);
  cudaFree(d_z);
  cudaFree(d_states);
  cudaFree(d_uniform);
  cudaFree(d_beta_square);
  cudaFree(d_shape);
  cudaFree(d_scale);
  cudaFree(d_p);
  cudaFree(d_tau_square_inverse);
  cudaFree(d_e_inverse);
  cudaFree(d_v_inverse);
  cudaFree(d_states);
  cudaFree(d_info);
  cudaFree(d_mu);
  cudaFree(d_one);
  cudaFree(d_zero);

  curandDestroyGenerator(gen);
  cusolverDnDestroy(solver_handle);
  cublasDestroy(handle);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  
  free(lambda_square);
  free(v_inverse);
}

