/* Basic API header files qsub -I -lselect=1:ncpus=1:mem=1gb:ngpus=1
nvcc -O3 -arch=sm_35 -G -lcublas -lcurand -lcusolver --shared -Xcompiler -fPIC -o cuHorse1.so cuHorse1.cu
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

/* Gibbs Sampler from GPU accelerated
*/
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
	double d_z_prop;
	double ratio;
	if (d_mu_star > 0.47) {
		d_s = curand_normal_double(&d_states[idx]);
		d_z_prop = d_s + d_mu_star;
		while (d_z_prop < 0) {
			d_s = curand_normal_double(&d_states[idx]);
			d_z_prop = d_s + d_mu_star;
		}
	}
	else {
		// Truncate to R+  assume alpha = 1 d_y_star takes care of region
		d_u_1 = curand_uniform_double(&d_states[idx]);
		d_u_2 = curand_uniform_double(&d_states[idx]);
		d_s = -log(d_u_1);
		ratio = exp(d_s - (d_s - d_mu[idx]) * (d_s - d_mu[idx]) / 2 - 1 - 2 * d_mu[idx]);
		while (d_u_2 > ratio) {
			d_u_1 = curand_uniform_double(&d_states[idx]);
			d_u_2 = curand_uniform_double(&d_states[idx]);
			d_s = -log(d_u_1);
			ratio = exp(d_s - (d_s - d_mu[idx]) * (d_s - d_mu[idx]) / 2 - 1 - 2 * d_mu[idx]);
		}
		d_z_prop = d_s * d_y_star;
	}
	d_z[idx] = d_z_prop;
}


__global__ void sample_others(double *d_uniform, double *d_Lambda_square_inverse, double *d_v_inverse, double *d_e_inverse, double *d_beta, double *d_tau_square_inverse, int *p) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	d_Lambda_square_inverse[idx + p[0] * idx] = -log(d_uniform[idx]) / (d_v_inverse[idx] + d_tau_square_inverse[0]*d_beta[idx] * d_beta[idx] / 2);
	d_v_inverse[idx] = -log(d_uniform[idx + p[0]]) / (1 + d_Lambda_square_inverse[idx + p[0] * idx]);
	if (idx == 0) {
		d_e_inverse[0] = -log(d_uniform[2 * p[0] - 1]) / (1 + d_tau_square_inverse[0]);
	}
}

__global__ void sample_tau(curandState_t *d_states, double *d_shape, double *d_rate, double *d_tau_square_inverse) {
	__shared__ double local_array[32];
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
			local_array[idx] = 1.0;
		}
		else {
			local_array[idx] = 0.0;
		}
		__syncthreads();
		if (idx == 0) {
			for (int i = 0; i < 32; i++) {
				if (local_array[i] == 1.0) {
					d_tau_square_inverse[0] = 1 / (d_rate[0] * d_tau_X[i]);
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
__global__ void prep_ab(double *d_shape, double *d_rate, int *d_p, double *d_e_inverse) {
	d_rate[0] = (d_p[0] + 1) / 2;
	d_shape[0] = d_e_inverse[0] + d_shape[0] / 2;
}
extern "C" /*__declspec(dllexport)*/
void gibbs_sampler(int *n_in, int *p_in, double *X, int *Y, double *store, double *debug) {
	int n = n_in[0];
	int p = p_in[0];

	cublasHandle_t handle;
	cublasStatus_t status;
	cudaError_t cudaStat;
	cusolverDnHandle_t solver_handle;
	cusolverStatus_t solver_status;
	curandGenerator_t gen;
	curandStatus_t curand_status;
	curandState_t *d_states;

	/* Device Storage */
	double *d_X;
	int *d_Y;
	double *d_XTX;
	double *d_Sigma_inverse;
	double *d_Lambda_square_inverse;
	double *d_lambda_square_inverse;
	double *d_Workspace;
	double *d_s;
	double *d_XTz;
	double *d_z;
	double *d_uniform;
	double *d_beta_square;
	double *d_rate;
	double *d_shape;
	double *d_store;
	int *d_p;
	double *d_tau_square_inverse;
	double *d_v_inverse;
	double *d_e_inverse;


	/*Host side variables */
	int Lwork;
	int devInfo = 0;
	double tau_square_inverse = 1.0;
	double e_inverse = 1.0;
	double *lambda_square = (double *)malloc(p * p * sizeof(double));
	double *v_inverse = (double *)malloc(p * sizeof(double));
	for (int i = 0; i < p; i++) {
		v_inverse[i] = 1.0;
		for (int j = 0; j < p; j++) {
			lambda_square[i + p * j] = 0.0;
		}
		lambda_square[i + p * i] = 1.0;
	}
	double *z = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++) {
		z[i] = .5;
	}

	/* Set up handles */
	status = cublasCreate(&handle);
	solver_status = cusolverDnCreate(&solver_handle);
	curand_status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
	curand_status = curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	/* Move X and Y to GPU */
	cudaStat = cudaMalloc((void **)&d_X, n * p * sizeof(double));
	cudaStat = cudaMalloc((void **)&d_Y, n * sizeof(int));

	status = cublasSetMatrix(n, p, sizeof(double), X, n, d_X, n);
	status = cublasSetVector(n, sizeof(int), Y, 1, d_Y, 1);

	/* Allocate space for XTX */
	cudaStat = cudaMalloc((void **)&d_XTX, p * p * sizeof(double));
	/* Allocate Space for Sigma inverse, R inverse */
	cudaStat = cudaMalloc((void **)&d_Sigma_inverse, p * p * sizeof(double));
	/* Allocate Space for Lambda square inverse */
	cudaStat = cudaMalloc((void **)&d_Lambda_square_inverse, p * p * sizeof(double));
	status = cublasSetMatrix(p, p, sizeof(double), lambda_square, p, d_Lambda_square_inverse, p);
	cudaStat = cudaMalloc((void **)&d_lambda_square_inverse, p * sizeof(double));
	status = cublasDcopy(handle, p, d_Lambda_square_inverse, p+1, d_lambda_square_inverse, 1);

	/* Allocate Space for s, v */
	cudaStat = cudaMalloc((void **)&d_s, p * sizeof(double));
	/* Allocate Space for XTz, mu */
	cudaStat = cudaMalloc((void **)&d_XTz, p * sizeof(double));
	/* Allocate Space for z */
	cudaStat = cudaMalloc((void **)&d_z, n * sizeof(double));
	status = cublasSetVector(n, sizeof(double), z, 1, d_z, 1);
	/* Allocate space for the random states */
	cudaStat = cudaMalloc((void **)&d_states, n * sizeof(curandState_t));
	/* Allocate space for the random uniform */
	cudaStat = cudaMalloc((void **)&d_uniform, (p + p + 1) * sizeof(double));
	/* Allocate space for beta_square */
	cudaStat = cudaMalloc((void **)&d_beta_square, p * sizeof(double));
	/* Allocated space for rate */
	cudaStat = cudaMalloc((void **)&d_rate, sizeof(double));
	/* Allocated space for shape */
	cudaStat = cudaMalloc((void **)&d_shape, sizeof(double));
	/* Allocate space for p */
	cudaStat = cudaMalloc((void **)&d_p, sizeof(int));
	cudaMemcpy(d_p, &p, sizeof(int), cudaMemcpyHostToDevice);
	/* Allocate space for tau_square_inverse */
	cudaStat = cudaMalloc((void **)&d_tau_square_inverse, sizeof(double));
	cudaMemcpy(d_tau_square_inverse, &tau_square_inverse, sizeof(double), cudaMemcpyHostToDevice);
	/* Allocate space for v_inverse */
	cudaStat = cudaMalloc((void **)&d_v_inverse, p * sizeof(double));
	status = cublasSetVector(p, sizeof(double), v_inverse, 1, d_v_inverse, 1);
	/* Allocate space for e_inverse */
	cudaStat = cudaMalloc((void **)&d_e_inverse, sizeof(double));
	cudaMemcpy(d_e_inverse, &e_inverse, sizeof(double), cudaMemcpyHostToDevice);
	/*Allocate space for store */
	cudaStat = cudaMalloc((void **)&d_store, n * n * sizeof(double));


	init <<<n, 1 >>>(0, d_states);
	/*Step i: Prevompute X^TX before starting the algorithm*/
	double one = 1.0;
	double zero = 0.0;
	status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
		p, p, n,
		&one, d_X, n, d_X, n, &zero,
		d_XTX, p);

	/*Step ii: Calculate SigmaInv = X^TX + tau^-2Lambda^-2*/
	status = cublasDcopy(handle, p * p, d_XTX, 1, d_Sigma_inverse, 1);
	status = cublasDaxpy(handle, p *p, &tau_square_inverse, d_Lambda_square_inverse, 1, d_Sigma_inverse, 1);

	/*Step iii: Perform a Cholesky decomposition to find R_inverse = Chol(Sigma_inverse)*/
	solver_status = cusolverDnDpotrf_bufferSize(solver_handle,
		CUBLAS_FILL_MODE_UPPER,
		p * p,
		d_Sigma_inverse,
		p,
		&Lwork);
	cudaStat = cudaMalloc((void **)&d_Workspace, Lwork * sizeof(double));

	for (int t = 0; t < 1; t++) {

		/*Step ii: Calculate SigmaInv = X^TX + tau^-2Lambda^-2*/
		status = cublasDcopy(handle, p * p, d_XTX, 1, d_Sigma_inverse, 1);
		status = cublasDaxpy(handle, p * p, &tau_square_inverse, d_Lambda_square_inverse, 1, d_Sigma_inverse, 1);

		solver_status = cusolverDnDpotrf(solver_handle,
			CUBLAS_FILL_MODE_UPPER,
			p * p,
			d_Sigma_inverse,
			p,
			d_Workspace,
			Lwork,
			&devInfo);
		/*Step iv: Draw s, a vector of IID standard normals */
		curand_status = curandGenerateNormalDouble(gen,
			d_s,
			p,
			0.0,
			1.0);
		
		/*Step v: Compute R * s by solving the trangular system R^-1 * v = s for v */
		status = cublasDtrsv(handle,
			CUBLAS_FILL_MODE_UPPER,
			CUBLAS_OP_N,
			CUBLAS_DIAG_NON_UNIT,
			p,
			d_Sigma_inverse,
			p,
			d_s,
			1);
		
		/*Step vi: Compute XTz */
		status = cublasDgemv(handle,
			CUBLAS_OP_T,
			n,
			p,
			&one,
			d_X,
			n,
			d_z,
			1,
			&zero,
			d_XTz,
			1);

		/*Step vii: Compute mu = Sigma * XT * z by solving the linear system Sigma_inv * mu = XT * z */
		solver_status = cusolverDnDpotrs(solver_handle,
			CUBLAS_FILL_MODE_UPPER,
			p,
			1,
			d_Sigma_inverse,
			p,
			d_XTz,
			p,
			&devInfo);

		/*Step viii: Compute Beta = Rs + mu */
		status = cublasDaxpy(handle,
			p,
			&one,
			d_s,
			1,
			d_XTz,
			1);

		sample_z <<<n, 1 >>>(d_states, d_Y, d_XTz, d_z);
		

		/* Sample lambda^2 v^-1 and e^-1*/
		curand_status = curandGenerateUniformDouble(gen, d_uniform, p + p + 1);
		sample_others <<<p, 1>>>(d_uniform, d_Lambda_square_inverse, d_v_inverse, d_e_inverse, d_XTz, d_tau_square_inverse, d_p);
		

		/* Sample tau^-2 */
		beta_square <<<p, 1 >>>(d_XTz, d_beta_square);
		status = cublasDcopy(handle, p, d_Lambda_square_inverse, p + 1, d_lambda_square_inverse, 1);
		status = cublasGetVector(p, sizeof(double), d_lambda_square_inverse, 1, debug, 1);
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
		status = cublasDdot(handle,
			p,
			d_beta_square, 1,
			d_lambda_square_inverse, 1,
			d_shape);
		cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

		//prep_ab <<<1, 1 >>>(d_shape, d_rate, d_p, d_e_inverse);

		//sample_tau <<<1, 32 >>>(d_states, d_shape, d_rate, d_tau_square_inverse);

		//status = cublasDcopy(handle, p, d_XTz, 1, d_store + t*p, 1);
	}

	status = cublasGetMatrix(n, n, sizeof(double), d_store, n, store, n);

	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_XTX);
	cudaFree(d_Sigma_inverse);
	cudaFree(d_Lambda_square_inverse);
	cudaFree(d_lambda_square_inverse);
	cudaFree(d_Workspace);
	cudaFree(d_s);
	cudaFree(d_z);
	cudaFree(d_states);
	cudaFree(d_uniform);
	cudaFree(d_beta_square);
	cudaFree(d_shape);
	cudaFree(d_rate);
	cudaFree(d_p);
	cudaFree(d_store);
	cudaFree(d_tau_square_inverse);
	cudaFree(d_e_inverse);
	cudaFree(d_v_inverse);
	cudaFree(d_states);

	curand_status = curandDestroyGenerator(gen);
	solver_status = cusolverDnDestroy(solver_handle);
	status = cublasDestroy(handle);

	free(lambda_square);
	free(v_inverse);
	free(z);
}
