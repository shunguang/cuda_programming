// This program performs sum reduction with an optimization
// that removes shared memory bank conflicts
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "../include/sumReduction.cuh"

#define SIZE 256
#define SHMEM_SIZE 256 * 4

__global__ void sum_reduction2a(int *v, int *v_r) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

__global__ void sum_reduction2b(int* v_r, int* v, int n) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	if (i + blockDim.x < n) {
		// Store first partial result instead of just the elements
		partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
		__syncthreads();

		// Start at 1/2 block stride and divide by two each iteration
		for (int s = blockDim.x / 2; s > 0; s >>= 1) {
			// Each thread does work unless it is further than the stride
			if (threadIdx.x < s) {
				partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
			}
			__syncthreads();
		}

		// Let the thread 0 for this block write it's result to main memory
		// Result is inexed by this block
		if (threadIdx.x == 0) {
			v_r[blockIdx.x] = partial_sum[0];
		}
	}
}

int test_sumReduction2a(int n) 
{
	assert(n / 2 == n * 2);

	// TB Size
	int TB_SIZE = SIZE;
	// Grid Size (cut in half) (No padding)
	int GRID_SIZE = n / TB_SIZE / 2;

	cudaError_t cuda_status;
	app::AppTicToc tt;

	// Vector size
	//int n = 1 << 16;
	size_t bytes = n * sizeof(int);
	size_t bytes_r = GRID_SIZE * sizeof(int);

	// Original vector and result vector
	int *h_v, *h_v_r;
	int *d_v, *d_v_r;

	// Allocate memory
	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes_r);   //wus1: changed from bytes to bytes_r

	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes_r);		//wus1: changed from bytes to bytes_r

	// Initialize vector
	int sum_gt= initialize_vector(h_v, n);

	// Copy to device
	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);


	// Call kernel
	sum_reduction2a << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);

	sum_reduction2a << <1, TB_SIZE >> > (d_v_r, d_v_r);

	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		printf("Kernel launch failed: %s", cudaGetErrorString(cuda_status));
		return -1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %s after launching addKernel!", cudaGetErrorString(cuda_status));
		return -2;
	}

	// Copy to host
	//wus1: we only need to copy 1 element, 
	//cudaMemcpy(h_v_r, d_v_r, bytes_r, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_v_r, d_v_r, sizeof(int), cudaMemcpyDeviceToHost);

	// Print the result
	//printf("Accumulated result is %d \n", h_v_r[0]);
	//scanf("Press enter to continue: ");
	assert(h_v_r[0] == sum_gt);

	printf("COMPLETED SUCCESSFULLY\n");
	cudaFree(d_v);
	cudaFree(d_v_r);
	free(h_v);
	free(h_v_r);

	return 0;
}

int test_sumReduction2b(const int n)
{
	app::AppTicToc tt;

	// Block and Grid Sizes
	int TB_SIZE = SIZE;
	int GRID_SIZE = (int)ceil(0.5 * n / TB_SIZE);

	//input data
	int* d_v;
	int  sum_gt = setInputData(n, &d_v);

	//output data
	int* d_v_r;
	int n_r = GRID_SIZE;
	cudaMalloc(&d_v_r, n_r * sizeof(int));


	// Call kernel
	tt.tic();
	sum_reduction2b << <GRID_SIZE, TB_SIZE >> > (d_v_r, d_v, n);
	sum_reduction2b << <1, TB_SIZE >> > (d_v_r, d_v_r, n_r);

	cudaError_t cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		printf("Kernel launch failed: %s", cudaGetErrorString(cuda_status));
		return -1;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %s after launching addKernel!", cudaGetErrorString(cuda_status));
		return -2;
	}
	tt.toc();

	// Copy to host;
	int h_r;
	cudaMemcpy( &h_r, d_v_r, sizeof(int), cudaMemcpyDeviceToHost);

	//Print the result
	assert(h_r == sum_gt);
	printf("COMPLETED SUCCESSFULLY: methodFlag=2, n=%d, %s\n", n, tt.toString("us", "timeUsed").c_str());

	cudaFree(d_v);
	cudaFree(d_v_r);

	return 0;
}
