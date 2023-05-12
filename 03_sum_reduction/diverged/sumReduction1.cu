// This program computes a sum reduction algortithm with warp divergence
// By: Nick from CoffeeBeforeArch

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

using std::accumulate;
using std::generate;
using std::cout;
using std::vector;

#include "../include/sumReduction.cuh"

#define SHMEM_SIZE 256

__global__ void sumReduction1a(int *v, int *v_r) {
	// Allocate shared memory
	__shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements into shared memory
	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	// Iterate of log base 2 the block dimension
	for (int s = 1; s < blockDim.x; s *= 2) {
		// Reduce the threads performing work by half previous the previous
		// iteration each cycle
		if (threadIdx.x % (2 * s) == 0) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	//wus: blockIdx.x [0, maxNumOfBlocksInGrid_x)
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

int test_sumReduction1a( int n ) {

	// Vector size
	assert(n / 2 == n * 2);

	// TB Size
	const int TB_SIZE = SHMEM_SIZE;
	const int GRID_SIZE = n / TB_SIZE;

	int N = n; //1 << 16;
	int N_r = GRID_SIZE;

	size_t bytes = N * sizeof(int);
	size_t bytes_r = N_r * sizeof(int);

	// Host data
	vector<int> h_v(N);

  // Initialize the input data
  generate(begin(h_v), end(h_v), [](){ return rand() % 10; });

	// Allocate device memory
	int *d_v, *d_v_r;
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes_r);
	
	// Copy to device
	cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);
	

	// Call kernels
	sumReduction1a<<<GRID_SIZE, TB_SIZE>>>(d_v, d_v_r);

	sumReduction1a<<<1, TB_SIZE>>> (d_v_r, d_v_r);

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


	// Copy to host;
	int h_r;
	cudaMemcpy( &h_r, d_v_r, sizeof(int), cudaMemcpyDeviceToHost);
	 
	//Print the result
	assert(h_r == std::accumulate(begin(h_v), end(h_v), 0));

	cout << "COMPLETED SUCCESSFULLY\n";

	cudaFree(d_v); 
	cudaFree(d_v_r);
	return 0;
}



__global__ void sumReduction1b(app_data_t* v_r, app_data_t* v, int n) {
	// Allocate shared memory
	__shared__ app_data_t partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		// Load elements into shared memory
		partial_sum[threadIdx.x] = v[tid];
		__syncthreads();

		// Iterate of log base 2 the block dimension
		for (int s = 1; s < blockDim.x; s *= 2) {
			// Reduce the threads performing work by half previous the previous
			// iteration each cycle
			if (threadIdx.x % (2 * s) == 0) {
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

int test_sumReduction1b(const int n)
{
	cudaError_t cuda_status;
	app::AppTicToc tt;

	//set up kernel threads 
	const int BLOCK_SIZE = SHMEM_SIZE;                     //# of threads in each block
	const int GRID_SIZE = (n - 1) / BLOCK_SIZE + 1;				 //# of blocks in a grid

	//input data
	app_data_t* d_v;
	app_data_t  sum_gt = setInputData(n, &d_v);

	//output data
	app_data_t* d_v_r;
	cudaMalloc(&d_v_r, GRID_SIZE * sizeof(app_data_t));


	// Call kernels
	tt.tic();
	sumReduction1b << <GRID_SIZE, BLOCK_SIZE >> > (d_v_r, d_v, n);
	sumReduction1b << <1, BLOCK_SIZE >> > (d_v_r, d_v_r, GRID_SIZE);

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
	// Copy results to host: we only need one element
	app_data_t h_r;
	cudaMemcpy((void*)&h_r, (void*)d_v_r, sizeof(app_data_t), cudaMemcpyDeviceToHost);
	tt.toc();

	//Print the result
	assert(h_r == sum_gt);

	printf("COMPLETED SUCCESSFULLY: methodFlag=1, n=%d, %s\n", n, tt.toString("us", "timeUsed").c_str());

	cudaFree(d_v);
	cudaFree(d_v_r);
	return 0;
}
