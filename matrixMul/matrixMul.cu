// This program computes a simple version of matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <cstdlib>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>

using std::cout;
using std::endl;
using std::generate;
using std::vector;

__global__ void matrixMul(int *a, int *b, int *c, int N) {
	// Compute each thread's global row and column
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Boundary check
	if ((row < N) && (col < N)) {
		// Iterate over row, and down column
	  int tmp = 0;
		for (int k = 0; k < N; k++) {
			// Accumulate result for a single element
			tmp += a[row * N + k] * b[k * N + col];
		}

		// Write back the results
		c[row * N + col] = tmp;
	}
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  // For every row...
  for(int i = 0; i < N; i++){
    // For every column...
    for(int j = 0; j < N; j++){
      // For every element in the row-column pair
      int tmp = 0;
      for(int k = 0; k < N; k++){
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {
	// Matrix size of 1024 x 1024;
	int N = 1 << 10;

	// Size (in bytes) of matrix
	size_t bytes = N * N * sizeof(int);

	// Host vectors
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

	// Initialize matrices
  generate(h_a.begin(), h_a.end(), [](){ return rand() % 100; });
  generate(h_b.begin(), h_b.end(), [](){ return rand() % 100; });

	// Allocate device memory
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Copy data to the device
	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

	// Threads per CTA dimension
	int THREADS = 32;

	// Blocks per grid dimension
	int BLOCKS = (N + THREADS - 1 ) / THREADS;

	// Use dim3 structs for block  and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 grid(BLOCKS, BLOCKS);

	// Launch kernel
	matrixMul <<<grid, threads >>> (d_a, d_b, d_c, N);

	// Copy back to the host
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	// Check result
	verify_result(h_a, h_b, h_c, N);

	printf("COMPLETED SUCCESSFULLY\n");
  
  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  
	return 0;
}
