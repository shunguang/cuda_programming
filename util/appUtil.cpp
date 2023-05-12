#include "appUtil.h"

app_data_t initialize_vector(app_data_t* v, int n) 
{
	app_data_t s = 0;
	for (int i = 0; i < n; i++) {
		v[i] = (app_data_t)(rand() % 100);
		s += v[i];
	}
	return s;
}

app_data_t  setInputData(int n, app_data_t** d_v)
{
	size_t bytes = n * sizeof(app_data_t);

	// Host data
	std::vector<app_data_t> h_v(n);

	// Initialize the input data
	app_data_t sum_gt = initialize_vector(h_v.data(), n);

	// Allocate device memory
	cudaMalloc(d_v, bytes);

	// Copy to device
	cudaMemcpy(*d_v, h_v.data(), bytes, cudaMemcpyHostToDevice);

	return sum_gt;
}


