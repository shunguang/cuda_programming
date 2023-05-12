#ifndef _SUM_REDUCTION_H_
#define _SUM_REDUCTION_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <iostream>

typedef int app_data_t;

app_data_t initialize_vector(app_data_t* v, int n);
app_data_t  setInputData(int n, app_data_t** d_v);

#endif