/*
 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================

 COMPILAR USANDO O SEGUINTE COMANDO:

 nvcc segmented_sort.cu -o segmented_sort -std=c++11 --expt-extended-lambda -I"/home/schmid/Dropbox/Unicamp/workspace/sorting_segments/moderngpu-master/src"

 */

#include <moderngpu/kernel_segsort.hxx>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cuda.h>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit (EXIT_FAILURE);
	}
}

template<typename T>
void print(T* vec, uint t, uint m) {
	std::cout << "\n";
	for (uint i = 0; i < t; i++) {
		for (uint j = 0; j < m; j++) {
			std::cout << vec[i * m + j] << " ";
		}
		std::cout << "\n";
	}

}

template<typename T>
void print(T* vec, uint t) {
	std::cout << "\n";
	for (uint i = 0; i < t; i++) {
		std::cout << vec[i] << " ";
	}
	std::cout << "\n";


}

__global__ void min_min(float* machines, float* completion_times, bool* task_map, bool* task_deleted,
			int t, int m, float MAX_FLOAT) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int vec[];

	float *s_comp_times = (float*)&vec[0];
	int *s_ind_t = (int*)&vec[t];
	int *s_ind_m = (int*)&vec[t+t];

	uint min = 0;
	uint imin = 0;
	uint jmin = 0;
	float min_value = 0;

	for(int k = 0; k < t; k++) {

		min_value = MAX_FLOAT;

		if(task_deleted[i]) {
			s_comp_times[i] = MAX_FLOAT;
		}
		else {

			for (int j = 0; j < m; j++) {

				if (completion_times[j] + machines[i * m + j] < min_value) {
					imin = i;
					jmin = j;
					min = imin * m + jmin;
					min_value = completion_times[jmin] + machines[min];
				}
			}

			s_comp_times[i] = min_value;
			s_ind_t[i] = imin;
			s_ind_m[i] = jmin;
		}

		__syncthreads();

		for(int e = t/2; e > 0; e/=2)
		{
			if (i < e) {
				if ((s_comp_times[i + e] < s_comp_times[i])
						|| (s_comp_times[i + e] == s_comp_times[i]
								&& s_ind_t[i + e] < s_ind_t[i])) {
					s_comp_times[i] = s_comp_times[i + e];
					s_ind_t[i] = s_ind_t[i + e];
					s_ind_m[i] = s_ind_m[i + e];
				}
			}
			__syncthreads();
		}

		if(i == 0) {
			min = s_ind_t[0] * m + s_ind_m[0];
			imin = s_ind_t[0];
			jmin = s_ind_m[0];

			task_deleted[imin] = true;
			task_map[min] = true;
			completion_times[jmin] = s_comp_times[0];
		}

		__syncthreads();

	}
}

int main(int argc, char** argv) {

	int t, m;

	/*if (argc < 3) {
			printf("Parameters missing: <number of tasks> <number of machines>\n\n");
			return 0;
	}
	t = atoi(argv[1]);
	m = atoi(argv[2]);
	*/
	int a = scanf("%d", &t);
	int a = scanf("%d", &m);

	uint mem_size_machines 			= sizeof(float) * (t * m);
	uint mem_size_completion_times 	= sizeof(float) * (m);
	uint mem_size_task_deleted 		= sizeof(bool) * (t);
	uint mem_size_task_map 			= sizeof(bool) * (t * m);

	float *machines				= (float *) malloc(mem_size_machines);
	float *completion_times 	= (float *) malloc(mem_size_completion_times);
	bool *task_deleted			= (bool  *) malloc(mem_size_task_deleted);
	bool *task_map 				= (bool  *) malloc(mem_size_task_map);

	float aux;
	for (int i = 0; i < t; i++) {
		for (int j = 0; j < m; j++) {
			int a = scanf("%f", &aux);

			machines[i * m + j] = aux;
			task_map[i * m + j] = false;
			completion_times[j] = 0;
		}
		task_deleted[i] = false;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *d_machines, *d_completion_times;
	bool *d_task_deleted, *d_task_map;
	float MAX_FLOAT = std::numeric_limits<float>::max();

	cudaTest(cudaMalloc((void **) &d_machines, mem_size_machines));
	cudaTest(cudaMalloc((void **) &d_completion_times, mem_size_completion_times));
	cudaTest(cudaMalloc((void **) &d_task_deleted, mem_size_task_deleted));
	cudaTest(cudaMalloc((void **) &d_task_map, mem_size_task_map));

	// copy host memory to device
	cudaTest(cudaMemcpy(d_machines, machines, mem_size_machines, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_completion_times, completion_times, mem_size_completion_times, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_deleted, task_deleted, mem_size_task_deleted, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_map, task_map, mem_size_task_map, cudaMemcpyHostToDevice));

	dim3 dimBlock(t);
	dim3 dimGrid(1);
	cudaEventRecord(start);
	min_min<<<dimGrid, dimBlock, t * sizeof(float) + t * sizeof(int) + t * sizeof(int) >>>(d_machines, d_completion_times,
			d_task_map, d_task_deleted, t, m, MAX_FLOAT);
	cudaEventRecord(stop);

	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

	if (ELAPSED_TIME == 1) {
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << milliseconds << "\n";
	}

	cudaDeviceSynchronize();

	cudaTest(cudaMemcpy(completion_times, d_completion_times, mem_size_completion_times, cudaMemcpyDeviceToHost));
	cudaTest(cudaMemcpy(task_map, d_task_map, mem_size_task_map, cudaMemcpyDeviceToHost));

	cudaFree(d_machines);
	cudaFree(d_completion_times);
	cudaFree(d_task_map);
	cudaFree(d_task_deleted);

	if (ELAPSED_TIME != 1) {
		//print(machines, m, t);
		//print(task_index, m, t);
		print(completion_times, m);
	}

	free(task_deleted);
	free(task_map);
	free(machines);
	free(completion_times);

	return 0;
}
