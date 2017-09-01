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

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
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
		float* d_reduc_comp, int* d_reduc_ind_t, int* d_reduc_ind_m,
			int m, int t, float MAX_FLOAT) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tId = threadIdx.x;
	extern __shared__ int vec[];

	float *s_comp_times = (float*)&vec[0];
	int *s_ind_t = (int*)&vec[BLOCK_SIZE];
	int *s_ind_m = (int*)&vec[2*BLOCK_SIZE];

	uint min = 0;
	uint imin = 0;
	uint jmin = 0;
	float min_value = MAX_FLOAT;

	if(!task_deleted[i]) {
		for (int j = 0; j < m; j++) {

			if (completion_times[j] + machines[j * t + i] < min_value) {
				imin = i;
				jmin = j;
				min = jmin * t + imin;
				min_value = completion_times[jmin] + machines[min];
			}
		}
	}

	s_comp_times[tId] = min_value;
	s_ind_t[tId] = imin;
	s_ind_m[tId] = jmin;

	__syncthreads();

	for(int e = BLOCK_SIZE/2; e > 0; e/=2)
	{
		if (tId < e) {
			if ((s_comp_times[tId + e] < s_comp_times[tId])
					|| (s_comp_times[tId + e] == s_comp_times[tId]
							&& s_ind_t[tId + e] < s_ind_t[tId])) {
				s_comp_times[tId] = s_comp_times[tId + e];
				s_ind_t[tId] = s_ind_t[tId + e];
				s_ind_m[tId] = s_ind_m[tId + e];
			}
		}
		__syncthreads();
	}

	if(tId == 0) {
		d_reduc_comp [blockIdx.x] = s_comp_times[0];
		d_reduc_ind_t[blockIdx.x] = s_ind_t[0];
		d_reduc_ind_m[blockIdx.x] = s_ind_m[0];
	}
}

__global__ void reduction(float* d_reduc_comp, int* d_reduc_ind_t, int* d_reduc_ind_m) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tId = threadIdx.x;

	extern __shared__ int vec[];
	float *s_comp_times = (float*)&vec[0];
	int *s_ind_t = (int*)&vec[BLOCK_SIZE];
	int *s_ind_m = (int*)&vec[2*BLOCK_SIZE];

	s_comp_times[tId] = d_reduc_comp[i];
	s_ind_t[tId] = d_reduc_ind_t[i];
	s_ind_m[tId] = d_reduc_ind_m[i];

	__syncthreads();

	for(int e = BLOCK_SIZE/2; e > 0; e/=2)
	{
		if (tId < e) {
			if ((s_comp_times[tId + e] < s_comp_times[tId])
					|| (s_comp_times[tId + e] == s_comp_times[tId]
							&& s_ind_t[tId + e] < s_ind_t[tId])) {
				s_comp_times[tId] = s_comp_times[tId + e];
				s_ind_t[tId] = s_ind_t[tId + e];
				s_ind_m[tId] = s_ind_m[tId + e];
			}
		}
		__syncthreads();
	}

	if(tId == 0) {
		d_reduc_comp [blockIdx.x] = s_comp_times[0];
		d_reduc_ind_t[blockIdx.x] = s_ind_t[0];
		d_reduc_ind_m[blockIdx.x] = s_ind_m[0];
	}
}

__global__ void block_reduction(float* completion_times, bool* task_map, bool* task_deleted,
		float* d_reduc_comp, int* d_reduc_ind_t, int* d_reduc_ind_m, int t) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tId = threadIdx.x;

	extern __shared__ int vec[];
	float *s_comp_times = (float*)&vec[0];
	int *s_ind_t = (int*)&vec[blockDim.x];
	int *s_ind_m = (int*)&vec[2*blockDim.x];

	s_comp_times[tId] = d_reduc_comp[i];
	s_ind_t[tId] = d_reduc_ind_t[i];
	s_ind_m[tId] = d_reduc_ind_m[i];

	__syncthreads();

	for(int e = blockDim.x/2; e > 0; e/=2)
	{
		if (tId < e) {
			if ((s_comp_times[tId + e] < s_comp_times[tId])
					|| (s_comp_times[tId + e] == s_comp_times[tId]
							&& s_ind_t[tId + e] < s_ind_t[tId])) {
				s_comp_times[tId] = s_comp_times[tId + e];
				s_ind_t[tId] = s_ind_t[tId + e];
				s_ind_m[tId] = s_ind_m[tId + e];
			}
		}
		__syncthreads();
	}

	if(tId == 0) {
		task_deleted[ s_ind_t[0] ] = true;
		task_map[ s_ind_m[0] * t + s_ind_t[0] ] = true;
		completion_times[ s_ind_m[0] ] = s_comp_times[0];
	}
}

int main(int argc, char** argv) {

	int t, m;

	if (argc < 3) {
			printf("Parameters missing: <number of tasks> <number of machines>\n\n");
			return 0;
	}

	t = atoi(argv[1]);
	m = atoi(argv[2]);

	uint mem_size_machines 			= sizeof(float) * (m * t);
	uint mem_size_completion_times 	= sizeof(float) * (m);
	uint mem_size_task_deleted 		= sizeof(bool) * (t);
	uint mem_size_task_map 			= sizeof(bool) * (m * t);
	uint mem_size_reduc_comp		= sizeof(float) * (t/BLOCK_SIZE);
	uint mem_size_reduc_ind			= sizeof(int) * (t/BLOCK_SIZE);

	float *machines				= (float *) malloc(mem_size_machines);
	float *completion_times 	= (float *) malloc(mem_size_completion_times);
	bool *task_deleted			= (bool  *) malloc(mem_size_task_deleted);
	bool *task_map 				= (bool  *) malloc(mem_size_task_map);

	float aux;
	for (int i = 0; i < t; i++) {
		for (int j = 0; j < m; j++) {
			int a = scanf("%f", &aux);

			machines[j * t + i] = aux;
			task_map[j * t + i] = false;
			completion_times[j] = 0;
		}
		task_deleted[i] = false;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *d_machines, *d_completion_times, *d_reduc_comp;
	int *d_reduc_ind_t, *d_reduc_ind_m;
	bool *d_task_deleted, *d_task_map;
	float MAX_FLOAT = std::numeric_limits<float>::max();

	cudaTest(cudaMalloc((void **) &d_machines, mem_size_machines));
	cudaTest(cudaMalloc((void **) &d_completion_times, mem_size_completion_times));
	cudaTest(cudaMalloc((void **) &d_task_deleted, mem_size_task_deleted));
	cudaTest(cudaMalloc((void **) &d_task_map, mem_size_task_map));

	cudaTest(cudaMalloc((void **) &d_reduc_comp, mem_size_reduc_comp));
	cudaTest(cudaMalloc((void **) &d_reduc_ind_t, mem_size_reduc_ind));
	cudaTest(cudaMalloc((void **) &d_reduc_ind_m, mem_size_reduc_ind));

	// copy host memory to device
	cudaTest(cudaMemcpy(d_machines, machines, mem_size_machines, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_completion_times, completion_times, mem_size_completion_times, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_deleted, task_deleted, mem_size_task_deleted, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_map, task_map, mem_size_task_map, cudaMemcpyHostToDevice));


	cudaEventRecord(start);
	for(int k = 0; k < t; k++) {
		dim3 dimBlock(BLOCK_SIZE);
		int dim = t/BLOCK_SIZE;
		dim3 dimGrid(dim);
		min_min<<<dimGrid, dimBlock, BLOCK_SIZE * sizeof(float) +
				BLOCK_SIZE * sizeof(int) + BLOCK_SIZE * sizeof(int) >>>
				(d_machines, d_completion_times, d_task_map, d_task_deleted,
						d_reduc_comp, d_reduc_ind_t, d_reduc_ind_m, m, t, MAX_FLOAT);

		for( ; dim > BLOCK_SIZE; dim/=BLOCK_SIZE) {
			dim3 block(BLOCK_SIZE);
			dim3 grid_b(dim/BLOCK_SIZE);
			reduction<<<grid_b, block, BLOCK_SIZE * sizeof(float) +
					BLOCK_SIZE * sizeof(int) + BLOCK_SIZE * sizeof(int) >>>
				(d_reduc_comp, d_reduc_ind_t, d_reduc_ind_m);
		}
		dim3 block(dim);
		dim3 grid_b(1);
		block_reduction<<<grid_b, block, dim * sizeof(float) + dim * sizeof(int) +
				dim * sizeof(int) >>> (d_completion_times, d_task_map, d_task_deleted,
				d_reduc_comp, d_reduc_ind_t, d_reduc_ind_m, t);
	}
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

	cudaFree(d_reduc_comp);
	cudaFree(d_reduc_ind_t);
	cudaFree(d_reduc_ind_m);

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
