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

#ifndef EXECUTIONS
#define EXECUTIONS 11
#endif

//using namespace cub;

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

void printSeg(int* host_data, uint num_seg, uint num_ele) {
	std::cout << "\n";
	for (uint i = 0; i < num_seg; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << num_ele << " ";
	std::cout << "\n";
}

__global__ void min_min_sorted(float* machines, uint* task_index, float* completion_times, bool* task_map,
		bool* task_deleted, uint* machine_current_index, int m, int t) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int vec[];

	float *s_comp_times = (float*)&vec[0];
	int *s_ind_max = (int*)&vec[m];

	uint min = 0;
	uint imin = 0;

	for(int k = 0; k < t; k++) {

		int j = machine_current_index[i];
		while (task_deleted[task_index[i * t + j]]) {
			j++;
		}
		machine_current_index[i] = j;

		s_comp_times[i] = completion_times[i] + machines[i * t + j];
		s_ind_max[i] = i;

		__syncthreads();

		for(int e = m/2; e > 0; e/=2)
		{
			if(i < e) {
				if (s_comp_times[i + e] == s_comp_times[i] && s_ind_max[i + e] < s_ind_max[i]) {
					s_comp_times[i] = s_comp_times[i + e];
					s_ind_max[i] = s_ind_max[i + e];
				}
				else if(s_comp_times[i + e] < s_comp_times[i]) {
					s_comp_times[i] = s_comp_times[i + e];
					s_ind_max[i] = s_ind_max[i + e];
				}
			}
			__syncthreads();
		}

		if(i == 0) {
			min = s_ind_max[0] * t + machine_current_index[s_ind_max[0]];
			imin = s_ind_max[0];

			completion_times[imin] = s_comp_times[0];
			task_deleted[task_index[min]] = true;
			task_map[task_index[min] * m + imin] = true;
		}

		__syncthreads();
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

	//uint mem_size_seg 				= sizeof(int) * (m + 1);
	uint mem_size_seg 				= sizeof(int) * (m);
	uint mem_size_machines 			= sizeof(float) * (m * t);
	uint mem_size_task_index		= sizeof(uint) * (m * t);

	uint mem_size_completion_times 	= sizeof(float) * (m);
	uint mem_size_machine_cur_index = sizeof(uint) * (m);

	uint mem_size_task_deleted 		= sizeof(bool) * (t);
	uint mem_size_task_map 			= sizeof(bool) * (t * m);

	int *segments 				= (int   *) malloc(mem_size_seg);
	float *machines				= (float *) malloc(mem_size_machines);
	uint *task_index 			= (uint  *) malloc(mem_size_task_index);

	float *completion_times 	= (float *) malloc(mem_size_completion_times);
	uint *machine_cur_index = (uint  *) malloc(mem_size_machine_cur_index);

	bool *task_deleted			= (bool  *) malloc(mem_size_task_deleted);
	bool *task_map 				= (bool  *) malloc(mem_size_task_map);

	float aux;
	for (int i = 0; i < t; i++) {
		for (int j = 0; j < m; j++) {
			int a = scanf("%f", &aux);

			task_index[j * t + i] = i;
			machines[j * t + i] = aux;
			segments[j] = j*t;

			task_map[i * m + j] = false;
			completion_times[j] = 0;
			machine_cur_index[j] = 0;
		}
		task_deleted[i] = false;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *d_completion_times;
	uint *d_machine_cur_index;
	bool *d_task_deleted, *d_task_map;

	cudaTest(cudaMalloc((void **) &d_completion_times, mem_size_completion_times));
	cudaTest(cudaMalloc((void **) &d_machine_cur_index, mem_size_machine_cur_index));
	cudaTest(cudaMalloc((void **) &d_task_deleted, mem_size_task_deleted));
	cudaTest(cudaMalloc((void **) &d_task_map, mem_size_task_map));

	// copy host memory to device
	cudaTest(cudaMemcpy(d_completion_times, completion_times, mem_size_completion_times, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_machine_cur_index, machine_cur_index, mem_size_machine_cur_index, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_deleted, task_deleted, mem_size_task_deleted, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_map, task_map, mem_size_task_map, cudaMemcpyHostToDevice));

	uint *d_task_index;
	int *d_segments;
	float *d_machines;

	cudaTest(cudaMalloc((void **) &d_segments, mem_size_seg));
	cudaTest(cudaMalloc((void **) &d_machines, mem_size_machines));
	cudaTest(cudaMalloc((void **) &d_task_index, mem_size_task_index));

	// copy host memory to device
	cudaTest(cudaMemcpy(d_segments, segments, mem_size_seg, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_machines, machines, mem_size_machines, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_index, task_index, mem_size_task_index, cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	mgpu::standard_context_t context;
	mgpu::segmented_sort(d_machines, d_task_index, m * t, d_segments, m, mgpu::less_t<float>(), context);

	dim3 dimBlock(m);
	dim3 dimGrid(1);
	min_min_sorted<<<dimGrid, dimBlock, m * sizeof(float) + m * sizeof(int) >>>(d_machines, d_task_index,
				d_completion_times,	d_task_map, d_task_deleted, d_machine_cur_index, m, t);

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

	cudaFree(d_segments);
	cudaFree(d_machines);
	cudaFree(d_task_index);

	if (ELAPSED_TIME != 1) {
		//print(machines, m, t);
		//print(task_index, m, t);
		print(completion_times, m);
	}

	free(task_deleted);
	free(task_map);
	free(machines);
	free(task_index);
	free(segments);
	free(completion_times);
	free(machine_cur_index);

	return 0;
}
