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
#define BLOCK_SIZE 1024
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

__global__ void min_min_sorted(float* machines, uint* task_index, float* completion_times, int* task_map,
		uint* machine_current_index, float* reduced_times, uint* reduced_indexes, int m, int t) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x;

	//printf("i=%d --- idx=%d\n", i, idx);
	extern __shared__ int vec[];

	float *s_comp_times = (float*)&vec[0];
	int *s_ind_max = (int*)&vec[blockDim.x];

	int j = machine_current_index[i];
	while (task_map[task_index[i * t + j]] != -1) {
		j++;
	}
	machine_current_index[i] = j;

	s_comp_times[idx] = completion_times[i] + machines[i * t + j];
	s_ind_max[idx] = i;
	
	__syncthreads();
//	printf("shared copy, ok!");

	for(int e = blockDim.x/2; e > 0; e/=2)
	{
		if (idx < e) {
			if ((s_comp_times[idx + e] < s_comp_times[idx])
					|| (s_comp_times[idx + e] == s_comp_times[idx]
							&& s_ind_max[idx + e] < s_ind_max[idx])) {
				s_comp_times[idx] = s_comp_times[idx + e];
				s_ind_max[idx] = s_ind_max[idx + e];
			}
		}
		__syncthreads();
	}

	if(idx == 0) {
		/*
		min = s_ind_max[0] * t + machine_current_index[s_ind_max[0]];
		imin = s_ind_max[0];
		completion_times[imin] = s_comp_times[0];
		task_map[task_index[min]] = imin;
		*/
		reduced_times[blockIdx.x] = s_comp_times[0];
		reduced_indexes[blockIdx.x] = s_ind_max[0];
//		printf("i=%d --- idx=%d\n", i, idx);
	}

}

__global__ void blocks_reduction(float* machines, uint* task_index, float* completion_times, int* task_map,
		uint* machine_current_index, float* reduced_times, uint* reduced_indexes, int m, int t) {

	//int i = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x;
	extern __shared__ int vec[];

	float *s_comp_times = (float*)&vec[0];
	int *s_ind_max = (int*)&vec[blockDim.x];

	uint min = 0;
	uint imin = 0;

	s_comp_times[idx] = reduced_times[idx];
	s_ind_max[idx] = reduced_indexes[idx];

	__syncthreads();

	for(int e = blockDim.x/2; e > 0; e/=2)
	{
		if (idx < e) {
			if ((s_comp_times[idx + e] < s_comp_times[idx])
					|| (s_comp_times[idx + e] == s_comp_times[idx]
							&& s_ind_max[idx + e] < s_ind_max[idx])) {
				s_comp_times[idx] = s_comp_times[idx + e];
				s_ind_max[idx] = s_ind_max[idx + e];
			}
		}
		__syncthreads();
	}

	if(idx == 0) {
		min = s_ind_max[0] * t + machine_current_index[s_ind_max[0]];
		imin = s_ind_max[0];

		completion_times[imin] = s_comp_times[0];
		task_map[task_index[min]] = imin;
	}

	//__syncthreads();

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
	a = scanf("%d", &m);


	//uint mem_size_seg 				= sizeof(int) * (m + 1);
	uint mem_size_seg 				= sizeof(int) * (m);
	uint mem_size_machines 			= sizeof(float) * (m * t);
	uint mem_size_task_index		= sizeof(uint) * (m * t);

	uint mem_size_completion_times 	= sizeof(float) * (m);
	uint mem_size_machine_cur_index = sizeof(uint) * (m);

	uint mem_size_task_map 			= sizeof(int) * (t);

	int *segments 				= (int   *) malloc(mem_size_seg);
	float *machines				= (float *) malloc(mem_size_machines);
	uint *task_index 			= (uint  *) malloc(mem_size_task_index);

	float *completion_times 	= (float *) malloc(mem_size_completion_times);
	uint *machine_cur_index = (uint  *) malloc(mem_size_machine_cur_index);

	int *task_map 				= (int  *) malloc(mem_size_task_map);

	float aux;
	for (int i = 0; i < t; i++) {
		for (int j = 0; j < m; j++) {
			int a = scanf("%f", &aux);

			task_index[j * t + i] = i;
			machines[j * t + i] = aux;
			segments[j] = j*t;


			completion_times[j] = 0;
			machine_cur_index[j] = 0;
		}
		task_map[i] = -1;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *d_completion_times;
	uint *d_machine_cur_index;
	int *d_task_map;
	//bool *d_task_deleted;

	cudaTest(cudaMalloc((void **) &d_completion_times, mem_size_completion_times));
	cudaTest(cudaMalloc((void **) &d_machine_cur_index, mem_size_machine_cur_index));
	cudaTest(cudaMalloc((void **) &d_task_map, mem_size_task_map));

	// copy host memory to device
	cudaTest(cudaMemcpy(d_completion_times, completion_times, mem_size_completion_times, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_machine_cur_index, machine_cur_index, mem_size_machine_cur_index, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_map, task_map, mem_size_task_map, cudaMemcpyHostToDevice));

	uint *d_task_index, *reduced_indexes;
	int *d_segments;
	float *d_machines, *reduced_times;

	cudaTest(cudaMalloc((void **) &d_segments, mem_size_seg));
	cudaTest(cudaMalloc((void **) &d_machines, mem_size_machines));
	cudaTest(cudaMalloc((void **) &d_task_index, mem_size_task_index));

	// copy host memory to device
	cudaTest(cudaMemcpy(d_segments, segments, mem_size_seg, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_machines, machines, mem_size_machines, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_index, task_index, mem_size_task_index, cudaMemcpyHostToDevice));

	int block, grid;
	
	if(BLOCK_SIZE > m){
		block = m;
		grid = 1;
	}
	else {
		block = BLOCK_SIZE;
		grid = m/BLOCK_SIZE;
	}

	dim3 dimBlock(block);
	dim3 dimGrid(grid);
	cudaTest(cudaMalloc((void **) &reduced_indexes, sizeof(uint) * (grid)));
	cudaTest(cudaMalloc((void **) &reduced_times, sizeof(float) * (grid)));


	cudaEventRecord(start);
	mgpu::standard_context_t context;
	mgpu::segmented_sort(d_machines, d_task_index, m * t, d_segments, m, mgpu::less_t<float>(), context);

	for(int k = 0; k < t; k++) {

		min_min_sorted<<<dimGrid, dimBlock, block * sizeof(float) + block * sizeof(int) >>>(d_machines, d_task_index,
						d_completion_times,	d_task_map, d_machine_cur_index, reduced_times, reduced_indexes, m, t);

		blocks_reduction<<<	1, grid, grid * sizeof(float) + grid * sizeof(int) >>>(d_machines, d_task_index,
				d_completion_times,	d_task_map, d_machine_cur_index, reduced_times, reduced_indexes, m, t);
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

	cudaFree(d_segments);
	cudaFree(d_machines);
	cudaFree(d_task_index);
	cudaFree(d_completion_times);
	cudaFree(d_task_map);

	if (ELAPSED_TIME != 1) {
		//print(machines, m, t);
		//print(task_index, m, t);
		print(completion_times, m);
	}

	free(task_map);
	free(machines);
	free(task_index);
	free(segments);
	free(completion_times);
	free(machine_cur_index);

	return 0;
}
