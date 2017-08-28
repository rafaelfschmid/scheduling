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

#include <cub/util_allocator.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <utility>
#include <iostream>
#include <bitset>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cuda.h>
//#include <cstdlib>
#include <iostream>
#include <chrono>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef EXECUTIONS
#define EXECUTIONS 11
#endif


using namespace std::chrono;
using namespace cub;

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

void min_min_sorted(float* machines, uint* task_index, float* completion_times, bool* task_map,
		bool* task_deleted, uint* machine_current_index, int m, int t) {

	uint min = 0;
	uint imin = 0;
	float min_value;

	for(int k = 0; k < t; k++) {

		min_value = std::numeric_limits<float>::max();

		for (int i = 0; i < m; i++) {

			int j = machine_current_index[i];
			while (task_deleted[task_index[i * t + j]]) {
				j++;
			}
			machine_current_index[i] = j;

			if (completion_times[i] + machines[i * t + j] < min_value) {
				min = i * t + j;
				imin = i;
				min_value = completion_times[imin] + machines[min];
			}
		}
		task_deleted[task_index[min]] = true;
		task_map[task_index[min] * m + imin] = true;
		completion_times[imin] = min_value;
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

	uint mem_size_seg = sizeof(int) * (m + 1);
	uint mem_size_machines = sizeof(float) * (m * t);
	uint mem_size_task_index = sizeof(uint) * (m * t);

	bool *task_deleted = (bool *) malloc(sizeof(bool) * t);
	bool *task_map = (bool *) malloc(sizeof(bool) * (t * m));

	int *segments = (int *) malloc(mem_size_seg);
	float *machines = (float *) malloc(mem_size_machines);
	uint *task_index = (uint *) malloc(mem_size_task_index);

	float *completion_times = (float *) malloc(sizeof(float) * (m));
	uint *machine_current_index = (uint *) malloc(sizeof(uint) * (m));

	float aux;
	for (int i = 0; i < t; i++) {
		for (int j = 0; j < m; j++) {
			int a = scanf("%f", &aux);

			task_index[j * t + i] = i;
			machines[j * t + i] = aux;
			segments[j] = j*t;

			task_map[i * m + j] = false;
			completion_times[j] = 0;
			machine_current_index[j] = 0;
		}
		task_deleted[i] = false;
	}
	segments[m] = m*t;

	//print(machines, m, t);
	//print(task_index, m, t);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_task_index, *d_task_index_out;
	int *d_segments;
	float *d_machines, *d_machines_out;
	void *d_temp = NULL;
	size_t temp_bytes = 0;

	cudaTest(cudaMalloc((void **) &d_segments, mem_size_seg));
	cudaTest(cudaMalloc((void **) &d_machines, mem_size_machines));
	cudaTest(cudaMalloc((void **) &d_machines_out, mem_size_machines));
	cudaTest(cudaMalloc((void **) &d_task_index, mem_size_task_index));
	cudaTest(cudaMalloc((void **) &d_task_index_out, mem_size_task_index));

	cudaEventRecord(start);
	// copy host memory to device
	cudaTest(cudaMemcpy(d_segments, segments, mem_size_seg, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_machines, machines, mem_size_machines, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_index, task_index, mem_size_task_index, cudaMemcpyHostToDevice));

	cub::DeviceSegmentedRadixSort::SortPairs(d_temp, temp_bytes, d_task_index, d_task_index,
			d_task_index, d_task_index_out, m * t,	m, d_segments, d_segments + 1);
	cudaTest(cudaMalloc((void **) &d_temp, temp_bytes));
	cub::DeviceSegmentedRadixSort::SortPairs(d_temp, temp_bytes, d_machines, d_machines_out,
			d_task_index, d_task_index_out, m * t,	m, d_segments, d_segments + 1);

	cudaTest(cudaMemcpy(task_index, d_task_index_out, mem_size_task_index, cudaMemcpyDeviceToHost));
	cudaTest(cudaMemcpy(machines, d_machines_out, mem_size_machines, cudaMemcpyDeviceToHost));

	min_min_sorted(machines, task_index, completion_times, task_map, task_deleted, machine_current_index, m, t);

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

	cudaFree(d_segments);
	cudaFree(d_machines);
	cudaFree(d_machines_out);
	cudaFree(d_task_index);
	cudaFree(d_task_index_out);
	cudaFree(d_temp);

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
	free(machine_current_index);

	return 0;
}
