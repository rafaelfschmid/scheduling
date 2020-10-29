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

#include <cub/util_type.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

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

struct Reduce {
	uint t;
	uint m;
	float value;

	Reduce(uint index_t, uint index_m, float value) {
		this->t = index_t;
		this->m = index_m;
		this->value = value;
	}

	Reduce() {
		this->t = 0;
		this->m = 0;
		this->value = 0.0;
	}

	static Reduce Max(){
		float max = std::numeric_limits<float>::max();
		return Reduce(0, 0, max);
	}
	//__host__ __device__
	//bool operator() (const Reduce& i,const Reduce& j) const { return (i.value < j.value); }

	__host__ __device__
	bool operator<(const Reduce& x) const
	{
		return this->value < x.value;
	}

	__host__ __device__
	bool operator>(const Reduce& x) const
	{
		return this->value > x.value;
	}
};

void print(Reduce* vec, uint t, uint m) {
	std::cout << "\n";
	for (uint i = 0; i < t; i++) {
		for (uint j = 0; j < m; j++) {
			std::cout << "t=" << vec[i * m + j].t << " m="
					<< vec[i * m + j].m << " value="
					<< vec[i * m + j].value << "\t||";
		}
		std::cout << "\n";
	}

}

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

__global__ void calc_completion_times(float* machines, float* completion_times, bool *task_deleted,
		Reduce* completion_aux, int m, int t, float MAX_FLOAT) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(!task_deleted[i]) {
		for (int j = 0; j < m; j++) {
			completion_aux[j * t + i].t = i;
			completion_aux[j * t + i].m = j;
			completion_aux[j * t + i].value = completion_times[j] + machines[j * t + i];
		}
	}
	else {
		for (int j = 0; j < m; j++) {
			completion_aux[j * t + i].t = i;
			completion_aux[j * t + i].m = j;
			completion_aux[j * t + i].value = MAX_FLOAT;
		}
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
	a = scanf("%d", &m);

	uint mem_size_machines 			= sizeof(float) * (m * t);
	uint mem_size_completion_times 	= sizeof(float) * (m);
	uint mem_size_task_deleted 		= sizeof(bool) * (t);
	uint mem_size_task_map 			= sizeof(bool) * (m * t);
	uint mem_size_completion_aux	= sizeof(Reduce) * (m * t);

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

	float *d_machines, *d_completion_times;
	bool *d_task_deleted, *d_task_map;
	Reduce *d_completion_aux;
	Reduce *d_min, *min;
	min = (Reduce *) malloc(sizeof(Reduce));
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	float MAX_FLOAT = std::numeric_limits<float>::max();

	cudaTest(cudaMalloc((void **) &d_min, sizeof(Reduce)));
	cudaTest(cudaMalloc((void **) &d_machines, mem_size_machines));
	cudaTest(cudaMalloc((void **) &d_completion_times, mem_size_completion_times));
	cudaTest(cudaMalloc((void **) &d_task_deleted, mem_size_task_deleted));
	cudaTest(cudaMalloc((void **) &d_task_map, mem_size_task_map));

	cudaTest(cudaMalloc((void **) &d_completion_aux, mem_size_completion_aux));

	// copy host memory to device
	cudaTest(cudaMemcpy(d_machines, machines, mem_size_machines, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_completion_times, completion_times, mem_size_completion_times, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_deleted, task_deleted, mem_size_task_deleted, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_map, task_map, mem_size_task_map, cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	for(int k = 0; k < t; k++) {
		int dim = t/BLOCK_SIZE;
		dim3 dimBlock(BLOCK_SIZE);
		dim3 dimGrid(dim);
		calc_completion_times<<<dimGrid, dimBlock>>>
				(d_machines, d_completion_times, d_task_deleted, d_completion_aux, m, t, MAX_FLOAT);

		thrust::device_vector<Reduce>::iterator iter = thrust::max_element(d_completion_aux, d_completion_aux+(m*t),);
		/*cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_completion_aux, d_min, m * t);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);	// Allocate temporary storage
		cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_completion_aux, d_min, m * t);*/

		cudaTest(cudaMemcpy(min, d_min, sizeof(Reduce), cudaMemcpyDeviceToHost));
		printf("min=%f t=%d m=%d\n", min->value, min->t, min->m);
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

	cudaFree(d_completion_aux);

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
