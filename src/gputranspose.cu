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

__global__ void transpose(const float *machines, float *machines_out,
		const uint *task_index, uint* task_index_out, int t, int m)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
  	int col = blockIdx.x * blockDim.x + threadIdx.x;

  	//for (int e = 0; e < t; e += BLOCK_SIZE)
  	machines_out[col * t + row] = machines[row * m + col];
  	task_index_out[col * t + row] = task_index[row * m + col];

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

	uint mem_size_machines 			= sizeof(float) * (m * t);
	uint mem_size_task_index		= sizeof(uint) * (m * t);

	float *machines				= (float *) malloc(mem_size_machines);
	uint *task_index 			= (uint  *) malloc(mem_size_task_index);

	float aux;
	for (int i = 0; i < t; i++) {
		for (int j = 0; j < m; j++) {
			int a = scanf("%f", &aux);

			task_index[i * m + j] = j;
			machines[i * m + j] = aux;
		}
	}

	print(machines, t, m);
	print(task_index, t, m);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_task_index, *d_task_index_out;
	float *d_machines, *d_machines_out;

	cudaTest(cudaMalloc((void **) &d_machines, mem_size_machines));
	cudaTest(cudaMalloc((void **) &d_machines_out, mem_size_machines));
	cudaTest(cudaMalloc((void **) &d_task_index, mem_size_task_index));
	cudaTest(cudaMalloc((void **) &d_task_index_out, mem_size_task_index));

	// copy host memory to device
	cudaTest(cudaMemcpy(d_machines, machines, mem_size_machines, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_task_index, task_index, mem_size_task_index, cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 grid(m/BLOCK_SIZE,t/BLOCK_SIZE,1);
	transposeCoalesced<<<block, grid>>>(d_machines, d_machines_out, d_task_index, d_task_index_out, t, m);
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

	cudaTest(cudaMemcpy(machines, d_machines_out, mem_size_machines, cudaMemcpyDeviceToHost));
	cudaTest(cudaMemcpy(task_index, d_task_index_out, mem_size_task_index, cudaMemcpyDeviceToHost));

	cudaFree(d_machines);
	cudaFree(d_machines_out);
	cudaFree(d_task_index);
	cudaFree(d_task_index_out);

	if (ELAPSED_TIME != 1) {
		print(machines, m, t);
		print(task_index, m, t);
	}

	free(machines);
	free(task_index);

	return 0;
}
