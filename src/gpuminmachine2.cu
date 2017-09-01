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
} myobj;

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

	extern __shared__ Reduce s_comp_times[];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	int tIdrow = threadIdx.y;
	int tIdcol = threadIdx.x;

	int iglobal = row * t + col;
	int ilocal = tIdrow * blockDim.x + tIdcol;

	if(!task_deleted[col]) {
		s_comp_times[ilocal].t = col;
		s_comp_times[ilocal].m = row;
		s_comp_times[ilocal].value = completion_times[row] + machines[iglobal];
	}
	else {
		s_comp_times[ilocal].t = col;
		s_comp_times[ilocal].m = row;
		s_comp_times[ilocal].value = MAX_FLOAT;
	}

	__syncthreads();

	for(int e = blockDim.x/2; e > 0; e/=2)
	{
		if (tIdcol < e) {
			if ((s_comp_times[ilocal + e].value < s_comp_times[ilocal].value)
					|| (s_comp_times[ilocal + e].value == s_comp_times[ilocal].value
							&& s_comp_times[ilocal + e].t < s_comp_times[ilocal].t)) {
				s_comp_times[ilocal].t = s_comp_times[ilocal + e].t;
				s_comp_times[ilocal].m = s_comp_times[ilocal + e].m;
				s_comp_times[ilocal].value = s_comp_times[ilocal + e].value;
			}
		}
		__syncthreads();
	}

	if(tIdcol == 0) {
		for(int e = blockDim.y/2; e > 0; e/=2)
		{
			if (tIdrow < e) {
				if ((s_comp_times[ilocal + e * blockDim.x].value < s_comp_times[ilocal].value)
						|| (s_comp_times[ilocal + e * blockDim.x].value == s_comp_times[ilocal].value
								&& s_comp_times[ilocal + e * blockDim.x].t < s_comp_times[ilocal].t)) {
					s_comp_times[ilocal].t = s_comp_times[ilocal + e * blockDim.x].t;
					s_comp_times[ilocal].m = s_comp_times[ilocal + e * blockDim.x].m;
					s_comp_times[ilocal].value = s_comp_times[ilocal + e * blockDim.x].value;
				}
			}
			__syncthreads();
		}
	}

	if(tIdrow == 0 && tIdcol == 0) {
		iglobal = blockIdx.y * gridDim.x + blockIdx.x;
		completion_aux[iglobal].t = s_comp_times[0].t;
		completion_aux[iglobal].m = s_comp_times[0].m;
		completion_aux[iglobal].value = s_comp_times[0].value;
	}
}

__global__ void reduction_two_dimensional(Reduce* completion_aux, int t) {

	extern __shared__ Reduce s_comp_times[];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tIdrow = threadIdx.y;
	int tIdcol = threadIdx.x;

	int iglobal = row * t + col;
	int ilocal = tIdrow * blockDim.x + tIdcol;

	s_comp_times[ilocal].t = completion_aux[iglobal].t;
	s_comp_times[ilocal].m = completion_aux[iglobal].m;
	s_comp_times[ilocal].value = completion_aux[iglobal].value;

	__syncthreads();

	for(int e = blockDim.x/2; e > 0; e/=2)
	{
		if (tIdcol < e) {
			if ((s_comp_times[ilocal + e].value < s_comp_times[ilocal].value)
					|| (s_comp_times[ilocal + e].value == s_comp_times[ilocal].value
							&& s_comp_times[ilocal + e].t < s_comp_times[ilocal].t)) {
				s_comp_times[ilocal].t = s_comp_times[ilocal + e].t;
				s_comp_times[ilocal].m = s_comp_times[ilocal + e].m;
				s_comp_times[ilocal].value = s_comp_times[ilocal + e].value;
			}
		}
		__syncthreads();
	}

	if(tIdcol == 0) {
		for(int e = blockDim.y/2; e > 0; e/=2)
		{
			if (tIdrow < e) {
				if ((s_comp_times[ilocal + e * BLOCK_SIZE].value < s_comp_times[ilocal].value)
						|| (s_comp_times[ilocal + e * BLOCK_SIZE].value == s_comp_times[ilocal].value
								&& s_comp_times[ilocal + e * BLOCK_SIZE].t < s_comp_times[ilocal].t)) {
					s_comp_times[ilocal].t = s_comp_times[ilocal + e * BLOCK_SIZE].t;
					s_comp_times[ilocal].m = s_comp_times[ilocal + e * BLOCK_SIZE].m;
					s_comp_times[ilocal].value = s_comp_times[ilocal + e * BLOCK_SIZE].value;
				}
			}
			__syncthreads();
		}
	}

	if(tIdrow == 0 && tIdcol == 0) {
		iglobal = blockIdx.y * gridDim.x + blockIdx.x;
		completion_aux[iglobal].t = s_comp_times[0].t;
		completion_aux[iglobal].m = s_comp_times[0].m;
		completion_aux[iglobal].value = s_comp_times[0].value;
	}
}

__global__ void reduction(Reduce* d_completion_aux) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tId = threadIdx.x;

	extern __shared__ Reduce s_comp_times[];

	s_comp_times[tId].t = d_completion_aux[i].t;
	s_comp_times[tId].m = d_completion_aux[i].m;
	s_comp_times[tId].value = d_completion_aux[i].value;

	__syncthreads();

	for(int e = blockDim.x/2; e > 0; e/=2)
	{
		if (tId < e) {
			if ((s_comp_times[tId + e].value < s_comp_times[tId].value)
					|| (s_comp_times[tId + e].value == s_comp_times[tId].value
							&& s_comp_times[tId + e].t < s_comp_times[tId].t)) {
				s_comp_times[tId].t = s_comp_times[tId + e].t;
				s_comp_times[tId].m = s_comp_times[tId + e].m;
				s_comp_times[tId].value = s_comp_times[tId + e].value;
			}
		}
		__syncthreads();
	}

	if(tId == 0) {
		d_completion_aux [blockIdx.x].t = s_comp_times[0].t;
		d_completion_aux[blockIdx.x].m = s_comp_times[0].m;
		d_completion_aux[blockIdx.x].value = s_comp_times[0].value;
	}
}

__global__ void block_reduction(float* completion_times, bool* task_map, bool* task_deleted,
		Reduce* d_completion_aux, int t) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tId = threadIdx.x;

	extern __shared__ Reduce s_comp_times[];

	s_comp_times[tId].t = d_completion_aux[i].t;
	s_comp_times[tId].m = d_completion_aux[i].m;
	s_comp_times[tId].value = d_completion_aux[i].value;

	__syncthreads();

	for(int e = blockDim.x/2; e > 0; e/=2)
	{
		if (tId < e) {
			if ((s_comp_times[tId + e].value < s_comp_times[tId].value)
					|| (s_comp_times[tId + e].value == s_comp_times[tId].value
							&& s_comp_times[tId + e].t < s_comp_times[tId].t)) {
				s_comp_times[tId].t = s_comp_times[tId + e].t;
				s_comp_times[tId].m = s_comp_times[tId + e].m;
				s_comp_times[tId].value = s_comp_times[tId + e].value;
			}
		}
		__syncthreads();
	}

	if(tId == 0) {
		task_deleted[ s_comp_times[0].t ] = true;
		task_map[ s_comp_times[0].m * t + s_comp_times[0].t ] = true;
		completion_times[ s_comp_times[0].m ] = s_comp_times[0].value;
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

	int dimCol = (t-1)/BLOCK_SIZE + 1;
	int dimRow = (m-1)/BLOCK_SIZE + 1;

	uint mem_size_completion_aux	= sizeof(Reduce) * (dimCol * dimRow);

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
	float MAX_FLOAT = std::numeric_limits<float>::max();

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
		dimCol = (t-1)/BLOCK_SIZE + 1;
		dimRow = (m-1)/BLOCK_SIZE + 1;

		dim3 dimB1(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimG1(dimCol, dimRow);
		calc_completion_times<<<dimG1, dimB1, BLOCK_SIZE * BLOCK_SIZE * sizeof(Reduce) >>>
				(d_machines, d_completion_times, d_task_deleted, d_completion_aux, m, t, MAX_FLOAT);



/*		Reduce *completion_aux 	= (Reduce *) malloc(mem_size_completion_aux);
		cudaTest(cudaMemcpy(completion_aux, d_completion_aux, mem_size_completion_aux, cudaMemcpyDeviceToHost));
		print(completion_aux, dimRow, dimCol);*/

/*		dimCol = (dimCol-1)/BLOCK_SIZE + 1;
		dimRow = (dimRow-1)/BLOCK_SIZE + 1;

		for( ; dimRow > BLOCK_SIZE; dimRow/=BLOCK_SIZE) {
			dim3 dimG2(dimCol, dimRow);

			reduction_two_dimensional<<<dimG2, dimB1, BLOCK_SIZE * BLOCK_SIZE * sizeof(Reduce) >>>	(d_completion_aux, t);

			dimCol /= BLOCK_SIZE;
		}

		dim3 dimB3(BLOCK_SIZE, dimRow);
		dim3 dimG3(dimCol);
		reduction_two_dimensional<<<dimG3, dimB3, dimRow * BLOCK_SIZE * sizeof(Reduce) >>> (d_completion_aux, t);

		cudaTest(cudaMemcpy(completion_aux, d_completion_aux, mem_size_completion_aux, cudaMemcpyDeviceToHost));
		print(completion_aux, dimRow, dimCol);

		for( ; dimCol > BLOCK_SIZE; dimCol/=BLOCK_SIZE) {
			dim3 dimB4(BLOCK_SIZE);
			dim3 dimG4(dimCol);

			reduction<<<dimG4, dimB4, BLOCK_SIZE * sizeof(Reduce) >>> (d_completion_aux);
		}

		dim3 dimB5(dimCol);
		dim3 dimG5(1);
		block_reduction<<<dimG5, dimB5, dimCol * sizeof(Reduce) >>> (d_completion_times, d_task_map, d_task_deleted,
				d_completion_aux, t);*/
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

/*
 *
 *
__global__ void block_reduction_two_dimensional(float* completion_times,
		bool* task_map, bool* task_deleted, Reduce* completion_aux, int t) {

	extern __shared__ Reduce s_comp_times[];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tIdrow = threadIdx.y;
	int tIdcol = threadIdx.x;

	int iglobal = row * t + col;
	int ilocal = tIdrow * blockDim.x + tIdcol;

	s_comp_times[ilocal].t = completion_aux[iglobal].t;
	s_comp_times[ilocal].m = completion_aux[iglobal].m;
	s_comp_times[ilocal].value = completion_aux[iglobal].value;

	__syncthreads();

	for(int e = blockDim.x/2; e > 0; e/=2)
	{
		if (tIdcol < e) {
			if ((s_comp_times[ilocal + e].value < s_comp_times[ilocal].value)
					|| (s_comp_times[ilocal + e].value == s_comp_times[ilocal].value
							&& s_comp_times[ilocal + e].t < s_comp_times[ilocal].t)) {
				s_comp_times[ilocal].t = s_comp_times[ilocal + e].t;
				s_comp_times[ilocal].m = s_comp_times[ilocal + e].m;
				s_comp_times[ilocal].value = s_comp_times[ilocal + e].value;
			}
		}
		__syncthreads();
	}

	if(tIdcol == 0) {
		for(int e = blockDim.y/2; e > 0; e/=2)
		{
			if (tIdrow < e) {
				if ((s_comp_times[ilocal + e * BLOCK_SIZE].value < s_comp_times[ilocal].value)
						|| (s_comp_times[ilocal + e * BLOCK_SIZE].value == s_comp_times[ilocal].value
								&& s_comp_times[ilocal + e * BLOCK_SIZE].t < s_comp_times[ilocal].t)) {
					s_comp_times[ilocal].t = s_comp_times[ilocal + e * BLOCK_SIZE].t;
					s_comp_times[ilocal].m = s_comp_times[ilocal + e * BLOCK_SIZE].m;
					s_comp_times[ilocal].value = s_comp_times[ilocal + e * BLOCK_SIZE].value;
				}
			}
			__syncthreads();
		}
	}

	if(tIdrow == 0 && tIdcol == 0) {
		task_deleted[ s_comp_times[0].t ] = true;
		task_map[ s_comp_times[0].m * t + s_comp_times[0].t ] = true;
		completion_times[ s_comp_times[0].m ] = s_comp_times[0].value;
	}
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

__global__ void reduction(Reduce* d_completion_aux) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tId = threadIdx.x;

	extern __shared__ Reduce s_comp_times[];

	s_comp_times[tId].t = d_completion_aux[i].t;
	s_comp_times[tId].m = d_completion_aux[i].m;
	s_comp_times[tId].value = d_completion_aux[i].value;

	__syncthreads();

	for(int e = blockDim.x/2; e > 0; e/=2)
	{
		if (tId < e) {
			if ((s_comp_times[tId + e].value < s_comp_times[tId].value)
					|| (s_comp_times[tId + e].value == s_comp_times[tId].value
							&& s_comp_times[tId + e].t < s_comp_times[tId].t)) {
				s_comp_times[tId].t = s_comp_times[tId + e].t;
				s_comp_times[tId].m = s_comp_times[tId + e].m;
				s_comp_times[tId].value = s_comp_times[tId + e].value;
			}
		}
		__syncthreads();
	}

	if(tId == 0) {
		d_completion_aux [blockIdx.x].t = s_comp_times[0].t;
		d_completion_aux[blockIdx.x].m = s_comp_times[0].m;
		d_completion_aux[blockIdx.x].value = s_comp_times[0].value;
	}
}

__global__ void block_reduction(float* completion_times, bool* task_map, bool* task_deleted,
		Reduce* d_completion_aux, int t) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tId = threadIdx.x;

	extern __shared__ Reduce s_comp_times[];

	s_comp_times[tId].t = d_completion_aux[i].t;
	s_comp_times[tId].m = d_completion_aux[i].m;
	s_comp_times[tId].value = d_completion_aux[i].value;

	__syncthreads();

	for(int e = blockDim.x/2; e > 0; e/=2)
	{
		if (tId < e) {
			if ((s_comp_times[tId + e].value < s_comp_times[tId].value)
					|| (s_comp_times[tId + e].value == s_comp_times[tId].value
							&& s_comp_times[tId + e].t < s_comp_times[tId].t)) {
				s_comp_times[tId].t = s_comp_times[tId + e].t;
				s_comp_times[tId].m = s_comp_times[tId + e].m;
				s_comp_times[tId].value = s_comp_times[tId + e].value;
			}
		}
		__syncthreads();
	}

	if(tId == 0) {
		task_deleted[ s_comp_times[0].t ] = true;
		task_map[ s_comp_times[0].m * t + s_comp_times[0].t ] = true;
		completion_times[ s_comp_times[0].m ] = s_comp_times[0].value;
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
	float MAX_FLOAT = std::numeric_limits<float>::max();

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
		int dimG = t * m;

		int dim = t/BLOCK_SIZE;
		dim3 dimBlock(BLOCK_SIZE);
		dim3 dimGrid(dim);
		calc_completion_times<<<dimGrid, dimBlock>>>
				(d_machines, d_completion_times, d_task_deleted, d_completion_aux, m, t, MAX_FLOAT);

		for( ; dimG > BLOCK_SIZE; dimG/=BLOCK_SIZE) {
			dim3 block(BLOCK_SIZE);
			dim3 grid_b(dimG/BLOCK_SIZE);
			reduction<<<grid_b, block, BLOCK_SIZE * sizeof(Reduce) >>>
				(d_completion_aux);
		}

		dim3 block(dimG);
		dim3 grid_b(1);
		block_reduction<<<grid_b, block, dimG * sizeof(Reduce) >>> (d_completion_times, d_task_map, d_task_deleted,
				d_completion_aux, t);
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
 *
 */
