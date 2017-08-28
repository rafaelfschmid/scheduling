/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <cuda.h>

void min_min(float* tasks, float* completion_times, bool* task_map, bool* task_deleted, int t, int m) {

	uint min = 0;
	uint imin = 0;
	uint jmin = 0;
	float min_value = 0;
	for(int k = 0; k < t; k++) {

		min_value = std::numeric_limits<float>::max();

		for (int i = 0; i < t; i++) {
			if (!task_deleted[i]) {
				for (int j = 0; j < m; j++) {

					if (completion_times[j] + tasks[i * m + j] < min_value) {
						imin = i;
						jmin = j;
						min = imin * m + jmin;
						min_value = completion_times[jmin] + tasks[min];
					}
				}
			}
		}
		task_deleted[imin] = true;
		task_map[min] = true;
		completion_times[jmin] = min_value;
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

int main(int argc, char **argv) {
	int t, m;

	if (argc < 3) {
			printf("Parameters missing: <number of tasks> <number of machines>\n\n");
			return 0;
	}

	t = atoi(argv[1]);
	m = atoi(argv[2]);
	//std::cout << "t=" << t << " m=" << m << "\n";

	bool *task_deleted = (bool *) malloc(sizeof(bool) * t);
	bool *task_map = (bool *) malloc(sizeof(bool) * (t * m));
	float *tasks = (float *) malloc(sizeof(float) * (t * m));
	float *completion_times = (float *) malloc(sizeof(float) * (m));

	float aux;
	for (int i = 0; i < t; i++) {
		for (int j = 0; j < m; j++) {
			int a = scanf("%f", &aux);
			tasks[i * m + j] = aux;
			task_map[i * m + j] = false;
			completion_times[j] = 0;
		}
		task_deleted[i] = false;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	min_min(tasks, completion_times, task_map, task_deleted, t, m);
	cudaEventRecord(stop);

	if (ELAPSED_TIME == 1) {
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << milliseconds << "\n";
	}
	else {
		//print(tasks, t, m);
		print(completion_times, m);
		//print(task_deleted, t);
		//print(task_map, t, m);
	}

	free(task_deleted);
	free(task_map);
	free(tasks);
	free(completion_times);

	return 0;
}

