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
#include <algorithm>    // std::sort
#include <limits>
#include <cuda.h>



struct Task {
	uint id;
	float time;

	Task(uint id, float time) {
		this->id = id;
		this->time = time;
	}

	Task() {
		this->id = 0;
		this->time = 0;
	}

	bool operator() (Task i,Task j) { return (i.time < j.time); }
} myobj;

bool taskComparation (Task i,Task j) { return (i.time < j.time); }

void print(Task* vec, uint t, uint m) {
	std::cout << "\n";
	for (uint i = 0; i < t; i++) {
		for (uint j = 0; j < m; j++) {
			std::cout << "id=" << vec[i * m + j].id << " time="
					<< vec[i * m + j].time << "\n";
		}
		std::cout << "\n";
	}

}

void segmented_sorting(Task* machines, int m, int t) {

	for(int i = 0; i < m; i++) {
		int j = i*t;
		std::stable_sort (&machines[j], &machines[j]+t, Task());
	}
}

void min_min_sorted(Task* machines, float* completion_times, int* task_map,
		uint* machine_current_index, int m, int t) {

	uint min = 0;
	uint imin = 0;
	float min_value;

	for(int k = 0; k < t; k++) {

		min_value = std::numeric_limits<float>::max();

		for (int i = 0; i < m; i++) {

			int j = machine_current_index[i];
			while (task_map[machines[i * t + j].id] != -1) {
				j++;
			}
			machine_current_index[i] = j;

			if (completion_times[i] + machines[i * t + j].time < min_value) {
				min = i * t + j;
				imin = i;
				min_value = completion_times[imin] + machines[min].time;
			}
		}
		task_map[machines[min].id] = imin;
		completion_times[imin] = min_value;
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

	int *task_map = (int *) malloc(sizeof(int) * (t));
	Task *machines = (Task *) malloc(sizeof(Task) * (m * t));
	float *completion_times = (float *) malloc(sizeof(float) * (m));
	uint *machine_current_index = (uint *) malloc(sizeof(uint) * (m));

	float aux;
	for (int i = 0; i < t; i++) {
		for (int j = 0; j < m; j++) {
			int a = scanf("%f", &aux);
			machines[j * t + i].id = i;
			machines[j * t + i].time = aux;
			completion_times[j] = 0;
			machine_current_index[j] = 0;
		}
		task_map[i] = -1;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	segmented_sorting(machines, m, t);
	//print(machines, t, m);
	min_min_sorted(machines, completion_times, task_map, machine_current_index, m, t);
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

	free(task_map);
	free(machines);
	free(completion_times);
	free(machine_current_index);

	return 0;
}

