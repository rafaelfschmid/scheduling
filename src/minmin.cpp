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

struct Task {
	uint machine;
	uint time;

	Task(uint id, uint time) {
		this->machine = id;
		this->time = time;
	}
};

void kernel(Task* tasks, uint* machine_index, int n, int m, int p, int q) {
	int d = 1 << (p - q);

	for (int j = 0; j < m; j++) {
		for (int i = 0; i < n; i++) {
			bool up = ((i >> p) & 2) == 0;

			if ((i & d) == 0
					&& (tasks[j * n + i].time
							> tasks[(j * n + i) | (j * n + d)].time) == up) {
				Task t = tasks[j * n + i];
				tasks[j * n + i] = tasks[(j * n + i) | (j * n + d)];
				tasks[(j * n + i) | (j * n + d)] = t;

				uint aux = machine_index[j * n + tasks[j * n + i].machine];
				machine_index[j * n + tasks[j * n + i].machine] =
						machine_index[j * n
								+ tasks[(j * n + i) | (j * n + d)].machine];
				machine_index[j * n + tasks[(j * n + i) | (j * n + d)].machine] =
						aux;
			}
		}
	}
}

void sorting(int logn, Task* a, uint* machine_index, int n, int m) {

	for (int p = 0; p < logn; p++) {
		for (int q = 0; q <= p; q++) {
			kernel(a, machine_index, n, m, p, q);
		}
	}
}

void min_min(uint* tasks, uint* completion_times, bool* task_map, bool* task_deleted, int n, int m) {

	min = 0;
	jmin = 0;
	for(int k = 0; k < m; k++) {

		completion_times[min] += tasks[min];
		task_map[min] = true;

		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m; i++) {
				if (!task_deleted[j] && completion_times[j * n + i] + tasks[j * n + i] < completion_times[min]) {
					completion_times[min] -= tasks[min];
					task_map[min] = false;

					jmin = j;
					min = j * n + i;
					completion_times[min] += tasks[min];
					task_map[min] = true;
				} else
					nextIt = j * n + i;
			}
		}
		min = nextIt;
		task_deleted[jmin] = true;
	}

}

void print(Task* vec, uint n, uint m) {
	std::cout << "\n";
	for (uint j = 0; j < m; j++) {
		for (uint i = 0; i < n; i++) {
			std::cout << "id=" << vec[j * n + i].machine << " time="
					<< vec[j * n + i].time << "\t";
		}
		std::cout << "\n";
	}

}

void print(uint* vec, uint n, uint m) {
	std::cout << "\n";
	for (uint j = 0; j < m; j++) {
		for (uint i = 0; i < n; i++) {
			std::cout << vec[j * n + i] << "\t";
		}
		std::cout << "\n";
	}

}

int main(int argc, char **argv) {
	uint num_of_elements;
	uint i;
	int n, m;

	scanf("%d", &m);
	scanf("%d", &n);

	uint mem_size = sizeof(Task) * (n * m);
	//Task *tasks = (Task *) malloc(mem_size);
	bool *task_map = (bool *) malloc(mem_size);
	//uint *machine_index = (uint *) malloc(mem_size);
	uint *tasks = (uint *) malloc(mem_size);
	uint *completion_times = (uint *) malloc(mem_size);

	uint aux;
	for (int j = 0; j < m; j++) {
		for (int i = 0; i < n; i++) {
			scanf("%d", &aux);
			//tasks[j*n + i].machine = i;
			//tasks[j*n + i].time = aux;
			tasks[j * n + i] = aux;
			//machine_index[j*n + i] = i;
			task_map[j * n + i] = false;
			completion_times[j * n + i] = 0;
		}
	}

	int logn = 0;
	for (int i = 1; i < n; i *= 2) {
		logn++;
	}

	printf("logn=%d\n", logn);

	//sorting(logn, tasks, machine_index, n, m);
	//min_min(tasks, machine_index, task_map, n, m);
	min_min(tasks, completion_times, task_map, n, m);

	print(tasks, n, m);
	print(machine_index, n, m);

	return 0;
}
