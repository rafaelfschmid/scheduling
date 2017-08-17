#include <time.h>
#include <algorithm>
#include <math.h>
#include <cstdlib>
#include <stdio.h>

#ifndef EXP_BITS_SIZE
#define EXP_BITS_SIZE 10
#endif

void segments_gen(int num_elements, int num_segments) {
	int* h_data = new int[num_segments];
	int *h_aux = new int[num_elements];

	h_data[0] = 0;
	h_aux[0] = 1;
	int i = 1;
	while(i < num_segments)
	{
		int x = rand() % num_elements;
		if(h_aux[x] == 0) {
			h_aux[x] = 1;
			h_data[i] = x;
			i++;
		}
	}
	free(h_aux);

	h_data[num_segments] = num_elements;

	std::sort(&h_data[0], &h_data[num_segments+1]);

	for (int i = 0; i < num_segments+1; i++)
	{
		printf("%d ", h_data[i]);
	}
}

void vectors_gen(int num_elements, int bits_size_elements) {
	for (int i = 0; i < num_elements; i++)
	{
		printf("%d ", rand() % bits_size_elements);
	}
}

int main(int argc, char** argv) {

	if (argc < 3) {
		printf(
				"Parameters needed: <number of segments> <number of elements>\n\n");
		return 0;
	}

	int number_of_segments = atoi(argv[1]);
	int number_of_elements = atoi(argv[2]);

	srand(time(NULL));
	printf("%d\n", number_of_segments);
	segments_gen(number_of_elements, number_of_segments);
	printf("\n");

	printf("%d\n", number_of_elements);
	vectors_gen(number_of_elements, pow(2, EXP_BITS_SIZE));
	printf("\n");

	return 0;
}

