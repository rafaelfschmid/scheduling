#include <time.h>
#include <algorithm>
#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <vector>

#ifndef EXP_BITS_SIZE
#define EXP_BITS_SIZE 4
#endif

void vector_gen(int num_elements, int bits_size_elements) {

	printf("%d\n", num_elements);
	for (int i = 0; i < num_elements; i++)
	{
		for (int j = 0; j < num_elements; j++)
		{
			std::cout << rand() % bits_size_elements;
			std::cout << " ";
		}
		//std::cout << "\n";
	}
}

int main(int argc, char** argv) {

	if (argc < 1) {
		printf(
				"Parameters needed: <number of elements>\n\n");
		return 0;
	}

	int number_of_elements = atoi(argv[1]);

	srand(time(NULL));
	vector_gen(number_of_elements, pow(2, EXP_BITS_SIZE));
	printf("\n");

	return 0;
}

