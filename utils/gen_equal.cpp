#include <time.h>
#include <algorithm>
#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <vector>

#ifndef EXP_BITS_SIZE
#define EXP_BITS_SIZE 12
#endif

void segments_gen(int num_segments, int size_segment) {
	for (int i = 0; i < num_segments+1; i++)
		printf("%d ", i * size_segment);
}

#ifdef RAND
void vectors_gen(int num_elements, int bits_size_elements, int number_of_segments = 0, int segment_size = 0) {

	for (int i = 0; i < num_elements; i++)
	{
		std::cout << rand() % bits_size_elements;
		std::cout << " ";
	}
}
#elif RANDMINMAX
void vectors_gen(int num_elements, int bits_size_elements, int number_of_segments, int segment_size) {
	std::vector<int> vecTotal;

	for(int j = 0; j < number_of_segments; j++) {
		std::vector<int> vec;

		vec.push_back(0);
		vec.push_back(bits_size_elements-1);

		for (int i = 2; i < segment_size; i++)
		{
			vec.push_back(rand() % bits_size_elements);
		}

		for (int i = 0; i < segment_size; i++)
		{
			int index = rand() % segment_size;
			int aux = vec[i];
			vec[i] = vec[index];
			vec[index] = aux;
		}

		vecTotal.insert(vecTotal.end(), vec.begin(), vec.end());
	}

	for (int i = 0; i < num_elements; i++)
	{
		std::cout << vecTotal[i];
		std::cout << " ";
	}
}
#elif SORTASC
void vectors_gen(int num_elements, int bits_size_elements, int number_of_segments = 0, int segment_size = 0) {
	std::vector<int> vec;

	for (int i = 0; i < num_elements; i++)
	{
		vec.push_back(rand() % bits_size_elements);
	}

	std::sort(vec.begin(), vec.end());

	for (int i = 0; i < num_elements; i++)
	{
		std::cout << vec[i];
		std::cout << " ";
	}
}
#elif SORTDESC
void vectors_gen(int num_elements, int bits_size_elements, int number_of_segments = 0, int segment_size = 0) {
	std::vector<int> vec;

	for (int i = 0; i < num_elements; i++)
	{
		vec.push_back(rand() % bits_size_elements);
	}

	std::sort(vec.begin(), vec.end(), std::greater<int>());

	for (int i = 0; i < num_elements; i++)
	{
		std::cout << vec[i];
		std::cout << " ";
	}
}
#endif

int main(int argc, char** argv) {

	if (argc < 3) {
		printf(
				"Parameters needed: <number of segments> <size of each segment> <exp size number>\n\n");
		return 0;
	}

	int number_of_segments = atoi(argv[1]);
	int size_of_segments = atoi(argv[2]);
	int number_of_elements = number_of_segments*size_of_segments;

	srand(time(NULL));
	printf("%d\n", number_of_segments);
	segments_gen(number_of_segments, size_of_segments);
	printf("\n");

	printf("%d\n", number_of_elements);
	vectors_gen(number_of_elements, pow(2, EXP_BITS_SIZE), number_of_segments, size_of_segments);
	printf("\n");

	return 0;
}

