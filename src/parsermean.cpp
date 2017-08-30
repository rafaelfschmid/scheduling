// basic file operations
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <string>

using namespace std;

#ifndef REPEAT
#define REPEAT 20
#endif

int main(int argc, char **argv) {

	if (argc < 3) {
		printf("Parameters missing: <file input> <file output>\n\n");
		return 0;
	}

	std::vector<std::vector<string> > matrix;

	ifstream input(argv[1]);
	ofstream output(argv[2]);

	string line;
	if (input.is_open()) {
		int k = 0;
		while (getline(input, line)) {
			std::vector<string> times;
			times.push_back(line);

			double mean = 0;
			for (int i = 0; i < REPEAT; i++) {
				getline(input, line);

				double value = stod(line);
				mean += value;
			}
			mean /= REPEAT;
			times.push_back(std::to_string(mean));
			matrix.push_back(times);

			getline(input, line);
		}
		input.close();

		for (int k = 0; k < 2; k++) {
			for (int j = 0; j < matrix.size(); j++) {
				output << std::fixed << matrix[j][k] << "\t";
			}
			output << "\n";
		}
		output.close();
	}

	return 0;
}
