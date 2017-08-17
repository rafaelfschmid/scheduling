// basic file operations
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <string>

using namespace std;

#ifndef REPEAT
#define REPEAT 10
#endif

int main(int argc, char **argv) {

	if (argc < 3) {
		printf("Parameters missing: <file input> <file output>\n\n");
		return 0;
	}

	std::vector<std::vector<double> > matrix;
	std::vector<double> segments;
	ifstream input(argv[1]);
	ofstream output(argv[2]);

	string line;
	if (input.is_open()) {
		int k = 0;
		while (getline(input, line)) {
			//cout << k++ << "\n";
			int size = stoi(line);

			std::vector<double> times;
			times.push_back(size);
			//cout << size << "\n";
			for (int i = 0; i < REPEAT; i++) {
				getline(input, line);
				//cout << line << "\n";
				times.push_back(stod(line));
			}
			matrix.push_back(times);

			getline(input, line);
			//cout << "\n";
		}
		input.close();

/*		for (int k = 0; k < 11; k++) {
			for (int j = 0; j < matrix[0].size(); j++) {
				output << std::fixed << matrix[0][j][k] << "\t";
//				cout << matrix[0][j][k] << ";";
			}
			output << "\n";
//			cout << "\n";
		}*/

		for (int k = 0; k < REPEAT; k++) {
			for (int j = 0; j < matrix.size(); j++) {
				output << std::fixed << matrix[j][k] << "\t";
				//cout << matrix[j][k] << ";";
			}
			output << "\n";
			//cout << "\n";
		}
		output.close();
	}

	return 0;
}
