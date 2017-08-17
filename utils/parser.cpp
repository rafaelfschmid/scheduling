// basic file operations
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <string>
using namespace std;

#ifndef SEGMENTS
#define SEGMENTS 1048576
#endif

int main(int argc, char **argv) {

	std::vector<std::vector<std::vector<double> > > matrix;
	std::vector<double> segments;
	ifstream input(argv[1]);
	ofstream output(argv[2]);

	string line;
	if (input.is_open()) {
		int k = 0;
		getline(input, line);
		while (getline(input, line)) {
			//cout << k++ << "\n";
			std::vector<std::vector<double> > multiple_times;
			int seg = stoi(line);
			//cout << seg << "\n";
			getline(input, line);
			int size = stoi(line);
			//cout << size << "\n";

			std::vector<double> segment;
			for (int i = 0; i < 11; i++) {
				segment.push_back(size);
			}
			multiple_times.push_back(segment);

			while (true) {
				std::vector<double> times;
				times.push_back(seg);
				//cout << seg << "\n";
				for (int i = 0; i < 10; i++) {
					getline(input, line);
					//cout << line << "\n";
					times.push_back(stod(line));
				}
				multiple_times.push_back(times);

				getline(input, line);

				if (seg >= SEGMENTS || seg >= (size/2) ) {
					break;
				}

				getline(input, line);
				seg = stoi(line);
				getline(input, line);
			}
			//cout << "\n";
			matrix.push_back(multiple_times);
		}
		input.close();

		for (int k = 0; k < 11; k++) {
			for (int j = 0; j < matrix[0].size(); j++) {
				output << std::fixed << matrix[0][j][k] << "\t";
//				cout << matrix[0][j][k] << ";";
			}
			output << "\n";
//			cout << "\n";
		}

		for (int i = 1; i < matrix.size(); i++) {
			for (int k = 1; k < 11; k++) {
				for (int j = 0; j < matrix[i].size(); j++) {
					output << std::fixed << matrix[i][j][k] << "\t";
//					cout << matrix[i][j][k] << ";";
				}
				output << "\n";
//				cout << "\n";
			}
		}
		output.close();
	}

	return 0;
}
